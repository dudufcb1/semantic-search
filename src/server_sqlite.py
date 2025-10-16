"""FastMCP server for semantic search in SQLite vectors.db."""
import sys
import sqlite3
from typing import Optional
from pathlib import Path
from fastmcp import FastMCP, Context
from fastmcp.exceptions import ToolError

# Import sqlite-vec for vector search support
try:
    import sqlite_vec
except ImportError:
    sqlite_vec = None

try:
    from .config import settings
    from .embedder import Embedder
    from .chunk_merger import smart_merge_search_results
except ImportError:
    from config import settings
    from embedder import Embedder
    from chunk_merger import smart_merge_search_results


# Initialize FastMCP server
mcp = FastMCP(
    name="sqlite-semantic-search",
    version="0.1.0"
)

# Initialize embedder service
embedder = Embedder(
    provider=settings.embedder_provider,
    api_key=settings.embedder_api_key,
    model_id=settings.embedder_model_id,
    base_url=settings.embedder_base_url
)


def _format_search_results(query: str, workspace_path: str, merged_results: dict) -> str:
    """Formatea los resultados de búsqueda semántica fusionados.

    Args:
        query: La consulta de búsqueda
        workspace_path: Ruta del workspace
        merged_results: Dict con resultados fusionados por archivo

    Returns:
        String formateado con los resultados (path: chunk)
    """
    if not merged_results:
        return f"""# Resultados de búsqueda semántica

**Workspace:** `{workspace_path}`
**Query:** `{query}`

No se encontraron resultados.
"""

    output = f"""# Resultados de búsqueda semántica

**Workspace:** `{workspace_path}`
**Query:** `{query}`
**Archivos encontrados:** {len(merged_results)}

---

"""

    # Agregar cada archivo con su contenido fusionado
    for idx, (file_path, data) in enumerate(merged_results.items(), 1):
        output += f"""## {idx}. `{file_path}`

```
{data['content']}
```

---

"""

    return output


@mcp.tool
async def semantic_search(
    workspace_path: str,
    query: str,
    max_results: int = 20,
    ctx: Context = None
) -> str:
    """Realiza búsqueda semántica en la base de datos SQLite del workspace (.codebase/vectors.db).

    Esta herramienta:
    1. Crea un embedding de la query usando el mismo modelo que indexó el código
    2. Busca los vectores más similares en vectors.db usando sqlite-vec
    3. Devuelve los resultados con score y código correspondiente

    Args:
        workspace_path: Ruta absoluta del workspace (el servidor inferirá .codebase/vectors.db)
        query: Texto natural a buscar en el código (ej: 'función de autenticación')
        max_results: Número máximo de resultados a devolver (default: 20)
        ctx: FastMCP context for logging

    Returns:
        Resultados formateados con score y código

    Raises:
        ToolError: Si hay errores de validación o ejecución
    """
    try:
        # Validar parámetros
        if not workspace_path or not workspace_path.strip():
            raise ToolError("El parámetro 'workspace_path' es requerido y no puede estar vacío.")

        if not query or not query.strip():
            raise ToolError("El parámetro 'query' es requerido y no puede estar vacío.")

        # Construir path a la base de datos: workspace_path/.codebase/vectors.db
        workspace = Path(workspace_path.strip()).resolve()
        db_path = workspace / ".codebase" / "vectors.db"

        # Verificar que el workspace existe
        if not workspace.exists():
            raise ToolError(f"El workspace no existe: {workspace}")

        if not workspace.is_dir():
            raise ToolError(f"El workspace no es un directorio: {workspace}")

        # Verificar que la base de datos existe
        if not db_path.exists():
            raise ToolError(f"La base de datos no existe: {db_path}\nAsegúrate de que el workspace esté indexado con .codebase/vectors.db")

        if not db_path.is_file():
            raise ToolError(f"La ruta de la base de datos no es un archivo: {db_path}")

        # Log de inicio
        if ctx:
            await ctx.info(f"[Semantic Search] Workspace: {workspace_path}, Query: {query}")
        else:
            print(f"[Semantic Search] Workspace: {workspace_path}, Query: {query}", file=sys.stderr)

        # Paso 1: Crear embedding de la query
        if ctx:
            await ctx.info(f"[Semantic Search] Creando embedding para query...")
        else:
            print(f"[Semantic Search] Creando embedding para query...", file=sys.stderr)

        vector = await embedder.create_embedding(query)

        if ctx:
            await ctx.info(f"[Semantic Search] Embedding creado: {len(vector)} dimensiones")
        else:
            print(f"[Semantic Search] Embedding creado: {len(vector)} dimensiones", file=sys.stderr)

        # Paso 2: Buscar en SQLite usando vec_distance_cosine
        if ctx:
            await ctx.info(f"[Semantic Search] Buscando en vectors.db...")
        else:
            print(f"[Semantic Search] Buscando en vectors.db...", file=sys.stderr)

        # Conectar a la base de datos
        conn = sqlite3.connect(str(db_path))

        try:
            # Cargar extensión sqlite-vec usando la librería Python
            if sqlite_vec is None:
                raise ToolError(
                    "La librería 'sqlite-vec' no está instalada.\n"
                    "Instálala con: pip install sqlite-vec\n"
                    "O con uv: uv pip install sqlite-vec"
                )

            # Cargar la extensión en la conexión
            sqlite_vec.load(conn)

            if ctx:
                await ctx.info(f"[Semantic Search] Extensión sqlite-vec cargada correctamente")
            else:
                print(f"[Semantic Search] Extensión sqlite-vec cargada correctamente", file=sys.stderr)

            cursor = conn.cursor()

            # Convertir vector a formato JSON para la consulta
            vector_json = "[" + ",".join(str(v) for v in vector) + "]"

            # Ejecutar búsqueda semántica usando MATCH (sqlite-vec syntax)
            # Nota: sqlite-vec usa MATCH en lugar de vec_distance_cosine
            sql_query = """
                SELECT
                    file_path,
                    code_chunk,
                    start_line,
                    end_line,
                    distance
                FROM code_vectors
                WHERE embedding MATCH ?
                ORDER BY distance ASC
                LIMIT ?
            """

            cursor.execute(sql_query, (vector_json, max_results))
            raw_results = cursor.fetchall()

            # Log de resultados crudos
            if ctx:
                await ctx.info(f"[Semantic Search] Búsqueda exitosa: {len(raw_results)} chunks encontrados")
            else:
                print(f"[Semantic Search] Búsqueda exitosa: {len(raw_results)} chunks encontrados", file=sys.stderr)

            # Paso 3: Fusionar chunks por archivo usando smart_merge
            if ctx:
                await ctx.info(f"[Semantic Search] Fusionando chunks por archivo...")
            else:
                print(f"[Semantic Search] Fusionando chunks por archivo...", file=sys.stderr)

            merged_results = smart_merge_search_results(
                workspace_path=str(workspace),
                search_results=raw_results,
                max_files=max_results  # Limitar a max_results archivos únicos
            )

            # Log de archivos únicos
            if ctx:
                await ctx.info(f"[Semantic Search] Procesados {len(merged_results)} archivos únicos")
            else:
                print(f"[Semantic Search] Procesados {len(merged_results)} archivos únicos", file=sys.stderr)

            # Formatear y retornar resultados
            return _format_search_results(query, str(workspace), merged_results)

        finally:
            # Cerrar cursor y conexión
            cursor.close()
            conn.close()

    except sqlite3.Error as e:
        error_msg = f"Error de SQLite: {str(e)}"
        if ctx:
            await ctx.error(f"[Semantic Search] {error_msg}")
        else:
            print(f"[Semantic Search] {error_msg}", file=sys.stderr)
        raise ToolError(error_msg)

    except ToolError:
        raise

    except Exception as e:
        error_msg = f"Error inesperado: {str(e)}"
        if ctx:
            await ctx.error(f"[Semantic Search] {error_msg}")
        else:
            print(f"[Semantic Search] {error_msg}", file=sys.stderr)
        raise ToolError(error_msg)


# Entry point for fastmcp CLI
if __name__ == "__main__":
    print("[MCP] Servidor `sqlite-semantic-search` inicializado y listo.", file=sys.stderr)
    mcp.run()

