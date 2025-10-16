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


async def _generate_refined_brief(query: str, merged_results: dict, ctx: Context = None) -> str:
    """Genera un brief refinado usando LLM para analizar los resultados.

    Usa la configuración del judge (judge_provider, judge_api_key, judge_base_url, judge_model_id).

    Args:
        query: La consulta de búsqueda original
        merged_results: Dict con resultados fusionados por archivo
        ctx: FastMCP context for logging

    Returns:
        Brief de 3-4 líneas analizando relevancia y gaps
    """
    try:
        # Verificar que hay API key configurada
        if not settings.judge_api_key:
            if ctx:
                await ctx.warning("[Refined Brief] No judge API key configured, skipping brief generation")
            else:
                print("[Refined Brief] No judge API key configured", file=sys.stderr)
            return ""

        import httpx

        if ctx:
            await ctx.info(f"[Refined Brief] Starting brief generation with provider: {settings.judge_provider}")
        else:
            print(f"[Refined Brief] Starting brief generation with provider: {settings.judge_provider}", file=sys.stderr)

        # Construir prompt con los resultados
        files_summary = []
        for idx, (file_path, data) in enumerate(merged_results.items(), 1):
            # Tomar primeras 100 líneas de cada archivo para el análisis
            content_lines = data['content'].split('\n')[:100]
            content_preview = '\n'.join(content_lines)
            files_summary.append(f"{idx}. {file_path}\n{content_preview}\n")

        prompt = f"""Analiza estos resultados de búsqueda semántica y genera un brief de 3-4 líneas que:

1. Identifica qué archivos SON relevantes para la query y por qué (1-2 líneas)
2. Identifica qué archivos NO son relevantes (ruido/boilerplate) y por qué (1 línea)
3. Detecta imports/referencias a archivos NO presentes en los resultados y sugiere revisarlos si son relevantes (1 línea)

Query del usuario: "{query}"

Resultados encontrados:
{''.join(files_summary)}

Genera un brief conciso en texto plano (sin markdown, sin formato especial). Enfócate primero en lo relevante, luego en lo no relevante, y finalmente en gaps."""

        # Determinar API endpoint y formato según provider
        if settings.judge_provider == "openai":
            # OpenAI API
            api_url = "https://api.openai.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {settings.judge_api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": settings.judge_model_id,
                "max_tokens": 500,
                "temperature": 0.0,
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            }
        elif settings.judge_provider == "openai-compatible":
            # OpenAI-compatible API (usa base_url custom)
            base_url = settings.judge_base_url.rstrip("/")
            api_url = f"{base_url}/chat/completions"
            headers = {
                "Authorization": f"Bearer {settings.judge_api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": settings.judge_model_id,
                "max_tokens": 500,
                "temperature": 0.0,
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            }
        else:
            # Unsupported provider
            if ctx:
                await ctx.warning(f"[Refined Brief] Unsupported judge provider: {settings.judge_provider}")
            return ""

        # Hacer request HTTP
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                api_url,
                headers=headers,
                json=payload
            )
            response.raise_for_status()

            data = response.json()

            # Extraer texto según formato de respuesta
            if settings.judge_provider in ["openai", "openai-compatible"]:
                brief = data["choices"][0]["message"]["content"].strip()
            else:
                brief = ""

        if ctx:
            await ctx.info(f"[Refined Brief] Generated brief: {len(brief)} chars")

        return brief

    except Exception as e:
        if ctx:
            await ctx.error(f"[Refined Brief] Error generating brief: {e}")
        else:
            print(f"[Refined Brief] Error generating brief: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
        # Si falla, retornar string vacío (no bloquear la búsqueda)
        return ""


def _format_search_results(query: str, workspace_path: str, merged_results: dict, refined_brief: str = "") -> str:
    """Formatea los resultados de búsqueda semántica fusionados.

    Args:
        query: La consulta de búsqueda
        workspace_path: Ruta del workspace
        merged_results: Dict con resultados fusionados por archivo
        refined_brief: Brief opcional generado por LLM

    Returns:
        String formateado con los resultados (TEXTO PLANO, sin markdown)
    """
    if not merged_results:
        return f"""Resultados de búsqueda semántica

Workspace: {workspace_path}
Query: {query}

No se encontraron resultados.
"""

    output = f"""Resultados de búsqueda semántica

Workspace: {workspace_path}
Query: {query}
Archivos encontrados: {len(merged_results)}

{'=' * 80}

"""

    # Agregar brief refinado si existe
    if refined_brief:
        output += f"""ANÁLISIS DE RELEVANCIA:

{refined_brief}

{'=' * 80}

"""

    # Agregar cada archivo con su contenido fusionado
    for idx, (file_path, data) in enumerate(merged_results.items(), 1):
        output += f"""{idx}. {file_path}

{data['content']}

{'=' * 80}

"""

    return output


@mcp.tool
async def semantic_search(
    workspace_path: str,
    query: str,
    max_results: int = 20,
    refined_answer: bool = False,
    ctx: Context = None
) -> str:
    """Realiza búsqueda semántica en la base de datos SQLite del workspace (.codebase/vectors.db).

    Esta herramienta:
    1. Crea un embedding de la query usando el mismo modelo que indexó el código
    2. Busca los vectores más similares en vectors.db usando sqlite-vec
    3. Devuelve los resultados con score y código correspondiente
    4. Opcionalmente genera un brief con análisis de relevancia usando LLM

    Args:
        workspace_path: Ruta absoluta del workspace (el servidor inferirá .codebase/vectors.db)
        query: Pregunta en lenguaje natural sobre QUÉ buscas o QUÉ hace el código.

               Ejemplos efectivos:
                 • "qué procesos automáticos existen"
                 • "cómo se procesan los datos de usuarios"
                 • "dónde se almacena la información"
                 • "implementación del sistema de notificaciones"

               Tips:
                 • Sé específico: "cómo se validan formularios" > "validación"
                 • Usa nombres exactos si los conoces: "UserService", "sendEmail"
                 • Pregunta por funcionalidad, no por archivos
                 • Combina conceptos: "autenticación y permisos de usuarios"

        max_results: Número máximo de resultados a devolver (default: 20)
        refined_answer: Si True, genera un brief con análisis de relevancia usando LLM.
                       El brief identifica archivos relevantes, ruido/boilerplate, y gaps
                       en imports/referencias. Se inserta antes de los resultados. (default: False)
        ctx: FastMCP context for logging

    Returns:
        Resultados formateados con score y código (opcionalmente con brief de análisis)

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

            # Generar brief refinado si se solicita
            refined_brief = ""
            if refined_answer and merged_results:
                if ctx:
                    await ctx.info(f"[Semantic Search] Generando brief refinado con LLM...")
                else:
                    print(f"[Semantic Search] Generando brief refinado con LLM...", file=sys.stderr)

                refined_brief = await _generate_refined_brief(query, merged_results, ctx)

            # Formatear y retornar resultados
            return _format_search_results(query, str(workspace), merged_results, refined_brief)

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

