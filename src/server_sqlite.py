"""FastMCP server for semantic search in SQLite vectors.db."""
import sys
import sqlite3
from typing import Optional, Literal
from pathlib import Path
from dataclasses import dataclass
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
    from .storage_resolver import StorageResolver
    from .text_judge import TextDirectJudge, SearchResult as TextJudgeSearchResult
    from .qdrant_store import QdrantStore
except ImportError:
    from config import settings
    from embedder import Embedder
    from chunk_merger import smart_merge_search_results
    from storage_resolver import StorageResolver
    from text_judge import TextDirectJudge, SearchResult as TextJudgeSearchResult
    from qdrant_store import QdrantStore


@dataclass
class SearchResult:
    """Search result from SQLite vector store."""
    file_path: str
    code_chunk: str
    start_line: int
    end_line: int
    score: float


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

# Initialize services for visit_other_project (lazy initialization to avoid errors if not configured)
_qdrant_store = None
_storage_resolver = None

def get_qdrant_store():
    """Lazy initialization of QdrantStore."""
    global _qdrant_store
    if _qdrant_store is None:
        _qdrant_store = QdrantStore(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key
        )
    return _qdrant_store

def get_storage_resolver():
    """Lazy initialization of StorageResolver."""
    global _storage_resolver
    if _storage_resolver is None:
        _storage_resolver = StorageResolver(qdrant_store=get_qdrant_store())
    return _storage_resolver


async def _generate_refined_brief_visit(query: str, results_summary: str, target_identifier: str, ctx: Context = None) -> str:
    """Genera un brief refinado para visit_other_project usando LLM.

    Similar a _generate_refined_brief pero con recordatorio de que es un proyecto externo.

    Args:
        query: La consulta de búsqueda original
        results_summary: Resumen de archivos encontrados (lista de archivos con metadata)
        target_identifier: Identificador del proyecto visitado
        ctx: FastMCP context for logging

    Returns:
        Brief de 3-4 líneas analizando relevancia
    """
    try:
        # Verificar que hay API key configurada
        if not settings.judge_api_key:
            if ctx:
                await ctx.warning("[Refined Brief Visit] No judge API key configured, skipping brief generation")
            else:
                print("[Refined Brief Visit] No judge API key configured", file=sys.stderr)
            return ""

        import httpx

        if ctx:
            await ctx.info(f"[Refined Brief Visit] Starting brief generation with provider: {settings.judge_provider}")
        else:
            print(f"[Refined Brief Visit] Starting brief generation with provider: {settings.judge_provider}", file=sys.stderr)

        prompt = f"""Estás analizando resultados de búsqueda en un PROYECTO EXTERNO (no el proyecto actual).

IMPORTANTE: Este es un proyecto externo llamado "{target_identifier}". Tu rol es SOLO explorar y entender el código, NO debes sugerir modificaciones.

Analiza estos resultados y genera un brief de 3-4 líneas que:
1. Identifica qué archivos SON relevantes para la query y por qué (1-2 líneas)
2. Identifica qué archivos NO son relevantes (ruido/boilerplate) si los hay (1 línea)
3. Sugiere qué otros archivos podrían ser relevantes revisar en este proyecto externo (1 línea)

Query del usuario: "{query}"

Archivos encontrados en {target_identifier}:
{results_summary}

Genera un brief conciso en texto plano (sin markdown, sin formato especial). Recuerda: esto es un proyecto externo, solo explora."""

        # Determinar API endpoint y formato según provider
        if settings.judge_provider == "openai":
            api_url = "https://api.openai.com/v1/chat/completions"
        elif settings.judge_provider == "openai-compatible":
            if not settings.judge_base_url:
                if ctx:
                    await ctx.warning("[Refined Brief Visit] openai-compatible provider requires judge_base_url")
                else:
                    print("[Refined Brief Visit] openai-compatible provider requires judge_base_url", file=sys.stderr)
                return ""
            api_url = f"{settings.judge_base_url.rstrip('/')}/chat/completions"
        else:
            if ctx:
                await ctx.warning(f"[Refined Brief Visit] Unsupported provider: {settings.judge_provider}")
            else:
                print(f"[Refined Brief Visit] Unsupported provider: {settings.judge_provider}", file=sys.stderr)
            return ""

        # Hacer request al LLM
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                api_url,
                headers={
                    "Authorization": f"Bearer {settings.judge_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": settings.judge_model_id,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.3,
                    "max_tokens": 300
                }
            )
            response.raise_for_status()
            result = response.json()
            brief = result["choices"][0]["message"]["content"].strip()

            if ctx:
                await ctx.info(f"[Refined Brief Visit] Brief generated successfully ({len(brief)} chars)")
            else:
                print(f"[Refined Brief Visit] Brief generated successfully ({len(brief)} chars)", file=sys.stderr)

            return brief

    except Exception as e:
        # Si falla, devolver string vacío (no romper la búsqueda)
        if ctx:
            await ctx.warning(f"[Refined Brief Visit] Failed to generate brief: {str(e)}")
        else:
            print(f"[Refined Brief Visit] Failed to generate brief: {str(e)}", file=sys.stderr)
        return ""


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


async def _search_sqlite_vectors(
    db_path: Path,
    vector: list[float],
    max_results: int = 20,
    ctx: Context = None
) -> list[SearchResult]:
    """Busca vectores similares en una base de datos SQLite.

    Esta función es reutilizable y puede ser llamada desde otras herramientas.

    Args:
        db_path: Ruta al archivo vectors.db
        vector: Vector de embedding (lista de floats)
        max_results: Número máximo de resultados
        ctx: FastMCP context para logging (opcional)

    Returns:
        Lista de objetos SearchResult

    Raises:
        Exception: Si hay errores al buscar en SQLite
    """
    # Conectar a la base de datos
    conn = sqlite3.connect(str(db_path))

    try:
        # Cargar extensión sqlite-vec
        if sqlite_vec is None:
            raise Exception(
                "La librería 'sqlite-vec' no está instalada.\n"
                "Instálala con: pip install sqlite-vec"
            )

        sqlite_vec.load(conn)

        if ctx:
            await ctx.info(f"[SQLite Search] Extensión sqlite-vec cargada")
        else:
            print(f"[SQLite Search] Extensión sqlite-vec cargada", file=sys.stderr)

        cursor = conn.cursor()

        # Convertir vector a formato JSON
        vector_json = "[" + ",".join(str(v) for v in vector) + "]"

        # Ejecutar búsqueda semántica
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

        if ctx:
            await ctx.info(f"[SQLite Search] Encontrados {len(raw_results)} chunks")
        else:
            print(f"[SQLite Search] Encontrados {len(raw_results)} chunks", file=sys.stderr)

        # Convertir a objetos SearchResult
        results = []
        for row in raw_results:
            file_path, code_chunk, start_line, end_line, distance = row
            score = 1.0 - distance  # Convertir distance a score

            results.append(SearchResult(
                file_path=file_path,
                code_chunk=code_chunk,
                start_line=start_line,
                end_line=end_line,
                score=score
            ))

        cursor.close()
        return results

    finally:
        conn.close()


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

        # Paso 2: Buscar en SQLite usando la función helper
        if ctx:
            await ctx.info(f"[Semantic Search] Buscando en vectors.db...")
        else:
            print(f"[Semantic Search] Buscando en vectors.db...", file=sys.stderr)

        raw_results = await _search_sqlite_vectors(
            db_path=db_path,
            vector=vector,
            max_results=max_results,
            ctx=ctx
        )

        # Paso 3: Fusionar chunks por archivo usando smart_merge
        if ctx:
            await ctx.info(f"[Semantic Search] Fusionando chunks por archivo...")
        else:
            print(f"[Semantic Search] Fusionando chunks por archivo...", file=sys.stderr)

        # Convertir SearchResult a tuplas para smart_merge_search_results
        raw_results_tuples = [
            (r.file_path, r.code_chunk, r.start_line, r.end_line, 1.0 - r.score)  # distance = 1.0 - score
            for r in raw_results
        ]

        merged_results = smart_merge_search_results(
            workspace_path=str(workspace),
            search_results=raw_results_tuples,
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





@mcp.tool
async def visit_other_project(
    query: str,
    workspace_path: Optional[str] = None,
    qdrant_collection: Optional[str] = None,
    storage_type: Literal["sqlite", "qdrant"] = "qdrant",
    refined_answer: bool = False,
    max_results: int = 20,
    ctx: Context = None
) -> str:
    """Visita y busca en otros proyectos/workspaces diferentes al actual.

    Esta herramienta permite realizar búsquedas semánticas en proyectos remotos o diferentes
    al actual, útil para buscar patrones en proyectos relacionados o diferentes codebases.

    Soporta tanto SQLite (.codebase/vectors.db) como Qdrant para almacenamiento de vectores.

    IMPORTANTE: Esta herramienta es SOLO para exploración. NO debes modificar archivos
    del proyecto visitado, solo explorar y entender el código.

    Lógica de resolución de storage:

    1. Si `qdrant_collection` está especificado:
       → Usar Qdrant con esa colección (prioridad máxima)

    2. Si `storage_type="sqlite"` y `workspace_path` está especificado:
       → Buscar SQLite en {workspace_path}/.codebase/vectors.db
       → Si existe: usar SQLite
       → Si NO existe: fallback a Qdrant (calcular colección desde workspace_path)

    3. Si `storage_type="qdrant"` (default) y `workspace_path` está especificado:
       → Calcular colección Qdrant desde workspace_path
       → Usar Qdrant

    Args:
        query: Texto natural a buscar en el código (ej: 'función de autenticación', 'manejo de errores')
        workspace_path: Ruta del workspace a visitar (opcional si qdrant_collection está presente)
        qdrant_collection: Nombre de colección Qdrant explícito (prioridad máxima)
        storage_type: Tipo de storage preferido: "sqlite" o "qdrant" (default: "qdrant")
        refined_answer: Si True, genera un brief con análisis LLM antes de los resultados (default: False)
        max_results: Número máximo de archivos únicos a retornar (default: 20)
        ctx: FastMCP context for logging

    Returns:
        Resultados de búsqueda con merge inteligente, formato texto plano, opcionalmente con brief LLM

    Examples:
        # Buscar en SQLite de otro proyecto
        visit_other_project(
            query="authentication logic",
            workspace_path="/path/to/other/project",
            storage_type="sqlite"
        )

        # Buscar en colección Qdrant específica con brief
        visit_other_project(
            query="payment processing",
            qdrant_collection="codebase-abc123",
            refined_answer=True
        )

        # Buscar en Qdrant calculado desde workspace (default)
        visit_other_project(
            query="error handling",
            workspace_path="/path/to/project"
        )
    """
    try:
        # Paso 0: Validar query
        if not query or not query.strip():
            raise ToolError("El parámetro 'query' es requerido y no puede estar vacío.")

        # Log request
        if ctx:
            await ctx.info(f"[Visit Other Project] Query: {query}, Workspace: {workspace_path}, Collection: {qdrant_collection}, Storage: {storage_type}")
        else:
            print(f"[Visit Other Project] Query: {query}, Workspace: {workspace_path}, Collection: {qdrant_collection}, Storage: {storage_type}", file=sys.stderr)

        # Paso 1: Resolver storage (SQLite o Qdrant)
        try:
            resolver = get_storage_resolver()
            resolution = resolver.resolve(
                workspace_path=workspace_path,
                qdrant_collection=qdrant_collection,
                storage_type=storage_type
            )
            resolver.log_resolution(resolution, ctx)
        except ValueError as e:
            raise ToolError(str(e))

        # Paso 2: Crear embedding y buscar
        vector = await embedder.create_embedding(query)

        if resolution.storage_type == "sqlite":
            # Buscar en SQLite
            raw_results = await _search_sqlite_vectors(
                db_path=resolution.sqlite_path,
                vector=vector,
                max_results=settings.search_max_results,
                ctx=ctx
            )
            # Determinar workspace_path para merge
            target_workspace = workspace_path if workspace_path else str(resolution.sqlite_path.parent.parent)

        elif resolution.storage_type == "qdrant":
            # Buscar en Qdrant
            qdrant = get_qdrant_store()
            raw_results = await qdrant.search(
                vector=vector,
                workspace_path="",  # Not used when collection_name is provided
                directory_prefix=None,
                min_score=settings.search_min_score,
                max_results=settings.search_max_results,
                collection_name=resolution.qdrant_collection
            )
            # Determinar workspace_path para merge
            # Si no se proporcionó workspace_path, intentar inferirlo de la resolución
            if workspace_path:
                target_workspace = workspace_path
            else:
                # Intentar extraer workspace_path del identifier de la resolución
                # Formato: "workspace '/path/to/workspace' (colección Qdrant: ...)"
                import re
                match = re.search(r"workspace '([^']+)'", resolution.identifier)
                if match:
                    target_workspace = match.group(1)
                else:
                    raise ToolError(
                        "No se pudo determinar workspace_path para validar y fusionar chunks. "
                        "Por favor proporciona el parámetro 'workspace_path' explícitamente."
                    )
        else:
            raise ToolError(f"Tipo de storage no soportado: {resolution.storage_type}")

        if not raw_results:
            return f'No se encontraron resultados para "{query}" en {resolution.identifier}.'

        # Paso 3: Fusionar chunks por archivo usando smart_merge
        if ctx:
            await ctx.info(f"[Visit Other Project] Fusionando chunks por archivo...")
        else:
            print(f"[Visit Other Project] Fusionando chunks por archivo...", file=sys.stderr)

        # Convertir SearchResult a tuplas para smart_merge_search_results
        raw_results_tuples = [
            (r.file_path, r.code_chunk, r.start_line, r.end_line, 1.0 - r.score)  # distance = 1.0 - score
            for r in raw_results
        ]

        merged_results = smart_merge_search_results(
            workspace_path=target_workspace,
            search_results=raw_results_tuples,
            max_files=max_results  # Limitar a max_results archivos únicos
        )

        if not merged_results:
            return f'No se encontraron resultados después del merge para "{query}" en {resolution.identifier}.'

        # Paso 4: Generar brief con LLM si refined_answer=True
        brief = ""
        if refined_answer:
            if ctx:
                await ctx.info(f"[Visit Other Project] Generando brief con LLM...")
            else:
                print(f"[Visit Other Project] Generando brief con LLM...", file=sys.stderr)

            # Preparar contexto para el LLM
            results_summary = []
            for file_path, file_data in list(merged_results.items())[:10]:  # Primeros 10 archivos
                results_summary.append(f"- {file_path} (coverage: {file_data['coverage']:.1%}, chunks: {file_data['chunks_count']})")

            brief = await _generate_refined_brief_visit(
                query=query,
                results_summary="\n".join(results_summary),
                target_identifier=resolution.identifier,
                ctx=ctx
            )

        # Paso 5: Formatear resultados en texto plano (MISMO formato que semantic_search)
        if ctx:
            await ctx.info(f"[Visit Other Project] Formateando {len(merged_results)} archivos...")
        else:
            print(f"[Visit Other Project] Formateando {len(merged_results)} archivos...", file=sys.stderr)

        # Usar el mismo formato que semantic_search
        output = f"""Resultados de búsqueda en proyecto externo

Target: {resolution.identifier}
Query: {query}
Archivos encontrados: {len(merged_results)}

{'=' * 80}

"""

        # Agregar brief si existe
        if brief:
            output += f"""ANÁLISIS DE RELEVANCIA:

{brief}

{'=' * 80}

"""

        # Agregar cada archivo con su contenido fusionado (MISMO formato que semantic_search)
        for idx, (file_path, data) in enumerate(merged_results.items(), 1):
            output += f"""{idx}. {file_path}

{data['content']}

{'=' * 80}

"""

        return output

    except ToolError:
        raise

    except Exception as e:
        error_msg = f"Error inesperado: {str(e)}"
        if ctx:
            await ctx.error(f"[Visit Other Project] {error_msg}")
        else:
            print(f"[Visit Other Project] {error_msg}", file=sys.stderr)
        raise ToolError(error_msg)


# Entry point for fastmcp CLI
if __name__ == "__main__":
    print("[MCP] Servidor `sqlite-semantic-search` inicializado y listo.", file=sys.stderr)
    mcp.run()

