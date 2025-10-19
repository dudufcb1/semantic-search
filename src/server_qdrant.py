"""FastMCP server for semantic search in Qdrant (reads state.json)."""
import sys
import json
import math
from typing import Optional, Literal
from pathlib import Path
from dataclasses import dataclass
from fastmcp import FastMCP, Context
from fastmcp.exceptions import ToolError

# Add src directory to sys.path to enable absolute imports
_current_file = Path(__file__).resolve()
_src_dir = _current_file.parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from config import settings
from embedder import Embedder
from chunk_merger import smart_merge_search_results
from storage_resolver import StorageResolver
from text_judge import TextDirectJudge, SearchResult as TextJudgeSearchResult
from qdrant_store import QdrantStore
from voyage_reranker import VoyageReranker


@dataclass
class SearchResult:
    """Search result from Qdrant vector store."""
    file_path: str
    code_chunk: str
    start_line: int
    end_line: int
    score: float


# Initialize FastMCP server
mcp = FastMCP(
    name="qdrant-semantic-search",
    version="0.1.0"
)

# Lazy initialization for all services to ensure env vars are loaded
_embedder = None
_qdrant_store = None
_storage_resolver = None
_voyage_reranker = None

def get_embedder():
    """Lazy initialization of Embedder."""
    global _embedder
    if _embedder is None:
        _embedder = Embedder(
            provider=settings.embedder_provider,
            api_key=settings.embedder_api_key,
            model_id=settings.embedder_model_id,
            base_url=settings.embedder_base_url,
            dimension=settings.embedder_dimension
        )
    return _embedder

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

def get_voyage_reranker():
    """Lazy initialization of VoyageReranker."""
    global _voyage_reranker
    if _voyage_reranker is None:
        if not settings.voyage_api_key:
            raise ValueError(
                "Voyage API key is required when native_rerank is enabled.\n"
                "Set MCP_CODEBASE_VOYAGE_API_KEY in your environment."
            )
        if not settings.voyage_rerank_model:
            raise ValueError(
                "Voyage rerank model is required when native_rerank is enabled.\n"
                "Set MCP_CODEBASE_VOYAGE_RERANK_MODEL in your environment."
            )
        _voyage_reranker = VoyageReranker(
            api_key=settings.voyage_api_key,
            model=settings.voyage_rerank_model,
            top_k=settings.voyage_top_k,
            truncation=settings.voyage_truncation
        )
    return _voyage_reranker


def _load_state_json(workspace_path: str) -> dict:
    """Lee .codebase/state.json para obtener qdrantCollection.

    Args:
        workspace_path: Ruta del workspace

    Returns:
        Dict con state.json content (workspacePath, qdrantCollection, etc.)

    Raises:
        ToolError: Si no existe state.json o no tiene qdrantCollection
    """
    workspace = Path(workspace_path)
    state_file = workspace / ".codebase" / "state.json"

    if not state_file.exists():
        raise ToolError(
            f"No se encontró {state_file}.\n\n"
            "Asegúrate de que el workspace ha sido indexado con codebase-index CLI.\n"
            "El archivo state.json debe contener la colección Qdrant."
        )

    try:
        with open(state_file, "r") as f:
            state = json.load(f)
    except json.JSONDecodeError as e:
        raise ToolError(f"Error al leer {state_file}: {e}")

    if "qdrantCollection" not in state:
        raise ToolError(
            f"El archivo {state_file} no contiene 'qdrantCollection'.\n"
            "Formato esperado:\n"
            "{\n"
            '  "workspacePath": "/path/to/workspace",\n'
            '  "qdrantCollection": "codebase-abc123...",\n'
            '  "createdAt": "...",\n'
            '  "updatedAt": "..."\n'
            "}"
        )

    return state


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


def _format_search_results(
    query: str,
    workspace_path: str,
    merged_results: dict,
    refined_brief: str = "",
    pagination: Optional[dict] = None
) -> str:
    """Formatea los resultados de búsqueda semántica fusionados.

    Args:
        query: La consulta de búsqueda
        workspace_path: Ruta del workspace
        merged_results: Dict con resultados fusionados por archivo (paginados)
        refined_brief: Brief opcional generado por LLM
        pagination: Información de paginación (total, página actual, tamaño)

    Returns:
        String formateado con los resultados (TEXTO PLANO, sin markdown)
    """
    total_available = pagination.get("total_files", len(merged_results)) if pagination else len(merged_results)
    current_page = pagination.get("page") if pagination else None
    total_pages = pagination.get("total_pages") if pagination else None
    page_size = pagination.get("page_size") if pagination else None

    if not merged_results:
        lines = [
            "Resultados de búsqueda semántica",
            "",
            f"Workspace: {workspace_path}",
            f"Query: {query}",
            f"Archivos encontrados (total): {total_available}",
        ]
        if pagination:
            lines.append(
                f"Paginación: página {current_page} de {max(total_pages, 1)} (tamaño {page_size})"
            )
        lines.append("")
        lines.append("No se encontraron resultados.")
        lines.append("")
        return "\n".join(lines)

    header_lines = [
        "Resultados de búsqueda semántica",
        "",
        f"Workspace: {workspace_path}",
        f"Query: {query}",
        f"Archivos encontrados (total): {total_available}",
    ]

    if pagination:
        header_lines.append(
            f"Mostrando {len(merged_results)} resultados (página {current_page} de {max(total_pages, 1)})"
        )
        header_lines.append(f"Tamaño de página: {page_size}")

    header_lines.extend(["", "{}".format('=' * 80), ""])
    output = "\n".join(header_lines)

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
    query: str,
    qdrant_collection: str,
    max_results: int = 20,
    refined_answer: bool = False,
    page: Optional[int] = None,
    page_size: Optional[int] = None,
    ctx: Context = None
) -> str:
    """Realiza búsqueda semántica en Qdrant.

    Esta herramienta:
    1. Crea un embedding de la query usando el mismo modelo que indexó el código
    2. Busca los vectores más similares en Qdrant
    3. Devuelve los resultados con score y código correspondiente
    4. Opcionalmente genera un brief con análisis de relevancia usando LLM

    Args:
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

        qdrant_collection: Nombre de la colección Qdrant (ej: "codebase-abc123"). REQUERIDO.
        max_results: Número máximo de resultados a devolver (default: 20)
        refined_answer: Si True, genera un brief con análisis de relevancia usando LLM.
                       El brief identifica archivos relevantes, ruido/boilerplate, y gaps
                       en imports/referencias. Se inserta antes de los resultados. (default: False)
        page: Número de página (1-based). Si no se envía, se usa 1 por defecto.
        page_size: Resultados por página. Default 10 (limitado por max_results y configuración).
        ctx: FastMCP context for logging

    Returns:
        Resultados formateados con score y código (opcionalmente con brief de análisis)

    Raises:
        ToolError: Si hay errores de validación o ejecución
    """
    try:
        # Validar parámetros
        if not query or not query.strip():
            raise ToolError("El parámetro 'query' es requerido y no puede estar vacío.")

        if not qdrant_collection or not qdrant_collection.strip():
            raise ToolError("El parámetro 'qdrant_collection' es requerido y no puede estar vacío.")

        collection_name = qdrant_collection.strip()

        # Configurar paginación (default página 1, tamaño 10)
        page = 1 if page is None else page
        page_size = 10 if page_size is None else page_size

        if page < 1:
            raise ToolError("El parámetro 'page' debe ser >= 1.")
        if page_size < 1:
            raise ToolError("El parámetro 'page_size' debe ser >= 1.")

        max_total_results = min(settings.search_max_results, max_results)
        if max_total_results < 1:
            raise ToolError("El parámetro 'max_results' debe ser >= 1.")

        effective_page_size = min(page_size, max_total_results)
        requested_limit = page * effective_page_size
        total_limit = min(max_total_results, requested_limit)

        if ctx:
            await ctx.info(
                f"[Semantic Search] Colección: {collection_name}, Query: {query}, "
                f"page={page}, page_size={effective_page_size}, limit={total_limit}"
            )
        else:
            print(
                f"[Semantic Search] Colección: {collection_name}, Query: {query}, "
                f"page={page}, page_size={effective_page_size}, limit={total_limit}",
                file=sys.stderr
            )

        # Paso 1: Crear embedding de la query
        if ctx:
            await ctx.info(f"[Semantic Search] Creando embedding para query...")
        else:
            print(f"[Semantic Search] Creando embedding para query...", file=sys.stderr)

        embedder = get_embedder()
        vector = await embedder.create_embedding(query)

        if ctx:
            await ctx.info(f"[Semantic Search] Embedding creado: {len(vector)} dimensiones")
        else:
            print(f"[Semantic Search] Embedding creado: {len(vector)} dimensiones", file=sys.stderr)

        # Paso 2: Buscar en Qdrant
        if ctx:
            await ctx.info(f"[Semantic Search] Buscando en Qdrant (colección: {collection_name})...")
        else:
            print(f"[Semantic Search] Buscando en Qdrant (colección: {collection_name})...", file=sys.stderr)

        qdrant = get_qdrant_store()

        raw_results_qdrant = await qdrant.search(
            vector=vector,
            workspace_path="",  # No usado cuando collection_name está presente
            directory_prefix=None,
            min_score=settings.search_min_score,
            max_results=total_limit,
            collection_name=collection_name
        )

        # Convertir resultados de Qdrant a SearchResult
        raw_results = [
            SearchResult(
                file_path=r.file_path,
                code_chunk=r.code_chunk,
                start_line=r.start_line,
                end_line=r.end_line,
                score=r.score
            )
            for r in raw_results_qdrant
        ]

        # Paso 2.5: RERANK con Voyage AI si está habilitado
        # Si native_rerank=True, usar Voyage en lugar de LLM judge
        if settings.native_rerank and raw_results:
            if ctx:
                await ctx.info(f"[Semantic Search] Reranking con Voyage AI (modelo: {settings.voyage_rerank_model})...")
            else:
                print(f"[Semantic Search] Reranking con Voyage AI (modelo: {settings.voyage_rerank_model})...", file=sys.stderr)

            try:
                reranker = get_voyage_reranker()
                raw_results = await reranker.rerank(
                    query=query,
                    results=raw_results,
                    top_k=max_results  # Limitar a max_results después del rerank
                )

                if ctx:
                    await ctx.info(f"[Semantic Search] Reranking completado: {len(raw_results)} resultados")
                else:
                    print(f"[Semantic Search] Reranking completado: {len(raw_results)} resultados", file=sys.stderr)

                # IMPORTANTE: Si usamos Voyage rerank, NO usamos LLM judge
                refined_answer = False

            except Exception as e:
                error_msg = f"Voyage reranking falló: {str(e)}"
                if ctx:
                    await ctx.error(f"[Semantic Search] {error_msg}")
                else:
                    print(f"[Semantic Search] {error_msg}", file=sys.stderr)
                raise ToolError(error_msg)

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
            workspace_path="",  # No workspace path available, merge without file validation
            search_results=raw_results_tuples,
            max_files=total_limit  # Limitar a los resultados pedidos (paginación incluida)
        )

        # Log de archivos únicos
        if ctx:
            await ctx.info(f"[Semantic Search] Procesados {len(merged_results)} archivos únicos")
        else:
            print(f"[Semantic Search] Procesados {len(merged_results)} archivos únicos", file=sys.stderr)

        # Aplicar paginación a los resultados fusionados
        merged_items = list(merged_results.items())
        total_files = len(merged_items)
        total_pages = math.ceil(total_files / effective_page_size) if total_files else 0

        if total_files > 0 and total_pages and page > total_pages:
            raise ToolError(
                f"La página solicitada ({page}) excede el total disponible ({total_pages})."
            )

        start_index = (page - 1) * effective_page_size
        end_index = start_index + effective_page_size
        paginated_items = merged_items[start_index:end_index] if merged_items else []
        paginated_results = dict(paginated_items)

        pagination_info = {
            "page": page,
            "page_size": effective_page_size,
            "total_files": total_files,
            "total_pages": total_pages,
        }

        if ctx:
            await ctx.info(
                f"[Semantic Search] Mostrando {len(paginated_results)} resultados de "
                f"{total_files} (página {page}/{max(total_pages, 1) if total_pages else 1})"
            )
        else:
            print(
                f"[Semantic Search] Mostrando {len(paginated_results)} resultados de "
                f"{total_files} (página {page}/{max(total_pages, 1) if total_pages else 1})",
                file=sys.stderr
            )

        # Generar brief refinado si se solicita
        refined_brief = ""
        if refined_answer and paginated_results:
            if ctx:
                await ctx.info(f"[Semantic Search] Generando brief refinado con LLM...")
            else:
                print(f"[Semantic Search] Generando brief refinado con LLM...", file=sys.stderr)

            refined_brief = await _generate_refined_brief(query, paginated_results, ctx)

        # Formatear y retornar resultados
        display_workspace = f"colección '{collection_name}'"
        return _format_search_results(
            query,
            display_workspace,
            paginated_results,
            refined_brief,
            pagination=pagination_info
        )

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

    IMPORTANTE: Esta herramienta es SOLO para exploración. NO debes modificar archivos
    del proyecto visitado, solo explorar y entender el código.

    Lógica de resolución de storage:

    1. Si `qdrant_collection` está especificado:
       → Usar Qdrant con esa colección (prioridad máxima)

    2. Si solo se proporciona `workspace_path`:
       → Calcular la colección Qdrant asociada al workspace
       → Si no está disponible, recurrir al índice local configurado para ese workspace

    Args:
        query: Texto natural a buscar en el código (ej: 'función de autenticación', 'manejo de errores')
        workspace_path: Ruta del workspace a visitar (opcional si qdrant_collection está presente)
        qdrant_collection: Nombre de colección Qdrant explícito (prioridad máxima)
        storage_type: Tipo de storage preferido (default: "qdrant")
        refined_answer: Si True, genera un brief con análisis LLM antes de los resultados (default: False)
        max_results: Número máximo de archivos únicos a retornar (default: 20)
        ctx: FastMCP context for logging

    Returns:
        Resultados de búsqueda con merge inteligente, formato texto plano, opcionalmente con brief LLM

    Examples:
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

        # Paso 1: Resolver storage (según disponibilidad detectada)
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
        embedder = get_embedder()
        vector = await embedder.create_embedding(query)

        if resolution.storage_type == "sqlite":
            # Buscar en índice local
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


@mcp.tool
async def search_commit_history(
    query: str,
    qdrant_collection: str,
    max_results: int = 10,
    ctx: Context = None
) -> str:
    """Search git commit history analyzed by LLM to find similar implementations, past decisions, and patterns.

    This tool provides historical context by searching through commits that have been:
    - Indexed with their metadata (author, date, message, changed files)
    - Analyzed and summarized by an LLM for semantic understanding
    - Stored with embeddings for semantic retrieval

    Use this to:
    - Find similar feature implementations from the past
    - Understand why a change was made (commit rationale)
    - Discover established patterns and conventions
    - Debug regressions by finding when something changed
    - Preserve and access institutional knowledge

    Examples:
        "feature flag implementation similar to checkoutFlowEnabled"
        "why did we bootstrap the Payment Class"
        "when did we add null checks to this parameter"
        "previous refactorings of the authentication system"

    Args:
        query: Natural language description of what you're looking for in commit history
        qdrant_collection: Qdrant collection name (e.g., "codebase-abc123"). REQUERIDO.
        max_results: Maximum number of commits to return (default: 10)
        ctx: FastMCP context for logging

    Returns:
        Formatted commit results with metadata, changed files, and LLM analysis

    Raises:
        ToolError: If validation or execution fails
    """
    try:
        # Validar parámetros
        if not query or not query.strip():
            raise ToolError("El parámetro 'query' es requerido y no puede estar vacío.")

        if not qdrant_collection or not qdrant_collection.strip():
            raise ToolError("El parámetro 'qdrant_collection' es requerido y no puede estar vacío.")

        collection_name = qdrant_collection.strip()

        if ctx:
            await ctx.info(f"[Commit History] Colección: {collection_name}, Query: {query}")

        # Paso 1: Crear embedding
        embedder = get_embedder()
        vector = await embedder.create_embedding(query)

        if ctx:
            await ctx.info(f"[Commit History] Embedding creado: {len(vector)} dims")

        # Paso 2: Buscar en Qdrant con filtro de tipo git-commit-analysis
        qdrant = get_qdrant_store()

        from qdrant_client.models import Filter, FieldCondition, MatchValue

        search_filter = Filter(
            must=[
                FieldCondition(
                    key="type",
                    match=MatchValue(value="git-commit-analysis")
                )
            ]
        )

        if ctx:
            await ctx.info(f"[Commit History] Buscando commits en Qdrant...")

        results = await qdrant.client.search(
            collection_name=collection_name,
            query_vector=vector,
            limit=max_results,
            query_filter=search_filter,
            with_payload=True
        )

        if not results:
            return f"""Git Commit History Search

Collection: {collection_name}
Query: {query}

No commits found matching your query.

This could mean:
- No commits have been indexed yet (check TRACK_GIT setting)
- No commits match the semantic query
- The collection doesn't have git commit tracking enabled
"""

        # Paso 3: Formatear resultados
        output = f"""Git Commit History Search

Collection: {collection_name}
Query: {query}
Commits found: {len(results)}

{'=' * 80}

"""

        for idx, result in enumerate(results, 1):
            payload = result.payload
            score = result.score

            commit_hash = payload.get('commitHash', 'unknown')[:7]
            branch = payload.get('branch', 'unknown')
            author = payload.get('author', 'unknown')
            date = payload.get('date', 'unknown')
            message = payload.get('message', '')
            files_changed = payload.get('filesChanged', 0)
            insertions = payload.get('insertions', 0)
            deletions = payload.get('deletions', 0)
            changed_paths = payload.get('changedFilePaths', [])
            analysis = payload.get('analysis', '')

            output += f"""{idx}. Commit {commit_hash} (similarity: {score * 100:.1f}%)

   Branch: {branch}
   Author: {author}
   Date: {date}
   Changes: {files_changed} files (+{insertions}/-{deletions})

   📝 Message:
"""
            for line in message.split('\n'):
                output += f"      {line}\n"

            output += f"\n   📁 Changed Files:\n"
            files_to_show = changed_paths[:5] if isinstance(changed_paths, list) else []
            for file_path in files_to_show:
                output += f"      • {file_path}\n"
            if len(changed_paths) > 5:
                output += f"      ... and {len(changed_paths) - 5} more files\n"

            output += f"\n   🤖 LLM Analysis:\n"
            for line in analysis.split('\n'):
                if line.strip():
                    output += f"      {line}\n"

            output += f"\n{'=' * 80}\n\n"

        if ctx:
            await ctx.info(f"[Commit History] Retornando {len(results)} commits")

        return output

    except ToolError:
        raise

    except Exception as e:
        error_msg = f"Error inesperado: {str(e)}"
        if ctx:
            await ctx.error(f"[Commit History] {error_msg}")
        raise ToolError(error_msg)


# Entry point for fastmcp CLI
if __name__ == "__main__":
    print("[MCP] Servidor `sqlite-semantic-search` inicializado y listo.", file=sys.stderr)
    mcp.run()
