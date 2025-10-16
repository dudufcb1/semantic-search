"""FastMCP server for codebase search."""
import sys
from typing import Optional, Literal
from fastmcp import FastMCP, Context
from fastmcp.exceptions import ToolError

try:
    from .config import settings
    from .embedder import Embedder
    from .judge import Judge, SearchResult as JudgeSearchResult
    from .text_judge import TextDirectJudge, SearchResult as TextJudgeSearchResult
    from .qdrant_store import QdrantStore, SearchResult
    from .storage_resolver import StorageResolver
except ImportError:
    from config import settings
    from embedder import Embedder
    from judge import Judge, SearchResult as JudgeSearchResult
    from text_judge import TextDirectJudge, SearchResult as TextJudgeSearchResult
    from qdrant_store import QdrantStore, SearchResult
    from storage_resolver import StorageResolver


# Initialize FastMCP server
mcp = FastMCP(
    name="codebase-search",
    version="0.1.0"
)

# Initialize services
embedder = Embedder(
    provider=settings.embedder_provider,
    api_key=settings.embedder_api_key,
    model_id=settings.embedder_model_id,
    base_url=settings.embedder_base_url
)

judge = Judge(
    provider=settings.judge_provider,
    api_key=settings.judge_api_key,
    model_id=settings.judge_model_id,
    max_tokens=settings.judge_max_tokens,
    temperature=settings.judge_temperature,
    base_url=settings.judge_base_url,
    system_prompt=settings.judge_system_prompt
)

# Experimental Text Direct Judge for explore_other_workspace
text_judge = TextDirectJudge(
    provider=settings.judge_provider,
    api_key=settings.judge_api_key,
    model_id=settings.judge_model_id,
    max_tokens=settings.judge_max_tokens,
    temperature=settings.judge_temperature,
    base_url=settings.judge_base_url
)

qdrant_store = QdrantStore(
    url=settings.qdrant_url,
    api_key=settings.qdrant_api_key
)

# Storage resolver for visit_other_project
storage_resolver = StorageResolver(qdrant_store=qdrant_store)


def _format_search_results(query: str, workspace_path: str, results: list[SearchResult]) -> str:
    """Format search results as text."""
    if not results:
        return f'No se encontraron coincidencias para "{query}" en el workspace "{workspace_path}".'
    
    formatted_parts = [f'Query: {query}', f'Workspace: {workspace_path}', '']
    
    for result in results:
        formatted_parts.append(f'Ruta: {result.file_path}')
        formatted_parts.append(f'Score: {result.score:.4f}')
        formatted_parts.append(f'Líneas: {result.start_line}-{result.end_line}')
        formatted_parts.append('---')
        formatted_parts.append(result.code_chunk.strip())
        formatted_parts.append('')
    
    return '\n'.join(formatted_parts)


def _format_rerank_results(query: str, workspace_path: str, reranked: list, summary: Optional[str] = None, usages: Optional[list[str]] = None) -> str:
    """Format reranked results as text."""
    if not reranked:
        return f'No se encontraron resultados relevantes para "{query}" en el workspace "{workspace_path}".'

    formatted_parts = [f'Query: {query}', f'Workspace: {workspace_path}', '']

    if summary:
        formatted_parts.append('=== RESUMEN ===')
        formatted_parts.append(summary)
        formatted_parts.append('')
        formatted_parts.append('=== RESULTADOS REORDENADOS ===')
        formatted_parts.append('')

    for result in reranked:
        formatted_parts.append(f'Ruta: {result.file_path}')
        formatted_parts.append(f'Relevancia: {result.relevancia:.4f}')
        formatted_parts.append(f'Score original: {result.score:.4f}')
        formatted_parts.append(f'Líneas: {result.start_line}-{result.end_line}')
        if result.razon:
            formatted_parts.append(f'Razón: {result.razon}')
        formatted_parts.append('---')
        formatted_parts.append(result.code_chunk.strip())
        formatted_parts.append('')

    # Add usages section if present
    if usages and len(usages) > 0:
        formatted_parts.append('=== USAGES DETECTADOS ===')
        for usage in usages:
            formatted_parts.append(f'• {usage}')
        formatted_parts.append('')

    return '\n'.join(formatted_parts)


def _format_search_results_explore(query: str, target_identifier: str, results: list[SearchResult]) -> str:
    """Format search results for explore_other_workspace tool."""
    if not results:
        return f'No se encontraron coincidencias para "{query}" en {target_identifier}.'

    formatted_parts = [f'Query: {query}', f'Target: {target_identifier}', '']

    for result in results:
        formatted_parts.append(f'Ruta: {result.file_path}')
        formatted_parts.append(f'Score: {result.score:.4f}')
        formatted_parts.append(f'Líneas: {result.start_line}-{result.end_line}')
        formatted_parts.append('---')
        formatted_parts.append(result.code_chunk.strip())
        formatted_parts.append('')

    return '\n'.join(formatted_parts)


def _format_rerank_results_explore(query: str, target_identifier: str, reranked: list, summary: Optional[str] = None, usages: Optional[list[str]] = None) -> str:
    """Format reranked results for explore_other_workspace tool."""
    if not reranked:
        return f'No se encontraron resultados relevantes para "{query}" en {target_identifier}.'

    formatted_parts = [f'Query: {query}', f'Target: {target_identifier}', '']

    if summary:
        formatted_parts.append('=== RESUMEN ===')
        formatted_parts.append(summary)
        formatted_parts.append('')
        formatted_parts.append('=== RESULTADOS REORDENADOS ===')
        formatted_parts.append('')

    for result in reranked:
        formatted_parts.append(f'Ruta: {result.file_path}')
        formatted_parts.append(f'Relevancia: {result.relevancia:.4f}')
        formatted_parts.append(f'Score original: {result.score:.4f}')
        formatted_parts.append(f'Líneas: {result.start_line}-{result.end_line}')
        if result.razon:
            formatted_parts.append(f'Razón: {result.razon}')
        formatted_parts.append('---')
        formatted_parts.append(result.code_chunk.strip())
        formatted_parts.append('')

    # Add usages section if present
    if usages and len(usages) > 0:
        formatted_parts.append('=== USAGES DETECTADOS ===')
        for usage in usages:
            formatted_parts.append(f'• {usage}')
        formatted_parts.append('')

    return '\n'.join(formatted_parts)


@mcp.tool
async def superior_codebase_search(
    query: str,
    workspace_path: str,
    ctx: Context = None
) -> str:
    """Realiza una búsqueda semántica en el código indexado utilizando los embeddings existentes.

    Args:
        query: Texto natural a buscar en el código (ej: 'función de autenticación', 'manejo de errores')
        workspace_path: Ruta absoluta del workspace a buscar (REQUERIDO)
        ctx: FastMCP context for logging

    Returns:
        Resultados de búsqueda formateados como texto
    """
    try:
        # Log request
        if ctx:
            await ctx.info(f"[Search] Query: {query}, Workspace: {workspace_path}")
        else:
            print(f"[Search] Query: {query}, Workspace: {workspace_path}", file=sys.stderr)

        # Validate inputs
        if not query or not query.strip():
            raise ToolError("El parámetro 'query' es requerido y no puede estar vacío.")

        if not workspace_path or not workspace_path.strip():
            raise ToolError("El parámetro 'workspace_path' es requerido.")

        # Create embedding
        vector = await embedder.create_embedding(query)

        # Search in Qdrant
        results = await qdrant_store.search(
            vector=vector,
            workspace_path=workspace_path,
            directory_prefix=None,  # No path filtering
            min_score=settings.search_min_score,
            max_results=settings.search_max_results
        )

        # Format and return results
        return _format_search_results(query, workspace_path, results)
        
    except ToolError:
        raise
    except Exception as e:
        error_msg = str(e)
        if ctx:
            await ctx.error(f"[Search] Error: {error_msg}")
        else:
            print(f"[Search] Error: {error_msg}", file=sys.stderr)
        raise ToolError("Error al buscar en el codebase")


@mcp.tool
async def explore_other_workspace(
    query: str,
    qdrant_collection: Optional[str] = None,
    workspace_path: Optional[str] = None,
    path: Optional[str] = None,
    mode: Literal["rerank", "summary"] = "rerank",
    ctx: Context = None
) -> str:
    """Explora y busca en colecciones de otros workspaces diferentes al actual.

    Esta herramienta permite realizar búsquedas semánticas en workspaces remotos o diferentes
    al actual, útil para buscar patrones en proyectos relacionados o diferentes codebases.

    EXPERIMENTAL: Usa formato de texto directo (no JSON) para eliminar problemas de parsing.
    Siempre realiza rerank por defecto para mejores resultados.

    Args:
        query: Texto natural a buscar en el código
        qdrant_collection: Nombre de colección Qdrant explícito (prioridad alta)
        workspace_path: Ruta del workspace alternativo (usado si no hay qdrant_collection)
        path: Prefijo de ruta opcional para filtrar resultados
        mode: Modo de operación - "rerank" solo reordena, "summary" incluye resumen
        ctx: FastMCP context for logging

    Returns:
        Resultados de búsqueda reordenados formateados como texto, opcionalmente con resumen
    """
    try:
        # Log request
        if ctx:
            await ctx.info(f"[Explore Other] Query: {query}, Collection: {qdrant_collection}, Workspace: {workspace_path}, Mode: {mode}")
        else:
            print(f"[Explore Other] Query: {query}, Collection: {qdrant_collection}, Workspace: {workspace_path}, Mode: {mode}", file=sys.stderr)

        # Validate inputs
        if not query or not query.strip():
            raise ToolError("El parámetro 'query' es requerido y no puede estar vacío.")

        # Determine collection to use
        final_collection_name = None
        target_identifier = None

        if qdrant_collection and qdrant_collection.strip():
            # Priority 1: Use explicit collection name
            final_collection_name = qdrant_collection.strip()
            target_identifier = f"colección '{final_collection_name}'"
        elif workspace_path and workspace_path.strip():
            # Priority 2: Calculate from workspace path
            normalized_workspace = qdrant_store._normalize_workspace_path(workspace_path.strip())
            final_collection_name = qdrant_store._get_collection_name(normalized_workspace)
            target_identifier = f"workspace '{workspace_path}' (colección: {final_collection_name})"
        else:
            raise ToolError(
                "Debes proporcionar 'qdrant_collection' O 'workspace_path'.\n\n"
                "- qdrant_collection: Nombre explícito de la colección (ej: 'codebase-f93e99958acc444e')\n"
                "- workspace_path: Ruta del workspace (se calculará la colección automáticamente)\n\n"
                "Si ambos están presentes, se usa qdrant_collection (prioridad alta)."
            )

        # Step 1: Perform initial search
        if ctx:
            await ctx.info(f"[Explore Other] Buscando en {target_identifier} para query: '{query}'")

        vector = await embedder.create_embedding(query)
        search_results = await qdrant_store.search(
            vector=vector,
            workspace_path="",  # Not used when collection_name is provided
            directory_prefix=path,
            min_score=settings.search_min_score,
            max_results=settings.search_max_results,
            collection_name=final_collection_name
        )

        if not search_results:
            return f'No se encontraron resultados para "{query}" en {target_identifier}.'

        # Step 2: Convert to TextDirectJudge format
        text_judge_results = [
            TextJudgeSearchResult(
                file_path=r.file_path,
                code_chunk=r.code_chunk,
                start_line=r.start_line,
                end_line=r.end_line,
                score=r.score
            )
            for r in search_results
        ]

        # Step 3: Rerank with Text Direct LLM (no JSON parsing)
        if ctx:
            await ctx.info(f"[Explore Other] Reordenando {len(text_judge_results)} resultados con TextDirectJudge")

        try:
            if mode == "summary":
                reranked, summary = await text_judge.rerank_with_summary(query, text_judge_results)
            else:
                reranked = await text_judge.rerank(query, text_judge_results)
                summary = None
        except Exception as e:
            # FALLBACK: Si el TextDirectJudge falla, devolver resultados originales
            if ctx:
                await ctx.info(f"[Explore Other] TextDirectJudge failed, fallback to original results: {str(e)}")
            else:
                print(f"[Explore Other] TextDirectJudge failed, fallback to original results: {str(e)}", file=sys.stderr)
            return _format_search_results_explore(query, target_identifier, search_results)

        # Apply max results limit
        reranked = reranked[:settings.reranking_max_results]

        # Format and return results
        return _format_rerank_results_explore(query, target_identifier, reranked, summary)

    except ToolError:
        raise
    except Exception as e:
        error_msg = str(e)
        if ctx:
            await ctx.info(f"[Explore Other] Fallback to search results due to: {error_msg}")
        else:
            print(f"[Explore Other] Fallback to search results due to: {error_msg}", file=sys.stderr)

        # Always return search results instead of error
        return _format_search_results_explore(query, target_identifier or "workspace desconocido", search_results)


@mcp.tool
async def superior_codebase_rerank(
    query: str,
    workspace_path: str,
    mode: Literal["rerank", "summary"] = "rerank",
    include_docs: bool = False,
    include_config: bool = False,
    search_intent: Optional[str] = None,
    ctx: Context = None
) -> str:
    """Realiza una búsqueda semántica y reordena los resultados usando un LLM Judge.

    Esta herramienta primero ejecuta una búsqueda semántica y luego usa un modelo de lenguaje
    para evaluar y reordenar los resultados según su relevancia real a la consulta.

    Args:
        query: Texto natural a buscar en el código
        workspace_path: Ruta absoluta del workspace a buscar (REQUERIDO)
        mode: Modo de operación - "rerank" solo reordena, "summary" incluye resumen
        include_docs: Si True, incluye archivos de documentación (.md, .txt, etc.). Por defecto False.
        include_config: Si True, incluye archivos de configuración (.json, .ini, .yaml, etc.). Por defecto False.
        search_intent: Palabra clave para concatenar a la query (ej: "implementation", "usage", "flow"). Se concatena como palabra adicional.
        ctx: FastMCP context for logging

    Returns:
        Resultados reordenados formateados como texto, opcionalmente con resumen
    """
    try:
        # Log request
        if ctx:
            await ctx.info(f"[Rerank] Query: {query}, Workspace: {workspace_path}, Mode: {mode}")
        else:
            print(f"[Rerank] Query: {query}, Workspace: {workspace_path}, Mode: {mode}", file=sys.stderr)

        # Validate inputs
        if not query or not query.strip():
            raise ToolError("El parámetro 'query' es requerido y no puede estar vacío.")

        if not workspace_path or not workspace_path.strip():
            raise ToolError("El parámetro 'workspace_path' es requerido.")

        # Concatenate search_intent to query if provided
        final_query = query
        if search_intent and search_intent.strip():
            final_query = f"{query} {search_intent.strip()}"
            print(f"[DEBUG Rerank] Query modificada con search_intent: '{final_query}'", file=sys.stderr)

        # Step 1: Perform initial search
        if ctx:
            await ctx.info(f"[Rerank] Buscando resultados para query: '{final_query}'")

        vector = await embedder.create_embedding(final_query)
        search_results = await qdrant_store.search(
            vector=vector,
            workspace_path=workspace_path,
            directory_prefix=None,  # No path filtering
            min_score=settings.search_min_score,
            max_results=settings.search_max_results
        )

        if not search_results:
            return f'No se encontraron resultados para "{final_query}" en el workspace "{workspace_path}".'

        # Step 2: Filter documentation files if include_docs=False
        if not include_docs:
            original_count = len(search_results)
            search_results = [
                r for r in search_results
                if not r.file_path.endswith(('.md', '.txt', '.rst', '.adoc', '.markdown'))
            ]
            filtered_count = original_count - len(search_results)
            if filtered_count > 0:
                print(f"[DEBUG] Filtrados {filtered_count} archivos de documentación", file=sys.stderr)
                if ctx:
                    await ctx.info(f"[Rerank] Filtrados {filtered_count} archivos de documentación")

        # Step 3: Filter configuration files if include_config=False
        if not include_config:
            original_count = len(search_results)
            search_results = [
                r for r in search_results
                if not r.file_path.endswith(('.json', '.ini', '.yaml', '.yml', '.toml', '.xml', '.env', '.config'))
            ]
            filtered_count = original_count - len(search_results)
            if filtered_count > 0:
                print(f"[DEBUG] Filtrados {filtered_count} archivos de configuración", file=sys.stderr)
                if ctx:
                    await ctx.info(f"[Rerank] Filtrados {filtered_count} archivos de configuración")

        if not search_results:
            return f'No se encontraron resultados de código para "{final_query}". Usa include_docs=True o include_config=True para incluir archivos de documentación/configuración.'

        # Step 3: Rerank with JSON Schema structured outputs
        print(f"[DEBUG] Rerankando con JSON Schema estructurado...", file=sys.stderr)

        if ctx:
            await ctx.info(f"[Rerank] Rerankando {len(search_results)} resultados con JSON Schema")

        # Convert to TextJudge format
        text_judge_results = [
            TextJudgeSearchResult(
                file_path=r.file_path,
                code_chunk=r.code_chunk,
                start_line=r.start_line,
                end_line=r.end_line,
                score=r.score
            )
            for r in search_results
        ]

        # Rerank with structured outputs
        summary = await text_judge.rerank_structured(query, text_judge_results)

        print(f"[DEBUG] Resumen estructurado generado: {len(summary.top_files)} archivos, {len(summary.code_fragments)} fragmentos", file=sys.stderr)

        # Format as markdown
        markdown = f"""# Resultados de búsqueda: {summary.query}

**Total de archivos:** {summary.total_files}
**Total de fragmentos:** {summary.total_fragments}

## Top {len(summary.top_files)} Archivos (calificación IA)

"""
        for i, file in enumerate(summary.top_files, 1):
            markdown += f"""### {i}. `{file.file_path}`
**Relevancia:** {file.relevance}
**Razón:** {file.reason}
**Líneas relevantes:** {file.relevant_lines}

"""

        if summary.code_fragments:
            markdown += f"\n## Fragmentos de Código ({len(summary.code_fragments)} fragmentos)\n\n"
            for i, frag in enumerate(summary.code_fragments, 1):
                # Detect language from file extension
                ext = frag.file_path.split('.')[-1] if '.' in frag.file_path else ''
                lang_map = {
                    'ts': 'typescript', 'js': 'javascript', 'py': 'python',
                    'java': 'java', 'cpp': 'cpp', 'c': 'c', 'go': 'go',
                    'rs': 'rust', 'rb': 'ruby', 'php': 'php', 'cs': 'csharp'
                }
                lang = lang_map.get(ext, '')

                markdown += f"""### {i}. `{frag.file_path}` (líneas {frag.start_line}-{frag.end_line})

**Explicación:** {frag.explanation}

```{lang}
{frag.code_snippet}
```

"""

        if summary.usages:
            markdown += f"\n## Usages ({len(summary.usages)} invocaciones)\n\n"
            for usage in summary.usages:
                markdown += f"- **{usage.function_name}** llamada en `{usage.file_path}:{usage.line_number}`  \n  {usage.context}\n\n"

        markdown += f"\n## Inferencia\n\n{summary.inference}\n"

        # Return formatted markdown
        return markdown
        
    except ToolError:
        raise
    except Exception as e:
        error_msg = str(e)
        if ctx:
            await ctx.info(f"[Rerank] Error: {error_msg}")
        else:
            print(f"[Rerank] Error: {error_msg}", file=sys.stderr)

        # Re-raise the exception instead of trying to use potentially undefined variables
        raise


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
            resolution = storage_resolver.resolve(
                workspace_path=workspace_path,
                qdrant_collection=qdrant_collection,
                storage_type=storage_type
            )
            storage_resolver.log_resolution(resolution, ctx)
        except ValueError as e:
            raise ToolError(str(e))

        # Paso 2: Crear embedding y buscar
        vector = await embedder.create_embedding(query)

        if resolution.storage_type == "sqlite":
            # Import SQLite search function
            try:
                from .server_sqlite import _search_sqlite_vectors
            except ImportError:
                from server_sqlite import _search_sqlite_vectors

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
            raw_results = await qdrant_store.search(
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

            # Import brief generator
            try:
                from .server_sqlite import _generate_refined_brief_visit
            except ImportError:
                from server_sqlite import _generate_refined_brief_visit

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
    print("[MCP] Servidor `codebase-search` inicializado y listo.", file=sys.stderr)
    mcp.run()

