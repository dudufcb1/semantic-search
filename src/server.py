"""FastMCP server for codebase search."""
import sys
from typing import Optional, Literal
from fastmcp import FastMCP, Context
from fastmcp.exceptions import ToolError

from .config import settings
from .embedder import Embedder
from .judge import Judge, SearchResult as JudgeSearchResult
from .qdrant_store import QdrantStore, SearchResult


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

qdrant_store = QdrantStore(
    url=settings.qdrant_url,
    api_key=settings.qdrant_api_key
)


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


def _format_rerank_results(query: str, workspace_path: str, reranked: list, summary: Optional[str] = None) -> str:
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
    
    return '\n'.join(formatted_parts)


@mcp.tool
async def superior_codebase_search(
    query: str,
    workspace_path: Optional[str] = None,
    path: Optional[str] = None,
    ctx: Context = None
) -> str:
    """Realiza una búsqueda semántica en el código indexado utilizando los embeddings existentes.
    
    IMPORTANTE: Debes especificar el workspace a buscar usando 'workspacePath'.
    Ejemplo: Si el usuario menciona 'independent-embeddings-indexer', usa esa ruta exacta.
    
    Args:
        query: Texto natural a buscar en el código (ej: 'función de autenticación', 'manejo de errores')
        workspace_path: Ruta absoluta del workspace a buscar (REQUERIDO)
        path: Prefijo de ruta opcional para filtrar resultados (ej: 'src/auth')
        ctx: FastMCP context for logging
        
    Returns:
        Resultados de búsqueda formateados como texto
    """
    try:
        # Log request
        if ctx:
            await ctx.info(f"[Search] Query: {query}, Workspace: {workspace_path}, Path: {path}")
        else:
            print(f"[Search] Query: {query}, Workspace: {workspace_path}, Path: {path}", file=sys.stderr)
        
        # Validate inputs
        if not query or not query.strip():
            raise ToolError("El parámetro 'query' es requerido y no puede estar vacío.")
        
        # Use default workspace if not provided
        effective_workspace = workspace_path or settings.default_workspace_path
        if not effective_workspace:
            raise ToolError(
                "Debes especificar 'workspacePath' o configurar MCP_CODEBASE_WORKSPACE.\n\n"
                "El cliente MCP debe enviar el workspace en los argumentos.\n\n"
                'Ejemplo: { "query": "login function", "workspacePath": "/path/to/project" }'
            )
        
        # Create embedding
        vector = await embedder.create_embedding(query)
        
        # Search in Qdrant
        results = await qdrant_store.search(
            vector=vector,
            workspace_path=effective_workspace,
            directory_prefix=path,
            min_score=settings.search_min_score,
            max_results=settings.search_max_results
        )
        
        # Format and return results
        return _format_search_results(query, effective_workspace, results)
        
    except ToolError:
        raise
    except Exception as e:
        error_msg = str(e)
        if ctx:
            await ctx.error(f"[Search] Error: {error_msg}")
        else:
            print(f"[Search] Error: {error_msg}", file=sys.stderr)
        raise ToolError(f"Error al buscar en el codebase: {error_msg}")


@mcp.tool
async def superior_codebase_rerank(
    query: str,
    workspace_path: Optional[str] = None,
    path: Optional[str] = None,
    mode: Literal["rerank", "summary"] = "rerank",
    ctx: Context = None
) -> str:
    """Realiza una búsqueda semántica y reordena los resultados usando un LLM Judge.
    
    Esta herramienta primero ejecuta una búsqueda semántica y luego usa un modelo de lenguaje
    para evaluar y reordenar los resultados según su relevancia real a la consulta.
    
    Args:
        query: Texto natural a buscar en el código
        workspace_path: Ruta absoluta del workspace a buscar (REQUERIDO)
        path: Prefijo de ruta opcional para filtrar resultados
        mode: Modo de operación - "rerank" solo reordena, "summary" incluye resumen
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
        
        # Use default workspace if not provided
        effective_workspace = workspace_path or settings.default_workspace_path
        if not effective_workspace:
            raise ToolError(
                "Debes especificar 'workspacePath' o configurar MCP_CODEBASE_WORKSPACE.\n\n"
                "El cliente MCP debe enviar el workspace en los argumentos.\n\n"
                'Ejemplo: { "query": "login function", "workspacePath": "/path/to/project" }'
            )
        
        # Step 1: Perform initial search
        if ctx:
            await ctx.info(f"[Rerank] Buscando resultados para query: '{query}'")
        
        vector = await embedder.create_embedding(query)
        search_results = await qdrant_store.search(
            vector=vector,
            workspace_path=effective_workspace,
            directory_prefix=path,
            min_score=settings.search_min_score,
            max_results=settings.search_max_results
        )
        
        if not search_results:
            return f'No se encontraron resultados para "{query}" en el workspace "{effective_workspace}".'
        
        # Step 2: Convert to Judge format
        judge_results = [
            JudgeSearchResult(
                file_path=r.file_path,
                code_chunk=r.code_chunk,
                start_line=r.start_line,
                end_line=r.end_line,
                score=r.score
            )
            for r in search_results
        ]
        
        # Step 3: Rerank with LLM
        if ctx:
            await ctx.info(f"[Rerank] Reordenando {len(judge_results)} resultados con LLM")
        
        reranked = await judge.rerank(query, judge_results)
        
        # Apply max results limit
        reranked = reranked[:settings.reranking_max_results]
        
        # Step 4: Generate summary if requested
        summary = None
        if mode == "summary":
            if ctx:
                await ctx.info("[Rerank] Generando resumen")
            summary = await judge.summarize(query, judge_results)
        
        # Format and return results
        return _format_rerank_results(query, effective_workspace, reranked, summary)
        
    except ToolError:
        raise
    except Exception as e:
        error_msg = str(e)
        if ctx:
            await ctx.error(f"[Rerank] Error: {error_msg}")
        else:
            print(f"[Rerank] Error: {error_msg}", file=sys.stderr)
        raise ToolError(f"Error al reordenar resultados: {error_msg}")


# Entry point for fastmcp CLI
if __name__ == "__main__":
    print("[MCP] Servidor `codebase-search` inicializado y listo.", file=sys.stderr)
    mcp.run()

