"""FastMCP server for codebase search."""
import sys
from typing import Optional, Literal
from fastmcp import FastMCP, Context
from fastmcp.exceptions import ToolError

try:
    from .config import settings
    from .embedder import Embedder
    from .judge import Judge, SearchResult as JudgeSearchResult
    from .qdrant_store import QdrantStore
except ImportError:
    from config import settings
    from embedder import Embedder
    from judge import Judge, SearchResult as JudgeSearchResult
    from qdrant_store import QdrantStore, SearchResult


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
async def superior_codebase_rerank(
    query: str,
    workspace_path: str,
    mode: Literal["rerank", "summary"] = "rerank",
    ctx: Context = None
) -> str:
    """Realiza una búsqueda semántica y reordena los resultados usando un LLM Judge.

    Esta herramienta primero ejecuta una búsqueda semántica y luego usa un modelo de lenguaje
    para evaluar y reordenar los resultados según su relevancia real a la consulta.

    Args:
        query: Texto natural a buscar en el código
        workspace_path: Ruta absoluta del workspace a buscar (REQUERIDO)
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

        if not workspace_path or not workspace_path.strip():
            raise ToolError("El parámetro 'workspace_path' es requerido.")

        # Step 1: Perform initial search
        if ctx:
            await ctx.info(f"[Rerank] Buscando resultados para query: '{query}'")

        vector = await embedder.create_embedding(query)
        search_results = await qdrant_store.search(
            vector=vector,
            workspace_path=workspace_path,
            directory_prefix=None,  # No path filtering
            min_score=settings.search_min_score,
            max_results=settings.search_max_results
        )
        
        if not search_results:
            return f'No se encontraron resultados para "{query}" en el workspace "{workspace_path}".'
        
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

        try:
            reranked = await judge.rerank(query, judge_results)
        except Exception as e:
            # FALLBACK: Si el Judge falla por JSON malformado, devolver respuesta cruda como texto
            if ctx:
                await ctx.info(f"[Rerank] Judge JSON parsing failed, returning raw response as text: {str(e)}")
            else:
                print(f"[Rerank] Judge JSON parsing failed, returning raw response as text: {str(e)}", file=sys.stderr)

            # Try to extract raw response from error message or judge's last response
            raw_response = None

            # Method 1: Extract from error message
            error_msg = str(e)
            if "Raw response: " in error_msg:
                raw_response = error_msg.split("Raw response: ", 1)[1]

            # Method 2: Check if judge stored the response
            elif hasattr(judge, '_last_raw_response'):
                raw_response = judge._last_raw_response

            if raw_response:
                # Return the raw response as text with context
                return f"""Query: {query}
Workspace: {workspace_path}

{raw_response.strip()}"""

            # If no raw response available, try to get fresh response
            try:
                user_prompt = judge._create_user_prompt(query, judge_results, include_summary=(mode == "summary"))
                raw_response = await judge._call_llm(user_prompt, include_summary=(mode == "summary"))

                # Return the raw response as text with context
                return f"""Query: {query}
Workspace: {workspace_path}

{raw_response.strip()}"""

            except Exception as inner_e:
                # Si todo falla, devolver resultados originales
                if ctx:
                    await ctx.info(f"[Rerank] Complete fallback to original results: {str(inner_e)}")
                else:
                    print(f"[Rerank] Complete fallback to original results: {str(inner_e)}", file=sys.stderr)
                return _format_search_results(query, workspace_path, search_results)
        
        # Apply max results limit
        reranked = reranked[:settings.reranking_max_results]
        
        # Step 4: Generate summary if requested
        summary = None
        if mode == "summary":
            if ctx:
                await ctx.info("[Rerank] Generando resumen")
            try:
                summary = await judge.summarize(query, judge_results)
            except Exception as e:
                # FALLBACK: Si el summary falla, no incluir summary
                if ctx:
                    await ctx.info(f"[Rerank] Summary failed, proceeding without summary: {str(e)}")
                else:
                    print(f"[Rerank] Summary failed, proceeding without summary: {str(e)}", file=sys.stderr)
                summary = None
        
        # Format and return results
        return _format_rerank_results(query, workspace_path, reranked, summary)
        
    except ToolError:
        raise
    except Exception as e:
        error_msg = str(e)
        if ctx:
            await ctx.info(f"[Rerank] Fallback to search results due to: {error_msg}")
        else:
            print(f"[Rerank] Fallback to search results due to: {error_msg}", file=sys.stderr)

        # Always return search results instead of error
        return _format_search_results(query, workspace_path, search_results)


# Entry point for fastmcp CLI
if __name__ == "__main__":
    print("[MCP] Servidor `codebase-search` inicializado y listo.", file=sys.stderr)
    mcp.run()

