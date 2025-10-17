"""Test script for all MCP tools (search and rerank)."""
import asyncio
import os
import re
from pathlib import Path


def parse_data_file(file_path: str) -> dict:
    """Parse data.txt file to extract test configuration."""
    with open(file_path, 'r') as f:
        contents = f.read()
    
    # Extract values using regex
    base_url_match = re.search(r'Base URL:\s*(.+)', contents, re.IGNORECASE)
    api_key_match = re.search(r'API_KEY:\s*(.+)', contents, re.IGNORECASE)
    model_match = re.search(r'Model:\s*(.+)', contents, re.IGNORECASE)
    workspace_match = re.search(r'Path de prueba:\s*(.+)', contents, re.IGNORECASE)
    query_match = re.search(r'Termino a buscar:\s*(.+)', contents, re.IGNORECASE)
    
    if not all([base_url_match, api_key_match, model_match, workspace_match, query_match]):
        raise ValueError("data.txt no contiene todos los campos requeridos")
    
    return {
        'base_url': base_url_match.group(1).strip(),
        'api_key': api_key_match.group(1).strip(),
        'model': model_match.group(1).strip(),
        'workspace_path': workspace_match.group(1).strip(),
        'query': query_match.group(1).strip(),
    }


def setup_environment():
    """Setup environment variables before importing config."""
    # Load data from data.txt
    data_path = Path(__file__).parent.parent / 'data.txt'
    data = parse_data_file(str(data_path))
    
    # Configure environment variables for embedder
    os.environ['MCP_CODEBASE_EMBEDDER_PROVIDER'] = 'openai-compatible'
    os.environ['MCP_CODEBASE_EMBEDDER_BASE_URL'] = data['base_url']
    os.environ['MCP_CODEBASE_EMBEDDER_API_KEY'] = data['api_key']
    os.environ['MCP_CODEBASE_EMBEDDER_MODEL_ID'] = data['model']
    
    # Set default Qdrant URL if not configured
    if 'MCP_CODEBASE_QDRANT_URL' not in os.environ and 'QDRANT_URL' not in os.environ:
        os.environ['MCP_CODEBASE_QDRANT_URL'] = 'http://localhost:6333'
    
    # Configure judge (use same endpoint for testing)
    if 'MCP_CODEBASE_JUDGE_API_KEY' not in os.environ and 'JUDGE_API_KEY' not in os.environ:
        # Try to use same API key as embedder
        os.environ['MCP_CODEBASE_JUDGE_API_KEY'] = data['api_key']
    
    if 'MCP_CODEBASE_JUDGE_BASE_URL' not in os.environ and 'JUDGE_BASE_URL' not in os.environ:
        # Try to use same base URL
        os.environ['MCP_CODEBASE_JUDGE_BASE_URL'] = data['base_url']
    
    if 'MCP_CODEBASE_JUDGE_MODEL_ID' not in os.environ and 'JUDGE_MODEL_ID' not in os.environ:
        # Use a default model
        os.environ['MCP_CODEBASE_JUDGE_MODEL_ID'] = 'gpt-4o-mini'
    
    return data


async def test_search(data: dict):
    """Test superior_codebase_search tool."""
    print("\n" + "="*80)
    print("TEST 1: superior_codebase_search")
    print("="*80)
    
    from src.config import settings
    from src.embedder import Embedder
    from src.qdrant_store import QdrantStore
    
    # Create services
    embedder = Embedder(
        provider=settings.embedder_provider,
        api_key=settings.embedder_api_key,
        model_id=settings.embedder_model_id,
        base_url=settings.embedder_base_url
    )
    
    qdrant_store = QdrantStore(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key
    )
    
    print(f"\n📋 Configuración:")
    print(f"  - Workspace: {data['workspace_path']}")
    print(f"  - Query: {data['query']}")
    print(f"  - Embedder: {settings.embedder_provider} ({settings.embedder_model_id})")
    print(f"  - Qdrant: {settings.qdrant_url}")
    
    # Create embedding
    print(f"\n🔄 Creando embedding...")
    vector = await embedder.create_embedding(data['query'])
    print(f"✅ Embedding creado: {len(vector)} dimensiones")
    
    # Search in Qdrant
    print(f"\n🔍 Buscando en Qdrant...")
    results = await qdrant_store.search(
        vector=vector,
        workspace_path=data['workspace_path'],
        min_score=settings.search_min_score,
        max_results=settings.search_max_results
    )
    
    print(f"✅ Resultados encontrados: {len(results)}")
    
    if results:
        print(f"\n📄 Top 3 resultados:")
        for i, result in enumerate(results[:3], 1):
            print(f"\n  {i}. {result.file_path}")
            print(f"     Score: {result.score:.4f}")
            print(f"     Líneas: {result.start_line}-{result.end_line}")
            print(f"     Extracto: {result.code_chunk.strip()[:100]}...")
    else:
        print("\n⚠️  No se encontraron resultados")
    
    return results


async def test_rerank(data: dict, search_results):
    """Test superior_codebase_rerank tool."""
    print("\n" + "="*80)
    print("TEST 2: superior_codebase_rerank")
    print("="*80)
    
    if not search_results:
        print("\n⚠️  No hay resultados para reordenar (test de búsqueda no encontró nada)")
        return
    
    from src.config import settings
    from src.judge import Judge, SearchResult as JudgeSearchResult
    
    # Create judge
    judge = Judge(
        provider=settings.judge_provider,
        api_key=settings.judge_api_key,
        model_id=settings.judge_model_id,
        max_tokens=settings.judge_max_tokens,
        temperature=settings.judge_temperature,
        base_url=settings.judge_base_url,
        system_prompt=settings.judge_system_prompt
    )
    
    print(f"\n📋 Configuración:")
    print(f"  - Query: {data['query']}")
    print(f"  - Judge: {settings.judge_provider} ({settings.judge_model_id})")
    print(f"  - Resultados a reordenar: {len(search_results)}")
    
    # Convert to Judge format
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
    
    # Rerank with LLM
    print(f"\n🔄 Reordenando con LLM...")
    try:
        reranked = await judge.rerank(data['query'], judge_results)
        print(f"✅ Resultados reordenados: {len(reranked)}")
        
        if reranked:
            print(f"\n📄 Top 3 resultados reordenados:")
            for i, result in enumerate(reranked[:3], 1):
                print(f"\n  {i}. {result.file_path}")
                print(f"     Relevancia: {result.relevancia:.4f}")
                print(f"     Score original: {result.score:.4f}")
                if result.razon:
                    print(f"     Razón: {result.razon}")
                print(f"     Extracto: {result.code_chunk.strip()[:100]}...")
        
        # Test summary mode
        print(f"\n🔄 Generando resumen...")
        summary = await judge.summarize(data['query'], judge_results[:5])
        print(f"✅ Resumen generado:")
        print(f"\n{summary}\n")
        
    except Exception as e:
        print(f"\n❌ Error en reranking: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("PRUEBA COMPLETA DE HERRAMIENTAS MCP")
    print("="*80)
    
    # Setup environment
    data = setup_environment()
    
    # Test 1: Search
    search_results = await test_search(data)
    
    # Test 2: Rerank
    await test_rerank(data, search_results)
    
    print("\n" + "="*80)
    print("✅ TODAS LAS PRUEBAS COMPLETADAS")
    print("="*80)


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"\n❌ Error durante las pruebas: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

