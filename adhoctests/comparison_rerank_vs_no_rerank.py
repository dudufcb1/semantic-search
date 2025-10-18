"""Test script to compare results WITH and WITHOUT Voyage rerank."""
import asyncio
import os
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# Configure environment variables BEFORE importing any modules
os.environ['MCP_CODEBASE_EMBEDDER_PROVIDER'] = 'openai-compatible'
os.environ['MCP_CODEBASE_EMBEDDER_BASE_URL'] = ''
os.environ['MCP_CODEBASE_EMBEDDER_MODEL_ID'] = 'text-embedding-3-small'
os.environ['MCP_CODEBASE_EMBEDDER_API_KEY'] = 'DUMMYISALOCAL'
os.environ['MCP_CODEBASE_EMBEDDER_DIMENSION'] = '1536'
os.environ['MCP_CODEBASE_QDRANT_URL'] = 'http://localhost:6333'
os.environ['MCP_CODEBASE_JUDGE_PROVIDER'] = 'openai-compatible'
os.environ['MCP_CODEBASE_JUDGE_BASE_URL'] = ''
os.environ['MCP_CODEBASE_JUDGE_MODEL_ID'] = 'gpt-4.1'
os.environ['MCP_CODEBASE_JUDGE_API_KEY'] = 'DUMMYISALOCAL'
os.environ['MCP_CODEBASE_VOYAGE_API_KEY'] = 'pa-'
os.environ['MCP_CODEBASE_VOYAGE_RERANK_MODEL'] = 'rerank-2.5'


async def test_without_rerank():
    """Test search WITHOUT Voyage rerank (pure vector similarity)."""
    print("\n" + "="*80)
    print("TEST 1: SIN VOYAGE RERANK (solo vector similarity)")
    print("="*80)

    # Temporarily disable rerank
    original_native_rerank = os.environ.get('MCP_CODEBASE_NATIVE_RERANK')
    os.environ['MCP_CODEBASE_NATIVE_RERANK'] = 'false'

    # Force reload of config to pick up env change
    import importlib
    import config
    importlib.reload(config)

    from embedder import Embedder
    from qdrant_store import QdrantStore
    from config import settings

    # Setup services
    embedder = Embedder(
        provider=settings.embedder_provider,
        api_key=settings.embedder_api_key,
        model_id=settings.embedder_model_id,
        base_url=settings.embedder_base_url,
        dimension=settings.embedder_dimension
    )

    qdrant = QdrantStore(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key
    )

    query = "voyage rerank implementation how does it work"
    collection = "codebase-1d85d0a83c1348b3be"

    print(f"\nQuery: {query}")
    print(f"Collection: {collection}")

    # Create embedding
    vector = await embedder.create_embedding(query)
    print(f"✅ Embedding creado: {len(vector)} dimensiones")

    # Search in Qdrant
    results = await qdrant.search(
        vector=vector,
        workspace_path="",
        directory_prefix=None,
        min_score=0.1,
        max_results=20,
        collection_name=collection
    )

    print(f"\n📊 Resultados (top 10):")
    for i, r in enumerate(results[:10], 1):
        print(f"\n{i}. {r.file_path}")
        print(f"   Score: {r.score:.4f}")
        print(f"   Líneas: {r.start_line}-{r.end_line}")

    # Restore original env
    if original_native_rerank:
        os.environ['MCP_CODEBASE_NATIVE_RERANK'] = original_native_rerank
    else:
        os.environ.pop('MCP_CODEBASE_NATIVE_RERANK', None)

    return results


async def test_with_rerank():
    """Test search WITH Voyage rerank."""
    print("\n" + "="*80)
    print("TEST 2: CON VOYAGE RERANK (semantic relevance)")
    print("="*80)

    # Enable rerank
    os.environ['MCP_CODEBASE_NATIVE_RERANK'] = 'true'

    # Force reload of config
    import importlib
    import config
    importlib.reload(config)

    from embedder import Embedder
    from qdrant_store import QdrantStore
    from voyage_reranker import VoyageReranker, SearchResult
    from config import settings

    # Setup services
    embedder = Embedder(
        provider=settings.embedder_provider,
        api_key=settings.embedder_api_key,
        model_id=settings.embedder_model_id,
        base_url=settings.embedder_base_url,
        dimension=settings.embedder_dimension
    )

    qdrant = QdrantStore(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key
    )

    reranker = VoyageReranker(
        api_key=settings.voyage_api_key,
        model=settings.voyage_rerank_model,
        top_k=None,  # Return all reranked
        truncation=True
    )

    query = "voyage rerank implementation how does it work"
    collection = "codebase-1d85d0a83c1348b3be"

    print(f"\nQuery: {query}")
    print(f"Collection: {collection}")
    print(f"Modelo Voyage: {settings.voyage_rerank_model}")

    # Create embedding
    vector = await embedder.create_embedding(query)
    print(f"✅ Embedding creado: {len(vector)} dimensiones")

    # Search in Qdrant
    raw_results = await qdrant.search(
        vector=vector,
        workspace_path="",
        directory_prefix=None,
        min_score=0.1,
        max_results=20,
        collection_name=collection
    )

    print(f"✅ Búsqueda Qdrant: {len(raw_results)} resultados")

    # Convert to SearchResult for reranker
    search_results = [
        SearchResult(
            file_path=r.file_path,
            code_chunk=r.code_chunk,
            start_line=r.start_line,
            end_line=r.end_line,
            score=r.score
        )
        for r in raw_results
    ]

    # Rerank with Voyage
    reranked = await reranker.rerank(query, search_results, top_k=10)

    print(f"\n📊 Resultados rerankeados (top 10):")
    for i, r in enumerate(reranked, 1):
        print(f"\n{i}. {r.file_path}")
        print(f"   Relevance (Voyage): {r.score:.4f}")
        print(f"   Líneas: {r.start_line}-{r.end_line}")

    return reranked


async def compare_results():
    """Compare results from both methods."""
    print("\n" + "="*80)
    print("COMPARACIÓN DE RESULTADOS")
    print("="*80)

    # Run both tests
    without_rerank = await test_without_rerank()
    with_rerank = await test_with_rerank()

    # Compare top 5
    print("\n" + "="*80)
    print("COMPARACIÓN TOP 5 ARCHIVOS")
    print("="*80)

    print("\nSIN RERANK (Vector Similarity):")
    for i, r in enumerate(without_rerank[:5], 1):
        print(f"{i}. {r.file_path} (score: {r.score:.4f})")

    print("\nCON VOYAGE RERANK (Semantic Relevance):")
    for i, r in enumerate(with_rerank[:5], 1):
        print(f"{i}. {r.file_path} (score: {r.score:.4f})")

    # Analyze differences
    print("\n" + "="*80)
    print("ANÁLISIS DE DIFERENCIAS")
    print("="*80)

    without_top5_files = [r.file_path for r in without_rerank[:5]]
    with_top5_files = [r.file_path for r in with_rerank[:5]]

    # Files that moved up with rerank
    print("\n✅ Archivos que SUBIERON con Voyage rerank:")
    for i, file in enumerate(with_top5_files, 1):
        try:
            old_pos = without_top5_files.index(file) + 1
            if old_pos > i:
                print(f"  • {file}: posición {old_pos} → {i} (+{old_pos - i} posiciones)")
        except ValueError:
            print(f"  • {file}: NO estaba en top 5 → ahora en posición {i}")

    # Files that moved down with rerank
    print("\n❌ Archivos que BAJARON con Voyage rerank:")
    for i, file in enumerate(without_top5_files, 1):
        try:
            new_pos = with_top5_files.index(file) + 1
            if new_pos > i:
                print(f"  • {file}: posición {i} → {new_pos} (-{new_pos - i} posiciones)")
        except ValueError:
            print(f"  • {file}: estaba en posición {i} → ahora fuera del top 5")


async def main():
    """Run comparison tests."""
    try:
        await compare_results()
        print("\n" + "="*80)
        print("✅ PRUEBAS COMPLETADAS")
        print("="*80)
    except Exception as e:
        print(f"\n❌ Error durante las pruebas: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    asyncio.run(main())
