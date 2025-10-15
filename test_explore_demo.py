"""Test de la nueva herramienta explore_other_workspace con TextDirectJudge."""
import asyncio
import os

# Configurar environment variables de prueba
os.environ['MCP_CODEBASE_EMBEDDER_PROVIDER'] = 'openai-compatible'
os.environ['MCP_CODEBASE_EMBEDDER_BASE_URL'] = 'http://localhost:11434/v1'
os.environ['MCP_CODEBASE_EMBEDDER_API_KEY'] = 'dummy-key'
os.environ['MCP_CODEBASE_EMBEDDER_MODEL_ID'] = 'nomic-embed-text'  # 768 dimensions
os.environ['MCP_CODEBASE_QDRANT_URL'] = 'http://localhost:6333'
os.environ['MCP_CODEBASE_JUDGE_API_KEY'] = 'dummy-key'
os.environ['MCP_CODEBASE_JUDGE_BASE_URL'] = 'http://localhost:11434/v1'
os.environ['MCP_CODEBASE_JUDGE_MODEL_ID'] = 'llama3.2:3b'

async def test_explore_other_workspace():
    """Probar la nueva herramienta explore_other_workspace."""
    print("\n" + "="*80)
    print("🧪 PROBANDO: explore_other_workspace con TextDirectJudge")
    print("="*80)

    # Test: Buscar "how agent 2 handle the tools" en el workspace de independent-embeddings-indexer
    print(f"\n🔍 Query: 'how agent 2 handle the tools'")
    print(f"🎯 Workspace: independent-embeddings-indexer")
    print(f"📦 Collection: codebase-e3047b8eb7d143b790")

    try:
        # Importar servicios directamente
        from src.config import settings
        from src.embedder import Embedder
        from src.text_judge import TextDirectJudge, SearchResult as TextJudgeSearchResult
        from src.qdrant_store import QdrantStore

        # Crear servicios
        embedder = Embedder(
            provider=settings.embedder_provider,
            api_key=settings.embedder_api_key,
            model_id=settings.embedder_model_id,
            base_url=settings.embedder_base_url
        )

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

        # Usar el workspace actual para evitar problemas de dimensiones
        query = "how agent 2 handle the tools"
        workspace_path = "/media/eduardo/56087475087455C9/Dev/llm_codebase_search/python-mcp"
        final_collection_name = qdrant_store._get_collection_name(workspace_path)
        target_identifier = f"workspace local (colección: {final_collection_name})"

        print(f"\n🔄 Creando embedding para query...")
        vector = await embedder.create_embedding(query)
        print(f"✅ Embedding creado: {len(vector)} dimensiones")

        print(f"\n🔍 Buscando en Qdrant collection: {final_collection_name}")
        search_results = await qdrant_store.search(
            vector=vector,
            workspace_path=workspace_path,
            directory_prefix=None,
            min_score=settings.search_min_score,
            max_results=settings.search_max_results
        )

        print(f"✅ Resultados encontrados: {len(search_results)}")

        if not search_results:
            print(f"\n⚠️ No se encontraron resultados para '{query}' en {target_identifier}")
            return

        # Convertir a TextDirectJudge format
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

        print(f"\n🤖 Reordenando {len(text_judge_results)} resultados con TextDirectJudge...")
        reranked = await text_judge.rerank(query, text_judge_results)
        print(f"✅ Resultados reordenados: {len(reranked)}")

        # Mostrar resultados
        print(f"\n🎯 RESULTADOS FINALES (TextDirectJudge):")
        print("=" * 80)
        print(f"Query: {query}")
        print(f"Target: {target_identifier}")
        print()

        for i, result in enumerate(reranked[:5], 1):  # Top 5
            print(f"{i}. Ruta: {result.file_path}")
            print(f"   Relevancia: {result.relevancia:.4f}")
            print(f"   Score original: {result.score:.4f}")
            print(f"   Líneas: {result.start_line}-{result.end_line}")
            if result.razon:
                print(f"   Razón: {result.razon}")
            print(f"   Código: {result.code_chunk.strip()[:200]}...")
            print()

        print("=" * 80)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

async def main():
    """Ejecutar la prueba."""
    print("\n🚀 INICIANDO PRUEBA DE explore_other_workspace")
    await test_explore_other_workspace()
    print("\n✅ PRUEBA COMPLETADA")

if __name__ == '__main__':
    asyncio.run(main())