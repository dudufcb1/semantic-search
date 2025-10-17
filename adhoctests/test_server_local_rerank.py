#!/usr/bin/env python3
"""
Script de prueba para replicar el flujo de superior_codebase_rerank de server_local.py
"""
import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import settings
from embedder import Embedder
from text_judge import TextDirectJudge, SearchResult as TextJudgeSearchResult
from qdrant_store import QdrantStore

# MODE from server_local.py
MODE = "text"


async def main():
    print("\n🚀 TEST DE superior_codebase_rerank (Replicando server_local.py)\n")
    print("=" * 70)

    # Parámetros de prueba
    query = "How does the RAG service retrieve context for chat messages"
    qdrant_collection = "codebase-e3047b8eb7d143b790"
    mode = "rerank"
    path = None  # directory_prefix

    print(f"\n📋 PARÁMETROS:")
    print(f"Query: \"{query}\"")
    print(f"Collection: {qdrant_collection}")
    print(f"Mode: {mode}")
    print(f"MODE: {MODE}")
    print(f"\n{'=' * 70}\n")

    # Initialize services (same as server_local.py)
    print("🔧 INICIALIZANDO SERVICIOS...")
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
    print("✅ Servicios inicializados\n")

    try:
        # Step 1: Perform initial search (same as server_local.py lines 379-416)
        print(f"[DEBUG] Buscando resultados para query: '{query}'")
        print(f"[DEBUG] Colección: {qdrant_collection}")
        print(f"[DEBUG] Min score: {settings.search_min_score}")
        print(f"[DEBUG] Max results: {settings.search_max_results}")

        vector = await embedder.create_embedding(query)

        print(f"[DEBUG] Vector generado: dimensión={len(vector)}, primeros 5 valores={vector[:5]}")

        search_results = await qdrant_store.search(
            vector=vector,
            workspace_path="",  # Not used when collection_name is provided
            directory_prefix=path,
            min_score=settings.search_min_score,
            max_results=settings.search_max_results,
            collection_name=qdrant_collection
        )

        print(f"[DEBUG] Resultados de Qdrant: {len(search_results)} encontrados")
        if search_results:
            print(f"[DEBUG] Mejor score: {max(r.score for r in search_results):.6f}")
            print(f"[DEBUG] Peor score: {min(r.score for r in search_results):.6f}")

            # Count unique files
            from collections import defaultdict
            grouped = defaultdict(list)
            for r in search_results:
                grouped[r.file_path].append(r)
            print(f"[DEBUG] Archivos únicos: {len(grouped)}")
            for file_path, file_results in sorted(grouped.items(), key=lambda x: max(r.score for r in x[1]), reverse=True)[:5]:
                print(f"[DEBUG]   - {file_path}: {len(file_results)} fragmentos")

        if not search_results:
            print(f"\n⚠️  No se encontraron resultados para \"{query}\" en la colección \"{qdrant_collection}\".")
            return

        # Step 2: Convert to TextDirectJudge format (same as server_local.py lines 418-427)
        print(f"\n[DEBUG] Convirtiendo {len(search_results)} resultados a formato TextDirectJudge...")
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

        # Step 3: Generate markdown summary (NO PARSING)
        print(f"\n[DEBUG] Generando resumen markdown con LLM...")

        markdown_response = await text_judge.generate_markdown_summary(query, text_judge_results)

        # Save full response to file
        with open('/tmp/llm_markdown_response.txt', 'w') as f:
            f.write(markdown_response)

        print(f"\n[DEBUG] Resumen markdown generado ({len(markdown_response)} caracteres)")
        print(f"💾 Respuesta completa guardada en: /tmp/llm_markdown_response.txt")

        # Step 4: Display markdown result
        print(f"\n{'=' * 70}")
        print(f"\n✅ RESULTADO MARKDOWN:\n")
        print(markdown_response)
        print(f"\n{'=' * 70}\n")

    except Exception as e:
        print(f"\n{'=' * 70}")
        print(f"\n❌ ERROR:\n")
        print(f"{type(e).__name__}: {e}")
        print(f"\n{'=' * 70}\n")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())

