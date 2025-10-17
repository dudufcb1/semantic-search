#!/usr/bin/env python3
"""
Script de prueba para verificar búsqueda en Qdrant usando la implementación Python.
Replica el comportamiento del test de Node.js.
"""
import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import settings
from embedder import Embedder
from qdrant_store import QdrantStore


async def main():
    print("\n🚀 TEST DE BÚSQUEDA PYTHON (Replicando test-qdrant.js)\n")
    print("=" * 60)
    
    # 1. Mostrar configuración
    print("\n📋 CONFIGURACIÓN:")
    print(f"Qdrant URL: {settings.qdrant_url}")
    print(f"Qdrant API Key: {'***' + settings.qdrant_api_key[-8:] if settings.qdrant_api_key else '(vacío)'}")
    print(f"Embed Provider: {settings.embedder_provider}")
    print(f"Embed Model: {settings.embedder_model_id}")
    print(f"Embed Dimension: {settings.embedder_dimension or 'auto-detect'}")
    print(f"Embed Base URL: {settings.embedder_base_url}")
    print(f"Search Min Score: {settings.search_min_score}")
    print(f"Search Max Results: {settings.search_max_results}")
    
    # 2. Crear servicios
    print("\n🔧 CREANDO SERVICIOS...")
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
    print("✅ Servicios creados")
    
    # 3. Parámetros de búsqueda (EXACTOS del test JS)
    query = "How does the RAG service retrieve context for chat messages"
    collection_name = "codebase-e3047b8eb7d143b790"

    # TEST: Probar con y sin intent
    search_intent = "implementation"  # ← CAMBIAR A None PARA PROBAR SIN INTENT

    print(f"\n🔍 PARÁMETROS DE BÚSQUEDA:")
    print(f"Query: \"{query}\"")
    print(f"Collection: {collection_name}")
    print(f"Search Intent: {search_intent or '(ninguno)'}")
    
    # 4. Crear embedding (con o sin intent)
    print(f"\n📊 CREANDO EMBEDDING...")
    try:
        # Aplicar intent enhancement si está presente (como hace el código real)
        enhanced_query = query
        if search_intent:
            enhanced_query = f"{query} [INTENT: {search_intent}]"
            print(f"Query mejorada: \"{enhanced_query}\"")

        vector = await embedder.create_embedding(enhanced_query)
        print(f"✅ Embedding creado")
        print(f"Dimensión del vector: {len(vector)}")
        print(f"Primeros 5 valores: {vector[:5]}")
    except Exception as e:
        print(f"❌ Error al crear embedding: {e}")
        return
    
    # 5. Realizar búsqueda
    print(f"\n🔎 REALIZANDO BÚSQUEDA EN QDRANT...")
    try:
        results = await qdrant_store.search(
            vector=vector,
            workspace_path="",  # No usado cuando se pasa collection_name
            directory_prefix=None,
            min_score=settings.search_min_score,
            max_results=settings.search_max_results,
            collection_name=collection_name
        )
        
        print(f"✅ Búsqueda exitosa")
        print(f"\nResultados encontrados: {len(results)}")
        
        if results:
            scores = [r.score for r in results]
            print(f"Mejor score: {max(scores):.6f}")
            print(f"Peor score: {min(scores):.6f}")
            
            print(f"\n📄 PRIMEROS 3 RESULTADOS:")
            for i, result in enumerate(results[:3], 1):
                print(f"\n  Resultado {i}:")
                print(f"    File: {result.file_path}")
                print(f"    Lines: {result.start_line}-{result.end_line}")
                print(f"    Score: {result.score:.6f}")
                preview = result.code_chunk[:100].replace('\n', ' ')
                print(f"    Preview: {preview}...")
        else:
            print("⚠️  No se encontraron resultados")
            print(f"\n💡 POSIBLES CAUSAS:")
            print(f"   - Score threshold muy alto ({settings.search_min_score})")
            print(f"   - Colección vacía")
            print(f"   - Dimensión del vector incorrecta")
            
    except Exception as e:
        print(f"❌ Error al buscar en Qdrant: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n✨ Prueba completada\n")


if __name__ == "__main__":
    asyncio.run(main())

