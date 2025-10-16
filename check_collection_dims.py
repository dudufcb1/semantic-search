#!/usr/bin/env python3
"""
Script para verificar las dimensiones de las colecciones en Qdrant.
"""
import asyncio
import sys
from pathlib import Path
import httpx

async def main():
    print("\n🔍 VERIFICANDO DIMENSIONES DE COLECCIONES EN QDRANT\n")
    print("=" * 60)
    
    qdrant_url = "http://localhost:6333"
    
    # 1. Listar todas las colecciones
    print(f"\n📋 LISTANDO COLECCIONES EN {qdrant_url}...")
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{qdrant_url}/collections")
        data = response.json()
        
        if "result" not in data or "collections" not in data["result"]:
            print("❌ No se pudieron obtener las colecciones")
            return
        
        collections = data["result"]["collections"]
        print(f"✅ Encontradas {len(collections)} colecciones\n")
        
        # 2. Mostrar info de cada colección
        for coll in collections:
            name = coll["name"]
            print(f"\n📦 Colección: {name}")
            
            # Obtener detalles de la colección
            detail_response = await client.get(f"{qdrant_url}/collections/{name}")
            detail_data = detail_response.json()
            
            if "result" in detail_data:
                result = detail_data["result"]
                
                # Dimensiones del vector
                if "config" in result and "params" in result["config"]:
                    params = result["config"]["params"]
                    if "vectors" in params:
                        vectors_config = params["vectors"]
                        
                        # Puede ser un dict directo o tener múltiples vectores
                        if isinstance(vectors_config, dict):
                            if "size" in vectors_config:
                                # Configuración simple
                                dims = vectors_config["size"]
                                distance = vectors_config.get("distance", "N/A")
                                print(f"   Dimensiones: {dims}")
                                print(f"   Distancia: {distance}")
                            else:
                                # Múltiples vectores
                                for vec_name, vec_config in vectors_config.items():
                                    dims = vec_config.get("size", "N/A")
                                    distance = vec_config.get("distance", "N/A")
                                    print(f"   Vector '{vec_name}':")
                                    print(f"     Dimensiones: {dims}")
                                    print(f"     Distancia: {distance}")
                
                # Número de puntos
                if "points_count" in result:
                    points = result["points_count"]
                    print(f"   Puntos indexados: {points}")
                
                # Status
                if "status" in result:
                    status = result["status"]
                    print(f"   Status: {status}")
        
        print("\n" + "=" * 60)
        
        # 3. Verificar colección específica
        target_collection = "codebase-e3047b8eb7d143b790"
        print(f"\n🎯 VERIFICANDO COLECCIÓN ESPECÍFICA: {target_collection}")
        
        found = False
        for coll in collections:
            if coll["name"] == target_collection:
                found = True
                detail_response = await client.get(f"{qdrant_url}/collections/{target_collection}")
                detail_data = detail_response.json()
                
                if "result" in detail_data:
                    result = detail_data["result"]
                    
                    if "config" in result and "params" in result["config"]:
                        params = result["config"]["params"]
                        if "vectors" in params:
                            vectors_config = params["vectors"]
                            
                            if isinstance(vectors_config, dict) and "size" in vectors_config:
                                dims = vectors_config["size"]
                                print(f"\n✅ Colección encontrada")
                                print(f"   Dimensiones configuradas: {dims}")
                                print(f"   Puntos indexados: {result.get('points_count', 'N/A')}")
                                
                                # Comparar con .env
                                print(f"\n⚠️  COMPARACIÓN:")
                                print(f"   Dimensiones en Qdrant: {dims}")
                                print(f"   Dimensiones en .env: 4096 (MCP_CODEBASE_EMBEDDER_DIMENSION)")
                                
                                if dims != 4096:
                                    print(f"\n🚨 PROBLEMA DETECTADO:")
                                    print(f"   La colección tiene {dims} dimensiones")
                                    print(f"   Pero estás buscando con vectores de 4096 dimensiones")
                                    print(f"\n💡 SOLUCIONES:")
                                    print(f"   1. Re-indexar con el modelo actual (Qwen 4096 dims)")
                                    print(f"   2. Cambiar a un modelo de {dims} dimensiones")
                                    print(f"   3. Usar una colección diferente indexada con 4096 dims")
                                else:
                                    print(f"\n✅ Las dimensiones coinciden!")
                break
        
        if not found:
            print(f"\n❌ Colección '{target_collection}' no encontrada")
            print(f"\nColecciones disponibles:")
            for coll in collections:
                print(f"   - {coll['name']}")
    
    print("\n✨ Verificación completada\n")


if __name__ == "__main__":
    asyncio.run(main())

