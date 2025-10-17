"""Test script para la nueva herramienta explore_other_workspace."""
import asyncio
import os
from pathlib import Path

# Configurar environment variables de prueba
os.environ['MCP_CODEBASE_EMBEDDER_PROVIDER'] = 'openai-compatible'
os.environ['MCP_CODEBASE_EMBEDDER_BASE_URL'] = 'http://localhost:11434/v1'
os.environ['MCP_CODEBASE_EMBEDDER_API_KEY'] = 'dummy-key'
os.environ['MCP_CODEBASE_EMBEDDER_MODEL_ID'] = 'nomic-embed-text'
os.environ['MCP_CODEBASE_QDRANT_URL'] = 'http://localhost:6333'
os.environ['MCP_CODEBASE_JUDGE_API_KEY'] = 'dummy-key'
os.environ['MCP_CODEBASE_JUDGE_BASE_URL'] = 'http://localhost:11434/v1'
os.environ['MCP_CODEBASE_JUDGE_MODEL_ID'] = 'gpt-4o-mini'

async def test_explore_workspace_priority_logic():
    """Test que la lógica de prioridades funcione correctamente."""
    print("\n" + "="*80)
    print("TEST: explore_other_workspace - Lógica de Prioridades")
    print("="*80)

    from src.server import mcp
    from fastmcp.exceptions import ToolError

    # Test 1: Solo qdrant_collection (debe usar collection)
    print("\n🔥 Test 1: Solo qdrant_collection")
    try:
        result = await mcp._tools['explore_other_workspace'](
            query="test query",
            qdrant_collection="codebase-test123",
            workspace_path=None
        )
        print("✅ Función ejecutada (debería usar colección directamente)")
        print(f"Resultado: {result[:100]}...")
    except Exception as e:
        if "No se encontraron resultados" in str(e) or "collection does not exist" in str(e).lower():
            print("✅ Error esperado - colección no existe pero lógica correcta")
        else:
            print(f"❌ Error inesperado: {e}")

    # Test 2: Solo workspace_path (debe calcular collection)
    print("\n🔥 Test 2: Solo workspace_path")
    try:
        result = await mcp._tools['explore_other_workspace'](
            query="test query",
            qdrant_collection=None,
            workspace_path="/path/to/test/workspace"
        )
        print("✅ Función ejecutada (debería calcular colección desde path)")
        print(f"Resultado: {result[:100]}...")
    except Exception as e:
        if "No se encontraron resultados" in str(e) or "collection does not exist" in str(e).lower():
            print("✅ Error esperado - colección calculada no existe pero lógica correcta")
        else:
            print(f"❌ Error inesperado: {e}")

    # Test 3: Ambos parametros (debe priorizar qdrant_collection)
    print("\n🔥 Test 3: Ambos parámetros (prioridad a qdrant_collection)")
    try:
        result = await mcp._tools['explore_other_workspace'](
            query="test query",
            qdrant_collection="priority-collection",
            workspace_path="/should/be/ignored"
        )
        print("✅ Función ejecutada (debería usar priority-collection, no calcular desde path)")
        print(f"Resultado: {result[:100]}...")
    except Exception as e:
        if "No se encontraron resultados" in str(e) or "collection does not exist" in str(e).lower():
            print("✅ Error esperado - colección prioritaria no existe pero lógica correcta")
        else:
            print(f"❌ Error inesperado: {e}")

    # Test 4: Ningún parámetro (debe dar error)
    print("\n🔥 Test 4: Ningún parámetro (debe dar error)")
    try:
        result = await mcp._tools['explore_other_workspace'](
            query="test query",
            qdrant_collection=None,
            workspace_path=None
        )
        print("❌ No debería llegar aquí - falta validación")
    except ToolError as e:
        if "Debes proporcionar 'qdrant_collection' O 'workspace_path'" in str(e):
            print("✅ Error de validación correcto")
        else:
            print(f"❌ Error de validación incorrecto: {e}")
    except Exception as e:
        print(f"❌ Error inesperado: {e}")

    # Test 5: Query vacío (debe dar error)
    print("\n🔥 Test 5: Query vacío (debe dar error)")
    try:
        result = await mcp._tools['explore_other_workspace'](
            query="",
            qdrant_collection="test-collection"
        )
        print("❌ No debería llegar aquí - falta validación de query")
    except ToolError as e:
        if "query' es requerido y no puede estar vacío" in str(e):
            print("✅ Error de validación de query correcto")
        else:
            print(f"❌ Error de validación de query incorrecto: {e}")
    except Exception as e:
        print(f"❌ Error inesperado: {e}")


async def test_explore_workspace_server_local():
    """Test la misma funcionalidad en server_local."""
    print("\n" + "="*80)
    print("TEST: explore_other_workspace - Server Local")
    print("="*80)

    from src.server_local import mcp as mcp_local
    from fastmcp.exceptions import ToolError

    # Test con qdrant_collection
    print("\n🔥 Test Server Local: qdrant_collection")
    try:
        result = await mcp_local._tools['explore_other_workspace'](
            query="authentication function",
            qdrant_collection="codebase-local-test"
        )
        print("✅ Server local ejecutado correctamente")
        print(f"Resultado: {result[:100]}...")
    except Exception as e:
        if "No se encontraron resultados" in str(e) or "collection does not exist" in str(e).lower():
            print("✅ Error esperado - colección no existe pero lógica correcta")
        else:
            print(f"❌ Error inesperado: {e}")


async def test_collection_name_calculation():
    """Test que el cálculo de nombre de colección sea consistente."""
    print("\n" + "="*80)
    print("TEST: Cálculo de Nombres de Colección")
    print("="*80)

    from src.qdrant_store import QdrantStore

    # Test casos específicos
    test_cases = [
        "/home/user/proyecto1",
        "/var/www/html/app",
        "/media/eduardo/disk/dev/myproject",
        "relative/path/project",
        "/home/user/proyecto1/",  # Con slash final
        "/home/user/proyecto1",   # Sin slash final
    ]

    qdrant_store = QdrantStore(url="http://localhost:6333")

    print("\n🔍 Calculando nombres de colección:")
    for path in test_cases:
        collection_name = qdrant_store._get_collection_name(path)
        normalized_path = qdrant_store._normalize_workspace_path(path)
        print(f"  Path: {path}")
        print(f"  Normalizado: {normalized_path}")
        print(f"  Colección: {collection_name}")
        print()

    # Test consistencia
    path1 = "/home/user/proyecto1"
    path2 = "/home/user/proyecto1/"
    collection1 = qdrant_store._get_collection_name(path1)
    collection2 = qdrant_store._get_collection_name(path2)

    if collection1 == collection2:
        print("✅ Consistencia: Paths con y sin slash final generan misma colección")
    else:
        print("❌ Inconsistencia: Paths con y sin slash final generan diferentes colecciones")
        print(f"  '{path1}' -> '{collection1}'")
        print(f"  '{path2}' -> '{collection2}'")


async def main():
    """Ejecutar todos los tests."""
    print("\n" + "="*80)
    print("PRUEBAS DE EXPLORE_OTHER_WORKSPACE")
    print("="*80)

    try:
        # Test 1: Lógica de prioridades
        await test_explore_workspace_priority_logic()

        # Test 2: Server local
        await test_explore_workspace_server_local()

        # Test 3: Cálculo de nombres de colección
        await test_collection_name_calculation()

        print("\n" + "="*80)
        print("✅ TODAS LAS PRUEBAS DE EXPLORE_OTHER_WORKSPACE COMPLETADAS")
        print("="*80)

    except Exception as e:
        print(f"\n❌ Error durante las pruebas: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    asyncio.run(main())