"""Test script for codebase search functionality."""
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

    # Set dummy judge credentials (not used in this test)
    if 'MCP_CODEBASE_JUDGE_API_KEY' not in os.environ and 'JUDGE_API_KEY' not in os.environ:
        os.environ['MCP_CODEBASE_JUDGE_API_KEY'] = 'DUMMY'

    return data


async def main():
    """Run search test."""
    data = setup_environment()

    # Import after setting environment variables
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
    
    print("Ejecutando búsqueda:")
    print(f" - Workspace: {data['workspace_path']}")
    print(f" - Query: {data['query']}")
    print()
    
    # Create embedding
    vector = await embedder.create_embedding(data['query'])
    print(f"✅ Embedding creado: {len(vector)} dimensiones")
    
    # Search in Qdrant
    results = await qdrant_store.search(
        vector=vector,
        workspace_path=data['workspace_path'],
        min_score=settings.search_min_score,
        max_results=settings.search_max_results
    )
    
    print(f"\n✅ Resultados encontrados: {len(results)}")
    
    if results:
        print("\nPrimer resultado:")
        first = results[0]
        print(f"  Ruta: {first.file_path}")
        print(f"  Score: {first.score:.4f}")
        print(f"  Líneas: {first.start_line}-{first.end_line}")
        print(f"  Extracto:\n{first.code_chunk.strip()}")
        
        if len(results) > 1:
            print(f"\n... y {len(results) - 1} resultados más")
    else:
        print("\n⚠️  No se encontraron resultados")


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"\n❌ Error durante la prueba de búsqueda: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

