#!/usr/bin/env python3
"""
Script de prueba para smart_merge_search_results.

Simula resultados de búsqueda semántica y prueba toda la lógica:
- Agrupación de chunks por archivo
- Validación "misma foto"
- Lógica de tamaño (completo vs fragmentos)
- Simulación de numeración
"""
import sys
from pathlib import Path

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from chunk_merger import smart_merge_search_results


def test_with_real_database():
    """
    Prueba con resultados reales de la base de datos SQLite.
    """
    import sqlite3
    import sqlite_vec
    
    # Configuración
    workspace = "/media/eduardo/56087475087455C9/Dev/Laravel/backend-paideia"
    db_path = f"{workspace}/.codebase/vectors.db"
    
    print("=" * 80)
    print("TEST: smart_merge_search_results con base de datos real")
    print("=" * 80)
    print(f"\nWorkspace: {workspace}")
    print(f"Database: {db_path}\n")
    
    # Conectar a la base de datos
    conn = sqlite3.connect(db_path)
    sqlite_vec.load(conn)
    cursor = conn.cursor()
    
    # Obtener algunos resultados de ejemplo (primeros 20 registros)
    cursor.execute("""
        SELECT file_path, code_chunk, start_line, end_line, 0.5 as distance
        FROM code_vectors
        LIMIT 20
    """)
    
    search_results = cursor.fetchall()
    conn.close()
    
    print(f"Resultados obtenidos: {len(search_results)}")
    print("\nArchivos en resultados:")
    files = set(r[0] for r in search_results)
    for f in sorted(files):
        count = sum(1 for r in search_results if r[0] == f)
        print(f"  - {f}: {count} chunks")
    
    # Procesar con smart_merge
    print("\n" + "-" * 80)
    print("Procesando con smart_merge_search_results...")
    print("-" * 80 + "\n")
    
    results = smart_merge_search_results(
        workspace_path=workspace,
        search_results=search_results,
        max_files=5  # Limitar a 5 archivos para la prueba
    )
    
    # Mostrar resultados
    print(f"\n{'=' * 80}")
    print(f"RESULTADOS PROCESADOS: {len(results)} archivos únicos")
    print(f"{'=' * 80}\n")
    
    for file_path, data in results.items():
        print(f"\n{'─' * 80}")
        print(f"📄 {file_path}")
        print(f"{'─' * 80}")
        print(f"  Chunks: {data['chunks_count']}")
        print(f"  Total líneas: {data['total_lines']}")
        print(f"  Cobertura: {data['coverage']:.2%}")
        print(f"  Mostrar completo: {data['show_complete']}")
        print(f"  Validado (misma foto): {data['validated']}")
        print(f"  Distancia: {data['distance']:.4f}")
        print(f"\n  Contenido (primeras 20 líneas):")
        print(f"  {'-' * 76}")
        
        # Mostrar primeras 20 líneas del contenido
        lines = data['content'].split('\n')[:20]
        for line in lines:
            print(f"  {line}")
        
        if len(data['content'].split('\n')) > 20:
            print(f"  ... ({len(data['content'].split('\n')) - 20} líneas más)")
    
    print(f"\n{'=' * 80}\n")


def test_with_simulated_data():
    """
    Prueba con datos simulados para validar la lógica.
    """
    print("=" * 80)
    print("TEST: smart_merge_search_results con datos simulados")
    print("=" * 80)
    
    workspace = "/media/eduardo/56087475087455C9/Dev/llm_codebase_search/python-mcp"
    
    # Simular resultados de búsqueda
    # Formato: (file_path, code_chunk, start_line, end_line, distance)
    search_results = [
        # Archivo 1: src/embedder.py (3 chunks)
        ("src/embedder.py", "import httpx\nfrom typing import Optional", 1, 3, 0.15),
        ("src/embedder.py", "class Embedder:\n    def __init__", 20, 30, 0.18),
        ("src/embedder.py", "async def create_embedding", 60, 75, 0.20),
        
        # Archivo 2: src/config.py (1 chunk)
        ("src/config.py", "from pydantic_settings import BaseSettings", 1, 10, 0.25),
        
        # Archivo 3: src/server.py (2 chunks)
        ("src/server.py", "from fastmcp import FastMCP", 1, 15, 0.30),
        ("src/server.py", "@mcp.tool\nasync def superior_codebase_search", 50, 80, 0.32),
    ]
    
    print(f"\nWorkspace: {workspace}")
    print(f"Resultados simulados: {len(search_results)}")
    print("\nArchivos en resultados:")
    files = set(r[0] for r in search_results)
    for f in sorted(files):
        count = sum(1 for r in search_results if r[0] == f)
        print(f"  - {f}: {count} chunks")
    
    # Procesar con smart_merge
    print("\n" + "-" * 80)
    print("Procesando con smart_merge_search_results...")
    print("-" * 80 + "\n")
    
    results = smart_merge_search_results(
        workspace_path=workspace,
        search_results=search_results,
        max_files=10
    )
    
    # Mostrar resultados
    print(f"\n{'=' * 80}")
    print(f"RESULTADOS PROCESADOS: {len(results)} archivos únicos")
    print(f"{'=' * 80}\n")
    
    for file_path, data in results.items():
        print(f"\n{'─' * 80}")
        print(f"📄 {file_path}")
        print(f"{'─' * 80}")
        print(f"  Chunks: {data['chunks_count']}")
        print(f"  Total líneas: {data['total_lines']}")
        print(f"  Cobertura: {data['coverage']:.2%}")
        print(f"  Mostrar completo: {data['show_complete']}")
        print(f"  Validado (misma foto): {data['validated']}")
        print(f"  Distancia: {data['distance']:.4f}")
        print(f"\n  Contenido (primeras 15 líneas):")
        print(f"  {'-' * 76}")
        
        # Mostrar primeras 15 líneas del contenido
        lines = data['content'].split('\n')[:15]
        for line in lines:
            print(f"  {line}")
        
        if len(data['content'].split('\n')) > 15:
            print(f"  ... ({len(data['content'].split('\n')) - 15} líneas más)")
    
    print(f"\n{'=' * 80}\n")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "real":
        # Prueba con base de datos real
        test_with_real_database()
    else:
        # Prueba con datos simulados
        print("\nUso:")
        print("  python test_smart_merge.py          # Prueba con datos simulados")
        print("  python test_smart_merge.py real     # Prueba con base de datos real\n")
        
        test_with_simulated_data()

