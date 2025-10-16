"""
Utilidad para concatenar múltiples chunks de un archivo con numeración de líneas.

Este módulo permite extraer y concatenar múltiples rangos de líneas de un archivo,
mostrando la numeración original y omitiendo las secciones intermedias.

Incluye lógica avanzada para:
- Validar si chunks coinciden con archivo real ("misma foto")
- Decidir mostrar archivo completo vs fragmentos según tamaño
- Simular numeración cuando archivo real no coincide
"""
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from collections import defaultdict

# Extensiones de archivos compilados/minificados que queremos evitar mostrar
COMPILED_EXTENSIONS = {
    '.min.js', '.min.css',  # Minificados
    '.map',                  # Source maps
    '.woff', '.woff2', '.ttf', '.eot',  # Fuentes
    '.sum', '.mod',          # Go modules
    '.svg',                  # SVG (a veces contienen mucho texto basura)
}

# Patrones en nombres de archivo que indican archivos compilados
COMPILED_PATTERNS = {
    'lock',  # package-lock.json, yarn.lock, composer.lock, etc.
    '.bundle.',  # Archivos bundle
    '.chunk.',   # Archivos chunk de webpack
}


def should_skip_file(file_path: str, content: str) -> bool:
    """
    Determina si un archivo debe ser omitido por ser compilado/minificado.

    Usa una estrategia híbrida:
    1. Extensiones conocidas de archivos compilados
    2. Patrones en nombres de archivo (lock, bundle, chunk)
    3. Ratio alto de números/puntos/guiones (>40%)
    4. Palabras promedio muy cortas (<3 caracteres)

    Args:
        file_path: Ruta del archivo
        content: Contenido del archivo o chunk

    Returns:
        True si el archivo debe ser omitido, False en caso contrario
    """
    file_path_lower = file_path.lower()

    # 1. Extensión conocida
    if any(file_path_lower.endswith(ext) for ext in COMPILED_EXTENSIONS):
        return True

    # 2. Patrones en nombre de archivo
    if any(pattern in file_path_lower for pattern in COMPILED_PATTERNS):
        return True

    # 3. Ratio de números/puntos/guiones alto (indica código compilado/minificado)
    if len(content) > 100:  # Solo aplicar si hay suficiente contenido
        numbers_and_symbols = sum(1 for c in content if c.isdigit() or c in '.-')
        ratio = numbers_and_symbols / len(content)
        if ratio > 0.4:  # Más del 40% son números/símbolos
            return True

    # 4. Palabras promedio muy cortas (indica minificación)
    words = content.split()
    if len(words) > 10:  # Solo aplicar si hay suficientes palabras
        avg_word_len = sum(len(w) for w in words) / len(words)
        if avg_word_len < 3.0:  # Palabras promedio < 3 caracteres
            return True

    return False


def merge_chunks(
    file_path: str,
    ranges: List[Tuple[int, int]],
    show_omitted: bool = True
) -> str:
    """
    Extrae y concatena múltiples rangos de líneas de un archivo.
    
    Args:
        file_path: Ruta al archivo a leer
        ranges: Lista de tuplas (start_line, end_line) con los rangos a extraer.
                Las líneas son 1-based (la primera línea es 1).
        show_omitted: Si True, muestra "... código omitido ..." entre chunks
        
    Returns:
        String con los chunks concatenados y numerados
        
    Example:
        >>> merge_chunks("app.py", [(1, 5), (10, 15), (20, 25)])
        1  import os
        2  import sys
        3  
        4  def main():
        5      print("Hello")
        
        ... código omitido (líneas 6-9) ...
        
        10     def helper():
        11         return True
        12     
        13     if helper():
        14         print("OK")
        15         return 0
        
        ... código omitido (líneas 16-19) ...
        
        20 def cleanup():
        21     pass
        22 
        23 if __name__ == "__main__":
        24     main()
        25     cleanup()
    """
    # Validar que el archivo existe
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"El archivo no existe: {file_path}")
    
    if not path.is_file():
        raise ValueError(f"La ruta no es un archivo: {file_path}")
    
    # Validar y ordenar rangos
    if not ranges:
        raise ValueError("Debes proporcionar al menos un rango")
    
    # Ordenar rangos por línea de inicio
    sorted_ranges = sorted(ranges, key=lambda r: r[0])
    
    # Validar que los rangos son válidos
    for start, end in sorted_ranges:
        if start < 1:
            raise ValueError(f"Las líneas deben ser >= 1, recibido: {start}")
        if end < start:
            raise ValueError(f"end_line debe ser >= start_line: ({start}, {end})")
    
    # Leer todas las líneas del archivo
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
    
    total_lines = len(lines)
    
    # Construir el resultado
    result = []
    last_end = 0
    
    for start, end in sorted_ranges:
        # Ajustar end si excede el total de líneas
        end = min(end, total_lines)

        # Mostrar indicador de código omitido si hay gap
        if show_omitted and last_end > 0 and start > last_end + 1:
            omitted_start = last_end + 1
            omitted_end = start - 1
            result.append("")  # Línea en blanco antes del mensaje
            result.append(f"... código omitido (líneas {omitted_start}-{omitted_end}) ...")
            result.append("")  # Línea en blanco después del mensaje

        # Agregar las líneas del chunk actual
        for line_num in range(start, end + 1):
            if line_num <= total_lines:
                # Índice en el array (0-based)
                idx = line_num - 1
                line_content = lines[idx].rstrip('\n')
                result.append(f"{line_num:4d}  {line_content}")

        last_end = end

    return '\n'.join(result)


def validate_chunk_matches_file(
    file_path: str,
    chunk_content: str,
    start_line: int,
    end_line: int,
    compare_chars: int = 100
) -> bool:
    """
    Valida si un chunk coincide con el archivo real ("misma foto").

    Compara los primeros N caracteres del chunk con las líneas correspondientes
    del archivo real para determinar si están sincronizados.

    Permite una tolerancia de hasta 2 líneas vacías al final del archivo.

    Args:
        file_path: Ruta al archivo real
        chunk_content: Contenido del chunk de la base de datos
        start_line: Línea de inicio del chunk (1-based)
        end_line: Línea final del chunk (1-based)
        compare_chars: Número de caracteres a comparar (default: 100)

    Returns:
        True si el chunk coincide con el archivo real, False en caso contrario
    """
    try:
        path = Path(file_path)
        if not path.exists() or not path.is_file():
            return False

        # Leer el archivo real
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()

        total_lines = len(lines)

        # Verificar que las líneas existen en el archivo
        # Permitir hasta 2 líneas de diferencia AL FINAL si están vacías
        if end_line > total_lines:
            diff = end_line - total_lines

            # Solo permitir diferencia de máximo 2 líneas
            if diff > 2:
                return False

            # Verificar que las líneas "faltantes" en el chunk serían vacías
            # (esto significa que el archivo real tiene menos líneas vacías al final)
            # En este caso, ajustamos end_line al total de líneas del archivo
            end_line = total_lines

        # Extraer las líneas correspondientes (convertir a 0-based)
        real_content = ''.join(lines[start_line - 1:end_line])

        # Comparar primeros N caracteres
        chunk_prefix = chunk_content[:compare_chars]
        real_prefix = real_content[:compare_chars]

        return chunk_prefix == real_prefix

    except Exception:
        return False


def calculate_coverage(ranges: List[Tuple[int, int]], total_lines: int) -> float:
    """
    Calcula la cobertura de chunks sobre el archivo total.

    Args:
        ranges: Lista de tuplas (start_line, end_line)
        total_lines: Total de líneas del archivo

    Returns:
        Cobertura como float entre 0.0 y 1.0
    """
    if total_lines == 0:
        return 0.0

    # Contar líneas únicas cubiertas (manejar solapamientos)
    covered_lines = set()
    for start, end in ranges:
        for line in range(start, end + 1):
            if line <= total_lines:
                covered_lines.add(line)

    return len(covered_lines) / total_lines


def should_show_complete_file(total_lines: int, coverage: float) -> bool:
    """
    Decide si mostrar el archivo completo o solo fragmentos.

    Lógica:
    - Archivos < 100 líneas: Siempre completo
    - Archivos < 300 líneas con cobertura > 60%: Completo
    - Otros casos: Solo fragmentos

    Args:
        total_lines: Total de líneas del archivo
        coverage: Cobertura de chunks (0.0 a 1.0)

    Returns:
        True si debe mostrar archivo completo, False para fragmentos
    """
    if total_lines < 100:
        return True
    elif total_lines < 300 and coverage > 0.6:
        return True
    else:
        return False


def merge_overlapping_ranges(ranges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Fusiona rangos overlapping o contiguos en rangos únicos.

    Args:
        ranges: Lista de tuplas (start_line, end_line)

    Returns:
        Lista de rangos fusionados, ordenados por start_line

    Ejemplo:
        [(1, 349), (10, 349), (11, 349), (296, 315)] → [(1, 349)]
        [(1, 10), (15, 20), (18, 25)] → [(1, 10), (15, 25)]
    """
    if not ranges:
        return []

    # Ordenar por start_line
    sorted_ranges = sorted(ranges, key=lambda r: r[0])

    merged = [sorted_ranges[0]]

    for current_start, current_end in sorted_ranges[1:]:
        last_start, last_end = merged[-1]

        # Si el rango actual overlaps o es contiguo con el último
        if current_start <= last_end + 1:
            # Fusionar: extender el último rango si es necesario
            merged[-1] = (last_start, max(last_end, current_end))
        else:
            # No overlap: agregar como nuevo rango
            merged.append((current_start, current_end))

    return merged


def simulate_chunk_numbering(
    chunks: List[Tuple[str, int, int]]
) -> str:
    """
    Simula numeración de líneas cuando no hay archivo real disponible.
    Fusiona chunks overlapping para evitar duplicación de contenido.

    Args:
        chunks: Lista de tuplas (chunk_content, start_line, end_line)

    Returns:
        String con chunks numerados y gaps omitidos
    """
    if not chunks:
        return ""

    # Extraer rangos y fusionar overlaps
    ranges = [(start, end) for _, start, end in chunks]
    merged_ranges = merge_overlapping_ranges(ranges)

    # Crear un mapa de líneas disponibles desde los chunks
    # Usamos el chunk más largo que cubra cada línea
    line_content = {}
    for chunk_content, start_line, end_line in chunks:
        lines = chunk_content.split('\n')
        for i, line in enumerate(lines):
            line_num = start_line + i
            if line_num <= end_line:
                # Si ya existe, mantener el más largo (más contexto)
                if line_num not in line_content or len(line) > len(line_content[line_num]):
                    line_content[line_num] = line

    # Generar output usando rangos fusionados
    result = []
    last_end = 0

    for start_line, end_line in merged_ranges:
        # Mostrar gap si existe
        if last_end > 0 and start_line > last_end + 1:
            result.append("")  # Línea en blanco antes
            result.append(f"... código omitido (líneas {last_end + 1}-{start_line - 1}) ...")
            result.append("")  # Línea en blanco después

        # Agregar líneas del rango fusionado
        for line_num in range(start_line, end_line + 1):
            if line_num in line_content:
                result.append(f"{line_num:4d}  {line_content[line_num]}")

        last_end = end_line

    return '\n'.join(result)


def smart_merge_search_results(
    workspace_path: str,
    search_results: List[Tuple[str, str, int, int, float]],
    max_files: int = 10
) -> Dict[str, Dict]:
    """
    Procesa resultados de búsqueda semántica con lógica inteligente.

    Esta función implementa la lógica completa:
    1. Agrupa chunks por archivo (múltiples chunks = 1 resultado)
    2. Construye path real: workspace / file_path
    3. Valida "misma foto" (primeros caracteres coinciden)
    4. Aplica lógica de tamaño (completo vs fragmentos)
    5. Usa merge_chunks o simula numeración según validación

    Args:
        workspace_path: Ruta base del workspace
        search_results: Lista de tuplas (file_path, code_chunk, start_line, end_line, distance)
        max_files: Número máximo de archivos únicos a procesar (default: 10)

    Returns:
        Dict con estructura:
        {
            "file_path": {
                "content": "código fusionado con numeración",
                "total_lines": 150,
                "coverage": 0.75,
                "show_complete": True,
                "chunks_count": 3,
                "validated": True,  # Si coincide con archivo real
                "distance": 0.123  # Distancia mínima de los chunks
            }
        }
    """
    workspace = Path(workspace_path)

    # Paso 0: Filtrar archivos compilados/minificados ANTES de agrupar
    filtered_results = []
    for file_path, code_chunk, start_line, end_line, distance in search_results:
        if not should_skip_file(file_path, code_chunk):
            filtered_results.append((file_path, code_chunk, start_line, end_line, distance))

    # Paso 1: Agrupar por archivo
    file_data = defaultdict(lambda: {
        'chunks': [],
        'ranges': [],
        'min_distance': float('inf')
    })

    for file_path, code_chunk, start_line, end_line, distance in filtered_results:
        file_data[file_path]['chunks'].append((code_chunk, start_line, end_line))
        file_data[file_path]['ranges'].append((start_line, end_line))
        file_data[file_path]['min_distance'] = min(
            file_data[file_path]['min_distance'],
            distance
        )

    # Paso 2: Procesar cada archivo (limitar a max_files)
    results = {}
    processed_count = 0

    # Ordenar archivos por distancia mínima (más relevantes primero)
    sorted_files = sorted(
        file_data.items(),
        key=lambda x: x[1]['min_distance']
    )

    for file_path, data in sorted_files:
        if processed_count >= max_files:
            break

        # Construir path real
        real_path = workspace / file_path

        # Paso 3: Validar "misma foto"
        validated = False
        if real_path.exists() and real_path.is_file():
            # Validar con el primer chunk
            first_chunk = data['chunks'][0]
            chunk_content, start_line, end_line = first_chunk
            validated = validate_chunk_matches_file(
                str(real_path),
                chunk_content,
                start_line,
                end_line
            )

        # Paso 4: Aplicar lógica de tamaño
        show_complete = False
        total_lines = 0
        coverage = 0.0

        if validated:
            # Archivo real disponible y validado
            try:
                with open(real_path, 'r', encoding='utf-8', errors='replace') as f:
                    total_lines = len(f.readlines())

                coverage = calculate_coverage(data['ranges'], total_lines)
                show_complete = should_show_complete_file(total_lines, coverage)

                # Paso 5a: Usar merge_chunks con archivo real
                if show_complete:
                    # Mostrar archivo completo
                    with open(real_path, 'r', encoding='utf-8', errors='replace') as f:
                        lines = f.readlines()
                    content = '\n'.join(
                        f"{i+1:4d}  {line.rstrip()}"
                        for i, line in enumerate(lines)
                    )
                else:
                    # Fusionar rangos overlapping antes de mostrar
                    merged_ranges = merge_overlapping_ranges(data['ranges'])
                    # Mostrar solo fragmentos
                    content = merge_chunks(str(real_path), merged_ranges)

            except Exception as e:
                # Fallback a simulación si hay error
                content = simulate_chunk_numbering(data['chunks'])
                validated = False
        else:
            # Paso 5b: Simular numeración con chunks
            content = simulate_chunk_numbering(data['chunks'])

        # Agregar resultado
        results[file_path] = {
            'content': content,
            'total_lines': total_lines,
            'coverage': coverage,
            'show_complete': show_complete,
            'chunks_count': len(data['chunks']),
            'validated': validated,
            'distance': data['min_distance']
        }

        processed_count += 1

    return results


def merge_chunks_from_search_results(
    workspace_path: str,
    results: List[Tuple[str, int, int]]
) -> dict:
    """
    Agrupa y fusiona chunks de búsqueda por archivo (versión simple).

    Esta función toma resultados de búsqueda semántica y agrupa los chunks
    que pertenecen al mismo archivo, luego los fusiona en una sola vista.

    Args:
        workspace_path: Ruta base del workspace
        results: Lista de tuplas (file_path, start_line, end_line)

    Returns:
        Dict con file_path como key y el contenido fusionado como value

    Example:
        >>> results = [
        ...     ("app.py", 1, 5),
        ...     ("app.py", 10, 15),
        ...     ("utils.py", 20, 30)
        ... ]
        >>> merged = merge_chunks_from_search_results("/workspace", results)
        >>> print(merged["app.py"])
    """
    workspace = Path(workspace_path)

    # Agrupar por archivo
    file_ranges = defaultdict(list)
    for file_path, start_line, end_line in results:
        file_ranges[file_path].append((start_line, end_line))

    # Fusionar chunks de cada archivo
    merged = {}
    for file_path, ranges in file_ranges.items():
        full_path = workspace / file_path
        try:
            merged[file_path] = merge_chunks(str(full_path), ranges)
        except Exception as e:
            merged[file_path] = f"Error al procesar {file_path}: {str(e)}"

    return merged


if __name__ == "__main__":
    # Ejemplo de uso
    import sys
    
    if len(sys.argv) < 2:
        print("Uso: python chunk_merger.py <archivo> <start1> <end1> [<start2> <end2> ...]")
        print("\nEjemplo:")
        print("  python chunk_merger.py app.py 1 5 10 15 20 25")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    # Parsear rangos de los argumentos
    ranges = []
    args = sys.argv[2:]
    
    if len(args) % 2 != 0:
        print("Error: Debes proporcionar pares de (start, end)")
        sys.exit(1)
    
    for i in range(0, len(args), 2):
        start = int(args[i])
        end = int(args[i + 1])
        ranges.append((start, end))
    
    try:
        result = merge_chunks(file_path, ranges)
        print(result)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

