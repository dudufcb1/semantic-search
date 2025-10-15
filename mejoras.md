# Mejoras para Judge - Identificación de Chunks Específicos

## Problema Principal

El Judge actual solo identifica archivos (`filePath`) pero no chunks específicos dentro del archivo. Esto causa que cuando hay múltiples chunks del mismo archivo (ej: `formatDate()` y `formatTime()`), el sistema tome el primero que encuentra en lugar del más relevante.

---

## Cambio 1: Agregar startLine/endLine al JSON Schema

### Ubicación: Líneas 127-146

### Código actual:
```python
json_structure = """{
    "reranked": [
      {
        "filePath": "ruta/al/archivo.js",
        "relevancia": 0.95,
        "razon": "Este archivo contiene la implementación principal de la funcionalidad buscada"
      }
    ],
    "summary": "Resumen global consolidado de cómo los fragmentos más relevantes responden a la consulta del usuario"
}"""
```

### Código mejorado:
```python
json_structure = """{
    "reranked": [
      {
        "filePath": "ruta/al/archivo.js",
        "startLine": 846,
        "endLine": 880,
        "relevancia": 0.95,
        "razon": "Este fragmento específico (líneas 846-880) contiene la función formatDate() que maneja Unix timestamps e ISO strings"
      }
    ],
    "summary": "Resumen global consolidado de cómo los fragmentos más relevantes responden a la consulta del usuario"
}"""
```

**Aplicar también en líneas 138-146 (versión sin summary)**

---

## Cambio 2: Actualizar Reglas de Relevancia

### Ubicación: Líneas 99-115

### Código actual:
```python
REGLAS OBLIGATORIAS DE RELEVANCIA (NO NEGOCIABLES):
1. CÓDIGO FUENTE (.ts, .js, .py, .java, .cpp, .go, .rs, .php, .rb, etc.):
      - Relevancia MÍNIMA: 0.81 si es relevante
      - SIEMPRE debe estar por encima de cualquier documentación

2. DOCUMENTACIÓN (.md, .txt, README):
      - Relevancia MÁXIMA: 0.8 (NUNCA más alta)
      - Solo incluir si coincide con el código y aporta valor
      - Si contradice el código → OMITIR completamente

3. ORDEN OBLIGATORIO:
      - Todos los archivos de código relevantes (0.81-1.0)
      - Luego archivos de configuración (0.5-0.8)
      - Luego documentación validada (máximo 0.8)
      - Finalmente otros archivos (0.1-0.5)

IMPORTANTE: Nunca un archivo de texto o documentación debe ir al inicio, asegurate de respetar el orden de calificaciones."""
```

### Código mejorado:
```python
REGLAS OBLIGATORIAS DE RELEVANCIA (NO NEGOCIABLES):
1. CÓDIGO FUENTE (.ts, .js, .py, .java, .cpp, .go, .rs, .php, .rb, etc.):
      - Relevancia MÍNIMA: 0.81 si es relevante
      - SIEMPRE debe estar por encima de cualquier documentación

2. DOCUMENTACIÓN (.md, .txt, README):
      - Relevancia MÁXIMA: 0.8 (NUNCA más alta)
      - Solo incluir si coincide con el código y aporta valor
      - Si contradice el código → OMITIR completamente

3. MÚLTIPLES FRAGMENTOS DEL MISMO ARCHIVO:
      - Evalúa CADA FRAGMENTO INDIVIDUALMENTE
      - Identifica cada fragmento por filePath + startLine + endLine
      - Un fragmento puede ser muy relevante (0.95) y otro del mismo archivo poco relevante (0.5)
      - Ejemplo: formatDate() puede ser 0.95, pero formatTime() solo 0.7 si la consulta es sobre formatDate

4. ORDEN OBLIGATORIO:
      - Todos los fragmentos de código relevantes (0.81-1.0)
      - Luego archivos de configuración (0.5-0.8)
      - Luego documentación validada (máximo 0.8)
      - Finalmente otros archivos (0.1-0.5)

IMPORTANTE: 
- Nunca un archivo de texto o documentación debe ir al inicio
- Identifica SIEMPRE cada fragmento por su ubicación exacta (startLine-endLine)
- Dos fragmentos del mismo archivo pueden tener relevancia MUY diferente"""
```

---

## Cambio 3: Actualizar User Prompt

### Ubicación: Líneas 155-171

### Código actual:
```python
def _create_user_prompt(self, query: str, results: list[SearchResult], include_summary: bool) -> str:
    """Create user prompt with query and results."""
    fragments = []
    for i, result in enumerate(results, 1):
        fragments.append(
            f"{i}. Archivo: {result.file_path} (líneas {result.start_line}-{result.end_line})\n"
            f"Score original: {result.score:.4f}\n"
            f"Código:\n{result.code_chunk.strip()}\n"
        )
    
    fragments_text = "\n".join(fragments)
    return f"""Consulta del usuario: "{query}"

Fragmentos encontrados:
{fragments_text}

Evalúa y reordena estos fragmentos según su relevancia para la consulta."""
```

### Código mejorado:
```python
def _create_user_prompt(self, query: str, results: list[SearchResult], include_summary: bool) -> str:
    """Create user prompt with query and results."""
    fragments = []
    for i, result in enumerate(results, 1):
        fragments.append(
            f"{i}. Archivo: {result.file_path} (líneas {result.start_line}-{result.end_line})\n"
            f"Score original: {result.score:.4f}\n"
            f"Código:\n{result.code_chunk.strip()}\n"
        )
    
    fragments_text = "\n".join(fragments)
    return f"""Consulta del usuario: "{query}"

Fragmentos encontrados:
{fragments_text}

INSTRUCCIONES:
1. Evalúa CADA FRAGMENTO INDIVIDUALMENTE según su relevancia para la consulta
2. Identifica cada fragmento por su archivo Y líneas exactas (startLine-endLine)
3. Dos fragmentos del mismo archivo pueden tener relevancia MUY diferente
4. Ejemplo: Si la consulta busca "formatDate", el fragmento con formatDate() es más relevante que el fragmento con formatTime()
5. Reordena los fragmentos de mayor a menor relevancia
6. Incluye SIEMPRE startLine y endLine en tu respuesta JSON"""
```

---

## Cambio 4: Actualizar _process_response para usar startLine/endLine

### Ubicación: Líneas 214-277

### Código actual:
```python
def _process_response(self, response: str, original_results: list[SearchResult]) -> list[RerankResult]:
    """Process LLM response and create reranked results."""
    # ... código de parsing JSON ...
    
    # Create lookup for original results
    results_map = {r.file_path: r for r in original_results}  # ← PROBLEMA

    # Build reranked results
    reranked = []
    for item in data["reranked"]:
        file_path = item.get("filePath")
        relevancia = item.get("relevancia", 0.0)
        razon = item.get("razon")

        if not file_path:
            continue

        original = results_map.get(file_path)  # ← PROBLEMA
        if not original:
            continue

        reranked.append(RerankResult(
            file_path=original.file_path,
            code_chunk=original.code_chunk,
            start_line=original.start_line,
            end_line=original.end_line,
            score=original.score,
            relevancia=relevancia,
            razon=razon
        ))

    return reranked
```

### Código mejorado:
```python
def _process_response(self, response: str, original_results: list[SearchResult]) -> list[RerankResult]:
    """Process LLM response and create reranked results."""
    # ... código de parsing JSON (sin cambios) ...
    
    # Create lookup for original results using filePath + lines as key
    results_map = {}
    for r in original_results:
        key = f"{r.file_path}:{r.start_line}-{r.end_line}"
        results_map[key] = r
    
    # Also create fallback map by filePath only (for backward compatibility)
    fallback_map = {}
    for r in original_results:
        if r.file_path not in fallback_map:
            fallback_map[r.file_path] = r

    # Build reranked results
    reranked = []
    for item in data["reranked"]:
        file_path = item.get("filePath")
        start_line = item.get("startLine")
        end_line = item.get("endLine")
        relevancia = item.get("relevancia", 0.0)
        razon = item.get("razon")

        if not file_path:
            continue

        # Try to find by exact match (filePath + lines)
        original = None
        if start_line is not None and end_line is not None:
            key = f"{file_path}:{start_line}-{end_line}"
            original = results_map.get(key)
        
        # Fallback to filePath only if exact match not found
        if not original:
            original = fallback_map.get(file_path)
            if original:
                print(f"[Judge] Warning: Using fallback match for {file_path} (no line numbers provided)", flush=True)
        
        if not original:
            print(f"[Judge] Warning: Could not find original result for {file_path}:{start_line}-{end_line}", flush=True)
            continue

        reranked.append(RerankResult(
            file_path=original.file_path,
            code_chunk=original.code_chunk,
            start_line=original.start_line,
            end_line=original.end_line,
            score=original.score,
            relevancia=relevancia,
            razon=razon
        ))

    return reranked
```

---

## Cambio 5: Mejorar Summary para identificar chunks específicos

### Ubicación: Líneas 324-347

### Código actual:
```python
summary_prompt = f"""Analiza los siguientes fragmentos de código y genera un resumen conciso explicando cómo responden a la consulta: "{query}"

VALIDACIÓN INTELIGENTE: Estos fragmentos ya han sido validados y filtrados. Solo incluye información que sea relevante y precisa.

Fragmentos encontrados:
"""

for i, r in enumerate(top_results, 1):
    is_markdown = r.file_path.endswith('.md') or r.file_path.endswith('.txt')
    prefix = '📄 [DOCUMENTACIÓN VALIDADA]' if is_markdown else '💻 [CÓDIGO]'
    summary_prompt += f"""
{i}. {prefix} {r.file_path} (líneas {r.start_line}-{r.end_line})
{r.code_chunk.strip()}

"""

summary_prompt += """
Genera un resumen de 2-3 párrafos explicando cómo estos fragmentos responden a la consulta del usuario:
- ENFÓCATE en las implementaciones reales encontradas en el código
- Si hay documentación incluida, es porque ya fue validada contra el código y es relevante
- Sé específico sobre qué funcionalidades y patrones encontraste

IMPORTANTE: Responde SOLO con el texto del resumen, SIN formato JSON, SIN código, SIN marcadores.
Solo texto natural en párrafos."""
```

### Código mejorado:
```python
summary_prompt = f"""Analiza los siguientes fragmentos de código y genera un resumen conciso explicando cómo responden a la consulta: "{query}"

VALIDACIÓN INTELIGENTE: Estos fragmentos ya han sido validados y filtrados. Solo incluye información que sea relevante y precisa.

Fragmentos encontrados:
"""

for i, r in enumerate(top_results, 1):
    is_markdown = r.file_path.endswith('.md') or r.file_path.endswith('.txt')
    prefix = '📄 [DOCUMENTACIÓN VALIDADA]' if is_markdown else '💻 [CÓDIGO]'
    summary_prompt += f"""
{i}. {prefix} {r.file_path} (líneas {r.start_line}-{r.end_line})
{r.code_chunk.strip()}

"""

summary_prompt += """
Genera un resumen de 2-3 párrafos explicando cómo estos fragmentos responden a la consulta del usuario:
- ENFÓCATE en las implementaciones reales encontradas en el código
- Si hay documentación incluida, es porque ya fue validada contra el código y es relevante
- Sé específico sobre qué funcionalidades y patrones encontraste
- IMPORTANTE: Si hay múltiples fragmentos del mismo archivo, especifica CUÁL es el más relevante y por qué
- Menciona las líneas específicas cuando sea relevante (ej: "El fragmento en líneas 846-880 contiene formatDate()")

FORMATO DE RESPUESTA:
1. Primer párrafo: Identifica los fragmentos más relevantes con sus ubicaciones exactas
2. Segundo párrafo: Explica cómo responden a la consulta
3. Tercer párrafo (opcional): Menciona fragmentos relacionados o contexto adicional

IMPORTANTE: Responde SOLO con el texto del resumen, SIN formato JSON, SIN código, SIN marcadores.
Solo texto natural en párrafos."""
```

---

## Resumen de Cambios

### Archivos a modificar:
- `src/judge.py` (o donde esté el código del Judge)

### Cambios por prioridad:

**Alta prioridad (implementar primero):**
1. ✅ Cambio 1: Agregar startLine/endLine al JSON schema
2. ✅ Cambio 4: Actualizar _process_response para usar líneas
3. ✅ Cambio 3: Actualizar user prompt

**Media prioridad:**
4. ⚠️  Cambio 2: Actualizar reglas de relevancia
5. ⚠️  Cambio 5: Mejorar summary

### Testing:

Después de implementar, probar con:
```python
query = "Where is the function that formats dates in the frontend? I need to find the formatDate function, how it works, what parameters it takes, and how it handles Unix timestamps versus ISO strings."
```

**Resultado esperado:**
- ✅ Debe mostrar `formatDate()` (líneas 846-880) con relevancia 0.95
- ✅ Debe mostrar `formatTime()` (líneas 882-901) con relevancia menor (0.7-0.8)
- ✅ El summary debe mencionar "líneas 846-880 contienen formatDate()"
- ✅ NO debe confundir formatDate con formatTime

---

## Validación

Después de implementar, verificar:

1. **JSON response del Judge incluye startLine/endLine:**
```json
{
  "reranked": [
    {
      "filePath": "frontend/app-chat.js",
      "startLine": 846,
      "endLine": 880,
      "relevancia": 0.95,
      "razon": "..."
    }
  ]
}
```

2. **results_map usa key compuesta:**
```python
key = "frontend/app-chat.js:846-880"
```

3. **Logs muestran warnings si falta info:**
```
[Judge] Warning: Using fallback match for frontend/app-chat.js (no line numbers provided)
```

4. **Summary menciona líneas específicas:**
```
El fragmento en líneas 846-880 de frontend/app-chat.js contiene la función formatDate()...
```

