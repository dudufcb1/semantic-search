#!/usr/bin/env python3
"""Test Structured Outputs with JSON Schema for code search results."""

import asyncio
import httpx
from typing import List
from pydantic import BaseModel


# Define Pydantic models for structured output
class TopFile(BaseModel):
    """Top ranked file with AI-assigned relevance."""
    file_path: str
    relevance: float  # AI-assigned relevance (0.0 to 1.0)
    reason: str  # Why this file is relevant
    relevant_lines: str  # e.g., "1-160, 322-459" (ranges of relevant lines)


class CodeFragment(BaseModel):
    """A code fragment with complete context."""
    file_path: str
    start_line: int
    end_line: int
    code_snippet: str  # COMPLETE code, NO truncation
    explanation: str  # What this code does and why it's relevant


class FunctionUsage(BaseModel):
    """Where a specific function/class is called/used."""
    function_name: str  # e.g., "formatDate()"
    file_path: str  # Where it's called
    line_number: int  # Specific line where it's called
    context: str  # Brief context of the call


class CodeSearchSummary(BaseModel):
    """Complete summary of code search results."""
    query: str
    total_files: int
    total_fragments: int
    top_files: List[TopFile]  # Top 10 files ranked by AI
    code_fragments: List[CodeFragment]  # Up to 10 code fragments with COMPLETE code
    usages: List[FunctionUsage]  # Where specific functions/classes are called
    inference: str  # Summary of findings and how it works


async def test_structured_output():
    """Test structured output with a simple example."""
    
    # Simulated search results (in real code, these come from Qdrant)
    mock_results = [
        {"file_path": "src/ragService.ts", "start_line": 1, "end_line": 160, "score": 0.95, "content": "class RAGService..."},
        {"file_path": "src/ragService.ts", "start_line": 161, "end_line": 321, "score": 0.92, "content": "selectBestSources()..."},
        {"file_path": "src/conversationService.ts", "start_line": 1, "end_line": 191, "score": 0.88, "content": "class ConversationService..."},
        {"file_path": "docs/timeline.md", "start_line": 95, "end_line": 253, "score": 0.75, "content": "# Timeline..."},
    ]
    
    query = "How does the RAG service retrieve context for chat messages"
    
    # Group by file
    from collections import defaultdict
    grouped = defaultdict(list)
    for r in mock_results:
        grouped[r["file_path"]].append(r)
    
    # Build prompt
    user_prompt = f"""Consulta del usuario: "{query}"

Total de archivos únicos: {len(grouped)}
Total de fragmentos: {len(mock_results)}

Archivos y fragmentos:
"""
    
    for file_path, file_results in grouped.items():
        is_code = not file_path.endswith(('.md', '.txt'))
        user_prompt += f"\nArchivo: {file_path} ({'CÓDIGO' if is_code else 'DOCUMENTACIÓN'})\n"
        for r in file_results:
            user_prompt += f"  - Líneas {r['start_line']}-{r['end_line']}, score: {r['score']:.2f}\n"
            user_prompt += f"    Contenido: {r['content'][:100]}...\n"
    
    user_prompt += """
INSTRUCCIONES:
1. Evalúa CADA FRAGMENTO INDIVIDUALMENTE según su relevancia para la consulta
2. Identifica cada fragmento por su archivo Y líneas exactas (startLine-endLine)
3. Dos fragmentos del mismo archivo pueden tener relevancia MUY diferente
4. Ejemplo: Si la consulta busca "formatDate", el fragmento con formatDate() es más relevante que el fragmento con formatTime()
5. Reordena los fragmentos de mayor a menor relevancia
6. Incluye SIEMPRE startLine, endLine y code_snippet en tu respuesta JSON
7. El code_snippet debe contener el código COMPLETO del fragmento (copia exacta de lo que te proporcioné)
8. NO trunces el código, incluye todo el fragmento
9. Analiza TODOS los fragmentos proporcionados

Genera el resumen estructurado completo ahora.
"""
    
    # Call LLM with structured output
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            "http://localhost:4141/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json={
                "model": "gpt-4.1",
                "messages": [
                    {
                        "role": "system",
                        "content": """Eres un asistente experto en análisis de código. Analiza los fragmentos de código y genera un resumen estructurado completo.

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

ESTRUCTURA DE RESPUESTA:

1. TOP_FILES (Top 10 archivos):
   - Lista los 10 archivos MÁS RELEVANTES según TU calificación (no el score original)
   - Para cada archivo: file_path, relevance (0.0-1.0), reason (por qué es relevante), relevant_lines (rangos de líneas relevantes, ej: "1-160, 322-459")
   - Ordena de mayor a menor relevancia

2. CODE_FRAGMENTS (Hasta 10 fragmentos de código):
   - Lista hasta 10 fragmentos de código COMPLETOS que explican cómo funciona el sistema
   - Para cada fragmento: file_path, start_line, end_line, code_snippet (código COMPLETO, SIN truncar), explanation (qué hace este código y por qué es relevante)
   - IMPORTANTE: code_snippet debe contener el código COMPLETO del fragmento, sin cortar
   - Si hay una función importante, AGREGA +5 líneas antes y +5 líneas después si están disponibles
   - Busca contexto, NO cortes el código

3. USAGES (Invocaciones):
   - Identifica dónde se LLAMA/INVOCA una función o clase específica
   - Ejemplo: Si encuentras formatDate() definida en utils.js, busca dónde se LLAMA formatDate() en otros archivos
   - Incluye: function_name, file_path (donde se llama), line_number (línea específica), context (contexto de la llamada)
   - Solo incluye invocaciones REALES que encuentres en el código, no inventes

4. INFERENCE (Resumen de hallazgos):
   - Resumen de cómo funciona el sistema basado en los fragmentos analizados
   - Explica cómo se infiere el funcionamiento
   - NO resumas, explica en detalle

IMPORTANTE:
- Nunca un archivo de texto o documentación debe ir al inicio de top_files
- Identifica SIEMPRE cada fragmento por su ubicación exacta (startLine-endLine)
- Dos fragmentos del mismo archivo pueden tener relevancia MUY diferente
- INCLUYE SIEMPRE el código real COMPLETO en code_snippet (copia el código exacto que te proporcioné)
- NO trunces el código, incluye todo el fragmento completo
- El code_snippet debe ser lo más completo posible para entender el contexto
- En usages, solo incluye invocaciones que REALMENTE encuentres en el código proporcionado"""
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ],
                "max_tokens": 32000,
                "temperature": 0.0,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "code_search_summary",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string"},
                                "total_files": {"type": "integer"},
                                "total_fragments": {"type": "integer"},
                                "top_files": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "file_path": {"type": "string"},
                                            "relevance": {"type": "number"},
                                            "reason": {"type": "string"},
                                            "relevant_lines": {"type": "string"}
                                        },
                                        "required": ["file_path", "relevance", "reason", "relevant_lines"],
                                        "additionalProperties": False
                                    }
                                },
                                "code_fragments": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "file_path": {"type": "string"},
                                            "start_line": {"type": "integer"},
                                            "end_line": {"type": "integer"},
                                            "code_snippet": {"type": "string"},
                                            "explanation": {"type": "string"}
                                        },
                                        "required": ["file_path", "start_line", "end_line", "code_snippet", "explanation"],
                                        "additionalProperties": False
                                    }
                                },
                                "usages": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "function_name": {"type": "string"},
                                            "file_path": {"type": "string"},
                                            "line_number": {"type": "integer"},
                                            "context": {"type": "string"}
                                        },
                                        "required": ["function_name", "file_path", "line_number", "context"],
                                        "additionalProperties": False
                                    }
                                },
                                "inference": {"type": "string"}
                            },
                            "required": ["query", "total_files", "total_fragments", "top_files", "code_fragments", "usages", "inference"],
                            "additionalProperties": False
                        }
                    }
                }
            }
        )
        
        if response.status_code != 200:
            print(f"ERROR {response.status_code}:")
            print(response.text)
            return

        data = response.json()
        
        # Extract structured content
        content = data["choices"][0]["message"]["content"]
        
        print("=" * 70)
        print("RESPUESTA ESTRUCTURADA (JSON):")
        print("=" * 70)
        print(content)
        print()
        
        # Parse as Pydantic model
        summary = CodeSearchSummary.model_validate_json(content)

        print("=" * 70)
        print("RESUMEN PARSEADO:")
        print("=" * 70)
        print(f"Query: {summary.query}")
        print(f"Total archivos: {summary.total_files}")
        print(f"Total fragmentos: {summary.total_fragments}")
        print()

        print("TOP 10 ARCHIVOS (calificación IA):")
        for i, file in enumerate(summary.top_files, 1):
            print(f"\n{i}. {file.file_path}")
            print(f"   Relevancia: {file.relevance}")
            print(f"   Razón: {file.reason}")
            print(f"   Líneas relevantes: {file.relevant_lines}")

        print(f"\n\nFRAGMENTOS DE CÓDIGO (hasta 10):")
        for i, frag in enumerate(summary.code_fragments, 1):
            print(f"\n{i}. {frag.file_path} (líneas {frag.start_line}-{frag.end_line})")
            print(f"   Explicación: {frag.explanation}")
            print(f"\n   CÓDIGO COMPLETO ({len(frag.code_snippet)} caracteres):")
            print("   " + "=" * 67)
            for line in frag.code_snippet.split('\n'):
                print(f"   {line}")
            print("   " + "=" * 67)

        if summary.usages:
            print(f"\n\nUSAGES (INVOCACIONES):")
            for usage in summary.usages:
                print(f"  - {usage.function_name} llamada en {usage.file_path}:{usage.line_number}")
                print(f"    Contexto: {usage.context}")

        print(f"\n\nINFERENCIA (Resumen de hallazgos):")
        print(f"{summary.inference}")


if __name__ == "__main__":
    asyncio.run(test_structured_output())

