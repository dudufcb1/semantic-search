#!/usr/bin/env python3
"""
Test JSON Schema structured outputs with REAL Qdrant data.
"""
import asyncio
import sys
from pathlib import Path
from typing import List
from pydantic import BaseModel

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import settings
from embedder import Embedder
from qdrant_store import QdrantStore
import httpx


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


async def main():
    """Test with real Qdrant data."""
    
    # Initialize services
    print("🔧 Inicializando servicios...")
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
    
    # Fetch real data
    query = "How does the RAG service retrieve context for chat messages"
    qdrant_collection = "codebase-e3047b8eb7d143b790"
    
    print(f"🔍 Buscando en Qdrant: '{query}'")
    vector = await embedder.create_embedding(query)
    search_results = await qdrant_store.search(
        vector=vector,
        workspace_path="",
        directory_prefix=None,
        min_score=settings.search_min_score,
        max_results=settings.search_max_results,
        collection_name=qdrant_collection
    )
    
    print(f"✅ Encontrados {len(search_results)} resultados")
    print(f"   Mejor score: {max(r.score for r in search_results):.6f}")
    print(f"   Peor score: {min(r.score for r in search_results):.6f}")
    
    # Count unique files
    from collections import defaultdict
    grouped = defaultdict(list)
    for r in search_results:
        grouped[r.file_path].append(r)
    print(f"   Archivos únicos: {len(grouped)}")
    
    # Build prompt with ALL results
    user_prompt = f"""Consulta del usuario: "{query}"

Total de archivos únicos: {len(grouped)}
Total de fragmentos: {len(search_results)}

Archivos y fragmentos:
"""
    
    for file_path, file_results in grouped.items():
        is_code = not file_path.endswith(('.md', '.txt'))
        user_prompt += f"\nArchivo: {file_path} ({'CÓDIGO' if is_code else 'DOCUMENTACIÓN'})\n"
        for r in file_results:
            user_prompt += f"  - Líneas {r.start_line}-{r.end_line}, score: {r.score:.2f}\n"
            user_prompt += f"    Código:\n{r.code_chunk}\n\n"
    
    user_prompt += """
INSTRUCCIONES:
1. Evalúa CADA FRAGMENTO INDIVIDUALMENTE según su relevancia para la consulta
2. Identifica cada fragmento por su archivo Y líneas exactas (startLine-endLine)
3. Dos fragmentos del mismo archivo pueden tener relevancia MUY diferente
4. INCLUYE SIEMPRE el código real COMPLETO en code_snippet (copia el código exacto que te proporcioné)
5. NO trunces el código, incluye todo el fragmento completo
6. En usages, solo incluye invocaciones que REALMENTE encuentres en el código proporcionado
"""
    
    print(f"\n📝 Prompt construido: {len(user_prompt)} caracteres")
    
    # Call LLM with JSON Schema
    print(f"\n🤖 Llamando a LLM con JSON Schema...")
    
    system_prompt = """Eres un experto en análisis de código. Tu tarea es analizar fragmentos de código y proporcionar un resumen estructurado.

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
    
    async with httpx.AsyncClient(timeout=300.0) as client:
        response = await client.post(
            "http://localhost:4141/v1/chat/completions",
            json={
                "model": "gpt-4.1",
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt
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
        
        data = response.json()
        
        # Extract structured content
        content = data["choices"][0]["message"]["content"]
        
        print("=" * 70)
        print("RESPUESTA ESTRUCTURADA (JSON):")
        print("=" * 70)
        print(content[:2000])
        print("...")
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
    asyncio.run(main())

