"""Text Direct Judge (LLM) client - JSON Schema structured outputs."""
import httpx
import re
from typing import Optional, List
from dataclasses import dataclass
from pydantic import BaseModel


@dataclass
class SearchResult:
    """Search result from vector store."""
    file_path: str
    code_chunk: str
    start_line: int
    end_line: int
    score: float


@dataclass
class RerankResult:
    """Reranked search result with relevance score."""
    file_path: str
    code_chunk: str
    start_line: int
    end_line: int
    score: float
    relevancia: float
    razon: Optional[str] = None


# Pydantic models for JSON Schema structured outputs
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


class TextDirectJudge:
    """LLM client that generates markdown summaries directly without parsing."""

    def __init__(
        self,
        provider: str,
        api_key: str,
        model_id: str,
        max_tokens: int = 15000,  # Increased for comprehensive markdown summaries
        temperature: float = 0.0,
        base_url: Optional[str] = None,
        system_prompt: Optional[str] = None
    ):
        """Initialize text-direct judge client."""
        self.provider = provider
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature

        # Determine endpoint
        if provider == "openai":
            self.endpoint = "https://api.openai.com/v1/chat/completions"
        else:
            if not base_url:
                raise ValueError("base_url is required for openai-compatible provider")
            self.endpoint = f"{base_url.rstrip('/')}/chat/completions"

        # Setup headers
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        self.system_prompt = system_prompt or self._get_structured_system_prompt()

    def _get_structured_system_prompt(self) -> str:
        """Get system prompt for JSON Schema structured outputs."""
        return """Eres un experto en análisis de código. Tu tarea es analizar fragmentos de código y proporcionar un resumen estructurado.

ESTRUCTURA DE RESPUESTA:

1. TOP_FILES (Top 10 archivos):
   - Lista los 10 archivos MÁS RELEVANTES según TU calificación (no el score original)
   - Para cada archivo: file_path, relevance (0.0-1.0), reason (por qué es relevante), relevant_lines (rangos de líneas relevantes, ej: "1-160, 322-459")
   - Ordena de mayor a menor relevancia

2. CODE_FRAGMENTS (Hasta 10 fragmentos de código):
   - SOLO incluye fragmentos de archivos de CÓDIGO (NO .md, .txt, .rst, .adoc, etc.)
   - Lista hasta 10 fragmentos de código COMPLETOS que explican cómo funciona el sistema
   - Para cada fragmento: file_path, start_line, end_line, code_snippet (código COMPLETO, SIN truncar), explanation (qué hace este código y por qué es relevante)
   - IMPORTANTE: code_snippet debe contener el código COMPLETO del fragmento, sin cortar
   - Si hay una función importante, AGREGA +5 líneas antes y +5 líneas después si están disponibles
   - Busca contexto, NO cortes el código
   - NUNCA incluyas archivos de documentación (.md, .txt, etc.) en CODE_FRAGMENTS

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
- NO truncues el código, incluye todo el fragmento completo
- El code_snippet debe ser lo más completo posible para entender el contexto
- En usages, solo incluye invocaciones que REALMENTE encuentres en el código proporcionado
- ARCHIVOS DE DOCUMENTACIÓN (.md, .txt, .rst, .adoc): NUNCA incluyas su contenido en code_snippet, solo menciónalos en top_files si son relevantes"""

    async def rerank_structured(self, query: str, results: list[SearchResult]) -> CodeSearchSummary:
        """Rerank search results using JSON Schema structured outputs.

        Args:
            query: The search query
            results: List of search results from vector store

        Returns:
            CodeSearchSummary with top files, code fragments, usages, and inference
        """
        if not results:
            return CodeSearchSummary(
                query=query,
                total_files=0,
                total_fragments=0,
                top_files=[],
                code_fragments=[],
                usages=[],
                inference="No se encontraron resultados para la consulta."
            )

        # Group results by file path
        from collections import defaultdict
        grouped_results = defaultdict(list)
        for result in results:
            grouped_results[result.file_path].append(result)

        # Build user prompt with ALL results
        user_prompt = f"""Consulta del usuario: "{query}"

Total de archivos únicos: {len(grouped_results)}
Total de fragmentos: {len(results)}

Archivos y fragmentos:
"""

        for file_path, file_results in grouped_results.items():
            is_code = not file_path.endswith(('.md', '.txt', '.rst', '.adoc'))
            user_prompt += f"\nArchivo: {file_path} ({'CÓDIGO' if is_code else 'DOCUMENTACIÓN'})\n"
            for r in file_results:
                user_prompt += f"  - Líneas {r.start_line}-{r.end_line}, score: {r.score:.2f}\n"
                if is_code:
                    user_prompt += f"    Código:\n{r.code_chunk}\n\n"
                else:
                    # Para documentación, solo mostrar primeras 3 líneas como preview
                    preview_lines = r.code_chunk.split('\n')[:3]
                    preview = '\n'.join(preview_lines)
                    user_prompt += f"    Preview (primeras 3 líneas):\n{preview}\n    [...resto del documento omitido...]\n\n"

        user_prompt += """
INSTRUCCIONES CRÍTICAS:
1. Evalúa CADA FRAGMENTO INDIVIDUALMENTE según su relevancia para la consulta
2. Identifica cada fragmento por su archivo Y líneas exactas (startLine-endLine)
3. Dos fragmentos del mismo archivo pueden tener relevancia MUY diferente
4. ARCHIVOS DE CÓDIGO: INCLUYE SIEMPRE el código real COMPLETO en code_snippet (copia el código exacto que te proporcioné)
5. ARCHIVOS DE DOCUMENTACIÓN (.md, .txt, .rst, .adoc): NUNCA los incluyas en CODE_FRAGMENTS, solo menciónalos en TOP_FILES si son relevantes
6. NO truncues el código de archivos de código, incluye todo el fragmento completo
7. En usages, solo incluye invocaciones que REALMENTE encuentres en el código proporcionado
"""

        # JSON Schema definition
        json_schema = {
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

        # Call LLM with JSON Schema
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                self.endpoint,
                headers=self.headers,
                json={
                    "model": self.model_id,
                    "messages": [
                        {"role": "system", "content": self._get_structured_system_prompt()},
                        {"role": "user", "content": user_prompt}
                    ],
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "code_search_summary",
                            "strict": True,
                            "schema": json_schema
                        }
                    }
                }
            )

            if not response.is_success:
                error_data = response.json() if response.content else {}
                error_msg = (
                    error_data.get("error", {}).get("message") or
                    error_data.get("error") or
                    error_data.get("message") or
                    response.text or
                    "Failed to call LLM"
                )
                raise Exception(f"TextDirectJudge error: {error_msg}")

            data = response.json()
            content = data["choices"][0]["message"]["content"]

            if not content:
                raise Exception("LLM returned empty response")

            # Parse as Pydantic model
            summary = CodeSearchSummary.model_validate_json(content)
            return summary