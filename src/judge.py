"""Judge (LLM) client for reranking and summarizing search results."""
import httpx
import json
import re
from typing import Optional
from dataclasses import dataclass


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


class Judge:
    """LLM client for reranking and summarizing search results."""
    
    def __init__(
        self,
        provider: str,
        api_key: str,
        model_id: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        base_url: Optional[str] = None,
        system_prompt: Optional[str] = None
    ):
        """Initialize judge client.
        
        Args:
            provider: Provider type ("openai" or "openai-compatible")
            api_key: API key for authentication
            model_id: Model identifier
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            base_url: Optional base URL for openai-compatible providers
            system_prompt: Optional custom system prompt
        """
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
        
        self.system_prompt = system_prompt or self._get_default_system_prompt()
    
    def _get_default_system_prompt(self) -> str:
        """Get default system prompt for reranking."""
        return """Eres un modelo experto en análisis de código.
Dada una consulta de usuario y varios fragmentos de código, evalúa qué fragmentos
realmente responden a la pregunta y ordénalos de mayor a menor relevancia, siempre los archivos más relevantes tienen que ser archivos de código fuente.

Debes devolver un JSON con la siguiente estructura exacta:
{
     "reranked": [
       {
         "filePath": "ruta/al/archivo.js",
         "relevancia": 0.95,
         "razon": "Este archivo contiene la implementación principal de la funcionalidad buscada"
       }
     ]
}"""
    
    def _get_dynamic_system_prompt(self, include_summary: bool) -> str:
        """Get dynamic system prompt based on whether summary is needed."""
        base_prompt = """Eres un modelo experto en análisis de código.
Dada una consulta de usuario y varios fragmentos de código, evalúa qué fragmentos
realmente responden a la pregunta y ordénalos de mayor a menor relevancia.

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
        
        summary_instruction = ""
        if include_summary:
            summary_instruction = """

CRÍTICO: Debes incluir un campo "summary" A NIVEL RAÍZ del JSON (no dentro de cada elemento del array).
El "summary" debe ser un resumen consolidado global que explique cómo los fragmentos más relevantes
responden a la consulta del usuario. NO incluyas "summary" dentro de cada objeto del array "reranked"."""
        
        json_structure = ""
        if include_summary:
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
        else:
            json_structure = """{
    "reranked": [
      {
        "filePath": "ruta/al/archivo.js",
        "relevancia": 0.95,
        "razon": "Este archivo contiene la implementación principal de la funcionalidad buscada"
      }
    ]
}"""
        
        return f"""{base_prompt}{summary_instruction}

Debes devolver un JSON válido con la siguiente estructura exacta:
{json_structure}

RESPONDE ÚNICAMENTE CON EL JSON, sin texto adicional."""
    
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
    
    async def _call_llm(self, user_prompt: str, include_summary: bool) -> str:
        """Call LLM API and return response text."""
        system_prompt = self._get_dynamic_system_prompt(include_summary)
        
        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.endpoint,
                headers=self.headers,
                json=payload,
                timeout=60.0
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
                raise Exception(f"Judge error: {error_msg}")
            
            data = response.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            if not content:
                raise Exception("LLM returned empty response")
            
            return content.strip()
    
    def _process_response(self, response: str, original_results: list[SearchResult]) -> list[RerankResult]:
        """Process LLM response and create reranked results."""
        # Try to extract JSON from response
        json_match = re.search(r'```(?:json)?\s*([\s\S]+?)\s*```', response)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = response
        
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise Exception(f"Failed to parse JSON response: {e}")
        
        if "reranked" not in data or not isinstance(data["reranked"], list):
            raise Exception("Response does not contain 'reranked' array")
        
        # Create lookup for original results
        results_map = {r.file_path: r for r in original_results}
        
        # Build reranked results
        reranked = []
        for item in data["reranked"]:
            file_path = item.get("filePath")
            relevancia = item.get("relevancia", 0.0)
            razon = item.get("razon")
            
            if not file_path:
                continue
            
            original = results_map.get(file_path)
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
    
    async def rerank(self, query: str, results: list[SearchResult]) -> list[RerankResult]:
        """Rerank search results using LLM.
        
        Args:
            query: User query
            results: Original search results
            
        Returns:
            Reranked results with relevance scores
        """
        if not results:
            return []
        
        user_prompt = self._create_user_prompt(query, results, include_summary=False)
        response = await self._call_llm(user_prompt, include_summary=False)
        return self._process_response(response, results)
    
    async def summarize(self, query: str, results: list[SearchResult]) -> str:
        """Generate summary of search results.
        
        Args:
            query: User query
            results: Search results to summarize
            
        Returns:
            Summary text
        """
        if not results:
            return "No hay resultados para resumir."
        
        # Use top 5 results for summary
        top_results = results[:5]
        
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

        response = await self._call_llm(summary_prompt, include_summary=False)

        # Clean up response
        summary = response.strip()

        # Try to extract from JSON if present (in case LLM ignores instructions)
        try:
            json_match = re.search(r'```(?:json)?\s*([\s\S]+?)\s*```', summary)
            if json_match:
                parsed = json.loads(json_match.group(1))
                # Try to extract summary from JSON
                if "summary" in parsed:
                    summary = parsed["summary"]
                elif "reranked" in parsed and isinstance(parsed["reranked"], list):
                    # Generate summary from reranked data
                    top_items = parsed["reranked"][:3]
                    summary_parts = [f"Basado en la consulta \"{query}\", los fragmentos más relevantes son:\n"]
                    for i, item in enumerate(top_items, 1):
                        file_path = item.get("filePath", "")
                        razon = item.get("razon", "")
                        summary_parts.append(f"{i}. **{file_path}**: {razon}")
                    summary = "\n\n".join(summary_parts)
                else:
                    summary = parsed.get("text") or parsed.get("content") or summary
            elif summary.startswith('{') and summary.endswith('}'):
                # Try to parse as JSON
                parsed = json.loads(summary)
                if "summary" in parsed:
                    summary = parsed["summary"]
                elif "reranked" in parsed and isinstance(parsed["reranked"], list):
                    # Generate summary from reranked data
                    top_items = parsed["reranked"][:3]
                    summary_parts = [f"Basado en la consulta \"{query}\", los fragmentos más relevantes son:\n"]
                    for i, item in enumerate(top_items, 1):
                        file_path = item.get("filePath", "")
                        razon = item.get("razon", "")
                        summary_parts.append(f"{i}. **{file_path}**: {razon}")
                    summary = "\n\n".join(summary_parts)
                else:
                    summary = parsed.get("text") or parsed.get("content") or summary
        except:
            pass

        # Remove "summary:" prefix if present
        summary = re.sub(r'^(summary|resumen):\s*', '', summary, flags=re.IGNORECASE).strip()

        # Remove JSON markers if still present
        summary = re.sub(r'^```(?:json)?\s*', '', summary)
        summary = re.sub(r'\s*```$', '', summary)

        return summary

