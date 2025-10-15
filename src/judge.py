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

3. MÚLTIPLES FRAGMENTOS DEL MISMO ARCHIVO:
      - Evalúa CADA FRAGMENTO INDIVIDUALMENTE
      - Identifica cada fragmento por filePath + startLine + endLine
      - Un fragmento puede ser muy relevante (0.95) y otro del mismo archivo poco relevante (0.5)
      - Ejemplo: Si buscan "autenticación de usuarios", el fragmento con login() puede ser 0.95, pero el fragmento con logout() solo 0.6
      - IMPORTANTE: En tu razón, sugiere al usuario ver líneas adicionales para contexto completo
        * Ejemplo: "Ver líneas 40-83 para contexto completo de la función"
        * Esto ayuda al usuario a entender mejor el código sin tener que expandir manualmente

4. USAGES (IMPORTANTE):
      - Si detectas fragmentos que son USAGES (usan la función/clase pero no la definen):
        * NO los incluyas en la lista principal de fragmentos reordenados
        * Agrégalos en una sección separada "USAGES:" al final
        * Formato: Lista breve de ubicaciones (filePath:lineNumber)
        * Ejemplo: "src/server.ts:539", "src/app.js:220"
      - Solo incluir si hay 3+ usages detectados
      - Máximo 10 usages en la lista
      - Los usages NO deben tener relevancia ni razón, solo ubicación

5. ORDEN OBLIGATORIO:
      - Todos los fragmentos de código relevantes (0.81-1.0)
      - Luego archivos de configuración (0.5-0.8)
      - Luego documentación validada (máximo 0.8)
      - Finalmente otros archivos (0.1-0.5)

IMPORTANTE:
- Nunca un archivo de texto o documentación debe ir al inicio
- Identifica SIEMPRE cada fragmento por su ubicación exacta (startLine-endLine)
- Dos fragmentos del mismo archivo pueden tener relevancia MUY diferente
- NO RECORTES EL CÓDIGO: Usa el fragmento completo tal como lo recibes, no lo resumas ni acortes
- El código relevante debe mostrarse COMPLETO, no solo 1-2 líneas"""

        summary_instruction = ""
        if include_summary:
            summary_instruction = """

CRÍTICO: Al final de tu respuesta, después de todos los fragmentos reordenados,
debes incluir las siguientes secciones opcionales (solo si aplican):

6. FLUJO END-TO-END (CUANDO APLIQUE):
   - Si detectas múltiples fragmentos que forman un flujo secuencial:
     * Agrega sección "FLUJO END-TO-END:" después de USAGES
     * Describe el flujo en 3-5 pasos numerados
     * Usa flechas → para mostrar secuencia
     * Ejemplo: "1. Usuario envía mensaje → 2. Se guarda sin embedding → 3. MessageIndexerService lo detecta → 4. Genera embedding en batch → 5. Almacena en SQLite"
   - Solo incluir si hay 3+ fragmentos relacionados secuencialmente

7. ARCHIVOS RELACIONADOS (CUANDO APLIQUE):
   - Si detectas fragmentos de múltiples archivos relacionados:
     * Agrega sección "ARCHIVOS RELACIONADOS:" después de FLUJO
     * Lista archivos con breve descripción de su rol
     * Formato: "• filePath - Descripción del rol"
     * Solo incluir archivos con relevancia > 0.7
     * Máximo 5 archivos

8. CONCEPTOS CLAVE (CUANDO APLIQUE):
   - Identifica conceptos técnicos importantes mencionados en los fragmentos
   * Agrega sección "CONCEPTOS CLAVE:" después de ARCHIVOS RELACIONADOS
   * Lista 3-5 conceptos con breve explicación
   * Formato: "• Concepto: Explicación breve"
   * Ejemplo: "• Batch Processing: Procesar múltiples items en una sola llamada API (20x más eficiente)"

9. MÉTRICAS DE COBERTURA (CUANDO APLIQUE):
   - Agrega sección "MÉTRICAS:" al final
   - Incluye:
     * Total de fragmentos evaluados
     * Fragmentos relevantes (relevancia > 0.8)
     * Número de archivos únicos
     * Si hay más resultados relevantes no mostrados
   - Formato: "📊 X de Y fragmentos relevantes (Z%)" en líneas separadas

RESUMEN GLOBAL:
[Resumen consolidado de 2-3 párrafos explicando cómo los fragmentos más relevantes responden a la consulta]"""

        text_structure = ""
        if include_summary:
            text_structure = """
FORMATO DE RESPUESTA (TEXTO PLANO):

1. filePath: src/auth/login.py
   startLine: 45
   endLine: 78
   relevancia: 0.95
   razon: Este fragmento (líneas 45-78) contiene la función principal de autenticación que valida credenciales y genera tokens. 💡 Ver líneas 40-83 para contexto completo de la función.

2. filePath: src/utils/validators.py
   startLine: 120
   endLine: 145
   relevancia: 0.88
   razon: Implementación auxiliar de validación de contraseñas utilizada por el sistema de autenticación. 💡 Ver líneas 115-150 para ver toda la clase de validación.

USAGES:
src/server.ts:539
src/app.js:220
src/routes/api.js:155

FLUJO END-TO-END:
1. Usuario envía credenciales → 2. login.py valida con validators.py → 3. Genera token JWT → 4. Almacena sesión en Redis → 5. Retorna token al cliente

ARCHIVOS RELACIONADOS:
• src/auth/login.py - Función principal de autenticación
• src/utils/validators.py - Validación de contraseñas y reglas
• src/middleware/auth.py - Middleware de verificación de tokens
• src/models/user.py - Modelo de usuario y permisos

CONCEPTOS CLAVE:
• JWT (JSON Web Token): Token firmado para autenticación stateless
• Password Hashing: Bcrypt con salt para almacenar contraseñas de forma segura
• Session Management: Redis para almacenar sesiones activas con TTL
• Role-Based Access Control: Permisos basados en roles de usuario

MÉTRICAS:
📊 8 de 15 fragmentos relevantes (53%)
📁 4 archivos únicos
⚠️  Hay más resultados relevantes no mostrados

RESUMEN GLOBAL:
[Aquí va el resumen consolidado de 2-3 párrafos explicando cómo los fragmentos más relevantes responden a la consulta]
"""
        else:
            text_structure = """
FORMATO DE RESPUESTA (TEXTO PLANO):

1. filePath: src/api/routes.py
   startLine: 180
   endLine: 215
   relevancia: 0.93
   razon: Este fragmento (líneas 180-215) contiene el endpoint principal que procesa las peticiones HTTP. 💡 Ver líneas 175-220 para contexto completo del router.

2. filePath: src/middleware/auth.py
   startLine: 55
   endLine: 88
   relevancia: 0.87
   razon: Middleware de autenticación que valida tokens JWT antes de procesar las peticiones. 💡 Ver líneas 50-95 para ver toda la clase de middleware.

USAGES:
src/server.ts:539
src/server.ts:540
src/app.js:220
"""

        return f"""{base_prompt}{summary_instruction}

Debes devolver TEXTO PLANO con la siguiente estructura exacta:
{text_structure}

IMPORTANTE:
- Usa EXACTAMENTE el formato mostrado arriba
- Cada fragmento debe tener: filePath, startLine, endLine, relevancia, razon
- La numeración debe ser consecutiva (1., 2., 3., etc.)
- NO uses JSON, solo texto plano estructurado
- SIEMPRE incluye startLine y endLine para identificar fragmentos específicos"""
    
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
2. Identifica cada fragmento por su archivo Y líneas exactas (startLine-endLine) pero añade su contexto adicional (contenido antes y después para mejor cobertura)
3. Dos fragmentos del mismo archivo pueden tener relevancia MUY diferente
4. Ejemplo: Si la consulta busca "validación de datos", el fragmento con validate_input() es más relevante que el fragmento con sanitize_output()
5. Reordena los fragmentos de mayor a menor relevancia
6. Incluye SIEMPRE startLine y endLine en tu respuesta
7. No incluyas codigo repetido. """
    
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
    
    def _process_response(self, response: str, original_results: list[SearchResult]) -> tuple[list[RerankResult], list[str]]:
        """Process LLM response and create reranked results + usages list.

        Parses plain text response instead of JSON for better reliability.

        Returns:
            Tuple of (reranked_results, usages_list)
        """
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

        # Parse plain text response
        reranked = []
        usages = []
        lines = response.strip().split('\n')

        current_item = {}
        in_usages_section = False

        for line in lines:
            line = line.strip()

            # Check if we're entering USAGES section
            if line.startswith('USAGES:'):
                # Save any pending item
                if current_item.get('filePath'):
                    reranked.append(self._create_rerank_result(current_item, results_map, fallback_map))
                    current_item = {}
                in_usages_section = True
                continue

            # Check if we're entering any of the summary sections (stop parsing reranked items)
            if line.startswith(('FLUJO END-TO-END:', 'ARCHIVOS RELACIONADOS:', 'CONCEPTOS CLAVE:', 'MÉTRICAS:', 'RESUMEN GLOBAL:')):
                # Save any pending item
                if current_item.get('filePath'):
                    reranked.append(self._create_rerank_result(current_item, results_map, fallback_map))
                    current_item = {}
                # Stop processing reranked items (summary sections are handled separately)
                break

            # Skip empty lines
            if not line:
                # If we have a complete item, process it
                if current_item.get('filePath') and not in_usages_section:
                    reranked.append(self._create_rerank_result(current_item, results_map, fallback_map))
                    current_item = {}
                continue

            # If we're in usages section, parse usage lines
            if in_usages_section:
                # Usage format: "src/server.ts:539" or just a file:line reference
                if ':' in line and not line.startswith('filePath:'):
                    usages.append(line)
                continue

            # Check if this is a new numbered item (e.g., "1.", "2.", etc.)
            if re.match(r'^\d+\.\s*filePath:', line):
                # Save previous item if exists
                if current_item.get('filePath'):
                    reranked.append(self._create_rerank_result(current_item, results_map, fallback_map))
                    current_item = {}

                # Extract filePath from this line
                match = re.search(r'filePath:\s*(.+)', line)
                if match:
                    current_item['filePath'] = match.group(1).strip()

            # Parse field lines
            elif line.startswith('filePath:'):
                # Save previous item if exists
                if current_item.get('filePath'):
                    reranked.append(self._create_rerank_result(current_item, results_map, fallback_map))
                    current_item = {}
                current_item['filePath'] = line.split(':', 1)[1].strip()

            elif line.startswith('startLine:'):
                try:
                    current_item['startLine'] = int(line.split(':', 1)[1].strip())
                except ValueError:
                    pass

            elif line.startswith('endLine:'):
                try:
                    current_item['endLine'] = int(line.split(':', 1)[1].strip())
                except ValueError:
                    pass

            elif line.startswith('relevancia:'):
                try:
                    current_item['relevancia'] = float(line.split(':', 1)[1].strip())
                except ValueError:
                    pass

            elif line.startswith('razon:'):
                current_item['razon'] = line.split(':', 1)[1].strip()

        # Don't forget the last item
        if current_item.get('filePath'):
            reranked.append(self._create_rerank_result(current_item, results_map, fallback_map))

        # If parsing failed completely, raise exception to trigger fallback
        if not reranked:
            print(f"[Judge] Warning: Could not parse response, will use raw LLM response", flush=True)
            raise ValueError(f"Failed to parse LLM response. Raw response will be returned.")

        return reranked, usages

    def _create_rerank_result(self, item: dict, results_map: dict, fallback_map: dict) -> Optional[RerankResult]:
        """Create a RerankResult from parsed item data."""
        file_path = item.get('filePath')
        start_line = item.get('startLine')
        end_line = item.get('endLine')
        relevancia = item.get('relevancia', 0.0)
        razon = item.get('razon')

        if not file_path:
            return None

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
            return None

        return RerankResult(
            file_path=original.file_path,
            code_chunk=original.code_chunk,
            start_line=original.start_line,
            end_line=original.end_line,
            score=original.score,
            relevancia=relevancia,
            razon=razon
        )
    
    async def rerank(self, query: str, results: list[SearchResult]) -> tuple[list[RerankResult], list[str]]:
        """Rerank search results using LLM.

        Args:
            query: User query
            results: Original search results

        Returns:
            Tuple of (reranked_results, usages_list)

        Raises:
            ValueError: If parsing fails, includes raw LLM response in exception message
        """
        if not results:
            return [], []

        user_prompt = self._create_user_prompt(query, results, include_summary=False)
        response = await self._call_llm(user_prompt, include_summary=False)

        try:
            return self._process_response(response, results)
        except ValueError as e:
            # Store raw response for fallback handling
            self._last_raw_response = response
            # Re-raise with raw response included
            raise ValueError(f"Failed to parse LLM response. Raw response: {response}") from e
    
    async def summarize(self, query: str, results: list[SearchResult]) -> str:
        """Generate enriched summary with flow, related files, concepts, and metrics.

        Args:
            query: User query
            results: Search results to summarize

        Returns:
            Enriched summary text with optional sections
        """
        if not results:
            return "No hay resultados para resumir."

        # Use top 10 results for enriched summary
        top_results = results[:10]

        summary_prompt = f"""Analiza los siguientes fragmentos de código y genera un resumen enriquecido explicando cómo responden a la consulta: "{query}"

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

        summary_prompt += f"""
Genera un resumen enriquecido con las siguientes secciones (SOLO incluye las que apliquen):

FLUJO END-TO-END: (Solo si hay 3+ fragmentos que forman un flujo secuencial)
Describe el flujo en 3-5 pasos numerados con flechas →
Ejemplo: "1. Usuario envía mensaje → 2. Se guarda sin embedding → 3. MessageIndexerService lo detecta → 4. Genera embedding en batch → 5. Almacena en SQLite"

ARCHIVOS RELACIONADOS: (Solo si hay múltiples archivos relacionados con relevancia > 0.7)
Lista archivos con breve descripción de su rol
Formato: "• filePath - Descripción del rol"
Máximo 5 archivos

CONCEPTOS CLAVE: (Solo si hay conceptos técnicos importantes)
Lista 3-5 conceptos con breve explicación
Formato: "• Concepto: Explicación breve"
Ejemplo: "• Batch Processing: Procesar múltiples items en una sola llamada API (20x más eficiente)"

MÉTRICAS:
📊 {len([r for r in results if r.score > 0.8])} de {len(results)} fragmentos relevantes ({int(len([r for r in results if r.score > 0.8])/len(results)*100)}%)
📁 {len(set(r.file_path for r in results))} archivos únicos
{"⚠️  Hay más resultados relevantes no mostrados" if len(results) >= 20 else "✅ Todos los resultados relevantes mostrados"}

RESUMEN GLOBAL:
Genera 2-3 párrafos explicando cómo estos fragmentos responden a la consulta:
- ENFÓCATE en las implementaciones reales encontradas en el código
- Sé específico sobre qué funcionalidades y patrones encontraste
- Menciona las líneas específicas cuando sea relevante

IMPORTANTE: Responde con las secciones en TEXTO PLANO, sin JSON, sin código, sin marcadores markdown.
Solo incluye las secciones que realmente apliquen (no inventes información)."""

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

