"""Text Direct Judge (LLM) client - Experimental text-only format without JSON parsing."""
import httpx
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


class TextDirectJudge:
    """Experimental LLM client that uses structured text output instead of JSON."""

    def __init__(
        self,
        provider: str,
        api_key: str,
        model_id: str,
        max_tokens: int = 2048,
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

        self.system_prompt = system_prompt or self._get_text_direct_system_prompt()

    def _get_text_direct_system_prompt(self) -> str:
        """Get system prompt for text-direct format."""
        return """You are an expert code analysis model.
Given a user query and several code fragments, evaluate which fragments
actually answer the question and rank them from highest to lowest relevance.

MANDATORY RELEVANCE RULES:
1. SOURCE CODE (.ts, .js, .py, .java, .cpp, .go, .rs, .php, .rb, etc.):
      - MINIMUM relevance: 0.81 if relevant
      - ALWAYS must be above any documentation

2. DOCUMENTATION (.md, .txt, README):
      - MAXIMUM relevance: 0.8 (NEVER higher)
      - Only include if it matches the code and adds value

3. MULTIPLE FRAGMENTS FROM SAME FILE:
      - Evaluate EACH FRAGMENT INDIVIDUALLY
      - Identify each fragment by filePath + startLine + endLine
      - One fragment can be very relevant (0.95) and another from same file less relevant (0.5)
      - IMPORTANT: In your REASON, suggest user to view additional lines for complete context
        * Example: "💡 View lines 40-83 for complete function context"
        * This helps user understand code better without manual expansion

4. USAGES (IMPORTANT):
      - If you detect fragments that are USAGES (use the function/class but don't define it):
        * DO NOT include them in the main ranked list
        * Add them in a separate "USAGES:" section at the end
        * Format: Brief list of locations (filePath:lineNumber)
        * Example: "src/server.ts:539", "src/app.js:220"
      - Only include if there are 3+ usages detected
      - Maximum 10 usages in the list
      - Usages should NOT have relevance or reason, only location

REQUIRED RESPONSE FORMAT:
For each relevant file, use EXACTLY this format:

FILE: path/to/file.js
RELEVANCE: 0.95
CODE_SNIPPET: [COMPLETE code fragment from the chunk - DO NOT truncate]
const login = (user) => {
    if (!user) return null;
    const token = generateToken(user);
    return { token, user };
}
REASON: Contains the main authentication implementation. 💡 View lines 40-83 for complete function context.

FILE: path/to/file2.py
RELEVANCE: 0.87
CODE_SNIPPET: [COMPLETE code fragment from the chunk - DO NOT truncate]
def process_user(data):
    validated = validate(data)
    return save_to_db(validated)
REASON: Handles user data processing logic. 💡 View lines 115-150 for complete class definition.

[Continue for all relevant files, ordered by relevance highest to lowest]

USAGES:
src/server.ts:539
src/app.js:220
src/utils.py:145

CRITICAL REQUIREMENTS:
- Use EXACTLY "FILE:", "RELEVANCE:", "CODE_SNIPPET:", "REASON:" as prefixes
- CODE_SNIPPET must show the COMPLETE relevant code fragment, DO NOT truncate or summarize
- Include ALL lines from the fragment that are relevant to the query
- One blank line between each file entry
- Order by relevance (highest to lowest)
- Only include truly relevant files (relevance >= 0.5)
- NO JSON, NO markdown format, ONLY the specified format
- DO NOT TRUNCATE CODE: Show the complete code fragment, not just 1-2 lines
- USAGES section is optional, only if 3+ usages are detected"""

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

        summary_instruction = ""
        if include_summary:
            summary_instruction = """

After listing all relevant files, include a summary with:

SUMMARY:
[Consolidated 2-3 paragraph summary explaining how the most relevant fragments answer the query]
"""

        return f"""User query: "{query}"

Code fragments found:
{fragments_text}

Evaluate and rerank these fragments according to their relevance to the query.{summary_instruction}"""

    async def _call_llm(self, user_prompt: str, include_summary: bool) -> str:
        """Call LLM API and return response text."""
        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": self.system_prompt},
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
                raise Exception(f"TextDirectJudge error: {error_msg}")

            data = response.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")

            if not content:
                raise Exception("LLM returned empty response")

            return content.strip()

    def _parse_text_response(self, response: str, original_results: list[SearchResult]) -> tuple[list[RerankResult], Optional[str], list[str]]:
        """Parse text-direct response format with CODE_SNIPPET support and USAGES.

        Returns:
            Tuple of (reranked_results, summary, usages_list)
        """
        # Create lookup for original results
        results_map = {r.file_path: r for r in original_results}

        # Split response into files, usages, and summary sections
        usages = []
        summary = None

        # First, extract SUMMARY if present
        parts = response.split("SUMMARY:")
        files_and_usages = parts[0].strip()
        if len(parts) > 1:
            summary = parts[1].strip()

        # Then, extract USAGES if present
        usages_parts = files_and_usages.split("USAGES:")
        files_section = usages_parts[0].strip()
        if len(usages_parts) > 1:
            usages_text = usages_parts[1].strip()
            # Parse usages (format: "src/file.ts:123")
            for line in usages_text.split('\n'):
                line = line.strip()
                if ':' in line and not line.startswith('FILE:'):
                    usages.append(line)

        # Parse files section
        reranked = []

        # Split by FILE: markers
        file_blocks = re.split(r'\nFILE:\s*', files_section)

        # Skip first empty block if present
        if file_blocks and not file_blocks[0].strip():
            file_blocks = file_blocks[1:]

        # Handle first block if it starts with FILE:
        if files_section.strip().startswith('FILE:'):
            first_file = files_section.split('FILE:', 1)[1]
            file_blocks = [first_file] + file_blocks[1:]

        for block in file_blocks:
            if not block.strip():
                continue

            lines = block.strip().split('\n')
            if not lines:
                continue

            # Parse FILE: line (could be the first line or already extracted)
            file_path = lines[0].strip()

            # Find RELEVANCE:, CODE_SNIPPET:, and REASON: lines
            relevancia = 0.5  # default
            code_snippet = None
            razon = None

            for line in lines[1:]:
                line = line.strip()
                if line.startswith('RELEVANCE:'):
                    try:
                        relevancia = float(line.split('RELEVANCE:', 1)[1].strip())
                    except (ValueError, IndexError):
                        relevancia = 0.5
                elif line.startswith('CODE_SNIPPET:'):
                    code_snippet = line.split('CODE_SNIPPET:', 1)[1].strip()
                elif line.startswith('REASON:'):
                    razon = line.split('REASON:', 1)[1].strip()
                # Also support Spanish versions for backward compatibility
                elif line.startswith('RELEVANCIA:'):
                    try:
                        relevancia = float(line.split('RELEVANCIA:', 1)[1].strip())
                    except (ValueError, IndexError):
                        relevancia = 0.5
                elif line.startswith('RAZON:'):
                    razon = line.split('RAZON:', 1)[1].strip()

            # Enhance reason with code snippet if available
            if code_snippet and razon:
                razon = f"{razon} | Code: {code_snippet}"
            elif code_snippet and not razon:
                razon = f"Relevant code: {code_snippet}"

            # Find corresponding original result
            original = results_map.get(file_path)
            if not original:
                # Try partial matching for similar file paths
                for orig_path, orig_result in results_map.items():
                    if orig_path.endswith(file_path) or file_path.endswith(orig_path):
                        original = orig_result
                        break

            if original and relevancia >= 0.5:
                reranked.append(RerankResult(
                    file_path=original.file_path,
                    code_chunk=original.code_chunk,
                    start_line=original.start_line,
                    end_line=original.end_line,
                    score=original.score,
                    relevancia=relevancia,
                    razon=razon
                ))

        # Sort by relevancia (highest first)
        reranked.sort(key=lambda x: x.relevancia, reverse=True)

        return reranked, summary, usages

    async def rerank(self, query: str, results: list[SearchResult]) -> tuple[list[RerankResult], list[str]]:
        """Rerank search results using text-direct format.

        Returns:
            Tuple of (reranked_results, usages_list)
        """
        if not results:
            return [], []

        user_prompt = self._create_user_prompt(query, results, include_summary=False)
        response = await self._call_llm(user_prompt, include_summary=False)

        reranked, _, usages = self._parse_text_response(response, results)
        return reranked, usages

    async def summarize(self, query: str, results: list[SearchResult]) -> str:
        """Generate summary using text-direct format."""
        if not results:
            return "No hay resultados para resumir."

        # Use top 5 results for summary
        top_results = results[:5]

        user_prompt = self._create_user_prompt(query, top_results, include_summary=True)
        response = await self._call_llm(user_prompt, include_summary=True)

        # Extract summary from response
        if "SUMMARY:" in response:
            summary = response.split("SUMMARY:", 1)[1].strip()
        else:
            # Fallback: use entire response as summary
            summary = response

        return summary

    async def rerank_with_summary(self, query: str, results: list[SearchResult]) -> tuple[list[RerankResult], str, list[str]]:
        """Rerank and summarize in one call using text-direct format.

        Returns:
            Tuple of (reranked_results, summary, usages_list)
        """
        if not results:
            return [], "No hay resultados para resumir.", []

        user_prompt = self._create_user_prompt(query, results, include_summary=True)
        response = await self._call_llm(user_prompt, include_summary=True)

        reranked, summary, usages = self._parse_text_response(response, results)

        if not summary:
            summary = f"Se encontraron {len(reranked)} archivos relevantes para la consulta '{query}'."

        return reranked, summary, usages