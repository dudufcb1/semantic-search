"""Embedder client for creating text embeddings."""
import httpx
from typing import Optional


# Query prefixes for specific models
QUERY_PREFIXES = {
    "openai": {
        "nomic-embed-code": "Represent this query for searching relevant code: ",
    },
    "openai-compatible": {
        "nomic-embed-code": "Represent this query for searching relevant code: ",
    },
}


class Embedder:
    """Client for creating embeddings using OpenAI-compatible API."""
    
    def __init__(
        self,
        provider: str,
        api_key: str,
        model_id: str,
        base_url: Optional[str] = None
    ):
        """Initialize embedder client.
        
        Args:
            provider: Provider type ("openai" or "openai-compatible")
            api_key: API key for authentication
            model_id: Model identifier
            base_url: Optional base URL for openai-compatible providers
        """
        self.provider = provider
        self.model_id = model_id
        
        # Determine endpoint
        if provider == "openai":
            self.endpoint = "https://api.openai.com/v1/embeddings"
        else:
            if not base_url:
                raise ValueError("base_url is required for openai-compatible provider")
            self.endpoint = f"{base_url.rstrip('/')}/embeddings"
        
        # Setup headers
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
    
    def _apply_query_prefix(self, text: str) -> str:
        """Apply model-specific query prefix if needed."""
        prefix = QUERY_PREFIXES.get(self.provider, {}).get(self.model_id)
        if prefix and not text.startswith(prefix):
            return f"{prefix}{text}"
        return text
    
    async def create_embedding(self, text: str) -> list[float]:
        """Create embedding for the given text.
        
        Args:
            text: Text to embed
            
        Returns:
            List of floats representing the embedding vector
            
        Raises:
            Exception: If the API request fails
        """
        # Apply query prefix
        processed_text = self._apply_query_prefix(text)
        
        # Prepare payload
        payload = {
            "model": self.model_id,
            "input": [processed_text],
        }
        
        # Make request
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.endpoint,
                headers=self.headers,
                json=payload,
                timeout=30.0
            )
            
            # Parse response
            if not response.is_success:
                error_data = response.json() if response.content else {}
                error_msg = (
                    error_data.get("error", {}).get("message") or
                    error_data.get("error") or
                    error_data.get("message") or
                    response.text or
                    "Failed to create embedding"
                )
                raise Exception(f"Embedder error: {error_msg}")
            
            data = response.json()
            
            # Extract embedding
            embedding = data.get("data", [{}])[0].get("embedding")
            if not embedding or not isinstance(embedding, list):
                raise Exception("Response does not contain valid embedding array")
            
            if len(embedding) == 0:
                raise Exception("Received empty embedding")
            
            # Convert to floats if needed
            result = []
            for i, value in enumerate(embedding):
                if isinstance(value, (int, float)):
                    result.append(float(value))
                elif isinstance(value, str):
                    try:
                        result.append(float(value))
                    except ValueError:
                        raise Exception(f"Cannot convert embedding value at index {i} to float")
                else:
                    raise Exception(f"Unsupported embedding type at index {i}: {type(value)}")
            
            return result

