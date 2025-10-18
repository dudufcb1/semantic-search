"""Voyage AI reranker client for semantic search results."""
import sys
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class SearchResult:
    """Search result from vector store."""
    file_path: str
    code_chunk: str
    start_line: int
    end_line: int
    score: float


class VoyageReranker:
    """Client for Voyage AI reranking API using native SDK."""

    def __init__(
        self,
        api_key: str,
        model: str = "rerank-2-lite",
        top_k: Optional[int] = None,
        truncation: bool = True
    ):
        """Initialize Voyage reranker client.

        Args:
            api_key: Voyage API key
            model: Rerank model to use. Options:
                   - "rerank-2.5" (best quality, ~300ms, max 8k tokens query)
                   - "rerank-2.5-lite" (balanced, ~200ms, max 8k tokens query)
                   - "rerank-2" (good, ~250ms, max 4k tokens query)
                   - "rerank-2-lite" (fast, ~150ms, max 2k tokens query)
                   - "rerank-1" (legacy, max 2k tokens query)
                   - "rerank-lite-1" (legacy, max 1k tokens query)
            top_k: Number of top results to return (None = return all reranked)
            truncation: Whether to truncate long documents automatically
        """
        try:
            import voyageai
        except ImportError:
            raise ImportError(
                "voyageai package is required for native reranking.\n"
                "Install it with: pip install voyageai"
            )

        self.model = model
        self.top_k = top_k
        self.truncation = truncation

        # Initialize Voyage client with API key
        self.client = voyageai.Client(api_key=api_key)

        print(f"[Voyage Reranker] Initialized with model: {model}", file=sys.stderr)

    async def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: Optional[int] = None
    ) -> List[SearchResult]:
        """Rerank search results using Voyage AI.

        Args:
            query: The search query
            results: List of SearchResult objects from vector search
            top_k: Override default top_k for this call

        Returns:
            List of SearchResult objects reranked by relevance

        Raises:
            Exception: If reranking fails
        """
        if not results:
            return []

        # Use provided top_k or fall back to instance default
        effective_top_k = top_k if top_k is not None else self.top_k

        # Extract documents (code chunks) from results
        documents = [r.code_chunk for r in results]

        print(
            f"[Voyage Reranker] Reranking {len(documents)} documents with model {self.model}",
            file=sys.stderr
        )

        try:
            # Call Voyage rerank API (synchronous SDK)
            reranking = self.client.rerank(
                query=query,
                documents=documents,
                model=self.model,
                top_k=effective_top_k,
                truncation=self.truncation
            )

            print(
                f"[Voyage Reranker] Reranking successful. Total tokens: {reranking.total_tokens}",
                file=sys.stderr
            )

            # Rebuild SearchResult objects in reranked order with new scores
            reranked_results = []
            for result in reranking.results:
                original_result = results[result.index]

                # Create new SearchResult with Voyage relevance score
                reranked_result = SearchResult(
                    file_path=original_result.file_path,
                    code_chunk=original_result.code_chunk,
                    start_line=original_result.start_line,
                    end_line=original_result.end_line,
                    score=result.relevance_score  # Replace with Voyage score
                )
                reranked_results.append(reranked_result)

            print(
                f"[Voyage Reranker] Returned {len(reranked_results)} reranked results",
                file=sys.stderr
            )

            return reranked_results

        except Exception as e:
            error_msg = f"Voyage reranking failed: {str(e)}"
            print(f"[Voyage Reranker] ERROR: {error_msg}", file=sys.stderr)
            raise Exception(error_msg) from e
