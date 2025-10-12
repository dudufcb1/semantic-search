"""Qdrant vector store client for codebase search."""
import hashlib
from pathlib import Path
from typing import Optional
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, SearchParams
from dataclasses import dataclass


@dataclass
class SearchResult:
    """Search result from vector store."""
    file_path: str
    code_chunk: str
    start_line: int
    end_line: int
    score: float


class QdrantStore:
    """Qdrant vector store for codebase search."""
    
    def __init__(self, url: str, api_key: Optional[str] = None):
        """Initialize Qdrant client.
        
        Args:
            url: Qdrant server URL
            api_key: Optional API key for authentication
        """
        self.client = AsyncQdrantClient(
            url=url,
            api_key=api_key,
            timeout=30.0
        )
    
    def _get_collection_name(self, workspace_path: str) -> str:
        """Generate collection name from workspace path.
        
        Args:
            workspace_path: Absolute path to workspace
            
        Returns:
            Collection name in format ws-{hash}
        """
        # Normalize path
        normalized = str(Path(workspace_path).resolve())
        
        # Create hash
        hash_obj = hashlib.sha256(normalized.encode('utf-8'))
        hash_hex = hash_obj.hexdigest()
        
        # Return collection name
        return f"ws-{hash_hex[:16]}"
    
    def _normalize_workspace_path(self, workspace_path: str) -> str:
        """Normalize workspace path to absolute path.
        
        Args:
            workspace_path: Workspace path (can be relative or absolute)
            
        Returns:
            Normalized absolute path
        """
        return str(Path(workspace_path).resolve())
    
    def _build_path_filter(self, directory_prefix: Optional[str] = None) -> Optional[Filter]:
        """Build Qdrant filter for directory prefix.
        
        Args:
            directory_prefix: Optional directory prefix to filter by
            
        Returns:
            Qdrant Filter object or None
        """
        if not directory_prefix:
            return None
        
        # Normalize prefix
        normalized_prefix = directory_prefix.strip().rstrip('/')
        if not normalized_prefix:
            return None
        
        # Split into segments
        segments = [s for s in normalized_prefix.split('/') if s]
        if not segments:
            return None
        
        # Build filter conditions for each segment
        conditions = []
        for i, segment in enumerate(segments):
            conditions.append(
                FieldCondition(
                    key=f"pathSegments[{i}]",
                    match=MatchValue(value=segment)
                )
            )
        
        return Filter(must=conditions)
    
    async def search(
        self,
        vector: list[float],
        workspace_path: str,
        directory_prefix: Optional[str] = None,
        min_score: float = 0.4,
        max_results: int = 20
    ) -> list[SearchResult]:
        """Search for similar code chunks in the vector store.
        
        Args:
            vector: Query embedding vector
            workspace_path: Workspace path to search in
            directory_prefix: Optional directory prefix to filter results
            min_score: Minimum similarity score threshold
            max_results: Maximum number of results to return
            
        Returns:
            List of search results
        """
        # Normalize workspace path and get collection name
        normalized_workspace = self._normalize_workspace_path(workspace_path)
        collection_name = self._get_collection_name(normalized_workspace)
        
        # Build filter
        filter_obj = self._build_path_filter(directory_prefix)
        
        # Perform search
        search_result = await self.client.search(
            collection_name=collection_name,
            query_vector=vector,
            query_filter=filter_obj,
            limit=max_results,
            score_threshold=min_score,
            search_params=SearchParams(
                hnsw_ef=128,
                exact=False
            ),
            with_payload=True
        )
        
        # Convert to SearchResult objects
        results = []
        for point in search_result:
            if not point.payload:
                continue
            
            results.append(SearchResult(
                file_path=point.payload.get("filePath", ""),
                code_chunk=point.payload.get("codeChunk", ""),
                start_line=point.payload.get("startLine", 0),
                end_line=point.payload.get("endLine", 0),
                score=point.score
            ))
        
        return results

