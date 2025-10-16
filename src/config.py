"""Configuration management using Pydantic Settings."""
import os
from pathlib import Path
from typing import Literal, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# Get the directory where this file is located
_CONFIG_DIR = Path(__file__).parent
# Go up one level to the project root
_PROJECT_ROOT = _CONFIG_DIR.parent
# Path to .env file
_ENV_FILE = _PROJECT_ROOT / ".env"


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    Supports both MCP_CODEBASE_* and short names for compatibility.
    """

    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # Default workspace path
    mcp_codebase_workspace: Optional[str] = None

    # Collection source mode: "default" (hash from workspace) or "codebase-indexer" (explicit collection)
    mcp_codebase_collection_source: str = "default"

    # Tools enable/disable flags
    mcp_codebase_enable_search: bool = True
    mcp_codebase_enable_rerank: bool = True

    # Qdrant configuration
    mcp_codebase_qdrant_url: str
    mcp_codebase_qdrant_api_key: Optional[str] = None

    # Embedder configuration
    mcp_codebase_embedder_provider: Literal["openai", "openai-compatible"] = "openai"
    mcp_codebase_embedder_api_key: str
    mcp_codebase_embedder_base_url: Optional[str] = None
    mcp_codebase_embedder_model_id: str = "text-embedding-3-small"

    # Judge (LLM) configuration
    mcp_codebase_judge_provider: Literal["openai", "openai-compatible", "mcp-internal"] = "openai"
    mcp_codebase_judge_api_key: str
    mcp_codebase_judge_base_url: Optional[str] = None
    mcp_codebase_judge_model_id: str = "gpt-4o-mini"
    mcp_codebase_judge_max_tokens: int = 32000
    mcp_codebase_judge_temperature: float = 0.0
    mcp_codebase_judge_system_prompt: Optional[str] = None

    # Reranking configuration
    mcp_codebase_reranking_max_results: int = 10
    mcp_codebase_reranking_include_reason: bool = True
    mcp_codebase_reranking_summarize: bool = False

    # Search configuration
    mcp_codebase_search_min_score: float = Field(default=0.1, ge=0.0, le=1.0)
    mcp_codebase_search_max_results: int = Field(default=20, ge=1, le=100)

    # LLM configuration for refined answer (defaults to Anthropic Claude)
    mcp_codebase_llm_api_key: Optional[str] = None
    mcp_codebase_llm_model_id: str = "claude-3-5-sonnet-20241022"

    # Properties for easier access
    @property
    def default_workspace_path(self) -> Optional[str]:
        return self.mcp_codebase_workspace

    @property
    def qdrant_url(self) -> str:
        return self.mcp_codebase_qdrant_url

    @property
    def qdrant_api_key(self) -> Optional[str]:
        return self.mcp_codebase_qdrant_api_key

    @property
    def embedder_provider(self) -> str:
        return self.mcp_codebase_embedder_provider

    @property
    def embedder_api_key(self) -> str:
        return self.mcp_codebase_embedder_api_key

    @property
    def embedder_base_url(self) -> Optional[str]:
        return self.mcp_codebase_embedder_base_url

    @property
    def embedder_model_id(self) -> str:
        return self.mcp_codebase_embedder_model_id

    @property
    def judge_provider(self) -> str:
        return self.mcp_codebase_judge_provider

    @property
    def judge_api_key(self) -> str:
        return self.mcp_codebase_judge_api_key

    @property
    def judge_base_url(self) -> Optional[str]:
        return self.mcp_codebase_judge_base_url

    @property
    def judge_model_id(self) -> str:
        return self.mcp_codebase_judge_model_id

    @property
    def judge_max_tokens(self) -> int:
        return self.mcp_codebase_judge_max_tokens

    @property
    def judge_temperature(self) -> float:
        return self.mcp_codebase_judge_temperature

    @property
    def judge_system_prompt(self) -> Optional[str]:
        return self.mcp_codebase_judge_system_prompt

    @property
    def reranking_max_results(self) -> int:
        return self.mcp_codebase_reranking_max_results

    @property
    def reranking_include_reason(self) -> bool:
        return self.mcp_codebase_reranking_include_reason

    @property
    def reranking_summarize(self) -> bool:
        return self.mcp_codebase_reranking_summarize

    @property
    def search_min_score(self) -> float:
        return self.mcp_codebase_search_min_score

    @property
    def search_max_results(self) -> int:
        return self.mcp_codebase_search_max_results

    @property
    def collection_source(self) -> str:
        return self.mcp_codebase_collection_source

    @property
    def enable_search(self) -> bool:
        return self.mcp_codebase_enable_search

    @property
    def enable_rerank(self) -> bool:
        return self.mcp_codebase_enable_rerank

    @property
    def llm_api_key(self) -> Optional[str]:
        return self.mcp_codebase_llm_api_key

    @property
    def llm_model_id(self) -> str:
        return self.mcp_codebase_llm_model_id


# Global settings instance
settings = Settings()

