"""
Configuration module for pharma-intelligence-ai.

Design decision: Flat Settings class instead of nested BaseSettings objects.
Pydantic-settings tries to JSON-parse nested model fields from env vars, which
breaks when the env var is a plain string like "chroma". Keeping everything flat
avoids this entirely — one class, one .env file, no surprises.

We expose grouped property accessors (.llm, .ct, .vs, .api) so call sites
still use clean namespaced access like settings.llm.provider.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Single flat Settings class — reads from .env file and environment variables.

    Usage:
        from src.core.config import get_settings
        s = get_settings()
        print(s.llm_provider)
        print(s.ct_page_size)
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM ---
    llm_provider: Literal["anthropic", "openai", "echo"] = Field(
    default="anthropic", alias="LLM_PROVIDER"
    )
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    anthropic_model: str = Field(default="claude-sonnet-4-5", alias="ANTHROPIC_MODEL")
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o", alias="OPENAI_MODEL")

    # --- ClinicalTrials.gov ---
    ct_base_url: str = Field(
        default="https://clinicaltrials.gov/api/v2", alias="CT_API_BASE_URL"
    )
    ct_rate_limit_per_sec: float = Field(
        default=1.0, alias="CT_RATE_LIMIT_PER_SEC"
    )
    ct_page_size: int = Field(
        default=100, alias="CT_PAGE_SIZE", ge=1, le=1000
    )
    ct_cache_dir: str = Field(default="./data/raw", alias="CT_CACHE_DIR")

    # --- Vector store ---
    vector_store_type: Literal["chroma", "pinecone"] = Field(
        default="chroma", alias="VECTOR_STORE"
    )
    chroma_persist_dir: str = Field(
        default="./chroma_db", alias="CHROMA_PERSIST_DIR"
    )
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2", alias="EMBEDDING_MODEL"
    )
    chroma_collection: str = Field(
        default="clinical_trials", alias="CHROMA_COLLECTION"
    )

    # --- API ---
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")
    api_cors_origins_raw: str = Field(
    default="http://localhost:3000", alias="API_CORS_ORIGINS"
    )
    api_rate_limit: str = Field(default="60/minute", alias="API_RATE_LIMIT")
    api_debug: bool = Field(default=False, alias="API_DEBUG")

    # --- Logging ---
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO", alias="LOG_LEVEL"
    )
    log_format: Literal["json", "console"] = Field(
        default="console", alias="LOG_FORMAT"
    )

    # --- Data dirs ---
    processed_data_dir: str = Field(
        default="./data/processed", alias="PROCESSED_DATA_DIR"
    )

    @property
    def api_cors_origins(self) -> list[str]:
        """Parse comma-separated CORS origins into a list."""
        return [o.strip() for o in self.api_cors_origins_raw.split(",")]

    # ------------------------------------------------------------------
    # Grouped accessors — so call sites can use settings.llm.provider
    # style access without triggering the nested BaseSettings JSON bug.
    # ------------------------------------------------------------------

    @property
    def llm(self) -> "LLMConfig":
        return LLMConfig(
            provider=self.llm_provider,
            anthropic_api_key=self.anthropic_api_key,
            anthropic_model=self.anthropic_model,
            openai_api_key=self.openai_api_key,
            openai_model=self.openai_model,
        )

    @property
    def ct(self) -> "CTConfig":
        return CTConfig(
            base_url=self.ct_base_url,
            rate_limit_per_sec=self.ct_rate_limit_per_sec,
            page_size=self.ct_page_size,
            cache_dir=self.ct_cache_dir,
        )

    @property
    def vs(self) -> "VectorStoreConfig":
        return VectorStoreConfig(
            store_type=self.vector_store_type,
            chroma_persist_dir=self.chroma_persist_dir,
            embedding_model=self.embedding_model,
            collection_name=self.chroma_collection,
        )

    @property
    def api(self) -> "APIConfig":
        return APIConfig(
            host=self.api_host,
            port=self.api_port,
            cors_origins=self.api_cors_origins,
            rate_limit=self.api_rate_limit,
            debug=self.api_debug,
        )

    @property
    def logging(self) -> "LoggingConfig":
        return LoggingConfig(
            level=self.log_level,
            format=self.log_format,
        )


# ---------------------------------------------------------------------------
# Plain dataclasses for the grouped accessors above.
# These are NOT BaseSettings — they're just typed containers.
# ---------------------------------------------------------------------------

@dataclass
class LLMConfig:
    provider: str
    anthropic_api_key: str
    anthropic_model: str
    openai_api_key: str
    openai_model: str


@dataclass
class CTConfig:
    base_url: str
    rate_limit_per_sec: float
    page_size: int
    cache_dir: str


@dataclass
class VectorStoreConfig:
    store_type: str
    chroma_persist_dir: str
    embedding_model: str
    collection_name: str


@dataclass
class APIConfig:
    host: str
    port: int
    cors_origins: list[str]
    rate_limit: str
    debug: bool


@dataclass
class LoggingConfig:
    level: str
    format: str


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the singleton Settings instance.
    Call get_settings.cache_clear() in tests to reset between cases.
    """
    return Settings()