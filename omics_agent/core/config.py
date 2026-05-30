"""
OmicsAgent configuration.

Reads from environment variables (or .env file) via Pydantic Settings.
Exposes get_llm_client() factory for both OpenAI and Anthropic backends.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import List, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class OmicsConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── LLM ──────────────────────────────────────────────────────────────────
    llm_provider: Literal["openai", "anthropic"] = "openai"
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    llm_model_openai: str = "gpt-4o"
    llm_model_anthropic: str = "claude-sonnet-4-6"
    llm_temperature: float = 0.1

    # ── LangSmith / Langfuse ─────────────────────────────────────────────────
    langchain_tracing_v2: bool = False
    langchain_api_key: str = Field(default="", alias="LANGCHAIN_API_KEY")
    langchain_project: str = "omics-agent"
    langfuse_public_key: str = Field(default="", alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str = Field(default="", alias="LANGFUSE_SECRET_KEY")

    # ── Vector store ─────────────────────────────────────────────────────────
    vector_backend: Literal["faiss", "pgvector"] = "faiss"
    database_url: str = "postgresql://omics:omicspass@localhost:5432/omicsdb"
    faiss_index_path: str = "data/faiss_index/patients.faiss"
    faiss_meta_db_path: str = "data/faiss_index/meta.db"

    # ── Models / caches ──────────────────────────────────────────────────────
    model_cache_dir: str = "./model_cache"
    embedding_cache_path: str = "data/processed/embeddings.h5"
    geneformer_model_name: str = "ctheodoris/Geneformer"
    geneformer_embedding_dim: int = 256
    geneformer_max_genes: int = 2048

    # ── Survival ─────────────────────────────────────────────────────────────
    survival_eval_times_months: List[int] = [12, 24, 36, 48, 60]
    conformal_alpha: float = 0.10
    resistance_threshold_kcal: float = 1.36   # re-used from ResistDTA convention

    # ── Saved model paths ────────────────────────────────────────────────────
    cox_model_path:      str = "model_cache/cox_model.pkl"
    rsf_model_path:      str = "model_cache/rsf_model.pkl"
    deepsurv_model_path: str = "model_cache/deepsurv_model.pkl"
    xgb_model_path:      str = "model_cache/xgb_model.pkl"
    ensemble_path:       str = "model_cache/ensemble.pkl"
    conformal_path:      str = "model_cache/conformal.pkl"
    shap_cox_path:       str = "model_cache/shap_cox.pkl"
    shap_rsf_path:       str = "model_cache/shap_rsf.pkl"
    shap_deepsurv_path:  str = "model_cache/shap_deepsurv.pkl"
    shap_xgb_path:       str = "model_cache/shap_xgb.pkl"

    # ── FAISS / DB paths (additional aliases used by agents) ────────────────
    faiss_db_path: str = "data/faiss_index/meta.db"

    # ── Retrieval ─────────────────────────────────────────────────────────────
    retrieval_top_k: int = 10
    max_genes:       int = 2048

    # ── Feature names ────────────────────────────────────────────────────────
    @property
    def feature_names_267(self) -> list:
        embed = [f"embed_{i}" for i in range(256)]
        clin  = [
            "age_at_diagnosis", "stage_i", "stage_ii", "stage_iii", "stage_iv",
            "grade_1", "grade_2", "grade_3",
            "treatment_chemo", "treatment_hormone", "treatment_surgery",
        ]
        return embed + clin

    @property
    def langchain_tracing(self) -> bool:
        return self.langchain_tracing_v2

    # ── Checkpointer ─────────────────────────────────────────────────────────
    checkpoint_db_path: str = "omics_checkpoints.db"

    @field_validator("survival_eval_times_months", mode="before")
    @classmethod
    def parse_eval_times(cls, v):
        if isinstance(v, str):
            return [int(x.strip()) for x in v.split(",")]
        return v

    def get_llm_client(self):
        """Return the appropriate LangChain LLM client based on LLM_PROVIDER."""
        if self.llm_provider == "openai":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=self.llm_model_openai,
                temperature=self.llm_temperature,
                api_key=self.openai_api_key or None,
            )
        else:
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(
                model=self.llm_model_anthropic,
                temperature=self.llm_temperature,
                api_key=self.anthropic_api_key or None,
            )

    def configure_tracing(self) -> None:
        """Set environment variables for LangSmith or Langfuse tracing."""
        if self.langchain_tracing_v2 and self.langchain_api_key:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_API_KEY"] = self.langchain_api_key
            os.environ["LANGCHAIN_PROJECT"] = self.langchain_project


@lru_cache(maxsize=1)
def get_config() -> OmicsConfig:
    cfg = OmicsConfig()
    cfg.configure_tracing()
    return cfg
