from __future__ import annotations

import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "sqlite:///./data/pico_sr.db"
    # LLM: "ollama" (local) or "groq" (cloud, no model download)
    llm_provider: str = "ollama"
    ollama_host: str = "http://127.0.0.1:11434"
    ollama_model: str = "llama3.1:8b"
    groq_api_key: str | None = None
    groq_model: str = "llama-3.1-8b-instant"
    groq_base_url: str = "https://api.groq.com/openai/v1"
    unpaywall_email: str = "you@example.com"
    ncbi_api_key: str | None = None
    pdf_dir: Path = Path("pdfs")
    forest_dir: Path = Path("output/forest")
    api_base: str = "http://127.0.0.1:8000"

    @property
    def sqlalchemy_url(self) -> str:
        url = self.database_url
        if url.startswith("sqlite:///./"):
            root = Path(__file__).resolve().parent.parent
            rel = url.replace("sqlite:///./", "")
            abs_path = (root / rel).resolve()
            abs_path.parent.mkdir(parents=True, exist_ok=True)
            return f"sqlite:///{abs_path.as_posix()}"
        return url


settings = Settings()
