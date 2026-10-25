import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    llm_base_url: str
    llm_api_key: str
    llm_model: str
    db_path: Path
    llm_timeout: float


def get_settings() -> Settings:
    return Settings(
        llm_base_url=os.getenv("LLM_BASE_URL", "http://localhost:8080/v1").rstrip("/"),
        llm_api_key=os.getenv("LLM_API_KEY", "local"),
        llm_model=os.getenv("LLM_MODEL", "qwen3.6-35b-a3b"),
        db_path=Path(os.getenv("RAGCHAT_DB", str(ROOT / "data" / "ragchat.db"))),
        llm_timeout=float(os.getenv("LLM_TIMEOUT", "300")),
    )
