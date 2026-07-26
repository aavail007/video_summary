from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    jobs_dir: Path = PROJECT_ROOT / "data" / "jobs"
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "").strip()
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "").strip()
    transcription_model: str = os.getenv(
        "TRANSCRIPTION_MODEL", "gpt-4o-transcribe"
    ).strip()
    summary_model: str = os.getenv("SUMMARY_MODEL", "gpt-5.6-terra").strip()
    summary_reasoning_effort: str = os.getenv(
        "SUMMARY_REASONING_EFFORT", "low"
    ).strip()
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-3.6-flash").strip()
    chunk_seconds: int = int(os.getenv("CHUNK_SECONDS", "900"))
    ytdlp_cookies_file: str = os.getenv("YTDLP_COOKIES_FILE", "").strip()

    def ensure_directories(self) -> None:
        self.jobs_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_directories()
