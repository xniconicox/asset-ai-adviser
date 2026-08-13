from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

POC_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(POC_ROOT / ".env")


def _optional_path(name: str) -> Path | None:
    value = os.getenv(name)
    return Path(value).expanduser() if value else None


DEFAULT_REPORT_OUTPUT_DIR = _optional_path("REPORT_OUTPUT_DIR") or Path(
    "out_report"
)
LEGACY_GOOGLE_DRIVE_REPORT_DIR = _optional_path("GOOGLE_DRIVE_REPORT_DIR")


@dataclass(frozen=True)
class Settings:
    root: Path = POC_ROOT
    data_dir: Path = POC_ROOT / "data"
    raw_dir: Path = POC_ROOT / "data" / "raw"
    published_dir: Path = POC_ROOT / "data" / "published"
    backup_dir: Path = POC_ROOT / "data" / "backups"
    log_dir: Path = POC_ROOT / "logs"
    db_path: Path = POC_ROOT / "data" / "poc.duckdb"
    pipeline_lock_path: Path = POC_ROOT / "data" / "pipeline.lock"
    jquants_api_key: str | None = os.getenv("JQUANTS_API_KEY") or None
    edinet_api_key: str | None = os.getenv("EDINET_API_KEY") or None
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY") or None
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-5.4-nano")
    report_output_dir: Path | None = DEFAULT_REPORT_OUTPUT_DIR
    google_drive_report_dir: Path | None = LEGACY_GOOGLE_DRIVE_REPORT_DIR
    tickers: tuple[str, ...] = tuple(
        value.strip()
        for value in os.getenv("POC_TICKERS", "72030,67580,99840,83060,94320").split(",")
        if value.strip()
    )
    price_from: str = os.getenv("POC_PRICE_FROM", "2025-01-01")

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.published_dir.mkdir(parents=True, exist_ok=True)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
