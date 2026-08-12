from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from asset_poc.config import Settings
from asset_poc.database import connect, initialize
from asset_poc.operations import (
    PipelineAlreadyRunning,
    create_backup,
    financial_catchup_dates,
    pipeline_lock,
    publish_snapshot,
    run_data_quality,
    start_batch,
)


def _settings(tmp_path) -> Settings:
    data_dir = tmp_path / "data"
    return Settings(
        root=tmp_path,
        data_dir=data_dir,
        raw_dir=data_dir / "raw",
        published_dir=data_dir / "published",
        backup_dir=data_dir / "backups",
        log_dir=tmp_path / "logs",
        db_path=data_dir / "poc.duckdb",
        pipeline_lock_path=data_dir / "pipeline.lock",
    )


def test_pipeline_lock_rejects_concurrent_writer(tmp_path) -> None:
    settings = _settings(tmp_path)
    with (
        pipeline_lock(settings),
        pytest.raises(PipelineAlreadyRunning),
        pipeline_lock(settings),
    ):
        pass


def test_financial_catchup_starts_at_target_without_watermark() -> None:
    assert financial_catchup_dates(None, date(2026, 5, 20)) == [date(2026, 5, 20)]


def test_financial_catchup_is_bounded_and_resumable() -> None:
    result = financial_catchup_dates("2026-05-01", date(2026, 5, 20), maximum_days=7)
    assert result == [date(2026, 5, day) for day in range(2, 9)]
    assert financial_catchup_dates("2026-05-20", date(2026, 5, 20)) == []


def test_quality_failure_does_not_prevent_atomic_snapshot_mechanism(tmp_path) -> None:
    settings = _settings(tmp_path)
    settings.ensure_dirs()
    with connect(settings.db_path) as connection:
        initialize(connection)
    batch_run_id = start_batch(settings, date(2026, 8, 12))

    quality = run_data_quality(settings, batch_run_id)
    assert quality["passed"] is False
    evidence_check = next(
        check for check in quality["checks"] if check["name"] == "qualitative_evidence_integrity"
    )
    assert evidence_check["status"] == "PASS"

    manifest = publish_snapshot(settings, batch_run_id)
    pointer = json.loads((settings.published_dir / "latest.json").read_text(encoding="utf-8"))
    assert pointer == manifest
    run_dir = settings.published_dir / manifest["run_dir"]
    assert (run_dir / "ranking_latest.parquet").exists()
    assert (run_dir / "data_quality.parquet").exists()
    assert (run_dir / "model_input_latest.parquet").exists()
    assert manifest["model_input_rows"] == 0


def test_backup_creates_database_copy_and_checksum(tmp_path) -> None:
    settings = _settings(tmp_path)
    settings.ensure_dirs()
    with connect(settings.db_path) as connection:
        initialize(connection)

    metadata = create_backup(settings)
    backup_path = Path(str(metadata["path"]))
    assert backup_path.exists()
    assert len(metadata["sha256"]) == 64
    assert backup_path.with_suffix(".json").exists()
