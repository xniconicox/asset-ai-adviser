from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from asset_poc.config import Settings
from asset_poc.report_delivery import publish_daily_report


def test_daily_report_is_versioned_and_copied_atomically(tmp_path, monkeypatch) -> None:
    settings = Settings(root=tmp_path, report_output_dir=tmp_path / "out")

    def fake_generate(_settings: Settings, output: Path) -> dict[str, object]:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"%PDF-1.4\ncurrent report\n")
        return {
            "snapshot_date": "2026-08-12",
            "report_kind": "daily_model_inference",
            "ranking_rows": 492,
            "model_version": "ridge-v2",
            "model_run": "run-1",
        }

    monkeypatch.setattr(
        "asset_poc.daily_ranking_report.generate_daily_ranking_report", fake_generate
    )
    result = publish_daily_report(settings, date(2026, 8, 13))

    local_pdf = Path(str(result["local_pdf"]))
    archive_pdf = (
        tmp_path
        / "out/archive/2026/08/asset-ai-model-ranking-20260813.pdf"
    )
    latest_pdf = tmp_path / "out/asset-ai-model-ranking-latest.pdf"
    assert local_pdf.read_bytes() == archive_pdf.read_bytes() == latest_pdf.read_bytes()
    metadata = json.loads(local_pdf.with_suffix(".json").read_text(encoding="utf-8"))
    assert metadata["report_date"] == "2026-08-13"
    assert metadata["snapshot_date"] == "2026-08-12"
    assert metadata["report_kind"] == "daily_model_inference"
    assert metadata["model_run"] == "run-1"
    assert len(metadata["sha256"]) == 64
    assert not list(tmp_path.rglob("*.tmp"))


def test_daily_report_keeps_local_copy_when_drive_is_not_configured(
    tmp_path, monkeypatch
) -> None:
    settings = Settings(
        root=tmp_path,
        report_output_dir=None,
        google_drive_report_dir=None,
    )

    def fake_generate(_settings: Settings, output: Path) -> dict[str, object]:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"%PDF-1.4\nlocal only\n")
        return {}

    monkeypatch.setattr(
        "asset_poc.daily_ranking_report.generate_daily_ranking_report", fake_generate
    )
    result = publish_daily_report(settings, date(2026, 8, 13))

    assert Path(str(result["local_pdf"])).exists()
    assert result["external_output"] == {"configured": False, "status": "skipped"}
