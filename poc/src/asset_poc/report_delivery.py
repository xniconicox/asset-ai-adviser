from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from asset_poc.config import Settings

JST = ZoneInfo("Asia/Tokyo")
REPORT_PREFIX = "asset-ai-model-ranking"


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def _atomic_json(payload: dict[str, object], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, destination)


def publish_daily_report(
    settings: Settings,
    report_date: date | None = None,
    report_dir: Path | None = None,
) -> dict[str, object]:
    """Generate a dated PDF and optionally copy it to an external synced folder."""
    from asset_poc.daily_ranking_report import generate_daily_ranking_report

    target_date = report_date or datetime.now(JST).date()
    date_text = target_date.strftime("%Y%m%d")
    relative_dir = Path(f"{target_date:%Y}") / f"{target_date:%m}"
    local_dir = settings.root / "output" / "pdf" / "daily" / relative_dir
    local_pdf = local_dir / f"{REPORT_PREFIX}-{date_text}.pdf"
    temporary_pdf = local_pdf.with_name(f".{local_pdf.name}.tmp")
    local_dir.mkdir(parents=True, exist_ok=True)

    summary = generate_daily_ranking_report(settings, temporary_pdf)
    os.replace(temporary_pdf, local_pdf)
    digest = hashlib.sha256(local_pdf.read_bytes()).hexdigest()
    metadata: dict[str, object] = {
        "report_date": target_date.isoformat(),
        "generated_at": datetime.now(JST).isoformat(),
        "file_name": local_pdf.name,
        "sha256": digest,
        "bytes": local_pdf.stat().st_size,
        "snapshot_date": summary.get("snapshot_date"),
        "report_kind": summary.get("report_kind"),
        "ranking_rows": summary.get("ranking_rows"),
        "model_version": summary.get("model_version"),
        "model_run": summary.get("model_run"),
    }
    local_json = local_pdf.with_suffix(".json")
    _atomic_json(metadata, local_json)

    destination_root = (
        report_dir or settings.report_output_dir or settings.google_drive_report_dir
    )
    output_result: dict[str, object] = {"configured": destination_root is not None}
    if destination_root is not None:
        destination_root = Path(destination_root)
        dated_dir = destination_root / "archive" / relative_dir
        drive_pdf = dated_dir / local_pdf.name
        drive_json = dated_dir / local_json.name
        _atomic_copy(local_pdf, drive_pdf)
        _atomic_copy(local_json, drive_json)
        _atomic_copy(local_pdf, destination_root / f"{REPORT_PREFIX}-latest.pdf")
        _atomic_copy(local_json, destination_root / f"{REPORT_PREFIX}-latest.json")
        output_result.update(
            {
                "status": "copied",
                "pdf": str(drive_pdf),
                "metadata": str(drive_json),
            }
        )
    else:
        output_result["status"] = "skipped"

    return {
        "status": "succeeded",
        "local_pdf": str(local_pdf),
        "local_metadata": str(local_json),
        "external_output": output_result,
        **metadata,
    }
