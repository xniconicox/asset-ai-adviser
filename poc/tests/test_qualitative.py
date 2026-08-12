from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from asset_poc.config import Settings
from asset_poc.database import connect
from asset_poc.qualitative import (
    EvidenceItem,
    QualitativeDisclosure,
    _analysis_record,
    calculate_qualitative_features,
    export_qualitative_evaluation,
    ingest_disclosure_text,
    structure_disclosures,
)


class _Usage:
    input_tokens = 100
    output_tokens = 20


class _Response:
    id = "response-1"
    model = "test-model-resolved"
    usage = _Usage()


def _settings(tmp_path: Path) -> Settings:
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
        openai_api_key=None,
    )


def test_structured_schema_rejects_out_of_range_score() -> None:
    with pytest.raises(ValidationError):
        QualitativeDisclosure(
            summary="test",
            outlook_score=101,
            demand_score=50,
            profitability_score=50,
            risk_control_score=50,
            earnings_quality_score=50,
            positive_factors=[],
            negative_factors=[],
            evidence=[],
        )


def test_latest_analysis_becomes_qualitative_feature() -> None:
    analyses = pd.DataFrame(
        [
            {
                "document_id": "old",
                "canonical_code": "7203",
                "disclosure_date": "2026-02-01",
                "analyzed_at": pd.Timestamp("2026-02-02", tz="UTC"),
                "outlook_score": 40,
                "demand_score": 40,
                "profitability_score": 40,
                "risk_control_score": 40,
                "earnings_quality_score": 40,
                "confidence": 0.6,
            },
            {
                "document_id": "latest",
                "canonical_code": "7203",
                "disclosure_date": "2026-05-01",
                "analyzed_at": pd.Timestamp("2026-05-02", tz="UTC"),
                "outlook_score": 80,
                "demand_score": 70,
                "profitability_score": 60,
                "risk_control_score": 50,
                "earnings_quality_score": 40,
                "confidence": 0.8,
            },
        ]
    )
    result = calculate_qualitative_features(analyses, "2026-08-12")

    assert len(result) == 1
    assert result.loc[0, "document_count"] == 2
    assert result.loc[0, "qualitative_score"] == pytest.approx(63.5)
    assert result.loc[0, "qualitative_confidence"] == 0.8


def test_analysis_record_keeps_only_evidence_found_in_source() -> None:
    document = pd.Series(
        {
            "document_id": "doc-1",
            "canonical_code": "7203",
            "disclosure_date": "2026-08-12",
            "source_url": "https://example.com/release",
            "text_content": "営業利益は前年同期を上回りました。\n需要は堅調です。",
        }
    )
    result = QualitativeDisclosure(
        summary="増益です。",
        outlook_score=50,
        demand_score=60,
        profitability_score=70,
        risk_control_score=50,
        earnings_quality_score=50,
        positive_factors=[],
        negative_factors=[],
        evidence=[
            EvidenceItem(
                factor="profitability",
                excerpt="営業利益は前年同期を上回りました。 需要は堅調です。",
                interpretation="増益と需要の強さを確認。",
            ),
            EvidenceItem(
                factor="outlook",
                excerpt="通期予想を上方修正しました。",
                interpretation="本文にない記述。",
            ),
        ],
    )

    record = _analysis_record(document, result, _Response(), "test-model")

    evidence = __import__("json").loads(record["evidence"])
    assert len(evidence) == 1
    assert evidence[0]["factor"] == "profitability"
    assert record["confidence"] == pytest.approx(0.08)


def test_earnings_release_wins_over_later_same_day_revision() -> None:
    analyses = pd.DataFrame(
        [
            {
                "document_id": "release",
                "canonical_code": "7203",
                "disclosure_date": "2026-08-12",
                "document_type": "earnings_release",
                "analyzed_at": pd.Timestamp("2026-08-12 09:00", tz="UTC"),
                "outlook_score": 80,
                "demand_score": 80,
                "profitability_score": 80,
                "risk_control_score": 80,
                "earnings_quality_score": 80,
                "confidence": 0.8,
            },
            {
                "document_id": "revision",
                "canonical_code": "7203",
                "disclosure_date": "2026-08-12",
                "document_type": "forecast_revision",
                "analyzed_at": pd.Timestamp("2026-08-12 10:00", tz="UTC"),
                "outlook_score": 20,
                "demand_score": 20,
                "profitability_score": 20,
                "risk_control_score": 20,
                "earnings_quality_score": 20,
                "confidence": 0.8,
            },
        ]
    )

    result = calculate_qualitative_features(analyses, "2026-08-12")

    assert result.loc[0, "qualitative_score"] == 80


def test_ingest_preserves_source_and_no_key_skips_llm(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    source_file = tmp_path / "release.txt"
    source_file.write_text("通期予想を上方修正しました。", encoding="utf-8")
    result = ingest_disclosure_text(
        settings,
        source_file,
        "7203",
        "2026-08-12",
        "決算短信",
        "https://example.com/release",
    )

    with connect(settings.db_path) as connection:
        row = connection.execute(
            "SELECT source_url, text_content FROM disclosure_texts WHERE document_id = ?",
            [result["document_id"]],
        ).fetchone()
    assert row == ("https://example.com/release", "通期予想を上方修正しました。")

    skipped = structure_disclosures(settings)
    assert skipped["requested_documents"] == 0
    assert "OPENAI_API_KEY未設定" in skipped["warnings"][0]


def test_empty_qualitative_evaluation_writes_review_files(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    output = tmp_path / "evaluation.csv"

    result = export_qualitative_evaluation(settings, output)

    assert result["documents"] == 0
    assert result["evidence_match_rate"] == 0.0
    assert output.exists()
    assert output.with_suffix(".summary.json").exists()
