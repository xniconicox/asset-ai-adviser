from __future__ import annotations

import hashlib
import json
import re
from datetime import date
from pathlib import Path
from typing import Literal

import duckdb
import pandas as pd
from pydantic import BaseModel, Field

from asset_poc.config import Settings
from asset_poc.database import (
    connect,
    finish_acquisition_run,
    initialize,
    insert_frame,
    start_acquisition_run,
)

PROMPT_VERSION = "qualitative_ja_v4"
SCHEMA_VERSION = "qualitative_v2"
FEATURE_VERSION = "qualitative_v2"


class EvidenceItem(BaseModel):
    factor: Literal["outlook", "demand", "profitability", "risk", "earnings_quality"]
    excerpt: str = Field(
        max_length=100,
        description="入力文書にそのまま存在する、連続した一箇所からの短い根拠抜粋",
    )
    interpretation: str = Field(description="抜粋からスコアを判断した理由")


class QualitativeDisclosure(BaseModel):
    summary: str = Field(description="投資家向けの簡潔な要約")
    outlook_score: float = Field(ge=0, le=100, description="会社見通し。50は中立")
    demand_score: float = Field(ge=0, le=100, description="需要・受注の強さ。50は中立")
    profitability_score: float = Field(
        ge=0, le=100, description="採算・価格転嫁・コスト状況。50は中立"
    )
    risk_control_score: float = Field(ge=0, le=100, description="リスクの低さと管理状況。50は中立")
    earnings_quality_score: float = Field(
        ge=0, le=100, description="利益の継続性・一過性要因の少なさ。50は中立"
    )
    positive_factors: list[str] = Field(max_length=5)
    negative_factors: list[str] = Field(max_length=5)
    evidence: list[EvidenceItem] = Field(max_length=10)


SYSTEM_PROMPT = """あなたは日本企業の決算開示を構造化するアナリストです。
入力文書に明記された内容だけを使い、推測で事実を補わないでください。
各スコアは0（非常に弱い）から100（非常に強い）、情報がない場合は50です。
risk_control_scoreはリスクが低く管理されているほど高くしてください。
earnings_quality_scoreは一過性利益への依存が少なく継続性が高いほど高くしてください。
根拠excerptは必ず入力文書の連続した一箇所をそのまま抜粋してください。
複数箇所の結合、要約、言い換え、省略記号（…）による途中省略、原文にない括弧の追加は禁止です。
各factorにつき一つ、100文字以内のexcerptを返し、複数の根拠を一つにまとめないでください。
将来株価、投資推奨、目標株価は生成しないでください。"""


def ingest_disclosure_text(
    settings: Settings,
    file_path: Path,
    canonical_code: str,
    disclosure_date: date | str,
    title: str,
    source_url: str,
    document_type: str = "earnings_release",
    source: str = "manual",
) -> dict[str, object]:
    text = file_path.read_text(encoding="utf-8")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    document_id = f"{source}:{canonical_code}:{disclosure_date}:{digest[:16]}"
    frame = pd.DataFrame(
        [
            {
                "document_id": document_id,
                "canonical_code": canonical_code.strip().removesuffix(".T"),
                "disclosure_date": pd.Timestamp(disclosure_date).date(),
                "disclosure_time": None,
                "title": title,
                "document_type": document_type,
                "source": source,
                "source_url": source_url,
                "raw_path": str(file_path.resolve()),
                "content_hash": digest,
                "text_content": text,
                "retrieved_at": pd.Timestamp.now(tz="UTC"),
            }
        ]
    )
    settings.ensure_dirs()
    with connect(settings.db_path) as connection:
        initialize(connection)
        insert_frame(connection, "disclosure_texts", frame)
    return {
        "document_id": document_id,
        "canonical_code": canonical_code,
        "characters": len(text),
        "source_url": source_url,
    }


def _analysis_record(
    document: pd.Series,
    result: QualitativeDisclosure,
    response: object,
    requested_model: str,
) -> dict:
    usage = getattr(response, "usage", None)
    source_text = str(document["text_content"])
    normalized_source = re.sub(r"\s+", "", source_text)
    validated_evidence = [
        item
        for item in result.evidence
        if re.sub(r"\s+", "", item.excerpt) in normalized_source
    ]
    evidence_ratio = (
        len(validated_evidence) / len(result.evidence) if result.evidence else 0.0
    )
    factor_coverage = len({item.factor for item in validated_evidence}) / 5.0
    validated_confidence = 0.8 * evidence_ratio * factor_coverage
    return {
        "document_id": document["document_id"],
        "canonical_code": document["canonical_code"],
        "disclosure_date": document["disclosure_date"],
        "model": requested_model,
        "resolved_model": getattr(response, "model", None),
        "prompt_version": PROMPT_VERSION,
        "schema_version": SCHEMA_VERSION,
        "summary": result.summary,
        "outlook_score": result.outlook_score,
        "demand_score": result.demand_score,
        "profitability_score": result.profitability_score,
        "risk_control_score": result.risk_control_score,
        "earnings_quality_score": result.earnings_quality_score,
        "confidence": validated_confidence,
        "positive_factors": json.dumps(result.positive_factors, ensure_ascii=False),
        "negative_factors": json.dumps(result.negative_factors, ensure_ascii=False),
        "evidence": json.dumps(
            [item.model_dump() for item in validated_evidence], ensure_ascii=False
        ),
        "response_id": getattr(response, "id", None),
        "input_tokens": getattr(usage, "input_tokens", None),
        "output_tokens": getattr(usage, "output_tokens", None),
        "source_url": document["source_url"],
        "analyzed_at": pd.Timestamp.now(tz="UTC"),
    }


def structure_disclosures(
    settings: Settings, limit: int = 10, canonical_code: str | None = None
) -> dict[str, object]:
    """Call the LLM only for source documents not yet analyzed by this model/prompt."""
    if not settings.openai_api_key:
        return {
            "warnings": ["OPENAI_API_KEY未設定: LLM構造化をスキップ"],
            "requested_documents": 0,
        }
    from openai import OpenAI

    settings.ensure_dirs()
    with connect(settings.db_path) as connection:
        initialize(connection)
        where_code = "AND d.canonical_code = ?" if canonical_code else ""
        parameters: list[object] = [settings.openai_model, PROMPT_VERSION]
        if canonical_code:
            parameters.append(canonical_code.strip().removesuffix(".T"))
        parameters.append(limit)
        documents = connection.execute(
            f"""
            SELECT d.* FROM disclosure_texts d
            LEFT JOIN qualitative_analyses a
              ON a.document_id = d.document_id AND a.model = ? AND a.prompt_version = ?
            WHERE a.document_id IS NULL
              AND d.document_type IN (
                  'earnings_release', 'earnings_presentation',
                  'forecast_revision', 'dividend_revision'
              )
              {where_code}
            ORDER BY d.disclosure_date,
                     CASE d.document_type
                       WHEN 'earnings_release' THEN 1
                       WHEN 'earnings_presentation' THEN 2
                       WHEN 'forecast_revision' THEN 3
                       WHEN 'dividend_revision' THEN 4
                       ELSE 5
                     END,
                     d.document_id
            LIMIT ?
            """,
            parameters,
        ).df()
        run_id = start_acquisition_run(
            connection, "openai", f"qualitative:{settings.openai_model}", len(documents)
        )

    client = OpenAI(api_key=settings.openai_api_key)
    processed = 0
    errors: list[str] = []
    for row in documents.itertuples(index=False):
        document = pd.Series(row._asdict())
        try:
            response = client.responses.parse(
                model=settings.openai_model,
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"銘柄コード: {document['canonical_code']}\n"
                            f"開示日: {document['disclosure_date']}\n"
                            f"表題: {document['title']}\n\n"
                            f"開示本文:\n{document['text_content']}"
                        ),
                    },
                ],
                text_format=QualitativeDisclosure,
            )
            parsed = response.output_parsed
            if parsed is None:
                raise RuntimeError("Structured Output was empty or refused")
            record = _analysis_record(document, parsed, response, settings.openai_model)
            with connect(settings.db_path) as connection:
                insert_frame(connection, "qualitative_analyses", pd.DataFrame([record]))
            processed += 1
        except Exception as error:  # noqa: BLE001 - one document must not stop the batch
            errors.append(f"{document['document_id']}:{error}")

    with connect(settings.db_path) as connection:
        finish_acquisition_run(
            connection,
            run_id,
            "succeeded" if not errors else "partial",
            processed,
            len(errors),
            "; ".join(errors)[:4000],
        )
    return {
        "requested_documents": len(documents),
        "structured_documents": processed,
        "errors": errors,
        "model": settings.openai_model,
        "prompt_version": PROMPT_VERSION,
        "run_id": run_id,
    }


def calculate_qualitative_features(analyses: pd.DataFrame, snapshot_date: object) -> pd.DataFrame:
    if analyses.empty:
        return pd.DataFrame()
    effective_date = pd.Timestamp(snapshot_date).date()
    frame = analyses.copy()
    frame["disclosure_date"] = pd.to_datetime(frame["disclosure_date"], errors="coerce")
    frame = frame[frame["disclosure_date"].dt.date <= effective_date]
    document_priority = {
        "dividend_revision": 1,
        "forecast_revision": 2,
        "earnings_presentation": 3,
        "earnings_release": 4,
    }
    frame["_document_priority"] = (
        frame["document_type"].map(document_priority).fillna(0)
        if "document_type" in frame
        else 0
    )
    frame = frame.sort_values(
        ["canonical_code", "disclosure_date", "_document_priority", "analyzed_at"]
    )
    if frame.empty:
        return pd.DataFrame()

    records: list[dict[str, object]] = []
    score_columns = [
        "outlook_score",
        "demand_score",
        "profitability_score",
        "risk_control_score",
        "earnings_quality_score",
    ]
    for code, group in frame.groupby("canonical_code"):
        document_count = group["document_id"].nunique()
        latest = group.iloc[-1]
        qualitative_score = (
            float(latest["outlook_score"]) * 0.30
            + float(latest["demand_score"]) * 0.20
            + float(latest["profitability_score"]) * 0.20
            + float(latest["risk_control_score"]) * 0.15
            + float(latest["earnings_quality_score"]) * 0.15
        )
        records.append(
            {
                "snapshot_date": effective_date,
                "canonical_code": code,
                "disclosure_date": latest["disclosure_date"].date(),
                "qualitative_score": qualitative_score,
                **{column: latest[column] for column in score_columns},
                "qualitative_confidence": latest["confidence"],
                "document_count": document_count,
                "source": "openai+source_document",
                "feature_version": FEATURE_VERSION,
                "calculated_at": pd.Timestamp.now(tz="UTC"),
            }
        )
    return pd.DataFrame(records)


def compute_and_store_qualitative_features(
    connection: duckdb.DuckDBPyConnection, snapshot_date: object
) -> pd.DataFrame:
    analyses = connection.execute(
        """
        SELECT a.*, d.document_type
        FROM qualitative_analyses a
        LEFT JOIN disclosure_texts d USING (document_id)
        """
    ).df()
    features = calculate_qualitative_features(analyses, snapshot_date)
    insert_frame(connection, "qualitative_feature_snapshots", features)
    return features


def export_qualitative_evaluation(
    settings: Settings, output_path: Path | None = None
) -> dict[str, object]:
    """Export machine checks and blank human-review columns for the active prompt."""
    settings.ensure_dirs()
    output_path = (
        output_path or settings.root / "output" / "evaluations" / "qualitative_evaluation.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with connect(settings.db_path) as connection:
        initialize(connection)
        analyses = connection.execute(
            """
            SELECT a.*, d.title, d.text_content,
                   s.company_name, s.sector33_name
            FROM qualitative_analyses a
            JOIN disclosure_texts d ON d.document_id = a.document_id
            LEFT JOIN securities s ON s.canonical_code = a.canonical_code
            WHERE a.model = ? AND a.prompt_version = ?
            ORDER BY a.disclosure_date, a.canonical_code, a.document_id
            """,
            [settings.openai_model, PROMPT_VERSION],
        ).df()

    records: list[dict[str, object]] = []
    matched_evidence = 0
    total_evidence = 0
    for row in analyses.itertuples(index=False):
        evidence = json.loads(row.evidence or "[]")
        normalized_source = re.sub(r"\s+", "", row.text_content or "")
        matches = [
            re.sub(r"\s+", "", item.get("excerpt", "")) in normalized_source
            for item in evidence
        ]
        matched_evidence += sum(matches)
        total_evidence += len(matches)
        records.append(
            {
                "canonical_code": row.canonical_code,
                "company_name": row.company_name,
                "sector33_name": row.sector33_name,
                "disclosure_date": row.disclosure_date,
                "title": row.title,
                "summary": row.summary,
                "outlook_score": row.outlook_score,
                "demand_score": row.demand_score,
                "profitability_score": row.profitability_score,
                "risk_control_score": row.risk_control_score,
                "earnings_quality_score": row.earnings_quality_score,
                "validated_confidence": row.confidence,
                "evidence_count": len(evidence),
                "evidence_match_rate": sum(matches) / len(matches) if matches else 0.0,
                "input_tokens": row.input_tokens,
                "output_tokens": row.output_tokens,
                "source_url": row.source_url,
                "evidence_json": json.dumps(evidence, ensure_ascii=False),
                "human_summary_ok": "",
                "human_scores_ok": "",
                "human_evidence_ok": "",
                "review_notes": "",
            }
        )

    evaluation = pd.DataFrame(records)
    evaluation.to_csv(output_path, index=False, encoding="utf-8-sig")
    summary_path = output_path.with_suffix(".summary.json")
    summary = {
        "model": settings.openai_model,
        "prompt_version": PROMPT_VERSION,
        "schema_version": SCHEMA_VERSION,
        "documents": len(evaluation),
        "companies": 0 if evaluation.empty else int(evaluation["canonical_code"].nunique()),
        "sectors": 0 if evaluation.empty else int(evaluation["sector33_name"].nunique()),
        "evidence_items": total_evidence,
        "evidence_match_rate": (
            matched_evidence / total_evidence if total_evidence else 0.0
        ),
        "average_confidence": (
            0.0 if evaluation.empty else float(evaluation["validated_confidence"].mean())
        ),
        "input_tokens": (
            0 if evaluation.empty else int(evaluation["input_tokens"].fillna(0).sum())
        ),
        "output_tokens": (
            0 if evaluation.empty else int(evaluation["output_tokens"].fillna(0).sum())
        ),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        **summary,
        "evaluation_csv": str(output_path.resolve()),
        "summary_json": str(summary_path.resolve()),
    }
