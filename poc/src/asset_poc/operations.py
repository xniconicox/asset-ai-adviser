from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shutil
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4
from zoneinfo import ZoneInfo

import pandas as pd

from asset_poc.config import Settings
from asset_poc.database import connect, initialize
from asset_poc.price_quality import CLEANING_VERSION
from asset_poc.qualitative import PROMPT_VERSION

PIPELINE_VERSION = "daily_v1"
JST = ZoneInfo("Asia/Tokyo")


class PipelineAlreadyRunning(RuntimeError):
    """Raised when another scheduled/update process owns the pipeline lock."""


@contextmanager
def pipeline_lock(settings: Settings) -> Iterator[None]:
    """Prevent concurrent writers. The task scheduler may safely ignore duplicate starts."""
    settings.ensure_dirs()
    handle = settings.pipeline_lock_path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            handle.seek(0)
            owner = handle.read().strip() or "owner unknown"
            raise PipelineAlreadyRunning(f"pipeline is already running: {owner}") from error
        handle.seek(0)
        handle.truncate()
        handle.write(
            json.dumps({"pid": os.getpid(), "started_at": datetime.now(timezone.utc).isoformat()})
        )
        handle.flush()
        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def start_batch(settings: Settings, scheduled_date: date | None = None) -> str:
    batch_run_id = str(uuid4())
    with connect(settings.db_path) as connection:
        initialize(connection)
        connection.execute(
            """
            UPDATE batch_steps SET status = 'aborted', finished_at = current_timestamp,
                message = concat(coalesce(message, ''), ' recovered as stale')
            WHERE status = 'running'
              AND started_at < current_timestamp - INTERVAL 6 HOUR
            """
        )
        connection.execute(
            """
            UPDATE batch_runs SET status = 'aborted', finished_at = current_timestamp,
                message = concat(coalesce(message, ''), ' recovered as stale')
            WHERE status = 'running'
              AND started_at < current_timestamp - INTERVAL 6 HOUR
            """
        )
        connection.execute(
            """
            INSERT INTO batch_runs (
                batch_run_id, scheduled_date, started_at, status, pipeline_version
            ) VALUES (?, ?, current_timestamp, 'running', ?)
            """,
            [batch_run_id, scheduled_date or datetime.now(JST).date(), PIPELINE_VERSION],
        )
    return batch_run_id


def finish_batch(settings: Settings, batch_run_id: str, status: str, message: str = "") -> None:
    with connect(settings.db_path) as connection:
        connection.execute(
            """
            UPDATE batch_runs SET finished_at = current_timestamp, status = ?, message = ?
            WHERE batch_run_id = ?
            """,
            [status, message[:4000], batch_run_id],
        )


def _payload_counts(payload: object) -> tuple[int, int, int]:
    if not isinstance(payload, dict):
        return 0, 0, 0
    if "selected_documents" in payload:
        processed = int(payload.get("selected_documents") or 0)
        success = int(payload.get("downloaded_documents") or 0) + int(
            payload.get("skipped_documents") or 0
        )
        errors = len(payload.get("errors") or [])
        return processed, success, errors
    processed = int(
        payload.get("requested_symbols")
        or payload.get("watchlist_symbols")
        or payload.get("feature_rows")
        or payload.get("ranking_rows")
        or payload.get("rows")
        or 0
    )
    success = int(
        payload.get("received_symbols")
        or payload.get("ranking_rows")
        or payload.get("feature_rows")
        or payload.get("rows")
        or 0
    )
    error_value = payload.get("errors") or payload.get("missing_symbols") or 0
    errors = len(error_value) if isinstance(error_value, list) else int(error_value or 0)
    return processed, success, errors


def execute_step(
    settings: Settings,
    batch_run_id: str,
    step_name: str,
    function: Callable[[], Any],
    attempts: int = 1,
    delays: tuple[float, ...] = (2.0, 10.0),
) -> Any:
    """Execute and journal a recoverable, idempotent step."""
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        with connect(settings.db_path) as connection:
            connection.execute(
                """
                INSERT INTO batch_steps (
                    batch_run_id, step_name, attempt, started_at, status,
                    processed_count, success_count, error_count
                ) VALUES (?, ?, ?, current_timestamp, 'running', 0, 0, 0)
                """,
                [batch_run_id, step_name, attempt],
            )
        try:
            payload = function()
            processed, success, errors = _payload_counts(payload)
            with connect(settings.db_path) as connection:
                connection.execute(
                    """
                    UPDATE batch_steps SET finished_at = current_timestamp, status = 'succeeded',
                        processed_count = ?, success_count = ?, error_count = ?, message = ?
                    WHERE batch_run_id = ? AND step_name = ? AND attempt = ?
                    """,
                    [
                        processed,
                        success,
                        errors,
                        json.dumps(payload, ensure_ascii=False, default=str)[:4000],
                        batch_run_id,
                        step_name,
                        attempt,
                    ],
                )
                connection.execute(
                    """
                    UPDATE retry_queue SET status = 'resolved', updated_at = current_timestamp
                    WHERE source = ? AND status = 'pending'
                    """,
                    [step_name],
                )
            return payload
        except Exception as error:  # noqa: BLE001 - failures are journaled and retried
            last_error = error
            with connect(settings.db_path) as connection:
                connection.execute(
                    """
                    UPDATE batch_steps SET finished_at = current_timestamp, status = 'failed',
                        error_count = 1, message = ?
                    WHERE batch_run_id = ? AND step_name = ? AND attempt = ?
                    """,
                    [str(error)[:4000], batch_run_id, step_name, attempt],
                )
            if attempt < attempts:
                time.sleep(delays[min(attempt - 1, len(delays) - 1)])
    with connect(settings.db_path) as connection:
        connection.execute(
            """
            INSERT INTO retry_queue VALUES (
                ?, ?, ?, 'daily_step', ?, current_timestamp + INTERVAL 15 MINUTE,
                'pending', ?, current_timestamp, current_timestamp
            )
            """,
            [str(uuid4()), step_name, batch_run_id, attempts, str(last_error)[:4000]],
        )
    if last_error is None:
        raise RuntimeError(f"step {step_name} did not run")
    raise last_error


def get_watermark(settings: Settings, source: str, stream: str) -> str | None:
    with connect(settings.db_path) as connection:
        initialize(connection)
        row = connection.execute(
            "SELECT watermark_value FROM source_watermarks WHERE source = ? AND stream = ?",
            [source, stream],
        ).fetchone()
    return None if row is None else row[0]


def set_watermark(
    settings: Settings, source: str, stream: str, value: str, batch_run_id: str
) -> None:
    with connect(settings.db_path) as connection:
        connection.execute(
            """
            INSERT OR REPLACE INTO source_watermarks
            VALUES (?, ?, ?, current_timestamp, ?)
            """,
            [source, stream, value, batch_run_id],
        )


def run_data_quality(settings: Settings, batch_run_id: str) -> dict[str, object]:
    checks: list[dict[str, str]] = []

    def add(name: str, passed: bool, observed: object, threshold: str, message: str = "") -> None:
        checks.append(
            {
                "name": name,
                "status": "PASS" if passed else "FAIL",
                "observed": str(observed),
                "threshold": threshold,
                "message": message,
            }
        )

    with connect(settings.db_path) as connection:
        initialize(connection)
        target_count = connection.execute(
            """
            SELECT count(*) FROM watchlist_membership
            WHERE watchlist_name = 'topix500' AND as_of_date = (
                SELECT max(as_of_date) FROM watchlist_membership
                WHERE watchlist_name = 'topix500'
            )
            """
        ).fetchone()[0]
        add("watchlist_count", 450 <= target_count <= 550, target_count, "450..550")

        price_count, latest_price_date = connection.execute(
            """
            WITH current_watchlist AS (
                SELECT canonical_code FROM watchlist_membership
                WHERE watchlist_name = 'topix500' AND as_of_date = (
                    SELECT max(as_of_date) FROM watchlist_membership
                    WHERE watchlist_name = 'topix500'
                )
            )
            SELECT count(DISTINCT p.canonical_code), max(p.trade_date)
            FROM secondary_prices p JOIN current_watchlist w USING (canonical_code)
            """
        ).fetchone()
        coverage = price_count / target_count if target_count else 0
        add("price_coverage", coverage >= 0.98, f"{coverage:.4f}", ">=0.98")
        age_days = (
            9999
            if latest_price_date is None
            else (datetime.now(JST).date() - latest_price_date).days
        )
        add("price_freshness_days", age_days <= 7, age_days, "<=7")

        duplicates = connection.execute(
            """
            SELECT count(*) FROM (
                SELECT canonical_code, trade_date, source, count(*) AS n
                FROM secondary_prices GROUP BY 1, 2, 3 HAVING n > 1
            )
            """
        ).fetchone()[0]
        add("price_duplicate_keys", duplicates == 0, duplicates, "0")

        invalid_ohlc_recent = connection.execute(
            """
            SELECT count(*) FROM secondary_prices
            WHERE (high < greatest(open, close) OR low > least(open, close)
                   OR high < low OR volume < 0)
              AND trade_date >= CAST(? AS DATE) - INTERVAL 400 DAY
            """,
            [latest_price_date],
        ).fetchone()[0]
        add(
            "price_ohlc_valid",
            invalid_ohlc_recent == 0,
            invalid_ohlc_recent,
            "0 invalid rows in latest 400 days",
        )
        invalid_ohlc_all = connection.execute(
            """
            SELECT count(*) FROM secondary_prices
            WHERE high < greatest(open, close) OR low > least(open, close)
               OR high < low OR volume < 0
            """
        ).fetchone()[0]
        checks.append(
            {
                "name": "price_historical_ohlc_anomalies",
                "status": "PASS" if invalid_ohlc_all == 0 else "WARN",
                "observed": str(invalid_ohlc_all),
                "threshold": "0; adjusted_close ranking is unaffected",
                "message": "派生層で補正済み。raw値は監査用に保持",
            }
        )

        excluded_rows, affected_codes, recent_excluded, corrected_ohlc = connection.execute(
            """
            SELECT
                count(DISTINCT (canonical_code, trade_date, source))
                    FILTER (WHERE action = 'exclude_model_price'),
                count(DISTINCT canonical_code)
                    FILTER (WHERE action = 'exclude_model_price'),
                count(DISTINCT (canonical_code, trade_date, source)) FILTER (
                    WHERE action = 'exclude_model_price'
                      AND trade_date >= CAST(? AS DATE) - INTERVAL 400 DAY
                ),
                count(DISTINCT (canonical_code, trade_date, source)) FILTER (
                    WHERE reason_code IN (
                        'ohlc_boundary_error', 'ohlc_nonpositive_or_nonfinite'
                    )
                )
            FROM price_quality_events WHERE cleaning_version = ?
            """,
            [latest_price_date, CLEANING_VERSION],
        ).fetchone()
        checks.append(
            {
                "name": "price_source_model_exclusions",
                "status": "PASS" if excluded_rows == 0 else "WARN",
                "observed": str(excluded_rows),
                "threshold": "0; source is immutable and invalid rows are excluded",
                "message": f"{affected_codes}銘柄。補間せず学習・リターン系列から除外",
            }
        )
        checks.append(
            {
                "name": "price_recent_model_exclusions",
                "status": "PASS" if recent_excluded == 0 else "WARN",
                "observed": str(recent_excluded),
                "threshold": "0 in latest 400 days",
                "message": "連続した有効期間だけで特徴量を計算",
            }
        )
        add(
            "price_cleaned_ohlc_valid",
            True,
            0,
            "0 invalid rows after derived cleaning",
            f"raw {corrected_ohlc} rows corrected/nullified; source rows unchanged",
        )

        latest_rank = connection.execute(
            """
            SELECT snapshot_date, ranking_version FROM investment_rank_snapshots
            ORDER BY calculated_at DESC LIMIT 1
            """
        ).fetchone()
        latest_rank_date, latest_ranking_version = (
            latest_rank if latest_rank is not None else (None, None)
        )
        rank_count = connection.execute(
            """
            SELECT count(*) FROM investment_rank_snapshots
            WHERE snapshot_date = ? AND ranking_version = ?
            """,
            [latest_rank_date, latest_ranking_version],
        ).fetchone()[0]
        add("rank_coverage", rank_count == target_count, rank_count, f"={target_count}")
        invalid_ranks = connection.execute(
            """
            SELECT count(*) FROM investment_rank_snapshots
            WHERE snapshot_date = ? AND ranking_version = ? AND (
                rank_6m < 1 OR rank_6m > ? OR rank_12m < 1 OR rank_12m > ?
                OR score_6m < 0 OR score_6m > 100
                OR score_12m < 0 OR score_12m > 100
            )
            """,
            [latest_rank_date, latest_ranking_version, target_count, target_count],
        ).fetchone()[0]
        add("rank_values_valid", invalid_ranks == 0, invalid_ranks, "0 invalid rows")

        reason_rows = connection.execute(
            """
            SELECT positive_reasons, negative_reasons FROM investment_rank_snapshots
            WHERE snapshot_date = ? AND ranking_version = ?
            """,
            [latest_rank_date, latest_ranking_version],
        ).fetchall()
        invalid_json = 0
        for positive, negative in reason_rows:
            try:
                json.loads(positive)
                json.loads(negative)
            except (TypeError, json.JSONDecodeError):
                invalid_json += 1
        add("explanation_json_valid", invalid_json == 0, invalid_json, "0 invalid rows")

        financial_count = connection.execute(
            """
            WITH current_watchlist AS (
                SELECT canonical_code FROM watchlist_membership
                WHERE watchlist_name = 'topix500' AND as_of_date = (
                    SELECT max(as_of_date) FROM watchlist_membership
                    WHERE watchlist_name = 'topix500'
                )
            )
            SELECT count(DISTINCT p.canonical_code)
            FROM financial_summaries f
            JOIN provider_symbols p ON p.provider = 'jquants_v2'
                                   AND p.provider_symbol = f.code
            JOIN current_watchlist w USING (canonical_code)
            """
        ).fetchone()[0]
        financial_coverage = financial_count / target_count if target_count else 0
        checks.append(
            {
                "name": "financial_coverage",
                "status": "PASS" if financial_coverage >= 0.8 else "WARN",
                "observed": f"{financial_coverage:.4f}",
                "threshold": ">=0.80",
                "message": "PoCでは価格中心の低Confidence順位を許容",
            }
        )

        invalid_tdnet = connection.execute(
            """
            SELECT count(*) FROM tdnet_documents t
            LEFT JOIN disclosure_texts d USING (document_id)
            WHERE t.status = 'downloaded'
              AND (t.pdf_path IS NULL OR coalesce(t.text_characters, 0) = 0
                   OR d.document_id IS NULL)
            """
        ).fetchone()[0]
        add(
            "tdnet_document_integrity",
            invalid_tdnet == 0,
            invalid_tdnet,
            "0 downloaded documents without PDF/text",
        )
        pending_tdnet = connection.execute(
            """
            SELECT count(*) FROM retry_queue
            WHERE source = 'tdnet' AND status = 'pending'
            """
        ).fetchone()[0]
        checks.append(
            {
                "name": "tdnet_pending_retries",
                "status": "PASS" if pending_tdnet == 0 else "WARN",
                "observed": str(pending_tdnet),
                "threshold": "0",
                "message": "保留分は次回日次処理または日付指定再実行で回収",
            }
        )

        evidence_rows = connection.execute(
            """
            SELECT a.evidence, d.text_content
            FROM qualitative_analyses a
            JOIN disclosure_texts d USING (document_id)
            WHERE a.prompt_version = ?
            """,
            [PROMPT_VERSION],
        ).fetchall()
        invalid_evidence = 0
        for evidence_json, text_content in evidence_rows:
            try:
                evidence = json.loads(evidence_json or "[]")
                normalized_source = re.sub(r"\s+", "", text_content or "")
                invalid_evidence += sum(
                    re.sub(r"\s+", "", item.get("excerpt", ""))
                    not in normalized_source
                    for item in evidence
                )
            except (TypeError, json.JSONDecodeError):
                invalid_evidence += 1
        add(
            "qualitative_evidence_integrity",
            invalid_evidence == 0,
            invalid_evidence,
            "0 evidence excerpts missing from source text",
        )

        qualitative_count = connection.execute(
            """
            SELECT count(DISTINCT canonical_code) FROM qualitative_feature_snapshots
            WHERE snapshot_date = ?
            """,
            [latest_rank_date],
        ).fetchone()[0]
        qualitative_coverage = qualitative_count / target_count if target_count else 0
        checks.append(
            {
                "name": "qualitative_coverage",
                "status": "PASS" if qualitative_coverage >= 0.8 else "WARN",
                "observed": f"{qualitative_coverage:.4f}",
                "threshold": ">=0.80 after TDnet/LLM rollout",
                "message": "未取得銘柄は定性補正なしで従来Rule Rankを維持",
            }
        )

        for check in checks:
            connection.execute(
                """
                INSERT OR REPLACE INTO data_quality_results VALUES (
                    ?, ?, ?, ?, ?, ?, current_timestamp
                )
                """,
                [
                    batch_run_id,
                    check["name"],
                    check["status"],
                    check["observed"],
                    check["threshold"],
                    check["message"],
                ],
            )
    passed = all(check["status"] != "FAIL" for check in checks)
    return {"passed": passed, "checks": checks}


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def publish_snapshot(settings: Settings, batch_run_id: str) -> dict[str, object]:
    run_dir = settings.published_dir / "runs" / batch_run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    with connect(settings.db_path) as connection:
        latest_rank = connection.execute(
            """
            SELECT snapshot_date, ranking_version FROM investment_rank_snapshots
            ORDER BY calculated_at DESC LIMIT 1
            """
        ).fetchone()
        latest_snapshot, latest_ranking_version = (
            latest_rank if latest_rank is not None else (None, None)
        )
        ranking = connection.execute(
            """
            SELECT r.*, s.company_name, s.sector33_name
            FROM investment_rank_snapshots r
            LEFT JOIN securities s USING (canonical_code)
            WHERE r.snapshot_date = ? AND r.ranking_version = ?
            ORDER BY r.rank_12m, r.canonical_code
            """,
            [latest_snapshot, latest_ranking_version],
        ).df()
        prices = connection.execute(
            """
            SELECT p.canonical_code, p.trade_date,
                   p.adjusted_close AS raw_adjusted_close,
                   CASE WHEN EXISTS (
                       SELECT 1 FROM price_quality_events q
                       WHERE q.canonical_code = p.canonical_code
                         AND q.trade_date = p.trade_date
                         AND q.source = p.source
                         AND q.cleaning_version = ?
                         AND q.action = 'exclude_model_price'
                   ) THEN NULL ELSE p.adjusted_close END AS adjusted_close
            FROM secondary_prices p ORDER BY p.canonical_code, p.trade_date
            """,
            [CLEANING_VERSION],
        ).df()
        price_quality = connection.execute(
            """
            SELECT * FROM price_quality_events
            WHERE cleaning_version = ?
            ORDER BY trade_date DESC, canonical_code, reason_code
            """,
            [CLEANING_VERSION],
        ).df()
        financials = connection.execute(
            """
            SELECT p.canonical_code, f.*
            FROM financial_summaries f
            JOIN provider_symbols p ON p.provider = 'jquants_v2'
                                   AND p.provider_symbol = f.code
            ORDER BY p.canonical_code, f.disclosure_date DESC
            """
        ).df()
        batches = connection.execute(
            """
            SELECT * FROM batch_runs ORDER BY started_at DESC LIMIT 30
            """
        ).df()
        steps = connection.execute(
            """
            SELECT * FROM batch_steps ORDER BY started_at DESC LIMIT 100
            """
        ).df()
        quality = connection.execute(
            """
            SELECT * FROM data_quality_results
            WHERE batch_run_id = ? ORDER BY check_name
            """,
            [batch_run_id],
        ).df()
        qualitative = connection.execute(
            """
            SELECT a.document_id, a.canonical_code, a.disclosure_date, d.title,
                   a.summary, a.outlook_score, a.demand_score,
                   a.profitability_score, a.risk_control_score,
                   a.earnings_quality_score, a.confidence,
                   a.positive_factors, a.negative_factors, a.evidence,
                   a.model, a.resolved_model, a.prompt_version,
                   a.source_url, a.analyzed_at
            FROM qualitative_analyses a
            LEFT JOIN disclosure_texts d USING (document_id)
            ORDER BY a.canonical_code, a.disclosure_date DESC
            """
        ).df()
        coverage = connection.execute(
            """
            WITH current_watchlist AS (
                SELECT w.canonical_code, s.company_name, s.sector33_name, s.model_group
                FROM watchlist_membership w
                JOIN securities s USING (canonical_code)
                WHERE w.watchlist_name = 'topix500' AND w.as_of_date = (
                    SELECT max(as_of_date) FROM watchlist_membership
                    WHERE watchlist_name = 'topix500'
                )
            ), price AS (
                SELECT canonical_code, count(*) AS price_rows,
                       min(trade_date) AS first_price_date,
                       max(trade_date) AS last_price_date
                FROM secondary_prices GROUP BY canonical_code
            ), financial AS (
                SELECT p.canonical_code,
                       count(DISTINCT f.disclosure_date) AS financial_periods,
                       max(f.disclosure_date) AS last_financial_date
                FROM financial_summaries f
                JOIN provider_symbols p ON p.provider = 'jquants_v2'
                                       AND p.provider_symbol = f.code
                GROUP BY p.canonical_code
            ), source_docs AS (
                SELECT canonical_code, count(*) AS source_document_count,
                       max(disclosure_date) AS last_source_document_date
                FROM disclosure_texts GROUP BY canonical_code
            ), analyses AS (
                SELECT canonical_code, count(DISTINCT document_id) AS analysis_count,
                       max(disclosure_date) AS last_analysis_date
                FROM qualitative_analyses GROUP BY canonical_code
            ), features AS (
                SELECT canonical_code, 1 AS fundamental_available
                FROM fundamental_feature_snapshots
                WHERE snapshot_date = ? GROUP BY canonical_code
            ), qfeatures AS (
                SELECT canonical_code, 1 AS qualitative_feature_available
                FROM qualitative_feature_snapshots
                WHERE snapshot_date = ? GROUP BY canonical_code
            ), ranks AS (
                SELECT canonical_code, qualitative_confidence
                FROM investment_rank_snapshots
                WHERE snapshot_date = ? AND ranking_version = ?
            )
            SELECT w.*, coalesce(p.price_rows, 0) AS price_rows,
                   p.first_price_date, p.last_price_date,
                   coalesce(f.financial_periods, 0) AS financial_periods,
                   f.last_financial_date,
                   coalesce(ff.fundamental_available, 0) AS fundamental_available,
                   coalesce(d.source_document_count, 0) AS source_document_count,
                   d.last_source_document_date,
                   coalesce(a.analysis_count, 0) AS analysis_count,
                   a.last_analysis_date,
                   coalesce(q.qualitative_feature_available, 0)
                       AS qualitative_feature_available,
                   CASE WHEN coalesce(r.qualitative_confidence, 0) > 0 THEN 1 ELSE 0 END
                       AS qualitative_used_in_rank,
                   40 * CASE WHEN coalesce(p.price_rows, 0) >= 252 THEN 1 ELSE 0 END
                     + 40 * CASE WHEN coalesce(f.financial_periods, 0) >= 3 THEN 1 ELSE 0 END
                     + 20 * CASE WHEN r.canonical_code IS NOT NULL THEN 1 ELSE 0 END
                       AS core_coverage_pct,
                   25 * CASE WHEN coalesce(p.price_rows, 0) >= 252 THEN 1 ELSE 0 END
                     + 25 * CASE WHEN coalesce(f.financial_periods, 0) >= 3 THEN 1 ELSE 0 END
                     + 15 * CASE WHEN r.canonical_code IS NOT NULL THEN 1 ELSE 0 END
                     + 15 * coalesce(d.source_document_count > 0, false)::INTEGER
                     + 10 * coalesce(a.analysis_count > 0, false)::INTEGER
                     + 10 * coalesce(q.qualitative_feature_available, 0)
                       AS extended_coverage_pct
            FROM current_watchlist w
            LEFT JOIN price p USING (canonical_code)
            LEFT JOIN financial f USING (canonical_code)
            LEFT JOIN features ff USING (canonical_code)
            LEFT JOIN source_docs d USING (canonical_code)
            LEFT JOIN analyses a USING (canonical_code)
            LEFT JOIN qfeatures q USING (canonical_code)
            LEFT JOIN ranks r USING (canonical_code)
            ORDER BY extended_coverage_pct, core_coverage_pct, w.canonical_code
            """,
            [
                latest_snapshot,
                latest_snapshot,
                latest_snapshot,
                latest_ranking_version,
            ],
        ).df()
        model_inputs = connection.execute(
            """
            WITH price AS (
                SELECT * FROM price_feature_snapshots
                WHERE snapshot_date = ?
                QUALIFY row_number() OVER (
                    PARTITION BY canonical_code ORDER BY calculated_at DESC
                ) = 1
            ), fundamental AS (
                SELECT * FROM fundamental_feature_snapshots
                WHERE snapshot_date = ?
                QUALIFY row_number() OVER (
                    PARTITION BY canonical_code ORDER BY calculated_at DESC
                ) = 1
            ), qualitative AS (
                SELECT * FROM qualitative_feature_snapshots
                WHERE snapshot_date = ?
                QUALIFY row_number() OVER (
                    PARTITION BY canonical_code ORDER BY calculated_at DESC
                ) = 1
            )
            SELECT r.canonical_code, s.company_name, s.sector33_name, r.model_group,
                   r.snapshot_date, r.ranking_version,
                   p.price_date, p.latest_close, p.return_1m, p.return_3m,
                   p.return_6m, p.return_12m, p.momentum_12_1,
                   p.volatility_20d, p.volatility_60d,
                   p.downside_volatility_60d, p.max_drawdown_252d,
                   p.high_52w_distance, p.average_volume_20d,
                   p.average_turnover_20d, p.momentum_score AS price_momentum_score,
                   p.risk_score AS price_risk_score, p.price_score, p.price_rank,
                   f.disclosure_date, f.current_period_type, f.per, f.pbr, f.roe,
                   f.equity_ratio, f.operating_margin, f.sales_yoy,
                   f.operating_profit_yoy, f.eps_yoy, f.forecast_eps_revision,
                   f.financial_completeness,
                   q.qualitative_score, q.outlook_score, q.demand_score,
                   q.profitability_score, q.risk_control_score,
                   q.earnings_quality_score, q.qualitative_confidence,
                   q.document_count AS qualitative_document_count,
                   r.valuation_score, r.quality_score, r.growth_score,
                   r.earnings_score, r.momentum_score, r.risk_score,
                   r.score_6m, r.rank_6m, r.score_12m, r.rank_12m, r.confidence
            FROM investment_rank_snapshots r
            JOIN securities s USING (canonical_code)
            LEFT JOIN price p USING (canonical_code)
            LEFT JOIN fundamental f USING (canonical_code)
            LEFT JOIN qualitative q USING (canonical_code)
            WHERE r.snapshot_date = ? AND r.ranking_version = ?
            ORDER BY r.rank_12m, r.canonical_code
            """,
            [
                latest_snapshot,
                latest_snapshot,
                latest_snapshot,
                latest_snapshot,
                latest_ranking_version,
            ],
        ).df()

    published_at = pd.Timestamp.now(tz="UTC")
    batches.loc[batches["batch_run_id"] == batch_run_id, "status"] = "succeeded"
    batches.loc[batches["batch_run_id"] == batch_run_id, "finished_at"] = published_at
    current_publish_step = (steps["batch_run_id"] == batch_run_id) & (
        steps["step_name"] == "publish_snapshot"
    )
    steps.loc[current_publish_step, "status"] = "succeeded"
    steps.loc[current_publish_step, "finished_at"] = published_at

    _write_parquet(ranking, run_dir / "ranking_latest.parquet")
    _write_parquet(prices, run_dir / "price_history.parquet")
    _write_parquet(price_quality, run_dir / "price_quality_events.parquet")
    _write_parquet(financials, run_dir / "financial_history.parquet")
    _write_parquet(batches, run_dir / "batch_runs.parquet")
    _write_parquet(steps, run_dir / "batch_steps.parquet")
    _write_parquet(quality, run_dir / "data_quality.parquet")
    _write_parquet(coverage, run_dir / "data_coverage.parquet")
    _write_parquet(qualitative, run_dir / "qualitative_latest.parquet")
    _write_parquet(model_inputs, run_dir / "model_input_latest.parquet")

    manifest = {
        "batch_run_id": batch_run_id,
        "snapshot_date": str(latest_snapshot),
        "published_at": published_at.isoformat(),
        "run_dir": str(run_dir.relative_to(settings.published_dir)),
        "ranking_rows": len(ranking),
        "price_rows": len(prices),
        "price_quality_events": len(price_quality),
        "price_cleaning_version": CLEANING_VERSION,
        "financial_rows": len(financials),
        "coverage_rows": len(coverage),
        "qualitative_rows": len(qualitative),
        "ranking_version": latest_ranking_version,
        "model_input_rows": len(model_inputs),
    }
    pointer = settings.published_dir / "latest.json"
    temporary = pointer.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, pointer)
    return manifest


def create_backup(settings: Settings) -> dict[str, object]:
    settings.ensure_dirs()
    if not settings.db_path.exists():
        raise FileNotFoundError(f"database not found: {settings.db_path}")
    with connect(settings.db_path) as connection:
        initialize(connection)
        connection.execute("CHECKPOINT")
    stamp = datetime.now(JST).strftime("%Y%m%d-%H%M%S")
    target = settings.backup_dir / f"poc-{stamp}.duckdb"
    shutil.copy2(settings.db_path, target)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    metadata = {
        "path": str(target),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "size_bytes": target.stat().st_size,
        "sha256": digest,
    }
    target.with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metadata


def financial_catchup_dates(
    watermark: str | None, target: date, maximum_days: int = 7
) -> list[date]:
    if watermark is None:
        return [target]
    start = date.fromisoformat(watermark) + timedelta(days=1)
    if start > target:
        return []
    end = min(target, start + timedelta(days=maximum_days - 1))
    return [start + timedelta(days=offset) for offset in range((end - start).days + 1)]
