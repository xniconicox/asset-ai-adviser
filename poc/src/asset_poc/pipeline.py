from __future__ import annotations

import argparse
import json
import time
from contextlib import nullcontext
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from asset_poc.collectors import (
    fetch_edinet_documents,
    fetch_jpx_universe,
    fetch_jquants_daily,
    fetch_jquants_financial_summary,
)
from asset_poc.config import Settings
from asset_poc.database import (
    add_manifest,
    connect,
    finish_acquisition_run,
    initialize,
    insert_frame,
    start_acquisition_run,
)
from asset_poc.features import FEATURE_VERSION as PRICE_FEATURE_VERSION
from asset_poc.features import compute_and_store_price_features
from asset_poc.operations import (
    PipelineAlreadyRunning,
    create_backup,
    execute_step,
    financial_catchup_dates,
    finish_batch,
    get_watermark,
    pipeline_lock,
    publish_snapshot,
    run_data_quality,
    set_watermark,
    start_batch,
)
from asset_poc.price_quality import (
    clean_price_history,
    store_price_quality_events,
    summarize_price_quality,
)
from asset_poc.prices import fetch_yahoo_prices
from asset_poc.qualitative import (
    export_qualitative_evaluation,
    ingest_disclosure_text,
    structure_disclosures,
)
from asset_poc.ranking import RANKING_VERSION, compute_and_store_investment_ranks
from asset_poc.raw_store import save_raw
from asset_poc.tdnet import collect_tdnet_disclosures
from asset_poc.watchlist import (
    WATCHLIST_NAME,
    build_watchlist,
    get_watchlist_symbols,
    to_yahoo_symbol,
)


def _save(connection, settings, source, suffix, frame, raw):
    path, digest = save_raw(settings.raw_dir, source, suffix, raw)
    add_manifest(connection, source, path, digest, len(frame))
    return path


def create_watchlist(settings: Settings, limit: int | None = None) -> dict[str, object]:
    settings.ensure_dirs()
    with connect(settings.db_path) as connection:
        initialize(connection)
        frame = build_watchlist(connection, limit=limit)
    return {
        "watchlist": WATCHLIST_NAME,
        "rows": len(frame),
        "as_of_date": frame["as_of_date"].max(),
        "general": int((frame["model_group"] == "general").sum()),
        "financial": int((frame["model_group"] == "financial").sum()),
    }


def collect_yahoo_prices(
    settings: Settings,
    period: str = "2y",
    limit: int | None = None,
    chunk_size: int = 25,
) -> dict[str, object]:
    settings.ensure_dirs()
    with connect(settings.db_path) as connection:
        initialize(connection)
        symbols = get_watchlist_symbols(connection, "yahoo_finance", limit=limit)
        if symbols.empty:
            build_watchlist(connection, limit=limit)
            symbols = get_watchlist_symbols(connection, "yahoo_finance", limit=limit)
        run_id = start_acquisition_run(
            connection, "yahoo_finance", f"prices:{period}", len(symbols)
        )
        received_codes: set[str] = set()
        rows = 0
        try:
            for offset in range(0, len(symbols), chunk_size):
                chunk = symbols.iloc[offset : offset + chunk_size]
                mapping = dict(zip(chunk["provider_symbol"], chunk["canonical_code"]))
                frame, raw = fetch_yahoo_prices(mapping, period=period)
                if not frame.empty:
                    path, digest = save_raw(
                        settings.raw_dir,
                        f"yahoo_prices_{offset // chunk_size + 1}",
                        "csv",
                        raw,
                    )
                    add_manifest(
                        connection,
                        "yahoo_finance",
                        path,
                        digest,
                        len(frame),
                        source_url="https://finance.yahoo.com/",
                        source_tier="C",
                        content_type="text/csv; normalized=true",
                        available_at=frame["available_at"].max(),
                    )
                    insert_frame(connection, "secondary_prices", frame)
                    received_codes.update(frame["canonical_code"].unique())
                    rows += len(frame)
                if offset + chunk_size < len(symbols):
                    time.sleep(2)
            missing = sorted(set(symbols["canonical_code"]) - received_codes)
            status_value = "succeeded" if not missing else "partial"
            finish_acquisition_run(
                connection,
                run_id,
                status_value,
                len(received_codes),
                len(missing),
                f"missing={','.join(missing)}" if missing else "",
            )
            return {
                "requested_symbols": len(symbols),
                "received_symbols": len(received_codes),
                "price_rows": rows,
                "missing_symbols": missing,
                "period": period,
                "run_id": run_id,
            }
        except Exception as error:
            finish_acquisition_run(
                connection,
                run_id,
                "failed",
                len(received_codes),
                len(symbols) - len(received_codes),
                str(error),
            )
            raise


def compute_features(settings: Settings) -> dict[str, object]:
    settings.ensure_dirs()
    with connect(settings.db_path) as connection:
        initialize(connection)
        frame = compute_and_store_price_features(connection)
    return {
        "feature_rows": len(frame),
        "snapshot_date": None if frame.empty else str(frame["snapshot_date"].max()),
        "feature_version": PRICE_FEATURE_VERSION,
    }


def analyze_price_quality(settings: Settings) -> dict[str, object]:
    settings.ensure_dirs()
    with connect(settings.db_path) as connection:
        initialize(connection)
        prices = connection.execute(
            """
            WITH latest AS (
                SELECT max(as_of_date) AS as_of_date FROM watchlist_membership
                WHERE watchlist_name = ?
            )
            SELECT p.* FROM secondary_prices p
            JOIN watchlist_membership w USING (canonical_code)
            JOIN latest l ON w.as_of_date = l.as_of_date
            WHERE w.watchlist_name = ?
            ORDER BY p.canonical_code, p.trade_date
            """,
            [WATCHLIST_NAME, WATCHLIST_NAME],
        ).df()
        _, events = clean_price_history(prices)
        store_price_quality_events(connection, events)
    return summarize_price_quality(events)


def compute_ranks(settings: Settings) -> dict[str, object]:
    settings.ensure_dirs()
    with connect(settings.db_path) as connection:
        initialize(connection)
        fundamentals, ranks = compute_and_store_investment_ranks(connection)
    return {
        "fundamental_feature_rows": len(fundamentals),
        "ranking_rows": len(ranks),
        "snapshot_date": None if ranks.empty else str(ranks["snapshot_date"].max()),
        "ranking_version": RANKING_VERSION,
    }


def refresh_topix500(settings: Settings, period: str = "2y") -> dict[str, object]:
    return {
        "watchlist": create_watchlist(settings),
        "prices": collect_yahoo_prices(settings, period=period),
        "features": compute_features(settings),
        "ranks": compute_ranks(settings),
    }


def _latest_price_date(settings: Settings) -> date | None:
    with connect(settings.db_path) as connection:
        initialize(connection)
        return connection.execute("SELECT max(trade_date) FROM secondary_prices").fetchone()[0]


def _incremental_price_period(settings: Settings) -> str:
    watermark = get_watermark(settings, "yahoo_finance", "daily_prices")
    if watermark is None:
        latest = _latest_price_date(settings)
        watermark = None if latest is None else str(latest)
    if watermark is None:
        return "2y"
    gap = (datetime.now(ZoneInfo("Asia/Tokyo")).date() - date.fromisoformat(watermark)).days
    if gap <= 30:
        return "1mo"
    if gap <= 90:
        return "3mo"
    return "2y"


def _collect_financial_catchup(settings: Settings, batch_run_id: str) -> dict[str, object]:
    if not settings.jquants_api_key:
        return {"warnings": ["JQUANTS_API_KEY未設定: 遅延決算の差分取得をスキップ"]}
    target = datetime.now(ZoneInfo("Asia/Tokyo")).date() - timedelta(weeks=12)
    watermark = get_watermark(settings, "jquants_v2", "financial_summary_date")
    targets = financial_catchup_dates(watermark, target)
    results: list[dict[str, object]] = []
    for index, catchup_date in enumerate(targets):
        result = collect_financials_by_date(settings, catchup_date)
        results.append(result)
        set_watermark(
            settings,
            "jquants_v2",
            "financial_summary_date",
            catchup_date.isoformat(),
            batch_run_id,
        )
        if index < len(targets) - 1:
            time.sleep(12.5)
    return {
        "catchup_dates": [str(value) for value in targets],
        "requested_dates": len(targets),
        "financial_rows": sum(int(value.get("watchlist_financial_rows", 0)) for value in results),
    }


def _collect_tdnet_catchup(settings: Settings, batch_run_id: str) -> dict[str, object]:
    today = datetime.now(ZoneInfo("Asia/Tokyo")).date()
    target = today - timedelta(days=1)
    watermark = get_watermark(settings, "tdnet_free_web", "disclosure_date")
    earliest = today - timedelta(days=30)
    if watermark and date.fromisoformat(watermark) < earliest - timedelta(days=1):
        watermark = (earliest - timedelta(days=1)).isoformat()
    targets = financial_catchup_dates(watermark, target)
    results: list[dict[str, object]] = []
    for catchup_date in targets:
        result = collect_tdnet_disclosures(settings, catchup_date, scope="earnings")
        if result.get("errors"):
            raise RuntimeError(f"TDnet partial failure for {catchup_date}: {result['errors']}")
        results.append(result)
        set_watermark(
            settings,
            "tdnet_free_web",
            "disclosure_date",
            catchup_date.isoformat(),
            batch_run_id,
        )
    return {
        "catchup_dates": [str(value) for value in targets],
        "requested_dates": len(targets),
        "selected_documents": sum(int(value.get("selected_documents", 0)) for value in results),
        "downloaded_documents": sum(int(value.get("downloaded_documents", 0)) for value in results),
        "skipped_documents": sum(int(value.get("skipped_documents", 0)) for value in results),
        "errors": [],
    }


def daily_refresh(settings: Settings, skip_network: bool = False) -> dict[str, object]:
    """Run the idempotent daily pipeline with retry, DQ gate and atomic publication."""
    settings.ensure_dirs()
    try:
        with pipeline_lock(settings):
            batch_run_id = start_batch(settings)
            steps: dict[str, object] = {}
            try:
                steps["watchlist"] = execute_step(
                    settings, batch_run_id, "build_watchlist", lambda: create_watchlist(settings)
                )
                if not skip_network:
                    steps["tdnet"] = execute_step(
                        settings,
                        batch_run_id,
                        "collect_tdnet",
                        lambda: _collect_tdnet_catchup(settings, batch_run_id),
                        attempts=3,
                    )
                    period = _incremental_price_period(settings)
                    steps["prices"] = execute_step(
                        settings,
                        batch_run_id,
                        "collect_prices",
                        lambda: collect_yahoo_prices(settings, period=period),
                        attempts=3,
                    )
                    latest = _latest_price_date(settings)
                    if latest is not None:
                        set_watermark(
                            settings,
                            "yahoo_finance",
                            "daily_prices",
                            latest.isoformat(),
                            batch_run_id,
                        )
                    steps["financials"] = execute_step(
                        settings,
                        batch_run_id,
                        "collect_financials",
                        lambda: _collect_financial_catchup(settings, batch_run_id),
                        attempts=3,
                    )
                else:
                    steps["network"] = {"skipped": True}

                steps["price_features"] = execute_step(
                    settings,
                    batch_run_id,
                    "compute_price_features",
                    lambda: compute_features(settings),
                )
                steps["ranks"] = execute_step(
                    settings, batch_run_id, "compute_ranks", lambda: compute_ranks(settings)
                )
                quality = execute_step(
                    settings,
                    batch_run_id,
                    "data_quality",
                    lambda: run_data_quality(settings, batch_run_id),
                )
                steps["quality"] = quality
                if not quality["passed"]:
                    failed = [
                        item["name"] for item in quality["checks"] if item["status"] == "FAIL"
                    ]
                    raise RuntimeError(f"data quality gate failed: {', '.join(failed)}")
                steps["publication"] = execute_step(
                    settings,
                    batch_run_id,
                    "publish_snapshot",
                    lambda: publish_snapshot(settings, batch_run_id),
                )
                finish_batch(settings, batch_run_id, "succeeded")
                return {"batch_run_id": batch_run_id, "status": "succeeded", "steps": steps}
            except Exception as error:
                finish_batch(settings, batch_run_id, "failed", str(error))
                raise
    except PipelineAlreadyRunning as error:
        return {"status": "skipped", "reason": str(error)}


def publish_current(settings: Settings) -> dict[str, object]:
    """Validate and publish current DB state without contacting providers."""
    return daily_refresh(settings, skip_network=True)


def backup_database(settings: Settings) -> dict[str, object]:
    try:
        with pipeline_lock(settings):
            return {"status": "succeeded", "backup": create_backup(settings)}
    except PipelineAlreadyRunning as error:
        return {"status": "skipped", "reason": str(error)}


def latest_price(settings: Settings, code: str) -> dict[str, object]:
    settings.ensure_dirs()
    canonical_code = code.strip().removesuffix(".T")
    symbol = to_yahoo_symbol(canonical_code)
    with connect(settings.db_path) as connection:
        initialize(connection)
        frame, raw = fetch_yahoo_prices({symbol: canonical_code}, period="5d")
        if frame.empty:
            raise RuntimeError(f"No Yahoo Finance price returned for {canonical_code}")
        path, digest = save_raw(settings.raw_dir, f"yahoo_latest_{canonical_code}", "csv", raw)
        add_manifest(
            connection,
            "yahoo_finance",
            path,
            digest,
            len(frame),
            source_url=f"https://finance.yahoo.com/quote/{symbol}/",
            source_tier="C",
            content_type="text/csv; normalized=true",
            available_at=frame["available_at"].max(),
        )
        insert_frame(connection, "secondary_prices", frame)
        latest = frame.sort_values("trade_date").iloc[-1]
    return {
        "canonical_code": canonical_code,
        "provider_symbol": symbol,
        "price": float(latest["close"]),
        "adjusted_close": float(latest["adjusted_close"]),
        "price_date": str(latest["trade_date"]),
        "retrieved_at": str(latest["retrieved_at"]),
        "source": "yahoo_finance",
        "source_tier": "C",
    }


def _collect_financials(settings: Settings, connection, codes: list[str] | None = None) -> int:
    codes = codes or list(settings.tickers)
    financial_rows = 0
    for index, code in enumerate(codes):
        frame, raw = fetch_jquants_financial_summary(settings.jquants_api_key, code)
        _save(connection, settings, f"jquants_financial_{code}", "json", frame, raw)
        insert_frame(connection, "financial_summaries", frame)
        financial_rows += len(frame)
        if index < len(codes) - 1:
            time.sleep(12.5)  # Free plan: 5 calls/minute
    return financial_rows


def collect_financials(settings: Settings) -> dict[str, object]:
    settings.ensure_dirs()
    result: dict[str, object] = {"warnings": []}
    if not settings.jquants_api_key:
        result["warnings"].append("JQUANTS_API_KEY未設定: 財務取得をスキップ")
        return result

    with connect(settings.db_path) as connection:
        initialize(connection)
        financial_rows = _collect_financials(settings, connection)
    result["financial_rows"] = financial_rows
    return result


def collect_financials_for_watchlist(
    settings: Settings,
    limit: int | None = None,
    min_periods: int = 3,
    force: bool = False,
    requests_per_minute: int = 5,
) -> dict[str, object]:
    if not settings.jquants_api_key:
        return {"warnings": ["JQUANTS_API_KEY未設定: 財務取得をスキップ"]}
    if requests_per_minute < 1:
        raise ValueError("requests_per_minute must be at least 1")
    settings.ensure_dirs()
    with connect(settings.db_path) as connection:
        initialize(connection)
        symbols = get_watchlist_symbols(connection, "jquants_v2", limit=limit)
        all_codes = symbols["provider_symbol"].tolist()
        coverage = dict(
            connection.execute(
                """
                SELECT code, count(DISTINCT disclosure_date)
                FROM financial_summaries GROUP BY code
                """
            ).fetchall()
        )
        codes = all_codes if force else [code for code in all_codes if coverage.get(code, 0) < min_periods]
        run_id = start_acquisition_run(connection, "jquants_v2", "financials:backfill", len(codes))
        rows = 0
        received = 0
        errors: list[str] = []
        request_interval_seconds = 60 / requests_per_minute
        try:
            for index, code in enumerate(codes):
                try:
                    frame, raw = fetch_jquants_financial_summary(settings.jquants_api_key, code)
                    path, digest = save_raw(settings.raw_dir, f"jquants_financial_{code}", "json", raw)
                    add_manifest(
                        connection,
                        "jquants_v2",
                        path,
                        digest,
                        len(frame),
                        source_url="https://api.jquants.com/v2/fins/summary",
                        source_tier="A",
                        content_type="application/json",
                        available_at=pd.Timestamp.now(tz="UTC"),
                    )
                    insert_frame(connection, "financial_summaries", frame)
                    rows += len(frame)
                    received += 1
                except Exception as error:  # noqa: BLE001 - one symbol must not stop the batch
                    errors.append(f"{code}:{error}")
                if index < len(codes) - 1:
                    time.sleep(request_interval_seconds)
        except KeyboardInterrupt:
            finish_acquisition_run(
                connection,
                run_id,
                "aborted",
                received,
                len(codes) - received,
                "interrupted by user",
            )
            raise
        status_value = "succeeded" if not errors else "partial"
        finish_acquisition_run(
            connection,
            run_id,
            status_value,
            received,
            len(errors),
            "; ".join(errors)[:4000],
        )
    return {
        "watchlist_symbols": len(all_codes),
        "requested_symbols": len(codes),
        "skipped_symbols": len(all_codes) - len(codes),
        "received_symbols": received,
        "financial_rows": rows,
        "errors": errors,
        "minimum_periods": min_periods,
        "forced": force,
        "requests_per_minute": requests_per_minute,
        "run_id": run_id,
    }


def collect_financials_by_date(settings: Settings, target_date: date | str) -> dict[str, object]:
    if not settings.jquants_api_key:
        return {"warnings": ["JQUANTS_API_KEY未設定: 財務取得をスキップ"]}
    target = str(target_date)
    settings.ensure_dirs()
    with connect(settings.db_path) as connection:
        initialize(connection)
        watchlist = get_watchlist_symbols(connection, "jquants_v2")
        wanted = set(watchlist["provider_symbol"])
        run_id = start_acquisition_run(
            connection, "jquants_v2", f"financials:date:{target}", len(wanted)
        )
        try:
            frame, raw = fetch_jquants_financial_summary(
                settings.jquants_api_key, target_date=target
            )
            path, digest = save_raw(
                settings.raw_dir, f"jquants_financial_date_{target}", "json", raw
            )
            add_manifest(
                connection,
                "jquants_v2",
                path,
                digest,
                len(frame),
                source_url="https://api.jquants.com/v2/fins/summary",
                source_tier="B",
                content_type="application/json",
                available_at=pd.Timestamp.now(tz="UTC"),
            )
            selected = frame[frame["code"].isin(wanted)].copy()
            insert_frame(connection, "financial_summaries", selected)
            received = selected["code"].nunique()
            finish_acquisition_run(connection, run_id, "succeeded", received, 0)
        except Exception as error:
            finish_acquisition_run(connection, run_id, "failed", 0, len(wanted), str(error))
            raise
    return {
        "target_date": target,
        "market_financial_rows": len(frame),
        "watchlist_financial_rows": len(selected),
        "watchlist_symbols": received,
        "run_id": run_id,
    }


def collect(settings: Settings) -> dict[str, object]:
    settings.ensure_dirs()
    result: dict[str, object] = {"warnings": []}
    with connect(settings.db_path) as connection:
        initialize(connection)

        universe, raw = fetch_jpx_universe()
        _save(connection, settings, "jpx_universe", "xls", universe, raw)
        insert_frame(connection, "universe", universe)
        result["universe_rows"] = len(universe)

        if settings.jquants_api_key:
            price_rows = 0
            for index, code in enumerate(settings.tickers):
                frame, raw = fetch_jquants_daily(
                    settings.jquants_api_key, code, settings.price_from
                )
                _save(connection, settings, f"jquants_daily_{code}", "json", frame, raw)
                insert_frame(connection, "daily_prices", frame)
                price_rows += len(frame)
                if index < len(settings.tickers) - 1:
                    time.sleep(12.5)  # Free plan: 5 calls/minute
            result["price_rows"] = price_rows
            time.sleep(12.5)
            result["financial_rows"] = _collect_financials(settings, connection)
        else:
            result["warnings"].append("JQUANTS_API_KEY未設定: 株価取得をスキップ")

        if settings.edinet_api_key:
            documents, raw = fetch_edinet_documents(
                settings.edinet_api_key,
                datetime.now(timezone.utc).date().isoformat(),
            )
            _save(connection, settings, "edinet_documents", "json", documents, raw)
            insert_frame(connection, "edinet_documents", documents)
            result["edinet_rows"] = len(documents)
        else:
            result["warnings"].append("EDINET_API_KEY未設定: 開示取得をスキップ")
    return result


def status(settings: Settings) -> dict[str, int]:
    settings.ensure_dirs()
    with connect(settings.db_path) as connection:
        initialize(connection)
        tables = [
            "universe",
            "daily_prices",
            "financial_summaries",
            "edinet_documents",
            "raw_manifest",
            "securities",
            "watchlist_membership",
            "secondary_prices",
            "price_feature_snapshots",
            "fundamental_feature_snapshots",
            "investment_rank_snapshots",
            "acquisition_runs",
            "batch_runs",
            "batch_steps",
            "source_watermarks",
            "retry_queue",
            "data_quality_results",
            "disclosure_texts",
            "tdnet_documents",
            "qualitative_analyses",
            "qualitative_feature_snapshots",
            "price_quality_events",
            "model_training_dataset",
            "trained_models",
            "model_predictions",
        ]
        return {
            table: connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
            for table in tables
        }


def _dispatch_command(args: argparse.Namespace, settings: Settings) -> dict[str, object]:
    if args.command == "collect":
        return collect(settings)
    if args.command == "collect-financials":
        return collect_financials(settings)
    if args.command == "backfill-financials":
        return collect_financials_for_watchlist(
            settings,
            args.limit,
            args.min_periods,
            args.force,
            args.requests_per_minute,
        )
    if args.command == "collect-financials-date":
        return collect_financials_by_date(settings, args.date)
    if args.command == "collect-tdnet-date":
        return collect_tdnet_disclosures(
            settings,
            args.date,
            scope=args.scope,
            watchlist_only=not args.market_wide,
            canonical_code=args.code,
            limit=args.limit,
            metadata_only=args.metadata_only,
            force=args.force,
        )
    if args.command == "build-watchlist":
        return create_watchlist(settings, args.limit)
    if args.command == "collect-yahoo":
        return collect_yahoo_prices(settings, args.period, args.limit)
    if args.command == "compute-features":
        return compute_features(settings)
    if args.command == "analyze-price-quality":
        return analyze_price_quality(settings)
    if args.command == "compute-ranks":
        return compute_ranks(settings)
    if args.command == "refresh-topix500":
        return refresh_topix500(settings, args.period)
    if args.command == "latest-price":
        return latest_price(settings, args.code)
    if args.command == "ingest-disclosure-text":
        return ingest_disclosure_text(
            settings,
            Path(args.file),
            args.code,
            args.date,
            args.title,
            args.source_url,
            args.document_type,
            args.source,
        )
    if args.command == "structure-disclosures":
        return structure_disclosures(settings, args.limit or 10, args.code)
    if args.command == "evaluate-qualitative":
        output = None if not args.output else Path(args.output)
        return export_qualitative_evaluation(settings, output)
    if args.command == "generate-model-report":
        from asset_poc.model_report import generate_model_eda_report

        output = None if not args.output else Path(args.output)
        return generate_model_eda_report(settings, output)
    if args.command == "generate-system-summary":
        from asset_poc.system_summary import generate_system_summary

        output = None if not args.output else Path(args.output)
        return generate_system_summary(settings, output)
    if args.command == "generate-data-coverage":
        from asset_poc.data_coverage_chart import generate_data_coverage_chart

        output = None if not args.output else Path(args.output)
        return generate_data_coverage_chart(
            settings, output, args.limit or 492, args.start, args.end
        )
    if args.command == "daily-report":
        from asset_poc.report_delivery import publish_daily_report

        report_dir_value = args.report_dir or args.drive_dir
        report_dir = None if not report_dir_value else Path(report_dir_value)
        report_date = None if not args.date else date.fromisoformat(args.date)
        return publish_daily_report(settings, report_date, report_dir)
    if args.command == "build-training-dataset":
        from asset_poc.learning import build_monthly_training_dataset

        return build_monthly_training_dataset(
            settings, args.start, args.end, args.dataset_version
        )
    if args.command == "train-model":
        from asset_poc.learning import train_models

        return train_models(
            settings,
            horizon=args.horizon,
            alpha_grid=args.alpha_grid,
            dataset_version=args.dataset_version,
        )
    if args.command == "evaluate-model":
        from asset_poc.learning import evaluate_model

        return evaluate_model(
            settings,
            model_id=args.model_id,
            horizon=args.horizon if args.horizon != "all" else "6m",
        )
    if args.command == "daily":
        return daily_refresh(settings, skip_network=args.skip_network)
    if args.command == "publish":
        return publish_current(settings)
    if args.command == "backup":
        return backup_database(settings)
    return status(settings)


def main() -> None:
    parser = argparse.ArgumentParser(description="Asset AI Adviser data PoC")
    parser.add_argument(
        "command",
        choices=[
            "collect",
            "collect-financials",
            "backfill-financials",
            "collect-financials-date",
            "collect-tdnet-date",
            "build-watchlist",
            "collect-yahoo",
            "compute-features",
            "analyze-price-quality",
            "compute-ranks",
            "refresh-topix500",
            "latest-price",
            "ingest-disclosure-text",
            "structure-disclosures",
            "evaluate-qualitative",
            "generate-model-report",
            "generate-system-summary",
            "generate-data-coverage",
            "daily-report",
            "build-training-dataset",
            "train-model",
            "evaluate-model",
            "daily",
            "publish",
            "backup",
            "status",
        ],
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--period", default="2y")
    parser.add_argument("--code")
    parser.add_argument("--date")
    parser.add_argument("--min-periods", type=int, default=3)
    parser.add_argument("--requests-per-minute", type=int, default=5)
    parser.add_argument("--file")
    parser.add_argument("--title")
    parser.add_argument("--source-url")
    parser.add_argument("--document-type", default="earnings_release")
    parser.add_argument("--source", default="manual")
    parser.add_argument("--output")
    parser.add_argument("--drive-dir")
    parser.add_argument("--report-dir")
    parser.add_argument("--scope", choices=["earnings", "all"], default="earnings")
    parser.add_argument("--market-wide", action="store_true")
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-network", action="store_true", help="reuse local data (daily only)")
    parser.add_argument("--start", help="first monthly evaluation date (YYYY-MM-DD)")
    parser.add_argument("--end", help="last monthly evaluation date (YYYY-MM-DD)")
    parser.add_argument(
        "--dataset-version", default="monthly_pit_v2_unadjusted_valuation"
    )
    parser.add_argument("--horizon", choices=["6m", "12m", "all"], default="all")
    parser.add_argument("--alpha-grid", default="0.1,1,10,100,1000")
    parser.add_argument("--model-id", default="latest")
    args = parser.parse_args()
    settings = Settings()
    if args.command == "collect-financials-date" and not args.date:
        parser.error("collect-financials-date requires --date, e.g. --date 2026-05-15")
    if args.command == "collect-tdnet-date" and not args.date:
        parser.error("collect-tdnet-date requires --date, e.g. --date 2026-08-11")
    if args.command == "latest-price" and not args.code:
        parser.error("latest-price requires --code, e.g. --code 7203")
    if args.command == "ingest-disclosure-text":
        required = ["file", "code", "date", "title", "source_url"]
        missing = [name for name in required if not getattr(args, name)]
        if missing:
            parser.error(
                "ingest-disclosure-text requires "
                + ", ".join(f"--{name.replace('_', '-')}" for name in missing)
            )

    lock_managed_in_command = {
        "daily",
        "publish",
        "backup",
        "generate-model-report",
        "generate-system-summary",
        "generate-data-coverage",
        "daily-report",
    }
    context = nullcontext() if args.command in lock_managed_in_command else pipeline_lock(settings)
    try:
        with context:
            payload = _dispatch_command(args, settings)
    except PipelineAlreadyRunning as error:
        payload = {"status": "skipped", "reason": str(error)}
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
