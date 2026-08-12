from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import duckdb
import pandas as pd


def connect(path: Path) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(path))


def initialize(connection: duckdb.DuckDBPyConnection) -> None:
    connection.execute("""
        CREATE TABLE IF NOT EXISTS raw_manifest (
            source VARCHAR, retrieved_at TIMESTAMPTZ, path VARCHAR,
            content_hash VARCHAR PRIMARY KEY, row_count BIGINT
        );
        CREATE TABLE IF NOT EXISTS universe (
            as_of_date VARCHAR, code VARCHAR, company_name VARCHAR,
            market_segment VARCHAR, sector33_code VARCHAR, sector33_name VARCHAR,
            sector17_code VARCHAR, sector17_name VARCHAR, size_code VARCHAR,
            size_name VARCHAR, source VARCHAR, retrieved_at TIMESTAMPTZ,
            PRIMARY KEY (as_of_date, code)
        );
        CREATE TABLE IF NOT EXISTS daily_prices (
            trade_date DATE, code VARCHAR, open DOUBLE, high DOUBLE, low DOUBLE,
            close DOUBLE, volume DOUBLE, turnover_value DOUBLE,
            adjustment_factor DOUBLE, adjusted_open DOUBLE, adjusted_high DOUBLE,
            adjusted_low DOUBLE, adjusted_close DOUBLE, adjusted_volume DOUBLE,
            source VARCHAR, retrieved_at TIMESTAMPTZ,
            PRIMARY KEY (trade_date, code, source)
        );
        CREATE TABLE IF NOT EXISTS financial_summaries (
            disclosure_date DATE, disclosure_time VARCHAR, code VARCHAR,
            disclosure_number VARCHAR PRIMARY KEY, document_type VARCHAR,
            current_period_type VARCHAR, current_period_start DATE,
            current_period_end DATE, current_fiscal_year_start DATE,
            current_fiscal_year_end DATE, sales DOUBLE,
            operating_profit DOUBLE, ordinary_profit DOUBLE, net_income DOUBLE,
            eps DOUBLE, total_assets DOUBLE, equity DOUBLE, equity_ratio DOUBLE,
            bps DOUBLE, cash_flow_operating DOUBLE, cash_flow_investing DOUBLE,
            cash_flow_financing DOUBLE, cash_and_equivalents DOUBLE,
            forecast_sales DOUBLE, forecast_operating_profit DOUBLE,
            forecast_ordinary_profit DOUBLE, forecast_net_income DOUBLE,
            forecast_eps DOUBLE, shares_outstanding DOUBLE, roe DOUBLE,
            source VARCHAR, retrieved_at TIMESTAMPTZ
        );
        CREATE TABLE IF NOT EXISTS edinet_documents (
            document_id VARCHAR PRIMARY KEY, edinet_code VARCHAR,
            security_code VARCHAR, filer_name VARCHAR, description VARCHAR,
            submitted_at TIMESTAMP, document_type_code VARCHAR,
            period_start DATE, period_end DATE, source VARCHAR,
            retrieved_at TIMESTAMPTZ
        );
        CREATE TABLE IF NOT EXISTS securities (
            canonical_code VARCHAR PRIMARY KEY, company_name VARCHAR,
            market_segment VARCHAR, sector33_name VARCHAR, size_name VARCHAR,
            model_group VARCHAR, updated_at TIMESTAMPTZ
        );
        CREATE TABLE IF NOT EXISTS provider_symbols (
            canonical_code VARCHAR, provider VARCHAR, provider_symbol VARCHAR,
            source_tier VARCHAR, updated_at TIMESTAMPTZ,
            PRIMARY KEY (canonical_code, provider)
        );
        CREATE TABLE IF NOT EXISTS watchlist_membership (
            watchlist_name VARCHAR, as_of_date VARCHAR, canonical_code VARCHAR,
            selection_rank INTEGER, selection_rule VARCHAR,
            created_at TIMESTAMPTZ,
            PRIMARY KEY (watchlist_name, as_of_date, canonical_code)
        );
        CREATE TABLE IF NOT EXISTS secondary_prices (
            trade_date DATE, canonical_code VARCHAR, provider_symbol VARCHAR,
            open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE,
            adjusted_close DOUBLE, volume DOUBLE, dividends DOUBLE,
            stock_splits DOUBLE, available_at TIMESTAMPTZ,
            retrieved_at TIMESTAMPTZ, source VARCHAR, source_tier VARCHAR,
            PRIMARY KEY (trade_date, canonical_code, source)
        );
        CREATE TABLE IF NOT EXISTS price_feature_snapshots (
            snapshot_date DATE, canonical_code VARCHAR, price_date DATE,
            latest_close DOUBLE, return_1m DOUBLE, return_3m DOUBLE,
            return_6m DOUBLE, return_12m DOUBLE, momentum_12_1 DOUBLE,
            volatility_20d DOUBLE, volatility_60d DOUBLE,
            downside_volatility_60d DOUBLE, max_drawdown_252d DOUBLE,
            high_52w_distance DOUBLE, average_volume_20d DOUBLE,
            average_turnover_20d DOUBLE, momentum_score DOUBLE,
            risk_score DOUBLE, price_score DOUBLE, price_rank INTEGER,
            source VARCHAR, source_tier VARCHAR, feature_version VARCHAR,
            calculated_at TIMESTAMPTZ,
            PRIMARY KEY (snapshot_date, canonical_code, feature_version)
        );
        CREATE TABLE IF NOT EXISTS fundamental_feature_snapshots (
            snapshot_date DATE, canonical_code VARCHAR, disclosure_date DATE,
            current_period_type VARCHAR, price_date DATE, latest_price DOUBLE,
            eps DOUBLE, bps DOUBLE, roe DOUBLE, equity_ratio DOUBLE,
            operating_margin DOUBLE, sales_yoy DOUBLE,
            operating_profit_yoy DOUBLE, eps_yoy DOUBLE,
            forecast_eps_revision DOUBLE, per DOUBLE, pbr DOUBLE,
            financial_completeness DOUBLE, source VARCHAR,
            feature_version VARCHAR, calculated_at TIMESTAMPTZ,
            PRIMARY KEY (snapshot_date, canonical_code, feature_version)
        );
        CREATE TABLE IF NOT EXISTS disclosure_texts (
            document_id VARCHAR PRIMARY KEY, canonical_code VARCHAR,
            disclosure_date DATE, disclosure_time VARCHAR, title VARCHAR,
            document_type VARCHAR, source VARCHAR, source_url VARCHAR,
            raw_path VARCHAR, content_hash VARCHAR, text_content VARCHAR,
            retrieved_at TIMESTAMPTZ
        );
        CREATE TABLE IF NOT EXISTS tdnet_documents (
            document_id VARCHAR PRIMARY KEY, canonical_code VARCHAR,
            tdnet_code VARCHAR, disclosure_date DATE, disclosure_time VARCHAR,
            available_at TIMESTAMPTZ, company_name VARCHAR, title VARCHAR,
            exchange VARCHAR, document_type VARCHAR, update_history VARCHAR,
            list_url VARCHAR, pdf_url VARCHAR, xbrl_url VARCHAR,
            pdf_path VARCHAR, xbrl_path VARCHAR, xbrl_extract_path VARCHAR,
            html_path VARCHAR, pdf_hash VARCHAR, xbrl_hash VARCHAR,
            text_characters BIGINT, status VARCHAR, retrieved_at TIMESTAMPTZ
        );
        CREATE TABLE IF NOT EXISTS qualitative_analyses (
            document_id VARCHAR, canonical_code VARCHAR, disclosure_date DATE,
            model VARCHAR, resolved_model VARCHAR, prompt_version VARCHAR,
            schema_version VARCHAR,
            summary VARCHAR, outlook_score DOUBLE, demand_score DOUBLE,
            profitability_score DOUBLE, risk_control_score DOUBLE,
            earnings_quality_score DOUBLE, confidence DOUBLE,
            positive_factors VARCHAR, negative_factors VARCHAR,
            evidence VARCHAR, response_id VARCHAR, input_tokens BIGINT,
            output_tokens BIGINT, source_url VARCHAR, analyzed_at TIMESTAMPTZ,
            PRIMARY KEY (document_id, model, prompt_version)
        );
        CREATE TABLE IF NOT EXISTS qualitative_feature_snapshots (
            snapshot_date DATE, canonical_code VARCHAR, disclosure_date DATE,
            qualitative_score DOUBLE, outlook_score DOUBLE,
            demand_score DOUBLE, profitability_score DOUBLE,
            risk_control_score DOUBLE, earnings_quality_score DOUBLE,
            qualitative_confidence DOUBLE, document_count BIGINT,
            source VARCHAR, feature_version VARCHAR, calculated_at TIMESTAMPTZ,
            PRIMARY KEY (snapshot_date, canonical_code, feature_version)
        );
        CREATE TABLE IF NOT EXISTS investment_rank_snapshots (
            snapshot_date DATE, canonical_code VARCHAR, model_group VARCHAR,
            price_date DATE, disclosure_date DATE, latest_price DOUBLE,
            per DOUBLE, pbr DOUBLE, roe DOUBLE, sales_yoy DOUBLE,
            operating_profit_yoy DOUBLE, valuation_score DOUBLE,
            quality_score DOUBLE, growth_score DOUBLE, earnings_score DOUBLE,
            momentum_score DOUBLE, risk_score DOUBLE, score_6m DOUBLE,
            rank_6m INTEGER, score_12m DOUBLE, rank_12m INTEGER,
            confidence DOUBLE, positive_reasons VARCHAR,
            negative_reasons VARCHAR, feature_version VARCHAR,
            ranking_version VARCHAR, calculated_at TIMESTAMPTZ,
            PRIMARY KEY (snapshot_date, canonical_code, ranking_version)
        );
        CREATE TABLE IF NOT EXISTS price_quality_events (
            trade_date DATE, canonical_code VARCHAR, source VARCHAR,
            cleaning_version VARCHAR, reason_code VARCHAR, severity VARCHAR,
            action VARCHAR, original_open DOUBLE, original_high DOUBLE,
            original_low DOUBLE, original_close DOUBLE,
            original_adjusted_close DOUBLE, original_volume DOUBLE,
            cleaned_high DOUBLE, cleaned_low DOUBLE, model_price DOUBLE,
            cleaned_volume DOUBLE, detected_at TIMESTAMPTZ,
            PRIMARY KEY (trade_date, canonical_code, source, cleaning_version, reason_code)
        );
        CREATE TABLE IF NOT EXISTS model_training_dataset (
            evaluation_date DATE, canonical_code VARCHAR, cutoff_at TIMESTAMPTZ,
            price_date DATE, disclosure_date DATE, sector33_name VARCHAR,
            model_group VARCHAR, return_1m DOUBLE, return_3m DOUBLE,
            return_6m DOUBLE, return_12m DOUBLE, momentum_12_1 DOUBLE,
            volatility_20d DOUBLE, volatility_60d DOUBLE,
            downside_volatility_60d DOUBLE, max_drawdown_252d DOUBLE,
            high_52w_distance DOUBLE, log_average_turnover_20d DOUBLE,
            per DOUBLE, pbr DOUBLE, roe DOUBLE, equity_ratio DOUBLE,
            operating_margin DOUBLE, sales_yoy DOUBLE,
            operating_profit_yoy DOUBLE, eps_yoy DOUBLE,
            forecast_eps_revision DOUBLE, financial_completeness DOUBLE,
            rule_score_6m DOUBLE, rule_score_12m DOUBLE,
            rule_confidence DOUBLE, label_start_date DATE,
            label_end_date_6m DATE, forward_return_6m DOUBLE,
            label_end_date_12m DATE, forward_return_12m DOUBLE,
            universe_as_of VARCHAR, survivor_bias_flag BOOLEAN,
            price_cleaning_version VARCHAR,
            fundamental_feature_version VARCHAR, rule_ranking_version VARCHAR,
            dataset_version VARCHAR, created_at TIMESTAMPTZ,
            PRIMARY KEY (evaluation_date, canonical_code, dataset_version)
        );
        CREATE TABLE IF NOT EXISTS trained_models (
            model_id VARCHAR PRIMARY KEY, horizon VARCHAR, algorithm VARCHAR,
            model_version VARCHAR, dataset_version VARCHAR,
            selected_alpha DOUBLE, validation_start DATE, test_start DATE,
            feature_names VARCHAR, coefficients VARCHAR, intercept DOUBLE,
            preprocessing VARCHAR, metrics VARCHAR, baseline_metrics VARCHAR,
            artifact_path VARCHAR, trained_at TIMESTAMPTZ
        );
        CREATE TABLE IF NOT EXISTS model_predictions (
            model_id VARCHAR, horizon VARCHAR, evaluation_date DATE,
            canonical_code VARCHAR, prediction DOUBLE, predicted_rank DOUBLE,
            actual_return DOUBLE, actual_rank DOUBLE, baseline_score DOUBLE,
            split VARCHAR, created_at TIMESTAMPTZ,
            PRIMARY KEY (model_id, evaluation_date, canonical_code, split)
        );
        CREATE TABLE IF NOT EXISTS acquisition_runs (
            run_id VARCHAR PRIMARY KEY, source VARCHAR, action VARCHAR,
            started_at TIMESTAMPTZ, finished_at TIMESTAMPTZ,
            status VARCHAR, requested_count BIGINT, received_count BIGINT,
            error_count BIGINT, message VARCHAR
        );
        CREATE TABLE IF NOT EXISTS batch_runs (
            batch_run_id VARCHAR PRIMARY KEY, scheduled_date DATE,
            started_at TIMESTAMPTZ, finished_at TIMESTAMPTZ,
            status VARCHAR, pipeline_version VARCHAR, message VARCHAR
        );
        CREATE TABLE IF NOT EXISTS batch_steps (
            batch_run_id VARCHAR, step_name VARCHAR, attempt INTEGER,
            started_at TIMESTAMPTZ, finished_at TIMESTAMPTZ,
            status VARCHAR, processed_count BIGINT, success_count BIGINT,
            error_count BIGINT, message VARCHAR,
            PRIMARY KEY (batch_run_id, step_name, attempt)
        );
        CREATE TABLE IF NOT EXISTS source_watermarks (
            source VARCHAR, stream VARCHAR, watermark_value VARCHAR,
            updated_at TIMESTAMPTZ, batch_run_id VARCHAR,
            PRIMARY KEY (source, stream)
        );
        CREATE TABLE IF NOT EXISTS retry_queue (
            retry_id VARCHAR PRIMARY KEY, source VARCHAR, item_key VARCHAR,
            action VARCHAR, attempt_count INTEGER, next_attempt_at TIMESTAMPTZ,
            status VARCHAR, last_error VARCHAR, created_at TIMESTAMPTZ,
            updated_at TIMESTAMPTZ
        );
        CREATE TABLE IF NOT EXISTS data_quality_results (
            batch_run_id VARCHAR, check_name VARCHAR, status VARCHAR,
            observed_value VARCHAR, threshold VARCHAR, message VARCHAR,
            checked_at TIMESTAMPTZ,
            PRIMARY KEY (batch_run_id, check_name)
        );
    """)
    connection.execute("ALTER TABLE raw_manifest ADD COLUMN IF NOT EXISTS source_url VARCHAR")
    connection.execute("ALTER TABLE raw_manifest ADD COLUMN IF NOT EXISTS source_tier VARCHAR")
    connection.execute("ALTER TABLE raw_manifest ADD COLUMN IF NOT EXISTS content_type VARCHAR")
    connection.execute("ALTER TABLE raw_manifest ADD COLUMN IF NOT EXISTS available_at TIMESTAMPTZ")
    connection.execute(
        "ALTER TABLE raw_manifest ADD COLUMN IF NOT EXISTS collector_version VARCHAR"
    )
    connection.execute(
        "ALTER TABLE investment_rank_snapshots ADD COLUMN IF NOT EXISTS qualitative_score DOUBLE"
    )
    connection.execute(
        "ALTER TABLE investment_rank_snapshots "
        "ADD COLUMN IF NOT EXISTS qualitative_confidence DOUBLE"
    )
    connection.execute(
        "ALTER TABLE investment_rank_snapshots "
        "ADD COLUMN IF NOT EXISTS qualitative_disclosure_date DATE"
    )
    connection.execute(
        "ALTER TABLE qualitative_analyses ADD COLUMN IF NOT EXISTS resolved_model VARCHAR"
    )


def insert_frame(connection: duckdb.DuckDBPyConnection, table: str, frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    connection.register("incoming_frame", frame)
    connection.execute(f"INSERT OR REPLACE INTO {table} BY NAME SELECT * FROM incoming_frame")
    connection.unregister("incoming_frame")


def add_manifest(
    connection: duckdb.DuckDBPyConnection,
    source: str,
    path: Path,
    content_hash: str,
    row_count: int,
    source_url: str | None = None,
    source_tier: str | None = None,
    content_type: str | None = None,
    available_at: object | None = None,
    collector_version: str = "poc_v1",
) -> None:
    connection.execute(
        """
        INSERT OR IGNORE INTO raw_manifest (
            source, retrieved_at, path, content_hash, row_count, source_url,
            source_tier, content_type, available_at, collector_version
        ) VALUES (?, current_timestamp, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            source,
            str(path),
            content_hash,
            row_count,
            source_url,
            source_tier,
            content_type,
            available_at,
            collector_version,
        ],
    )


def start_acquisition_run(
    connection: duckdb.DuckDBPyConnection,
    source: str,
    action: str,
    requested_count: int,
) -> str:
    run_id = str(uuid4())
    connection.execute(
        """
        INSERT INTO acquisition_runs (
            run_id, source, action, started_at, status, requested_count,
            received_count, error_count
        ) VALUES (?, ?, ?, current_timestamp, 'running', ?, 0, 0)
        """,
        [run_id, source, action, requested_count],
    )
    return run_id


def finish_acquisition_run(
    connection: duckdb.DuckDBPyConnection,
    run_id: str,
    status: str,
    received_count: int,
    error_count: int,
    message: str = "",
) -> None:
    connection.execute(
        """
        UPDATE acquisition_runs
        SET finished_at = current_timestamp, status = ?, received_count = ?,
            error_count = ?, message = ?
        WHERE run_id = ?
        """,
        [status, received_count, error_count, message, run_id],
    )
