from __future__ import annotations

import json
from collections.abc import Iterable

import duckdb
import pandas as pd

from asset_poc.database import insert_frame
from asset_poc.features import FEATURE_VERSION as PRICE_FEATURE_VERSION
from asset_poc.qualitative import (
    FEATURE_VERSION as QUALITATIVE_FEATURE_VERSION,
)
from asset_poc.qualitative import (
    compute_and_store_qualitative_features,
)
from asset_poc.watchlist import WATCHLIST_NAME

FUNDAMENTAL_FEATURE_VERSION = "fundamental_v3_unadjusted_valuation"
RANKING_VERSION = "rule_rank_v6_unadjusted_valuation"
QUALITATIVE_WEIGHT_6M = 0.10
QUALITATIVE_WEIGHT_12M = 0.08

WEIGHTS_6M = {
    "valuation_score": 0.10,
    "quality_score": 0.10,
    "growth_score": 0.15,
    "earnings_score": 0.25,
    "momentum_score": 0.30,
    "risk_score": 0.10,
}
WEIGHTS_12M = {
    "valuation_score": 0.20,
    "quality_score": 0.20,
    "growth_score": 0.20,
    "earnings_score": 0.20,
    "momentum_score": 0.10,
    "risk_score": 0.10,
}

FACTOR_LABELS = {
    "valuation_score": "Valuation",
    "quality_score": "Quality",
    "growth_score": "Growth",
    "earnings_score": "Earnings",
    "momentum_score": "Momentum",
    "risk_score": "Risk",
    "qualitative_score": "Qualitative",
}

FINANCIAL_VALUE_COLUMNS = (
    "sales",
    "operating_profit",
    "ordinary_profit",
    "net_income",
    "eps",
    "total_assets",
    "equity",
    "equity_ratio",
    "bps",
    "cash_flow_operating",
    "cash_flow_investing",
    "cash_flow_financing",
    "cash_and_equivalents",
    "forecast_sales",
    "forecast_operating_profit",
    "forecast_ordinary_profit",
    "forecast_net_income",
    "forecast_eps",
    "shares_outstanding",
    "roe",
)
ACTUAL_RESULT_COLUMNS = ("sales", "operating_profit", "ordinary_profit", "net_income", "eps")


def _canonical_code(value: object) -> str:
    code = str(value).strip()
    return code[:-1] if len(code) == 5 and code.endswith("0") else code


def _valid_number(value: object) -> float | None:
    number = pd.to_numeric(value, errors="coerce")
    return None if pd.isna(number) else float(number)


def _growth(current: object, previous: object) -> float | None:
    current_value = _valid_number(current)
    previous_value = _valid_number(previous)
    if current_value is None or previous_value is None or abs(previous_value) < 1e-12:
        return None
    return (current_value - previous_value) / abs(previous_value)


def _latest_non_null(group: pd.DataFrame, column: str) -> float | None:
    values = pd.to_numeric(group[column], errors="coerce").dropna()
    return None if values.empty else float(values.iloc[-1])


def select_preferred_financial_rows(financials: pd.DataFrame) -> pd.DataFrame:
    """Return one deterministic end-of-day record per accounting period.

    J-Quants can return a financial statement, an earnings forecast revision, and a
    dividend revision for the same company/date/period. The most complete row is the
    base, while later non-null values from that day are overlaid. Stable tie-breakers
    make Point-in-Time feature generation independent of API row order.
    """
    if financials.empty:
        return financials.copy()

    data = financials.copy()
    value_columns = [column for column in FINANCIAL_VALUE_COLUMNS if column in data]
    data["_value_count"] = data[value_columns].notna().sum(axis=1)
    if "disclosure_time" in data:
        disclosure_time = pd.to_timedelta(
            data["disclosure_time"].fillna("00:00:00").astype(str), errors="coerce"
        )
        data["_time_sort"] = disclosure_time.dt.total_seconds().fillna(-1)
    else:
        data["_time_sort"] = -1
    if "disclosure_number" not in data:
        data["disclosure_number"] = data.index.astype(str)
    data["_number_sort"] = data["disclosure_number"].fillna("").astype(str)
    if "retrieved_at" in data:
        data["_retrieved_sort"] = pd.to_datetime(
            data["retrieved_at"], errors="coerce", utc=True
        )
    else:
        data["_retrieved_sort"] = pd.NaT

    group_columns = [
        "code",
        "disclosure_date",
        "current_period_type",
        "current_period_end",
        "current_fiscal_year_end",
    ]
    duplicate_mask = data.duplicated(group_columns, keep=False)
    selected = [data.loc[~duplicate_mask]]
    duplicate_rows = data.loc[duplicate_mask]
    for _, group in duplicate_rows.groupby(group_columns, dropna=False, sort=False):
        preferred_order = group.sort_values(
            ["_value_count", "_time_sort", "_number_sort", "_retrieved_sort"],
            ascending=[False, False, False, False],
            kind="mergesort",
        )
        preferred = preferred_order.iloc[0].copy()
        latest_order = group.sort_values(
            ["_time_sort", "_number_sort", "_retrieved_sort"],
            ascending=[False, False, False],
            kind="mergesort",
        )
        for column in value_columns:
            values = latest_order[column].dropna()
            if not values.empty:
                preferred[column] = values.iloc[0]
        latest = latest_order.iloc[0]
        for column in ("disclosure_time", "disclosure_number", "retrieved_at"):
            if column in data:
                preferred[column] = latest[column]
        selected.append(preferred.to_frame().T)

    return (
        pd.concat(selected, ignore_index=True)
        .drop(
            columns=["_value_count", "_time_sort", "_number_sort", "_retrieved_sort"],
            errors="ignore",
        )
        .reset_index(drop=True)
    )


def calculate_fundamental_features(
    financials: pd.DataFrame,
    price_features: pd.DataFrame,
    snapshot_date: object | None = None,
) -> pd.DataFrame:
    if price_features.empty:
        return pd.DataFrame()

    prices = price_features.copy()
    prices["canonical_code"] = prices["canonical_code"].astype(str)
    effective_date = pd.Timestamp(
        snapshot_date if snapshot_date is not None else prices["price_date"].max()
    ).date()
    price_lookup = prices.set_index("canonical_code")

    if financials.empty:
        return pd.DataFrame()
    data = financials.copy()
    data["canonical_code"] = data["code"].map(_canonical_code)
    data["disclosure_date"] = pd.to_datetime(data["disclosure_date"], errors="coerce")
    data["current_period_end"] = pd.to_datetime(data["current_period_end"], errors="coerce")
    data["current_fiscal_year_end"] = pd.to_datetime(
        data["current_fiscal_year_end"], errors="coerce"
    )
    data = data[data["disclosure_date"].dt.date <= effective_date]
    data = select_preferred_financial_rows(data)
    data = data.sort_values(
        [
            "canonical_code",
            "disclosure_date",
            "disclosure_time",
            "current_period_end",
            "current_fiscal_year_end",
            "current_period_type",
            "disclosure_number",
        ],
        kind="mergesort",
    )

    records: list[dict[str, object]] = []
    for code, group in data.groupby("canonical_code"):
        if code not in price_lookup.index:
            continue
        actual_columns = [column for column in ACTUAL_RESULT_COLUMNS if column in group]
        actuals = group[group[actual_columns].notna().any(axis=1)]
        latest = group.iloc[-1] if actuals.empty else actuals.iloc[-1]
        same_period = group[
            (group["current_period_type"] == latest["current_period_type"])
            & (group["current_period_end"] < latest["current_period_end"])
        ]
        prior = None if same_period.empty else same_period.iloc[-1]

        forecast_rows = group[group["forecast_eps"].notna()]
        forecast_eps = None
        forecast_history = forecast_rows
        if not forecast_rows.empty:
            latest_forecast = forecast_rows.iloc[-1]
            forecast_eps = _valid_number(latest_forecast["forecast_eps"])
            forecast_history = forecast_rows[
                forecast_rows["current_fiscal_year_end"]
                == latest_forecast["current_fiscal_year_end"]
            ]
        prior_forecast_eps = (
            None if len(forecast_history) < 2 else forecast_history.iloc[-2]["forecast_eps"]
        )
        latest_disclosure_date = latest["disclosure_date"]
        if not forecast_rows.empty:
            latest_disclosure_date = max(
                latest_disclosure_date, forecast_rows.iloc[-1]["disclosure_date"]
            )

        latest_price = _valid_number(price_lookup.loc[code, "latest_close"])
        actual_eps = _valid_number(latest["eps"])
        full_year_eps_values = pd.to_numeric(
            group.loc[group["current_period_type"] == "FY", "eps"], errors="coerce"
        ).dropna()
        full_year_eps = None if full_year_eps_values.empty else float(full_year_eps_values.iloc[-1])
        valuation_eps = forecast_eps if forecast_eps and forecast_eps > 0 else full_year_eps
        bps = _latest_non_null(group, "bps")
        roe = _latest_non_null(group, "roe")
        equity_ratio = _latest_non_null(group, "equity_ratio")
        sales = _valid_number(latest["sales"])
        operating_profit = _valid_number(latest["operating_profit"])
        operating_margin = (
            operating_profit / sales
            if sales not in (None, 0) and operating_profit is not None
            else None
        )
        sales_yoy = _growth(latest["sales"], None if prior is None else prior["sales"])
        operating_profit_yoy = _growth(
            latest["operating_profit"], None if prior is None else prior["operating_profit"]
        )
        eps_yoy = _growth(latest["eps"], None if prior is None else prior["eps"])
        forecast_eps_revision = _growth(forecast_eps, prior_forecast_eps)
        per = (
            latest_price / valuation_eps
            if latest_price is not None and valuation_eps is not None and valuation_eps > 0
            else None
        )
        pbr = (
            latest_price / bps
            if latest_price is not None and bps is not None and bps > 0
            else None
        )
        completeness_values = [
            per,
            pbr,
            roe,
            equity_ratio,
            operating_margin,
            sales_yoy,
            operating_profit_yoy,
            eps_yoy,
            forecast_eps_revision,
        ]
        records.append(
            {
                "snapshot_date": effective_date,
                "canonical_code": code,
                "disclosure_date": latest_disclosure_date.date(),
                "current_period_type": latest["current_period_type"],
                "price_date": price_lookup.loc[code, "price_date"],
                "latest_price": latest_price,
                "eps": actual_eps,
                "bps": bps,
                "roe": roe,
                "equity_ratio": equity_ratio,
                "operating_margin": operating_margin,
                "sales_yoy": sales_yoy,
                "operating_profit_yoy": operating_profit_yoy,
                "eps_yoy": eps_yoy,
                "forecast_eps_revision": forecast_eps_revision,
                "per": per,
                "pbr": pbr,
                "financial_completeness": sum(value is not None for value in completeness_values)
                / len(completeness_values),
                "source": "jquants_v2+yahoo_finance",
                "feature_version": FUNDAMENTAL_FEATURE_VERSION,
                "calculated_at": pd.Timestamp.now(tz="UTC"),
            }
        )
    return pd.DataFrame(records)


def _metric_score(frame: pd.DataFrame, column: str, *, higher_is_better: bool) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    clipped = values.copy()
    for indexes in frame.groupby("model_group").groups.values():
        valid = values.loc[indexes].dropna()
        if len(valid) >= 10:
            lower, upper = valid.quantile([0.025, 0.975])
            clipped.loc[indexes] = values.loc[indexes].clip(lower, upper)

    ascending = higher_is_better
    group_score = clipped.groupby(frame["model_group"]).rank(pct=True, ascending=ascending)
    sector_score = clipped.groupby([frame["model_group"], frame["sector33_name"]]).rank(
        pct=True, ascending=ascending
    )
    sector_count = clipped.groupby([frame["model_group"], frame["sector33_name"]]).transform(
        "count"
    )
    sector_score = sector_score.where(sector_count >= 5, group_score)
    return (sector_score * 0.7 + group_score * 0.3) * 100


def _mean_score(frame: pd.DataFrame, columns: Iterable[str]) -> pd.Series:
    return frame[list(columns)].mean(axis=1, skipna=True).fillna(50.0)


def _reason_payload(row: pd.Series, weights: dict[str, float], positive: bool) -> str:
    reasons = []
    for factor, weight in weights.items():
        if factor == "qualitative_score":
            weight *= float(row.get("qualitative_confidence", 0.0) or 0.0)
        contribution = weight * (float(row[factor]) - 50.0)
        if (positive and contribution > 0) or (not positive and contribution < 0):
            reasons.append(
                {
                    "factor": FACTOR_LABELS[factor],
                    "score": round(float(row[factor]), 1),
                    "contribution": round(contribution, 1),
                }
            )
    reasons.sort(key=lambda item: abs(item["contribution"]), reverse=True)
    return json.dumps(reasons[:3], ensure_ascii=False)


def calculate_investment_ranks(
    securities: pd.DataFrame,
    price_features: pd.DataFrame,
    fundamental_features: pd.DataFrame,
    qualitative_features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if price_features.empty:
        return pd.DataFrame()
    frame = securities.merge(price_features, on="canonical_code", how="inner")
    if not fundamental_features.empty:
        fundamental_columns = [
            column
            for column in fundamental_features.columns
            if column
            not in {
                "snapshot_date",
                "price_date",
                "latest_price",
                "source",
                "feature_version",
                "calculated_at",
            }
        ]
        frame = frame.merge(
            fundamental_features[fundamental_columns], on="canonical_code", how="left"
        )
    if qualitative_features is not None and not qualitative_features.empty:
        qualitative_columns = [
            "canonical_code",
            "disclosure_date",
            "qualitative_score",
            "qualitative_confidence",
        ]
        qualitative = qualitative_features[qualitative_columns].rename(
            columns={"disclosure_date": "qualitative_disclosure_date"}
        )
        frame = frame.merge(qualitative, on="canonical_code", how="left")

    if "qualitative_score" not in frame:
        frame["qualitative_score"] = 50.0
    if "qualitative_confidence" not in frame:
        frame["qualitative_confidence"] = 0.0
    if "qualitative_disclosure_date" not in frame:
        frame["qualitative_disclosure_date"] = None
    frame["qualitative_score"] = pd.to_numeric(frame["qualitative_score"], errors="coerce").fillna(
        50.0
    )
    frame["qualitative_confidence"] = pd.to_numeric(
        frame["qualitative_confidence"], errors="coerce"
    ).fillna(0.0)

    required_fundamental = [
        "disclosure_date",
        "per",
        "pbr",
        "roe",
        "equity_ratio",
        "operating_margin",
        "sales_yoy",
        "operating_profit_yoy",
        "eps_yoy",
        "forecast_eps_revision",
        "financial_completeness",
    ]
    for column in required_fundamental:
        if column not in frame:
            frame[column] = None

    frame["per_score"] = _metric_score(frame, "per", higher_is_better=False)
    frame["pbr_score"] = _metric_score(frame, "pbr", higher_is_better=False)
    frame["roe_score"] = _metric_score(frame, "roe", higher_is_better=True)
    frame["equity_ratio_score"] = _metric_score(frame, "equity_ratio", higher_is_better=True)
    frame["operating_margin_score"] = _metric_score(
        frame, "operating_margin", higher_is_better=True
    )
    frame["sales_yoy_score"] = _metric_score(frame, "sales_yoy", higher_is_better=True)
    frame["operating_profit_yoy_score"] = _metric_score(
        frame, "operating_profit_yoy", higher_is_better=True
    )
    frame["eps_yoy_score"] = _metric_score(frame, "eps_yoy", higher_is_better=True)
    frame["forecast_revision_score"] = _metric_score(
        frame, "forecast_eps_revision", higher_is_better=True
    )

    frame["valuation_score"] = _mean_score(frame, ["per_score", "pbr_score"])
    frame["quality_score"] = _mean_score(
        frame, ["roe_score", "equity_ratio_score", "operating_margin_score"]
    )
    frame["growth_score"] = _mean_score(
        frame, ["sales_yoy_score", "operating_profit_yoy_score", "eps_yoy_score"]
    )
    frame["earnings_score"] = _mean_score(
        frame,
        ["forecast_revision_score", "operating_profit_yoy_score", "eps_yoy_score"],
    )
    frame["momentum_score"] = frame["momentum_score"].fillna(50.0)
    frame["risk_score"] = frame["risk_score"].fillna(50.0)
    frame["confidence"] = 0.4 + 0.6 * pd.to_numeric(
        frame["financial_completeness"], errors="coerce"
    ).fillna(0.0)

    raw_6m = sum(frame[factor] * weight for factor, weight in WEIGHTS_6M.items())
    raw_12m = sum(frame[factor] * weight for factor, weight in WEIGHTS_12M.items())
    base_6m = 50.0 + frame["confidence"] * (raw_6m - 50.0)
    base_12m = 50.0 + frame["confidence"] * (raw_12m - 50.0)
    qualitative_delta = frame["qualitative_confidence"] * (frame["qualitative_score"] - 50.0)
    frame["score_6m"] = (base_6m + QUALITATIVE_WEIGHT_6M * qualitative_delta).clip(0, 100)
    frame["score_12m"] = (base_12m + QUALITATIVE_WEIGHT_12M * qualitative_delta).clip(0, 100)
    frame["rank_6m"] = frame["score_6m"].rank(method="min", ascending=False).astype("Int64")
    frame["rank_12m"] = frame["score_12m"].rank(method="min", ascending=False).astype("Int64")
    explanation_weights = {**WEIGHTS_12M, "qualitative_score": QUALITATIVE_WEIGHT_12M}
    frame["positive_reasons"] = frame.apply(
        _reason_payload, axis=1, weights=explanation_weights, positive=True
    )
    frame["negative_reasons"] = frame.apply(
        _reason_payload, axis=1, weights=explanation_weights, positive=False
    )
    frame["ranking_version"] = RANKING_VERSION
    frame["feature_version"] = (
        f"{PRICE_FEATURE_VERSION}+{FUNDAMENTAL_FEATURE_VERSION}+{QUALITATIVE_FEATURE_VERSION}"
    )
    frame["calculated_at"] = pd.Timestamp.now(tz="UTC")

    output_columns = [
        "snapshot_date",
        "canonical_code",
        "model_group",
        "price_date",
        "disclosure_date",
        "latest_close",
        "per",
        "pbr",
        "roe",
        "sales_yoy",
        "operating_profit_yoy",
        "valuation_score",
        "quality_score",
        "growth_score",
        "earnings_score",
        "momentum_score",
        "risk_score",
        "qualitative_score",
        "qualitative_confidence",
        "qualitative_disclosure_date",
        "score_6m",
        "rank_6m",
        "score_12m",
        "rank_12m",
        "confidence",
        "positive_reasons",
        "negative_reasons",
        "feature_version",
        "ranking_version",
        "calculated_at",
    ]
    output = frame[output_columns].rename(columns={"latest_close": "latest_price"})
    return output.sort_values(["rank_12m", "canonical_code"])


def compute_and_store_investment_ranks(
    connection: duckdb.DuckDBPyConnection,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    price_features = connection.execute(
        """
        SELECT * FROM price_feature_snapshots
        WHERE feature_version = ?
          AND snapshot_date = (
            SELECT max(snapshot_date) FROM price_feature_snapshots
            WHERE feature_version = ?
          )
        """,
        [PRICE_FEATURE_VERSION, PRICE_FEATURE_VERSION],
    ).df()
    securities = connection.execute(
        """
        WITH latest AS (
            SELECT max(as_of_date) AS as_of_date FROM watchlist_membership
            WHERE watchlist_name = ?
        )
        SELECT s.canonical_code, s.company_name, s.sector33_name, s.model_group
        FROM watchlist_membership w
        JOIN latest l ON w.as_of_date = l.as_of_date
        JOIN securities s ON s.canonical_code = w.canonical_code
        WHERE w.watchlist_name = ?
        """,
        [WATCHLIST_NAME, WATCHLIST_NAME],
    ).df()
    financials = connection.execute("SELECT * FROM financial_summaries").df()
    fundamentals = calculate_fundamental_features(financials, price_features)
    snapshot_date = price_features["snapshot_date"].max()
    qualitative = compute_and_store_qualitative_features(connection, snapshot_date)
    ranks = calculate_investment_ranks(securities, price_features, fundamentals, qualitative)
    insert_frame(connection, "fundamental_feature_snapshots", fundamentals)
    insert_frame(connection, "investment_rank_snapshots", ranks)
    return fundamentals, ranks
