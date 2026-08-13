from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import uuid4

import duckdb
import numpy as np
import pandas as pd

from asset_poc.config import Settings
from asset_poc.database import connect, initialize, insert_frame
from asset_poc.price_quality import CLEANING_VERSION, clean_price_history
from asset_poc.ranking import (
    FUNDAMENTAL_FEATURE_VERSION,
    RANKING_VERSION,
    calculate_fundamental_features,
    calculate_investment_ranks,
)
from asset_poc.watchlist import WATCHLIST_NAME

DATASET_VERSION = "monthly_pit_v2_unadjusted_valuation"
MODEL_VERSION = "ridge_rank_v2_unadjusted_valuation"

PRICE_FEATURE_COLUMNS = [
    "return_1m",
    "return_3m",
    "return_6m",
    "return_12m",
    "momentum_12_1",
    "volatility_20d",
    "volatility_60d",
    "downside_volatility_60d",
    "max_drawdown_252d",
    "high_52w_distance",
    "log_average_turnover_20d",
]
FUNDAMENTAL_FEATURE_COLUMNS = [
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
MODEL_FEATURE_COLUMNS = PRICE_FEATURE_COLUMNS + FUNDAMENTAL_FEATURE_COLUMNS

HORIZONS = {
    "6m": {
        "months": 6,
        "label": "forward_return_6m",
        "label_end": "label_end_date_6m",
        "baseline": "rule_score_6m",
    },
    "12m": {
        "months": 12,
        "label": "forward_return_12m",
        "label_end": "label_end_date_12m",
        "baseline": "rule_score_12m",
    },
}


def _period_return(values: np.ndarray, days: int) -> float | None:
    if len(values) <= days:
        return None
    return float(values[-1] / values[-days - 1] - 1.0)


def _annualized_volatility(values: np.ndarray, days: int, downside: bool = False) -> float | None:
    if len(values) < 3:
        return None
    returns = np.diff(values) / values[:-1]
    recent = returns[-days:]
    if downside:
        recent = recent[recent < 0]
    if len(recent) < 2:
        return None
    return float(np.std(recent, ddof=1) * math.sqrt(252))


def _max_drawdown(values: np.ndarray) -> float | None:
    recent = values[-252:]
    if len(recent) < 2:
        return None
    running_max = np.maximum.accumulate(recent)
    return float(np.min(recent / running_max - 1.0))


def _forward_label(
    dates: np.ndarray,
    prices: np.ndarray,
    segments: np.ndarray,
    evaluation_date: pd.Timestamp,
    months: int,
) -> tuple[object | None, object | None, float | None]:
    """Return next-trading-day entry and first trading day on/after target month."""
    evaluation = np.datetime64(evaluation_date.normalize(), "ns")
    start_position = int(np.searchsorted(dates, evaluation, side="right"))
    if start_position >= len(dates):
        return None, None, None
    start_date = pd.Timestamp(dates[start_position])
    if (start_date - evaluation_date.normalize()).days > 10:
        return None, None, None

    target = evaluation_date.normalize() + pd.DateOffset(months=months)
    end_position = int(np.searchsorted(dates, np.datetime64(target, "ns"), side="left"))
    if end_position >= len(dates):
        return start_date.date(), None, None
    end_date = pd.Timestamp(dates[end_position])
    if (end_date - target).days > 10 or segments[start_position] != segments[end_position]:
        return start_date.date(), None, None
    return (
        start_date.date(),
        end_date.date(),
        float(prices[end_position] / prices[start_position] - 1.0),
    )


def _monthly_evaluation_dates(
    prices: pd.DataFrame,
    financials: pd.DataFrame,
    start: str | None,
    end: str | None,
) -> list[pd.Timestamp]:
    return_column = "return_price" if "return_price" in prices else "adjusted_close"
    valid_dates = pd.to_datetime(
        prices.loc[
            pd.to_numeric(prices[return_column], errors="coerce") > 0,
            "trade_date",
        ]
    ).dropna()
    if valid_dates.empty:
        return []
    default_start = valid_dates.min() + pd.DateOffset(years=1)
    if not financials.empty:
        first_financial = pd.to_datetime(financials["disclosure_date"], errors="coerce").min()
        if pd.notna(first_financial):
            default_start = max(default_start, first_financial)
    start_date = pd.Timestamp(start) if start else default_start
    end_date = pd.Timestamp(end) if end else valid_dates.max()
    calendar = pd.DataFrame({"trade_date": valid_dates.drop_duplicates().sort_values()})
    calendar = calendar[
        (calendar["trade_date"] >= start_date) & (calendar["trade_date"] <= end_date)
    ]
    if calendar.empty:
        return []
    return list(calendar.groupby(calendar["trade_date"].dt.to_period("M"))["trade_date"].max())


def _build_price_rows(
    prices: pd.DataFrame,
    evaluation_dates: list[pd.Timestamp],
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for code, raw_group in prices.groupby("canonical_code", sort=False):
        group = (
            raw_group.sort_values("trade_date")
            .drop_duplicates("trade_date", keep="last")
            .reset_index(drop=True)
        )
        all_dates = pd.to_datetime(group["trade_date"], errors="coerce")
        return_column = "return_price" if "return_price" in group else "adjusted_close"
        return_price = pd.to_numeric(group[return_column], errors="coerce")
        valid = return_price.notna() & np.isfinite(return_price) & (return_price > 0)
        date_gaps = all_dates.diff().dt.days.gt(10).fillna(False)
        segment_all = ((~valid) | date_gaps).cumsum().to_numpy(dtype=int)
        valid_positions = np.flatnonzero(valid.to_numpy())
        if not len(valid_positions):
            continue
        dates = all_dates.iloc[valid_positions].to_numpy(dtype="datetime64[ns]")
        values = return_price.iloc[valid_positions].to_numpy(dtype=float)
        segments = segment_all[valid_positions]
        valuation_column = (
            "valuation_price" if "valuation_price" in group else "clean_close"
        )
        valuation_price = pd.to_numeric(
            group[valuation_column], errors="coerce"
        ).to_numpy()[valid_positions]
        clean_volume = pd.to_numeric(group["clean_volume"], errors="coerce").to_numpy()[
            valid_positions
        ]

        for evaluation_date in evaluation_dates:
            position = int(
                np.searchsorted(
                    dates, np.datetime64(evaluation_date.normalize(), "ns"), side="right"
                )
                - 1
            )
            if position < 0:
                continue
            price_date = pd.Timestamp(dates[position])
            if (evaluation_date.normalize() - price_date).days > 10:
                continue
            segment = segments[position]
            segment_start = int(np.searchsorted(segments, segment, side="left"))
            history = values[segment_start : position + 1]
            history_valuation = valuation_price[segment_start : position + 1]
            history_volume = clean_volume[segment_start : position + 1]
            recent_252 = history[-252:]
            turnover = history_valuation[-20:] * history_volume[-20:]
            average_turnover = (
                float(np.nanmean(turnover)) if np.isfinite(turnover).any() else None
            )
            start_6m, end_6m, return_6m = _forward_label(
                dates, values, segments, evaluation_date, 6
            )
            start_12m, end_12m, return_12m = _forward_label(
                dates, values, segments, evaluation_date, 12
            )
            records.append(
                {
                    "evaluation_date": evaluation_date.date(),
                    "snapshot_date": evaluation_date.date(),
                    "canonical_code": str(code),
                    "price_date": price_date.date(),
                    "latest_close": (
                        float(history_valuation[-1])
                        if np.isfinite(history_valuation[-1])
                        and history_valuation[-1] > 0
                        else None
                    ),
                    "return_1m": _period_return(history, 21),
                    "return_3m": _period_return(history, 63),
                    "return_6m": _period_return(history, 126),
                    "return_12m": _period_return(history, 252),
                    "momentum_12_1": (
                        float(history[-22] / history[-253] - 1.0)
                        if len(history) > 252
                        else None
                    ),
                    "volatility_20d": _annualized_volatility(history, 20),
                    "volatility_60d": _annualized_volatility(history, 60),
                    "downside_volatility_60d": _annualized_volatility(
                        history, 60, downside=True
                    ),
                    "max_drawdown_252d": _max_drawdown(history),
                    "high_52w_distance": (
                        float(history[-1] / np.max(recent_252) - 1.0)
                        if len(recent_252)
                        else None
                    ),
                    "average_turnover_20d": average_turnover,
                    "log_average_turnover_20d": (
                        float(np.log1p(average_turnover))
                        if average_turnover is not None and average_turnover >= 0
                        else None
                    ),
                    "label_start_date": start_6m or start_12m,
                    "label_end_date_6m": end_6m,
                    "forward_return_6m": return_6m,
                    "label_end_date_12m": end_12m,
                    "forward_return_12m": return_12m,
                }
            )
    frame = pd.DataFrame(records)
    if frame.empty:
        return frame
    momentum_inputs = ["return_3m", "return_6m", "return_12m", "momentum_12_1"]
    frame["momentum_score"] = (
        frame.groupby("evaluation_date")[momentum_inputs].rank(pct=True).mean(axis=1) * 100
    )
    volatility_safety = 1 - frame.groupby("evaluation_date")["volatility_60d"].rank(
        pct=True
    )
    drawdown_safety = frame.groupby("evaluation_date")["max_drawdown_252d"].rank(pct=True)
    frame["risk_score"] = pd.concat([volatility_safety, drawdown_safety], axis=1).mean(
        axis=1
    ) * 100
    return frame


def _load_training_sources(
    connection: duckdb.DuckDBPyConnection,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    universe_as_of = connection.execute(
        """
        SELECT max(as_of_date) FROM watchlist_membership
        WHERE watchlist_name = ?
        """,
        [WATCHLIST_NAME],
    ).fetchone()[0]
    if universe_as_of is None:
        raise RuntimeError("watchlist_membership is empty; run asset-poc build-watchlist first")
    prices = connection.execute(
        """
        SELECT p.* FROM secondary_prices p
        JOIN watchlist_membership w ON w.canonical_code = p.canonical_code
        WHERE w.watchlist_name = ? AND w.as_of_date = ?
        ORDER BY p.canonical_code, p.trade_date
        """,
        [WATCHLIST_NAME, universe_as_of],
    ).df()
    financials = connection.execute(
        """
        SELECT f.* FROM financial_summaries f
        JOIN watchlist_membership w
          ON (CASE WHEN length(f.code) = 5 AND right(f.code, 1) = '0'
                   THEN left(f.code, 4) ELSE f.code END) = w.canonical_code
        WHERE w.watchlist_name = ? AND w.as_of_date = ?
        """,
        [WATCHLIST_NAME, universe_as_of],
    ).df()
    securities = connection.execute(
        """
        SELECT s.* FROM securities s
        JOIN watchlist_membership w ON w.canonical_code = s.canonical_code
        WHERE w.watchlist_name = ? AND w.as_of_date = ?
        """,
        [WATCHLIST_NAME, universe_as_of],
    ).df()
    return prices, financials, securities, str(universe_as_of)


def build_monthly_training_dataset(
    settings: Settings,
    start: str | None = None,
    end: str | None = None,
    dataset_version: str = DATASET_VERSION,
) -> dict[str, object]:
    """Build derived Point-in-Time features and forward labels without mutating raw tables."""
    settings.ensure_dirs()
    with connect(settings.db_path) as connection:
        initialize(connection)
        raw_prices, financials, securities, universe_as_of = _load_training_sources(connection)
        cleaned_prices, _ = clean_price_history(raw_prices)
        evaluation_dates = _monthly_evaluation_dates(cleaned_prices, financials, start, end)
        price_rows = _build_price_rows(cleaned_prices, evaluation_dates)
        if price_rows.empty:
            raise RuntimeError("No monthly price rows were generated for the requested period")

        monthly_frames: list[pd.DataFrame] = []
        for evaluation_date, month_prices in price_rows.groupby("evaluation_date", sort=True):
            fundamentals = calculate_fundamental_features(
                financials, month_prices, snapshot_date=evaluation_date
            )
            ranks = calculate_investment_ranks(securities, month_prices, fundamentals)
            fundamental_columns = ["canonical_code", "disclosure_date"] + FUNDAMENTAL_FEATURE_COLUMNS
            available_fundamental_columns = [
                column for column in fundamental_columns if column in fundamentals
            ]
            month = month_prices.merge(
                fundamentals[available_fundamental_columns],
                on="canonical_code",
                how="left",
            )
            month = month.merge(
                ranks[["canonical_code", "score_6m", "score_12m", "confidence"]].rename(
                    columns={
                        "score_6m": "rule_score_6m",
                        "score_12m": "rule_score_12m",
                        "confidence": "rule_confidence",
                    }
                ),
                on="canonical_code",
                how="left",
            )
            monthly_frames.append(month)

        normalized_frames = [frame.dropna(axis=1, how="all") for frame in monthly_frames]
        dataset = pd.concat(normalized_frames, ignore_index=True)
        dataset = dataset.merge(
            securities[["canonical_code", "sector33_name", "model_group"]],
            on="canonical_code",
            how="left",
        )
        cutoff = pd.to_datetime(dataset["evaluation_date"]).dt.tz_localize(
            "Asia/Tokyo"
        ) + pd.Timedelta(hours=23, minutes=59, seconds=59)
        dataset["cutoff_at"] = cutoff.dt.tz_convert("UTC")
        dataset["universe_as_of"] = universe_as_of
        dataset["survivor_bias_flag"] = True
        dataset["price_cleaning_version"] = CLEANING_VERSION
        dataset["fundamental_feature_version"] = FUNDAMENTAL_FEATURE_VERSION
        dataset["rule_ranking_version"] = RANKING_VERSION
        dataset["dataset_version"] = dataset_version
        dataset["created_at"] = pd.Timestamp.now(tz="UTC")

        keep_columns = [
            "evaluation_date",
            "canonical_code",
            "cutoff_at",
            "price_date",
            "disclosure_date",
            "sector33_name",
            "model_group",
            *MODEL_FEATURE_COLUMNS,
            "rule_score_6m",
            "rule_score_12m",
            "rule_confidence",
            "label_start_date",
            "label_end_date_6m",
            "forward_return_6m",
            "label_end_date_12m",
            "forward_return_12m",
            "universe_as_of",
            "survivor_bias_flag",
            "price_cleaning_version",
            "fundamental_feature_version",
            "rule_ranking_version",
            "dataset_version",
            "created_at",
        ]
        for column in keep_columns:
            if column not in dataset:
                dataset[column] = None
        dataset = dataset[keep_columns].sort_values(["evaluation_date", "canonical_code"])
        connection.execute(
            "DELETE FROM model_training_dataset WHERE dataset_version = ?",
            [dataset_version],
        )
        insert_frame(connection, "model_training_dataset", dataset)

    output_dir = settings.root / "output" / "training"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{dataset_version}.parquet"
    dataset.to_parquet(output_path, index=False)
    return {
        "dataset_version": dataset_version,
        "rows": len(dataset),
        "codes": int(dataset["canonical_code"].nunique()),
        "evaluation_months": int(dataset["evaluation_date"].nunique()),
        "evaluation_start": str(dataset["evaluation_date"].min()),
        "evaluation_end": str(dataset["evaluation_date"].max()),
        "labeled_6m_rows": int(dataset["forward_return_6m"].notna().sum()),
        "labeled_12m_rows": int(dataset["forward_return_12m"].notna().sum()),
        "financial_coverage": float(dataset["financial_completeness"].notna().mean()),
        "universe": f"current_fixed:{universe_as_of}",
        "survivor_bias": True,
        "output": str(output_path),
    }


@dataclass
class RidgeFit:
    feature_names: list[str]
    means: np.ndarray
    scales: np.ndarray
    coefficients: np.ndarray
    intercept: float
    alpha: float

    def predict(self, values: pd.DataFrame) -> np.ndarray:
        matrix = values[self.feature_names].to_numpy(dtype=float)
        standardized = (matrix - self.means) / self.scales
        return self.intercept + standardized @ self.coefficients

    def as_dict(self) -> dict[str, object]:
        return {
            "feature_names": self.feature_names,
            "means": self.means.tolist(),
            "scales": self.scales.tolist(),
            "coefficients": self.coefficients.tolist(),
            "intercept": self.intercept,
            "alpha": self.alpha,
        }


def _rank_features(frame: pd.DataFrame) -> pd.DataFrame:
    values = frame[MODEL_FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    ranked = values.groupby(frame["evaluation_date"]).rank(pct=True, method="average")
    ranked = ranked.fillna(0.5)
    ranked.columns = [f"{column}_pct" for column in ranked.columns]
    return ranked


def _fit_ridge(features: pd.DataFrame, target: pd.Series, alpha: float) -> RidgeFit:
    matrix = features.to_numpy(dtype=float)
    values = pd.to_numeric(target, errors="coerce").to_numpy(dtype=float)
    means = matrix.mean(axis=0)
    scales = matrix.std(axis=0)
    scales[scales < 1e-12] = 1.0
    standardized = (matrix - means) / scales
    intercept = float(values.mean())
    centered_target = values - intercept
    penalty = np.eye(standardized.shape[1]) * float(alpha)
    try:
        coefficients = np.linalg.solve(
            standardized.T @ standardized + penalty,
            standardized.T @ centered_target,
        )
    except np.linalg.LinAlgError:
        coefficients = np.linalg.pinv(standardized.T @ standardized + penalty) @ (
            standardized.T @ centered_target
        )
    return RidgeFit(
        list(features.columns), means, scales, coefficients, intercept, float(alpha)
    )


def evaluate_ranking(
    frame: pd.DataFrame,
    prediction_column: str,
    actual_column: str,
) -> dict[str, object]:
    data = frame[["evaluation_date", "canonical_code", prediction_column, actual_column]].dropna()
    if data.empty:
        return {"rows": 0, "months": 0}
    monthly: list[dict[str, float]] = []
    previous_top: set[str] | None = None
    turnovers: list[float] = []
    for _, group in data.groupby("evaluation_date", sort=True):
        if len(group) < 10:
            continue
        size = max(5, math.ceil(len(group) * 0.10))
        ordered = group.sort_values(prediction_column, ascending=False)
        top = ordered.head(size)
        bottom = ordered.tail(size)
        universe_return = float(group[actual_column].mean())
        spearman_ic = group[prediction_column].rank(method="average").corr(
            group[actual_column].rank(method="average")
        )
        top_codes = set(top["canonical_code"].astype(str))
        if previous_top is not None:
            turnovers.append(1.0 - len(previous_top & top_codes) / max(len(top_codes), 1))
        previous_top = top_codes
        monthly.append(
            {
                "ic": float(spearman_ic),
                "top_return": float(top[actual_column].mean()),
                "universe_return": universe_return,
                "top_excess": float(top[actual_column].mean() - universe_return),
                "bottom_return": float(bottom[actual_column].mean()),
                "long_short": float(top[actual_column].mean() - bottom[actual_column].mean()),
            }
        )
    monthly_frame = pd.DataFrame(monthly)
    return {
        "rows": len(data),
        "months": len(monthly_frame),
        "mean_spearman_ic": float(monthly_frame["ic"].mean()),
        "median_spearman_ic": float(monthly_frame["ic"].median()),
        "positive_ic_rate": float((monthly_frame["ic"] > 0).mean()),
        "top_decile_return": float(monthly_frame["top_return"].mean()),
        "universe_return": float(monthly_frame["universe_return"].mean()),
        "top_decile_excess": float(monthly_frame["top_excess"].mean()),
        "top_decile_excess_win_rate": float((monthly_frame["top_excess"] > 0).mean()),
        "long_short_spread": float(monthly_frame["long_short"].mean()),
        "mean_top_decile_turnover": float(np.mean(turnovers)) if turnovers else None,
    }


def _temporal_split_masks(
    frame: pd.DataFrame,
    label_end_column: str,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Timestamp, pd.Timestamp]:
    dates = sorted(pd.to_datetime(frame["evaluation_date"]).dropna().unique())
    if len(dates) < 24:
        raise RuntimeError("At least 24 labeled evaluation months are required")
    test_start = pd.Timestamp(dates[max(2, int(len(dates) * 0.85))])
    evaluation = pd.to_datetime(frame["evaluation_date"])
    label_end = pd.to_datetime(frame[label_end_column])
    eligible_validation_dates = sorted(
        evaluation[(evaluation < test_start) & (label_end < test_start)].dropna().unique()
    )
    validation_months = max(12, math.ceil(len(dates) * 0.15))
    if len(eligible_validation_dates) <= validation_months:
        raise RuntimeError("Not enough non-overlapping months for validation")
    validation_start = pd.Timestamp(eligible_validation_dates[-validation_months])
    train = (evaluation < validation_start) & (label_end < validation_start)
    validation = (
        (evaluation >= validation_start)
        & (evaluation < test_start)
        & (label_end < test_start)
    )
    final_train = (evaluation < test_start) & (label_end < test_start)
    return train, validation, final_train, validation_start, test_start


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value


def _train_one_horizon(
    dataset: pd.DataFrame,
    horizon: str,
    alpha_grid: list[float],
) -> tuple[dict[str, object], pd.DataFrame, RidgeFit]:
    config = HORIZONS[horizon]
    label = str(config["label"])
    label_end = str(config["label_end"])
    baseline = str(config["baseline"])
    frame = dataset[dataset[label].notna() & dataset[label_end].notna()].copy()
    frame["evaluation_date"] = pd.to_datetime(frame["evaluation_date"])
    frame[label_end] = pd.to_datetime(frame[label_end])
    ranked = _rank_features(frame)
    frame = pd.concat([frame.reset_index(drop=True), ranked.reset_index(drop=True)], axis=1)
    ranked_columns = list(ranked.columns)
    frame["target_excess"] = frame[label] - frame.groupby("evaluation_date")[label].transform(
        "median"
    )
    train_mask, validation_mask, final_train_mask, validation_start, test_start = (
        _temporal_split_masks(frame, label_end)
    )
    test_mask = frame["evaluation_date"] >= test_start

    candidates: list[dict[str, object]] = []
    for alpha in alpha_grid:
        fit = _fit_ridge(frame.loc[train_mask, ranked_columns], frame.loc[train_mask, "target_excess"], alpha)
        validation = frame.loc[validation_mask].copy()
        validation["prediction"] = fit.predict(validation[ranked_columns])
        metrics = evaluate_ranking(validation, "prediction", label)
        candidates.append({"alpha": alpha, "metrics": metrics})
    selected = max(
        candidates,
        key=lambda item: (
            float(item["metrics"].get("mean_spearman_ic") or -1e9),
            float(item["metrics"].get("top_decile_excess") or -1e9),
        ),
    )
    selected_alpha = float(selected["alpha"])
    evaluation_fit = _fit_ridge(
        frame.loc[final_train_mask, ranked_columns],
        frame.loc[final_train_mask, "target_excess"],
        selected_alpha,
    )
    test = frame.loc[test_mask].copy()
    test["prediction"] = evaluation_fit.predict(test[ranked_columns])
    test_metrics = evaluate_ranking(test, "prediction", label)
    baseline_metrics = evaluate_ranking(test, baseline, label)

    deployment_fit = _fit_ridge(ranked, frame["target_excess"], selected_alpha)
    latest_date = pd.to_datetime(dataset["evaluation_date"]).max()
    latest = dataset[pd.to_datetime(dataset["evaluation_date"]) == latest_date].copy()
    latest_ranked = _rank_features(latest)
    latest["prediction"] = deployment_fit.predict(latest_ranked)

    prediction_columns = ["evaluation_date", "canonical_code", label, baseline]
    test_predictions = test[prediction_columns + ["prediction"]].copy()
    test_predictions["split"] = "test"
    latest_predictions = latest[["evaluation_date", "canonical_code", label, baseline, "prediction"]].copy()
    latest_predictions["split"] = "latest"
    predictions = pd.concat([test_predictions, latest_predictions], ignore_index=True)
    predictions = predictions.rename(columns={label: "actual_return", baseline: "baseline_score"})
    predictions["predicted_rank"] = predictions.groupby("evaluation_date")["prediction"].rank(
        method="min", ascending=False
    )
    predictions["actual_rank"] = predictions.groupby("evaluation_date")["actual_return"].rank(
        method="min", ascending=False
    )

    payload = {
        "horizon": horizon,
        "algorithm": "ridge",
        "model_version": MODEL_VERSION,
        "selected_alpha": selected_alpha,
        "alpha_candidates": candidates,
        "split": {
            "validation_start": validation_start,
            "test_start": test_start,
            "purged_train_rows": int(train_mask.sum()),
            "validation_rows": int(validation_mask.sum()),
            "final_train_rows": int(final_train_mask.sum()),
            "test_rows": int(test_mask.sum()),
        },
        "test_metrics": test_metrics,
        "rule_baseline_test_metrics": baseline_metrics,
        "evaluation_fit": evaluation_fit.as_dict(),
        "deployment_fit": deployment_fit.as_dict(),
    }
    return payload, predictions, deployment_fit


def train_models(
    settings: Settings,
    horizon: str = "all",
    alpha_grid: str = "0.1,1,10,100,1000",
    dataset_version: str = DATASET_VERSION,
) -> dict[str, object]:
    alphas = sorted({float(value.strip()) for value in alpha_grid.split(",") if value.strip()})
    if not alphas or any(value < 0 for value in alphas):
        raise ValueError("alpha-grid must contain non-negative numbers")
    requested = list(HORIZONS) if horizon == "all" else [horizon]
    with connect(settings.db_path) as connection:
        initialize(connection)
        dataset = connection.execute(
            "SELECT * FROM model_training_dataset WHERE dataset_version = ?",
            [dataset_version],
        ).df()
    if dataset.empty:
        raise RuntimeError("Training dataset is empty; run asset-poc build-training-dataset first")

    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_root = settings.root / "output" / "models" / run_stamp
    output_root.mkdir(parents=True, exist_ok=True)
    results: dict[str, object] = {}
    with connect(settings.db_path) as connection:
        initialize(connection)
        for item in requested:
            payload, predictions, deployment_fit = _train_one_horizon(dataset, item, alphas)
            model_id = f"{MODEL_VERSION}_{item}_{run_stamp}_{uuid4().hex[:6]}"
            model_dir = output_root / item
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / "model.json"
            metrics_path = model_dir / "metrics.json"
            predictions_path = model_dir / "predictions.parquet"
            coefficients_path = model_dir / "coefficients.csv"
            model_document = {
                "model_id": model_id,
                "dataset_version": dataset_version,
                **payload,
            }
            model_path.write_text(
                json.dumps(_json_safe(model_document), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            metrics_path.write_text(
                json.dumps(
                    _json_safe(
                        {
                            "model": payload["test_metrics"],
                            "rule_baseline": payload["rule_baseline_test_metrics"],
                        }
                    ),
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            predictions["model_id"] = model_id
            predictions["horizon"] = item
            predictions["created_at"] = pd.Timestamp.now(tz="UTC")
            predictions.to_parquet(predictions_path, index=False)
            pd.DataFrame(
                {
                    "feature": deployment_fit.feature_names,
                    "coefficient": deployment_fit.coefficients,
                }
            ).assign(absolute_coefficient=lambda x: x["coefficient"].abs()).sort_values(
                "absolute_coefficient", ascending=False
            ).to_csv(coefficients_path, index=False)

            split = payload["split"]
            model_row = pd.DataFrame(
                [
                    {
                        "model_id": model_id,
                        "horizon": item,
                        "algorithm": "ridge",
                        "model_version": MODEL_VERSION,
                        "dataset_version": dataset_version,
                        "selected_alpha": payload["selected_alpha"],
                        "validation_start": split["validation_start"],
                        "test_start": split["test_start"],
                        "feature_names": json.dumps(deployment_fit.feature_names),
                        "coefficients": json.dumps(deployment_fit.coefficients.tolist()),
                        "intercept": deployment_fit.intercept,
                        "preprocessing": json.dumps(
                            {
                                "means": deployment_fit.means.tolist(),
                                "scales": deployment_fit.scales.tolist(),
                                "missing_fill": 0.5,
                                "transform": "monthly_cross_sectional_percentile",
                            }
                        ),
                        "metrics": json.dumps(_json_safe(payload["test_metrics"])),
                        "baseline_metrics": json.dumps(
                            _json_safe(payload["rule_baseline_test_metrics"])
                        ),
                        "artifact_path": str(model_path),
                        "trained_at": pd.Timestamp.now(tz="UTC"),
                    }
                ]
            )
            insert_frame(connection, "trained_models", model_row)
            insert_frame(connection, "model_predictions", predictions)
            results[item] = {
                "model_id": model_id,
                "selected_alpha": payload["selected_alpha"],
                "split": _json_safe(split),
                "test_metrics": _json_safe(payload["test_metrics"]),
                "rule_baseline_test_metrics": _json_safe(
                    payload["rule_baseline_test_metrics"]
                ),
                "artifact_dir": str(model_dir),
            }
    return {"dataset_version": dataset_version, "models": results}


def evaluate_model(
    settings: Settings,
    model_id: str = "latest",
    horizon: str = "6m",
) -> dict[str, object]:
    with connect(settings.db_path) as connection:
        initialize(connection)
        if model_id == "latest":
            row = connection.execute(
                """
                SELECT * FROM trained_models WHERE horizon = ?
                ORDER BY trained_at DESC LIMIT 1
                """,
                [horizon],
            ).df()
        else:
            row = connection.execute(
                "SELECT * FROM trained_models WHERE model_id = ?",
                [model_id],
            ).df()
        if row.empty:
            raise RuntimeError("No matching trained model was found")
        selected_id = str(row.iloc[0]["model_id"])
        predictions = connection.execute(
            """
            SELECT evaluation_date, canonical_code, prediction, actual_return,
                   baseline_score
            FROM model_predictions WHERE model_id = ? AND split = 'test'
            """,
            [selected_id],
        ).df()
    model_metrics = evaluate_ranking(predictions, "prediction", "actual_return")
    baseline_metrics = evaluate_ranking(predictions, "baseline_score", "actual_return")
    return {
        "model_id": selected_id,
        "horizon": str(row.iloc[0]["horizon"]),
        "dataset_version": str(row.iloc[0]["dataset_version"]),
        "test_start": str(row.iloc[0]["test_start"]),
        "model": _json_safe(model_metrics),
        "rule_baseline": _json_safe(baseline_metrics),
        "artifact_path": str(row.iloc[0]["artifact_path"]),
    }
