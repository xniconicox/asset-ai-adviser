from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from asset_poc.learning import (
    DATASET_VERSION,
    MODEL_FEATURE_COLUMNS,
    MODEL_VERSION,
    _build_price_rows,
    _fit_ridge,
    _forward_label,
    _temporal_split_masks,
    evaluate_ranking,
)


def test_forward_label_starts_next_trade_and_uses_calendar_month_target() -> None:
    dates = pd.to_datetime(
        ["2025-01-31", "2025-02-03", "2025-07-31", "2026-01-30"]
    ).to_numpy(dtype="datetime64[ns]")
    prices = np.array([100.0, 101.0, 121.2, 131.3])
    segments = np.array([0, 0, 0, 0])

    start, end, result = _forward_label(
        dates, prices, segments, pd.Timestamp("2025-01-31"), 6
    )

    assert str(start) == "2025-02-03"
    assert str(end) == "2025-07-31"
    assert np.isclose(result, 0.2)


def test_forward_label_does_not_cross_invalid_price_segment() -> None:
    dates = pd.to_datetime(["2025-01-31", "2025-02-03", "2025-07-31"]).to_numpy(
        dtype="datetime64[ns]"
    )
    start, end, result = _forward_label(
        dates,
        np.array([100.0, 101.0, 121.2]),
        np.array([0, 0, 1]),
        pd.Timestamp("2025-01-31"),
        6,
    )

    assert str(start) == "2025-02-03"
    assert end is None
    assert result is None


def test_monthly_rows_use_raw_close_for_valuation_and_adjusted_for_return() -> None:
    dates = pd.bdate_range("2024-01-01", periods=260)
    prices = pd.DataFrame(
        {
            "trade_date": dates,
            "canonical_code": "1111",
            "return_price": np.linspace(80.0, 120.0, len(dates)),
            "valuation_price": 150.0,
            "clean_volume": 1000.0,
        }
    )

    result = _build_price_rows(prices, [dates[-1]])

    assert result.loc[0, "latest_close"] == 150.0
    assert result.loc[0, "return_12m"] == pytest.approx(120.0 / prices.loc[7, "return_price"] - 1)


def test_ridge_recovers_linear_direction() -> None:
    features = pd.DataFrame({"a": np.linspace(0, 1, 100), "b": np.linspace(1, 0, 100)})
    target = features["a"] * 2 - features["b"]

    model = _fit_ridge(features, target, alpha=0.1)
    prediction = model.predict(features)

    assert np.corrcoef(prediction, target)[0, 1] > 0.999
    assert model.coefficients[0] > 0
    assert model.coefficients[1] < 0


def test_temporal_split_purges_overlapping_labels() -> None:
    dates = pd.date_range("2020-01-31", periods=40, freq="ME")
    frame = pd.DataFrame(
        {
            "evaluation_date": dates,
            "label_end": dates + pd.DateOffset(months=12),
        }
    )

    train, validation, final_train, validation_start, test_start = _temporal_split_masks(
        frame, "label_end"
    )

    assert (frame.loc[train, "label_end"] < validation_start).all()
    assert (frame.loc[final_train, "label_end"] < test_start).all()
    assert (frame.loc[validation, "evaluation_date"] >= validation_start).all()


def test_ranking_metrics_reward_correct_order() -> None:
    rows = []
    for month in pd.date_range("2024-01-31", periods=3, freq="ME"):
        for index in range(20):
            rows.append(
                {
                    "evaluation_date": month,
                    "canonical_code": str(index),
                    "prediction": float(index),
                    "actual": float(index) / 100,
                }
            )
    metrics = evaluate_ranking(pd.DataFrame(rows), "prediction", "actual")

    assert metrics["mean_spearman_ic"] == 1.0
    assert metrics["top_decile_excess"] > 0
    assert metrics["long_short_spread"] > 0


def test_model_feature_list_contains_price_and_fundamentals() -> None:
    assert "momentum_12_1" in MODEL_FEATURE_COLUMNS
    assert "per" in MODEL_FEATURE_COLUMNS
    assert "forecast_eps_revision" in MODEL_FEATURE_COLUMNS
    assert DATASET_VERSION == "monthly_pit_v2_unadjusted_valuation"
    assert MODEL_VERSION == "ridge_rank_v2_unadjusted_valuation"
