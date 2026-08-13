import pandas as pd

from asset_poc.features import calculate_price_features
from asset_poc.price_quality import clean_price_history, summarize_price_quality


def _row(day: str, **updates: object) -> dict[str, object]:
    row: dict[str, object] = {
        "trade_date": pd.Timestamp(day).date(),
        "canonical_code": "1111",
        "source": "yahoo_finance",
        "open": 100.0,
        "high": 105.0,
        "low": 95.0,
        "close": 102.0,
        "adjusted_close": 100.0,
        "volume": 1000.0,
    }
    row.update(updates)
    return row


def test_cleaning_preserves_source_and_records_each_action() -> None:
    source = pd.DataFrame(
        [
            _row("2026-01-05"),
            _row("2026-01-06", high=101.0, close=102.0),
            _row("2026-01-07", adjusted_close=-10.0),
            _row("2026-01-08", adjusted_close=5000.0),
            _row(
                "2026-01-09",
                open=1e10,
                high=1e10,
                low=1e10,
                close=1e10,
                adjusted_close=1e10,
                volume=0.0,
            ),
        ]
    )
    original = source.copy(deep=True)

    cleaned, events = clean_price_history(source)

    pd.testing.assert_frame_equal(source, original)
    boundary = cleaned.loc[cleaned["trade_date"] == pd.Timestamp("2026-01-06").date()].iloc[0]
    assert boundary["clean_high"] == 102.0
    assert boundary["quality_status"] == "corrected"
    assert boundary["valuation_price"] == boundary["clean_close"]
    assert boundary["return_price"] == boundary["adjusted_close"]
    assert cleaned.loc[cleaned["trade_date"] == pd.Timestamp("2026-01-07").date(), "model_price"].isna().all()
    assert cleaned.loc[cleaned["trade_date"] == pd.Timestamp("2026-01-08").date(), "model_price"].isna().all()
    assert cleaned.loc[cleaned["trade_date"] == pd.Timestamp("2026-01-09").date(), "model_price"].isna().all()
    assert set(events["reason_code"]) == {
        "ohlc_boundary_error",
        "adjusted_close_nonpositive",
        "adjustment_ratio_outlier",
        "zero_volume_scale_outlier",
    }
    summary = summarize_price_quality(events)
    assert summary["affected_rows"] == 4
    assert summary["excluded_model_rows"] == 3


def test_feature_history_does_not_cross_invalid_or_long_gap() -> None:
    old_dates = pd.bdate_range("2024-01-01", periods=260)
    new_dates = pd.bdate_range("2025-12-18", periods=130)
    old = pd.DataFrame(
        [_row(str(day.date()), adjusted_close=100.0 + index) for index, day in enumerate(old_dates)]
    )
    invalid = pd.DataFrame(
        [
            _row(
                "2025-12-17",
                open=1e10,
                high=1e10,
                low=1e10,
                close=1e10,
                adjusted_close=1e10,
                volume=0.0,
            )
        ]
    )
    new = pd.DataFrame(
        [_row(str(day.date()), adjusted_close=200.0 + index) for index, day in enumerate(new_dates)]
    )
    source = pd.concat([old, invalid, new], ignore_index=True)
    cleaned, _ = clean_price_history(source)

    result = calculate_price_features(cleaned)

    assert len(result) == 1
    assert pd.isna(result.loc[0, "return_12m"])
    assert result.loc[0, "return_6m"] > 0
    assert result.loc[0, "price_date"] == new_dates[-1].date()


def test_valuation_uses_unadjusted_close_while_returns_use_adjusted_close() -> None:
    dates = pd.bdate_range("2026-01-01", periods=22)
    source = pd.DataFrame(
        [
            _row(
                str(day.date()),
                close=120.0,
                open=120.0,
                high=120.0,
                low=120.0,
                adjusted_close=80.0 + index,
            )
            for index, day in enumerate(dates)
        ]
    )
    cleaned, _ = clean_price_history(source)

    result = calculate_price_features(cleaned)

    assert result.loc[0, "latest_close"] == 120.0
    assert result.loc[0, "return_1m"] == (101.0 / 80.0) - 1.0
