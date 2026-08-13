from __future__ import annotations

import numpy as np
import pandas as pd

from asset_poc.database import insert_frame

CLEANING_VERSION = "price_clean_v2_dual_price"


def _finite(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return pd.Series(np.isfinite(values), index=series.index)


def _event_rows(
    data: pd.DataFrame,
    cleaned: pd.DataFrame,
    mask: pd.Series,
    reason_code: str,
    severity: str,
    action: str,
    detected_at: pd.Timestamp,
) -> pd.DataFrame:
    columns = [
        "trade_date",
        "canonical_code",
        "source",
        "open",
        "high",
        "low",
        "close",
        "adjusted_close",
        "volume",
    ]
    selected = data.loc[mask, columns].copy()
    if selected.empty:
        return selected
    selected = selected.rename(
        columns={
            "open": "original_open",
            "high": "original_high",
            "low": "original_low",
            "close": "original_close",
            "adjusted_close": "original_adjusted_close",
            "volume": "original_volume",
        }
    )
    selected["cleaned_high"] = cleaned.loc[mask, "clean_high"].to_numpy()
    selected["cleaned_low"] = cleaned.loc[mask, "clean_low"].to_numpy()
    selected["model_price"] = cleaned.loc[mask, "model_price"].to_numpy()
    selected["cleaned_volume"] = cleaned.loc[mask, "clean_volume"].to_numpy()
    selected["cleaning_version"] = CLEANING_VERSION
    selected["reason_code"] = reason_code
    selected["severity"] = severity
    selected["action"] = action
    selected["detected_at"] = detected_at
    return selected


def clean_price_history(prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create an auditable model-input layer without changing source rows."""
    if prices.empty:
        return prices.copy(), pd.DataFrame()

    data = prices.copy()
    numeric_columns = ["open", "high", "low", "close", "adjusted_close", "volume"]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    cleaned = data.copy()
    cleaned["clean_open"] = data["open"]
    cleaned["clean_high"] = data["high"]
    cleaned["clean_low"] = data["low"]
    cleaned["clean_close"] = data["close"]
    cleaned["clean_volume"] = data["volume"]
    # Keep the two economic meanings separate.  Adjusted close is suitable for
    # return calculations, while valuation ratios must use the price that was
    # actually quoted on that date.  ``model_price`` remains as a compatibility
    # alias for the return series used by older callers.
    cleaned["return_price"] = data["adjusted_close"]
    cleaned["model_price"] = cleaned["return_price"]

    finite_ohlc = pd.concat(
        [_finite(data[column]) for column in ("open", "high", "low", "close")],
        axis=1,
    ).all(axis=1)
    positive_ohlc = (data[["open", "high", "low", "close"]] > 0).all(axis=1)
    invalid_ohlc = ~(finite_ohlc & positive_ohlc)
    observed_high = data[["open", "high", "low", "close"]].max(axis=1)
    observed_low = data[["open", "high", "low", "close"]].min(axis=1)
    boundary_error = (~invalid_ohlc) & (
        (data["high"] < observed_high) | (data["low"] > observed_low)
    )

    cleaned.loc[boundary_error, "clean_high"] = observed_high[boundary_error]
    cleaned.loc[boundary_error, "clean_low"] = observed_low[boundary_error]
    cleaned.loc[
        invalid_ohlc, ["clean_open", "clean_high", "clean_low", "clean_close"]
    ] = np.nan
    cleaned["valuation_price"] = cleaned["clean_close"]

    finite_adjusted = _finite(data["adjusted_close"])
    finite_close = _finite(data["close"])
    adjusted_nonfinite = ~finite_adjusted
    adjusted_nonpositive = finite_adjusted & (data["adjusted_close"] <= 0)
    ratio = data["adjusted_close"] / data["close"]
    ratio_outlier = (
        finite_adjusted
        & finite_close
        & (data["adjusted_close"] > 0)
        & (data["close"] > 0)
        & ((ratio < 0.001) | (ratio > 10.0))
    )
    zero_volume_scale_outlier = (
        finite_close
        & (data["volume"] == 0)
        & (data["close"] > 1_000_000_000)
    )
    invalid_model_price = (
        adjusted_nonfinite
        | adjusted_nonpositive
        | ratio_outlier
        | zero_volume_scale_outlier
        | ~finite_close
        | (data["close"] <= 0)
    )
    cleaned.loc[invalid_model_price, ["return_price", "model_price"]] = np.nan

    negative_volume = _finite(data["volume"]) & (data["volume"] < 0)
    cleaned.loc[negative_volume, "clean_volume"] = np.nan

    corrected = boundary_error | invalid_ohlc | negative_volume
    cleaned["quality_status"] = np.select(
        [invalid_model_price, corrected], ["excluded", "corrected"], default="clean"
    )
    cleaned["cleaning_version"] = CLEANING_VERSION

    detected_at = pd.Timestamp.now(tz="UTC")
    event_specs = [
        (boundary_error, "ohlc_boundary_error", "warning", "expand_ohlc_envelope"),
        (invalid_ohlc, "ohlc_nonpositive_or_nonfinite", "error", "null_ohlc"),
        (adjusted_nonfinite, "adjusted_close_nonfinite", "error", "exclude_model_price"),
        (adjusted_nonpositive, "adjusted_close_nonpositive", "error", "exclude_model_price"),
        (ratio_outlier, "adjustment_ratio_outlier", "error", "exclude_model_price"),
        (
            zero_volume_scale_outlier,
            "zero_volume_scale_outlier",
            "error",
            "exclude_model_price",
        ),
        (negative_volume, "negative_volume", "error", "null_volume"),
    ]
    events = [
        _event_rows(data, cleaned, mask, reason, severity, action, detected_at)
        for mask, reason, severity, action in event_specs
    ]
    events = [event for event in events if not event.empty]
    return cleaned, pd.concat(events, ignore_index=True) if events else pd.DataFrame()


def store_price_quality_events(connection, events: pd.DataFrame) -> None:
    connection.execute(
        "DELETE FROM price_quality_events WHERE cleaning_version = ?",
        [CLEANING_VERSION],
    )
    insert_frame(connection, "price_quality_events", events)


def summarize_price_quality(events: pd.DataFrame) -> dict[str, object]:
    if events.empty:
        return {"cleaning_version": CLEANING_VERSION, "event_rows": 0, "by_reason": {}}
    return {
        "cleaning_version": CLEANING_VERSION,
        "event_rows": len(events),
        "affected_rows": int(
            events[["canonical_code", "trade_date", "source"]].drop_duplicates().shape[0]
        ),
        "affected_codes": int(events["canonical_code"].nunique()),
        "excluded_model_rows": int((events["action"] == "exclude_model_price").sum()),
        "by_reason": events.groupby("reason_code").size().sort_index().to_dict(),
    }
