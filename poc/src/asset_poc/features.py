from __future__ import annotations

import math

import duckdb
import pandas as pd

from asset_poc.database import insert_frame
from asset_poc.price_quality import clean_price_history, store_price_quality_events
from asset_poc.watchlist import WATCHLIST_NAME

FEATURE_VERSION = "price_v3_dual_price"


def _period_return(prices: pd.Series, days: int) -> float | None:
    clean = prices.dropna()
    if len(clean) <= days:
        return None
    return float(clean.iloc[-1] / clean.iloc[-days - 1] - 1)


def _latest_usable_segment(group: pd.DataFrame) -> pd.DataFrame:
    """Use only the latest continuous run of valid total-return prices."""
    ordered = group.sort_values("trade_date").drop_duplicates("trade_date", keep="last")
    price_column = "return_price" if "return_price" in ordered else "adjusted_close"
    return_price = pd.to_numeric(ordered[price_column], errors="coerce")
    valid = return_price.notna() & return_price.map(math.isfinite) & (return_price > 0)
    dates = pd.to_datetime(ordered["trade_date"], errors="coerce")
    breaks = (~valid) | dates.diff().dt.days.gt(10).fillna(False)
    segment_id = breaks.cumsum()
    if not valid.any():
        return ordered.iloc[0:0].copy()
    latest_segment = segment_id.loc[valid[valid].index[-1]]
    result = ordered.loc[(segment_id == latest_segment) & valid].copy()
    result["return_price"] = return_price.loc[result.index]
    return result


def calculate_price_features(prices: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for code, group in prices.groupby("canonical_code"):
        group = _latest_usable_segment(group)
        if group.empty:
            continue
        return_price = group["return_price"].astype(float)
        valuation_column = "valuation_price" if "valuation_price" in group else "close"
        valuation_price = pd.to_numeric(group[valuation_column], errors="coerce")
        latest_valuation = valuation_price.iloc[-1]
        latest_close = (
            float(latest_valuation)
            if pd.notna(latest_valuation)
            and math.isfinite(float(latest_valuation))
            and float(latest_valuation) > 0
            else None
        )
        returns = return_price.pct_change(fill_method=None)
        recent_252 = return_price.tail(252)
        recent_returns_60 = returns.tail(60)
        downside = recent_returns_60[recent_returns_60 < 0]
        max_drawdown = (recent_252 / recent_252.cummax() - 1).min()
        momentum_12_1 = None
        if len(return_price.dropna()) > 252:
            momentum_12_1 = float(return_price.iloc[-22] / return_price.iloc[-253] - 1)
        records.append(
            {
                "canonical_code": str(code),
                "price_date": group["trade_date"].iloc[-1],
                "latest_close": latest_close,
                "return_1m": _period_return(return_price, 21),
                "return_3m": _period_return(return_price, 63),
                "return_6m": _period_return(return_price, 126),
                "return_12m": _period_return(return_price, 252),
                "momentum_12_1": momentum_12_1,
                "volatility_20d": float(returns.tail(20).std() * math.sqrt(252)),
                "volatility_60d": float(recent_returns_60.std() * math.sqrt(252)),
                "downside_volatility_60d": (
                    float(downside.std() * math.sqrt(252)) if len(downside) > 1 else None
                ),
                "max_drawdown_252d": float(max_drawdown),
                "high_52w_distance": float(
                    return_price.iloc[-1] / recent_252.max() - 1
                ),
                "average_volume_20d": float(
                    group.get("clean_volume", group["volume"]).tail(20).mean()
                ),
                "average_turnover_20d": float(
                    (
                        group.get("clean_close", group["close"])
                        * group.get("clean_volume", group["volume"])
                    ).tail(20).mean()
                ),
                "source": "yahoo_finance+price_clean_v2_dual_price",
                "source_tier": "C",
            }
        )
    frame = pd.DataFrame(records)
    if frame.empty:
        return frame

    momentum_inputs = ["return_3m", "return_6m", "return_12m", "momentum_12_1"]
    momentum_percentiles = frame[momentum_inputs].rank(pct=True)
    frame["momentum_score"] = momentum_percentiles.mean(axis=1, skipna=True) * 100
    volatility_safety = 1 - frame["volatility_60d"].rank(pct=True)
    drawdown_safety = frame["max_drawdown_252d"].rank(pct=True)
    frame["risk_score"] = pd.concat([volatility_safety, drawdown_safety], axis=1).mean(axis=1) * 100
    frame["price_score"] = frame["momentum_score"] * 0.7 + frame["risk_score"] * 0.3
    frame["price_rank"] = frame["price_score"].rank(method="min", ascending=False).astype("Int64")
    frame["snapshot_date"] = max(frame["price_date"])
    frame["feature_version"] = FEATURE_VERSION
    frame["calculated_at"] = pd.Timestamp.now(tz="UTC")
    return frame


def compute_and_store_price_features(
    connection: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    prices = connection.execute(
        """
        WITH latest_watchlist AS (
            SELECT max(as_of_date) AS as_of_date FROM watchlist_membership
            WHERE watchlist_name = ?
        )
        SELECT p.* FROM secondary_prices p
        JOIN watchlist_membership w ON w.canonical_code = p.canonical_code
        JOIN latest_watchlist l ON w.as_of_date = l.as_of_date
        WHERE w.watchlist_name = ?
        ORDER BY p.canonical_code, p.trade_date
        """,
        [WATCHLIST_NAME, WATCHLIST_NAME],
    ).df()
    cleaned, events = clean_price_history(prices)
    store_price_quality_events(connection, events)
    frame = calculate_price_features(cleaned)
    insert_frame(connection, "price_feature_snapshots", frame)
    return frame
