from __future__ import annotations

from collections.abc import Mapping
from datetime import time

import pandas as pd
import yfinance as yf

YAHOO_SOURCE = "yahoo_finance"


def _symbol_frame(download: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if not isinstance(download.columns, pd.MultiIndex):
        return download.copy()
    if symbol in download.columns.get_level_values(-1):
        return download.xs(symbol, axis=1, level=-1, drop_level=True).copy()
    if symbol in download.columns.get_level_values(0):
        return download.xs(symbol, axis=1, level=0, drop_level=True).copy()
    return pd.DataFrame(index=download.index)


def normalize_yahoo_download(
    download: pd.DataFrame,
    symbol_to_code: Mapping[str, str],
    retrieved_at: pd.Timestamp | None = None,
) -> pd.DataFrame:
    retrieved_at = retrieved_at or pd.Timestamp.now(tz="UTC")
    rows: list[pd.DataFrame] = []
    for symbol, canonical_code in symbol_to_code.items():
        source = _symbol_frame(download, symbol)
        if source.empty or "Close" not in source:
            continue
        source = source.reset_index()
        date_column = source.columns[0]
        source = source.rename(
            columns={
                date_column: "trade_date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Adj Close": "adjusted_close",
                "Volume": "volume",
                "Dividends": "dividends",
                "Stock Splits": "stock_splits",
            }
        )
        source["trade_date"] = pd.to_datetime(source["trade_date"], errors="coerce").dt.date
        source = source.dropna(subset=["trade_date", "close"])
        source["canonical_code"] = canonical_code
        source["provider_symbol"] = symbol
        if "adjusted_close" not in source:
            source["adjusted_close"] = source["close"]
        for column in ["open", "high", "low", "volume", "dividends", "stock_splits"]:
            if column not in source:
                source[column] = 0.0 if column in {"dividends", "stock_splits"} else None
        for column in [
            "open",
            "high",
            "low",
            "close",
            "adjusted_close",
            "volume",
            "dividends",
            "stock_splits",
        ]:
            source[column] = pd.to_numeric(source[column], errors="coerce")
        source["available_at"] = pd.to_datetime(source["trade_date"]).map(
            lambda value: pd.Timestamp.combine(value.date(), time(15, 30)).tz_localize("Asia/Tokyo")
        )
        source["retrieved_at"] = retrieved_at
        source["source"] = YAHOO_SOURCE
        source["source_tier"] = "C"
        rows.append(source)

    columns = [
        "trade_date",
        "canonical_code",
        "provider_symbol",
        "open",
        "high",
        "low",
        "close",
        "adjusted_close",
        "volume",
        "dividends",
        "stock_splits",
        "available_at",
        "retrieved_at",
        "source",
        "source_tier",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.concat(rows, ignore_index=True)[columns]


def fetch_yahoo_prices(
    symbol_to_code: Mapping[str, str], period: str = "2y"
) -> tuple[pd.DataFrame, bytes]:
    symbols = list(symbol_to_code)
    download = yf.download(
        tickers=symbols,
        period=period,
        interval="1d",
        auto_adjust=False,
        actions=True,
        group_by="column",
        threads=True,
        progress=False,
        timeout=45,
    )
    frame = normalize_yahoo_download(download, symbol_to_code)
    raw = frame.to_csv(index=False).encode("utf-8")
    return frame, raw
