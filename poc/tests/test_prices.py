import numpy as np
import pandas as pd

from asset_poc.features import calculate_price_features
from asset_poc.prices import normalize_yahoo_download


def test_normalize_yahoo_multi_index_download() -> None:
    dates = pd.date_range("2026-08-10", periods=2, freq="B")
    columns = pd.MultiIndex.from_product(
        [["Open", "High", "Low", "Close", "Adj Close", "Volume"], ["7203.T"]]
    )
    download = pd.DataFrame(
        [[100, 105, 99, 104, 103, 1000], [104, 108, 102, 107, 107, 1200]],
        index=dates,
        columns=columns,
    )
    frame = normalize_yahoo_download(download, {"7203.T": "7203"})
    assert len(frame) == 2
    assert frame.loc[1, "canonical_code"] == "7203"
    assert frame.loc[1, "adjusted_close"] == 107
    assert str(frame.loc[1, "available_at"].tzinfo) == "Asia/Tokyo"


def test_calculate_price_features_has_momentum_and_snapshot() -> None:
    dates = pd.date_range("2025-01-01", periods=260, freq="B")
    prices = pd.DataFrame(
        {
            "trade_date": dates.date,
            "canonical_code": "7203",
            "close": np.linspace(100, 200, len(dates)),
            "adjusted_close": np.linspace(100, 200, len(dates)),
            "volume": 1000.0,
        }
    )
    frame = calculate_price_features(prices)
    assert len(frame) == 1
    assert frame.loc[0, "return_12m"] > 0
    assert frame.loc[0, "momentum_12_1"] > 0
    assert frame.loc[0, "snapshot_date"] == dates[-1].date()
