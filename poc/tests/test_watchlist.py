from pathlib import Path

import pandas as pd

from asset_poc.database import connect, initialize, insert_frame
from asset_poc.watchlist import build_watchlist, get_watchlist_symbols


def test_watchlist_prefers_topix_size_groups(tmp_path: Path) -> None:
    rows = []
    for code, size in [
        ("1001", "TOPIX Mid400"),
        ("1002", "TOPIX Large70"),
        ("1003", "TOPIX Core30"),
        ("1004", "TOPIX Mid400"),
    ]:
        rows.append(
            {
                "as_of_date": "2026-07-31",
                "code": code,
                "company_name": f"Company {code}",
                "market_segment": "プライム（内国株式）",
                "sector33_code": "5250",
                "sector33_name": "情報・通信業",
                "sector17_code": "10",
                "sector17_name": "情報通信・サービスその他",
                "size_code": "-",
                "size_name": size,
                "source": "jpx",
                "retrieved_at": pd.Timestamp("2026-08-12", tz="UTC"),
            }
        )
    with connect(tmp_path / "test.duckdb") as connection:
        initialize(connection)
        insert_frame(connection, "universe", pd.DataFrame(rows))
        frame = build_watchlist(connection, limit=3)
        assert frame["canonical_code"].tolist() == ["1003", "1002", "1001"]
        symbols = get_watchlist_symbols(connection, "yahoo_finance")
        assert symbols["provider_symbol"].tolist() == ["1003.T", "1002.T", "1001.T"]
