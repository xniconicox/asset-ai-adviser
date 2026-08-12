from pathlib import Path

import pandas as pd

from asset_poc.collectors import fetch_jquants_financial_summary
from asset_poc.database import connect, initialize, insert_frame


def test_universe_upsert_is_idempotent(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        [
            {
                "as_of_date": "2026-07-31",
                "code": "7203",
                "company_name": "トヨタ自動車",
                "market_segment": "プライム（内国株式）",
                "sector33_code": "3700",
                "sector33_name": "輸送用機器",
                "sector17_code": "6",
                "sector17_name": "自動車・輸送機",
                "size_code": "1",
                "size_name": "TOPIX Core30",
                "source": "jpx",
                "retrieved_at": pd.Timestamp("2026-08-12", tz="UTC"),
            }
        ]
    )
    with connect(tmp_path / "test.duckdb") as connection:
        initialize(connection)
        insert_frame(connection, "universe", frame)
        insert_frame(connection, "universe", frame)
        assert connection.execute("SELECT count(*) FROM universe").fetchone()[0] == 1


def test_financial_summary_upsert_is_idempotent(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        [
            {
                "disclosure_date": "2026-05-08",
                "disclosure_time": "15:00:00",
                "code": "72030",
                "disclosure_number": "20260508555555",
                "document_type": "FYFinancialStatements_Consolidated_JP",
                "current_period_type": "FY",
                "current_period_start": "2025-04-01",
                "current_period_end": "2026-03-31",
                "current_fiscal_year_start": "2025-04-01",
                "current_fiscal_year_end": "2026-03-31",
                "sales": 100.0,
                "operating_profit": 10.0,
                "ordinary_profit": 11.0,
                "net_income": 7.0,
                "eps": 50.0,
                "total_assets": 200.0,
                "equity": 100.0,
                "equity_ratio": 0.5,
                "bps": 500.0,
                "cash_flow_operating": 12.0,
                "cash_flow_investing": -5.0,
                "cash_flow_financing": -3.0,
                "cash_and_equivalents": 20.0,
                "forecast_sales": 110.0,
                "forecast_operating_profit": 12.0,
                "forecast_ordinary_profit": 13.0,
                "forecast_net_income": 8.0,
                "forecast_eps": 55.0,
                "shares_outstanding": 1000.0,
                "roe": 0.08,
                "source": "jquants_v2",
                "retrieved_at": pd.Timestamp("2026-08-12", tz="UTC"),
            }
        ]
    )
    with connect(tmp_path / "test.duckdb") as connection:
        initialize(connection)
        insert_frame(connection, "financial_summaries", frame)
        insert_frame(connection, "financial_summaries", frame)
        assert connection.execute("SELECT count(*) FROM financial_summaries").fetchone()[0] == 1


def test_financial_summary_normalizes_empty_numbers(monkeypatch) -> None:
    class FakeResponse:
        content = b"{}"

        def json(self):
            return {
                "data": [
                    {
                        "DiscDate": "2026-05-08",
                        "DiscNo": "1",
                        "Code": "72030",
                        "Sales": "100",
                        "OP": "",
                    }
                ]
            }

    monkeypatch.setattr("asset_poc.collectors._get", lambda *args, **kwargs: FakeResponse())
    frame, _ = fetch_jquants_financial_summary("key", "72030")
    assert frame.loc[0, "sales"] == 100
    assert pd.isna(frame.loc[0, "operating_profit"])
