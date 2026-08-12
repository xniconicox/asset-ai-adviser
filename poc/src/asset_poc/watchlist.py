from __future__ import annotations

import duckdb
import pandas as pd

WATCHLIST_NAME = "topix500"
WATCHLIST_RULE = "All TOPIX Core30, Large70 and Mid400 ordered by size group and code"
DOMESTIC_MARKETS = (
    "プライム（内国株式）",
    "スタンダード（内国株式）",
    "グロース（内国株式）",
)
FINANCIAL_SECTORS = {
    "銀行業",
    "保険業",
    "証券、商品先物取引業",
    "その他金融業",
}


def to_jquants_code(canonical_code: str) -> str:
    code = str(canonical_code).strip()
    return code if len(code) >= 5 else f"{code}0"


def to_yahoo_symbol(canonical_code: str) -> str:
    return f"{str(canonical_code).strip()}.T"


def build_watchlist(
    connection: duckdb.DuckDBPyConnection,
    limit: int | None = None,
    watchlist_name: str = WATCHLIST_NAME,
) -> pd.DataFrame:
    limit_sql = " LIMIT ?" if limit else ""
    parameters: list[object] = [*DOMESTIC_MARKETS]
    if limit:
        parameters.append(limit)
    frame = connection.execute(
        f"""
        WITH latest AS (
            SELECT max(as_of_date) AS as_of_date FROM universe
        )
        SELECT u.as_of_date, u.code AS canonical_code, u.company_name,
               u.market_segment, u.sector33_name, u.size_name
        FROM universe u, latest
        WHERE u.as_of_date = latest.as_of_date
          AND u.market_segment IN (?, ?, ?)
          AND u.size_name IN ('TOPIX Core30', 'TOPIX Large70', 'TOPIX Mid400')
        ORDER BY CASE u.size_name
                   WHEN 'TOPIX Core30' THEN 1
                   WHEN 'TOPIX Large70' THEN 2
                   ELSE 3
                 END,
                 u.code
        {limit_sql}
        """,
        parameters,
    ).df()
    if frame.empty:
        raise RuntimeError("Universe is empty. Run `asset-poc collect` first.")

    now = pd.Timestamp.now(tz="UTC")
    frame["selection_rank"] = range(1, len(frame) + 1)
    frame["selection_rule"] = WATCHLIST_RULE
    frame["watchlist_name"] = watchlist_name
    frame["model_group"] = frame["sector33_name"].map(
        lambda value: "financial" if value in FINANCIAL_SECTORS else "general"
    )
    frame["created_at"] = now

    securities = frame[
        [
            "canonical_code",
            "company_name",
            "market_segment",
            "sector33_name",
            "size_name",
            "model_group",
        ]
    ].copy()
    securities["updated_at"] = now
    connection.register("watchlist_securities", securities)
    connection.execute(
        "INSERT OR REPLACE INTO securities BY NAME SELECT * FROM watchlist_securities"
    )
    connection.unregister("watchlist_securities")

    membership = frame[
        [
            "watchlist_name",
            "as_of_date",
            "canonical_code",
            "selection_rank",
            "selection_rule",
            "created_at",
        ]
    ]
    connection.register("watchlist_rows", membership)
    connection.execute(
        "INSERT OR REPLACE INTO watchlist_membership BY NAME SELECT * FROM watchlist_rows"
    )
    connection.unregister("watchlist_rows")

    symbols = []
    for code in frame["canonical_code"]:
        symbols.extend(
            [
                {
                    "canonical_code": code,
                    "provider": "jquants_v2",
                    "provider_symbol": to_jquants_code(code),
                    "source_tier": "A",
                    "updated_at": now,
                },
                {
                    "canonical_code": code,
                    "provider": "yahoo_finance",
                    "provider_symbol": to_yahoo_symbol(code),
                    "source_tier": "C",
                    "updated_at": now,
                },
            ]
        )
    symbol_frame = pd.DataFrame(symbols)
    connection.register("provider_symbol_rows", symbol_frame)
    connection.execute(
        "INSERT OR REPLACE INTO provider_symbols BY NAME SELECT * FROM provider_symbol_rows"
    )
    connection.unregister("provider_symbol_rows")
    return frame


def get_watchlist_symbols(
    connection: duckdb.DuckDBPyConnection,
    provider: str,
    limit: int | None = None,
    watchlist_name: str = WATCHLIST_NAME,
) -> pd.DataFrame:
    limit_sql = " LIMIT ?" if limit else ""
    return connection.execute(
        f"""
        WITH latest AS (
            SELECT max(as_of_date) AS as_of_date
            FROM watchlist_membership WHERE watchlist_name = ?
        )
        SELECT w.canonical_code, p.provider_symbol, w.selection_rank
        FROM watchlist_membership w
        JOIN latest l ON w.as_of_date = l.as_of_date
        JOIN provider_symbols p ON p.canonical_code = w.canonical_code
        WHERE w.watchlist_name = ? AND p.provider = ?
        ORDER BY w.selection_rank{limit_sql}
        """,
        [watchlist_name, watchlist_name, provider, *([limit] if limit else [])],
    ).df()
