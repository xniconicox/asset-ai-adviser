from __future__ import annotations

import io
from datetime import date, datetime, timezone
from typing import Any

import pandas as pd
import requests

JPX_LIST_URL = (
    "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls"
)
JQUANTS_DAILY_URL = "https://api.jquants.com/v2/equities/bars/daily"
JQUANTS_FINANCIAL_SUMMARY_URL = "https://api.jquants.com/v2/fins/summary"
EDINET_DOCUMENTS_URL = "https://api.edinet-fsa.go.jp/api/v2/documents.json"
USER_AGENT = "asset-ai-adviser-poc/0.1 (local research)"


def _get(url: str, **kwargs: Any) -> requests.Response:
    headers = {"User-Agent": USER_AGENT, **kwargs.pop("headers", {})}
    response = requests.get(url, headers=headers, timeout=45, **kwargs)
    response.raise_for_status()
    return response


def fetch_jpx_universe() -> tuple[pd.DataFrame, bytes]:
    response = _get(JPX_LIST_URL)
    frame = pd.read_excel(io.BytesIO(response.content), dtype=str)
    frame.columns = [str(column).strip() for column in frame.columns]
    aliases = {
        "日付": "as_of_date",
        "コード": "code",
        "銘柄名": "company_name",
        "市場・商品区分": "market_segment",
        "33業種コード": "sector33_code",
        "33業種区分": "sector33_name",
        "17業種コード": "sector17_code",
        "17業種区分": "sector17_name",
        "規模コード": "size_code",
        "規模区分": "size_name",
    }
    frame = frame.rename(columns=aliases)
    expected = list(aliases.values())
    for column in expected:
        if column not in frame:
            frame[column] = None
    frame = frame[expected].copy()
    frame["code"] = frame["code"].str.strip()
    frame["source"] = "jpx"
    frame["retrieved_at"] = pd.Timestamp.now(tz="UTC")
    return frame, response.content


def fetch_jquants_daily(api_key: str, code: str, from_date: str) -> tuple[pd.DataFrame, bytes]:
    params = {"code": code, "from": from_date.replace("-", "")}
    response = _get(JQUANTS_DAILY_URL, params=params, headers={"x-api-key": api_key})
    payload = response.json()
    rows = payload.get("data") or payload.get("daily_quotes") or []
    frame = pd.DataFrame(rows)
    aliases = {
        "Date": "trade_date",
        "Code": "code",
        "O": "open",
        "Open": "open",
        "H": "high",
        "High": "high",
        "L": "low",
        "Low": "low",
        "C": "close",
        "Close": "close",
        "Vo": "volume",
        "Volume": "volume",
        "Va": "turnover_value",
        "TurnoverValue": "turnover_value",
        "AdjFactor": "adjustment_factor",
        "AdjustmentFactor": "adjustment_factor",
        "AdjO": "adjusted_open",
        "AdjustmentOpen": "adjusted_open",
        "AdjH": "adjusted_high",
        "AdjustmentHigh": "adjusted_high",
        "AdjL": "adjusted_low",
        "AdjustmentLow": "adjusted_low",
        "AdjC": "adjusted_close",
        "AdjustmentClose": "adjusted_close",
        "AdjVo": "adjusted_volume",
        "AdjustmentVolume": "adjusted_volume",
    }
    frame = frame.rename(columns=aliases)
    columns = [
        "trade_date",
        "code",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "turnover_value",
        "adjustment_factor",
        "adjusted_open",
        "adjusted_high",
        "adjusted_low",
        "adjusted_close",
        "adjusted_volume",
    ]
    for column in columns:
        if column not in frame:
            frame[column] = None
    frame = frame[columns]
    frame["source"] = "jquants_v2"
    frame["retrieved_at"] = pd.Timestamp.now(tz="UTC")
    return frame, response.content


def fetch_jquants_financial_summary(
    api_key: str,
    code: str | None = None,
    target_date: date | str | None = None,
) -> tuple[pd.DataFrame, bytes]:
    if bool(code) == bool(target_date):
        raise ValueError("Specify exactly one of code or target_date")
    params = {"code": code} if code else {"date": str(target_date).replace("-", "")}
    response = _get(
        JQUANTS_FINANCIAL_SUMMARY_URL,
        params=params,
        headers={"x-api-key": api_key},
    )
    frame = pd.DataFrame(response.json().get("data", []))
    aliases = {
        "DiscDate": "disclosure_date",
        "DiscTime": "disclosure_time",
        "Code": "code",
        "DiscNo": "disclosure_number",
        "DocType": "document_type",
        "CurPerType": "current_period_type",
        "CurPerSt": "current_period_start",
        "CurPerEn": "current_period_end",
        "CurFYSt": "current_fiscal_year_start",
        "CurFYEn": "current_fiscal_year_end",
        "Sales": "sales",
        "OP": "operating_profit",
        "OdP": "ordinary_profit",
        "NP": "net_income",
        "EPS": "eps",
        "TA": "total_assets",
        "Eq": "equity",
        "EqAR": "equity_ratio",
        "BPS": "bps",
        "CFO": "cash_flow_operating",
        "CFI": "cash_flow_investing",
        "CFF": "cash_flow_financing",
        "CashEq": "cash_and_equivalents",
        "FSales": "forecast_sales",
        "FOP": "forecast_operating_profit",
        "FOdP": "forecast_ordinary_profit",
        "FNP": "forecast_net_income",
        "FEPS": "forecast_eps",
        "ShOutFY": "shares_outstanding",
        "ROE": "roe",
    }
    frame = frame.rename(columns=aliases)
    columns = list(aliases.values())
    for column in columns:
        if column not in frame:
            frame[column] = None
    frame = frame[columns]
    date_columns = [
        "disclosure_date",
        "current_period_start",
        "current_period_end",
        "current_fiscal_year_start",
        "current_fiscal_year_end",
    ]
    numeric_columns = [
        "sales",
        "operating_profit",
        "ordinary_profit",
        "net_income",
        "eps",
        "total_assets",
        "equity",
        "equity_ratio",
        "bps",
        "cash_flow_operating",
        "cash_flow_investing",
        "cash_flow_financing",
        "cash_and_equivalents",
        "forecast_sales",
        "forecast_operating_profit",
        "forecast_ordinary_profit",
        "forecast_net_income",
        "forecast_eps",
        "shares_outstanding",
        "roe",
    ]
    for column in date_columns:
        frame[column] = pd.to_datetime(frame[column], errors="coerce").dt.date
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["source"] = "jquants_v2"
    frame["retrieved_at"] = pd.Timestamp.now(tz="UTC")
    return frame, response.content


def fetch_edinet_documents(
    api_key: str, target_date: str | None = None
) -> tuple[pd.DataFrame, bytes]:
    target_date = target_date or datetime.now(timezone.utc).date().isoformat()
    response = _get(
        EDINET_DOCUMENTS_URL,
        params={"date": target_date, "type": 2, "Subscription-Key": api_key},
    )
    payload = response.json()
    frame = pd.DataFrame(payload.get("results", []))
    aliases = {
        "docID": "document_id",
        "edinetCode": "edinet_code",
        "secCode": "security_code",
        "filerName": "filer_name",
        "docDescription": "description",
        "submitDateTime": "submitted_at",
        "docTypeCode": "document_type_code",
        "periodStart": "period_start",
        "periodEnd": "period_end",
    }
    frame = frame.rename(columns=aliases)
    columns = list(aliases.values())
    for column in columns:
        if column not in frame:
            frame[column] = None
    frame = frame[columns]
    frame["source"] = "edinet_v2"
    frame["retrieved_at"] = pd.Timestamp.now(tz="UTC")
    return frame, response.content
