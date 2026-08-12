import json

import pandas as pd
import pytest

from asset_poc.ranking import (
    calculate_fundamental_features,
    calculate_investment_ranks,
    select_preferred_financial_rows,
)


def _financial_row(
    code: str,
    disclosure_date: str,
    period_end: str,
    sales: float,
    operating_profit: float,
    eps: float,
    bps: float,
    roe: float,
) -> dict[str, object]:
    return {
        "code": f"{code}0",
        "disclosure_date": disclosure_date,
        "disclosure_time": "15:00:00",
        "current_period_type": "FY",
        "current_period_end": period_end,
        "current_fiscal_year_end": period_end,
        "sales": sales,
        "operating_profit": operating_profit,
        "eps": eps,
        "bps": bps,
        "roe": roe,
        "equity_ratio": 0.5,
        "forecast_eps": None,
    }


def test_fundamental_features_use_comparable_period() -> None:
    financials = pd.DataFrame(
        [
            _financial_row("1111", "2025-05-10", "2025-03-31", 100, 10, 20, 200, 0.10),
            _financial_row("1111", "2026-05-10", "2026-03-31", 120, 15, 30, 240, 0.12),
        ]
    )
    prices = pd.DataFrame(
        [
            {
                "canonical_code": "1111",
                "price_date": pd.Timestamp("2026-08-12").date(),
                "latest_close": 1200.0,
            }
        ]
    )

    result = calculate_fundamental_features(financials, prices)

    assert len(result) == 1
    assert result.loc[0, "per"] == 40.0
    assert result.loc[0, "pbr"] == 5.0
    assert result.loc[0, "roe"] == 0.12
    assert result.loc[0, "sales_yoy"] == 0.2
    assert result.loc[0, "operating_profit_yoy"] == 0.5


def test_rule_rank_outputs_horizons_confidence_and_reasons() -> None:
    securities = pd.DataFrame(
        [
            {
                "canonical_code": code,
                "company_name": code,
                "sector33_name": "機械",
                "model_group": "general",
            }
            for code in ["1111", "2222", "3333"]
        ]
    )
    prices = pd.DataFrame(
        [
            {
                "canonical_code": code,
                "snapshot_date": pd.Timestamp("2026-08-12").date(),
                "price_date": pd.Timestamp("2026-08-12").date(),
                "latest_close": 1000.0,
                "momentum_score": momentum,
                "risk_score": risk,
            }
            for code, momentum, risk in [
                ("1111", 90.0, 80.0),
                ("2222", 50.0, 50.0),
                ("3333", 10.0, 20.0),
            ]
        ]
    )
    fundamentals = pd.DataFrame(
        [
            {
                "canonical_code": code,
                "disclosure_date": pd.Timestamp("2026-05-10").date(),
                "per": per,
                "pbr": pbr,
                "roe": roe,
                "equity_ratio": 0.5,
                "operating_margin": margin,
                "sales_yoy": growth,
                "operating_profit_yoy": growth,
                "eps_yoy": growth,
                "forecast_eps_revision": growth,
                "financial_completeness": completeness,
            }
            for code, per, pbr, roe, margin, growth, completeness in [
                ("1111", 10.0, 1.0, 0.20, 0.20, 0.30, 1.0),
                ("2222", 20.0, 2.0, 0.10, 0.10, 0.05, 1.0),
                ("3333", None, None, None, None, None, 0.0),
            ]
        ]
    )

    result = calculate_investment_ranks(securities, prices, fundamentals)
    first = result.set_index("canonical_code").loc["1111"]
    missing = result.set_index("canonical_code").loc["3333"]

    assert first["rank_6m"] == 1
    assert first["rank_12m"] == 1
    assert first["score_6m"] > 50
    assert first["score_12m"] > 50
    assert missing["confidence"] == 0.4
    assert json.loads(first["positive_reasons"])


def test_per_does_not_use_partial_period_eps_without_forecast() -> None:
    financials = pd.DataFrame(
        [
            {
                **_financial_row("1111", "2025-05-10", "2025-03-31", 100, 10, 20, 200, 0.10),
                "forecast_eps": None,
            },
            {
                **_financial_row("1111", "2026-02-10", "2026-03-31", 90, 9, 15, 220, 0.11),
                "current_period_type": "3Q",
                "forecast_eps": None,
            },
        ]
    )
    prices = pd.DataFrame(
        [
            {
                "canonical_code": "1111",
                "price_date": pd.Timestamp("2026-08-12").date(),
                "latest_close": 1000.0,
            }
        ]
    )

    result = calculate_fundamental_features(financials, prices)

    assert result.loc[0, "eps"] == 15
    assert result.loc[0, "per"] == 50


def test_same_day_financial_rows_are_coalesced_deterministically() -> None:
    statement = {
        **_financial_row("1111", "2026-05-10", "2026-03-31", 120, 15, 30, 240, 0.12),
        "disclosure_number": "100",
        "document_type": "FYFinancialStatements_Consolidated_JP",
        "forecast_eps": None,
    }
    revision = {
        **statement,
        "disclosure_time": "16:00:00",
        "disclosure_number": "101",
        "document_type": "EarnForecastRevision",
        "sales": None,
        "operating_profit": None,
        "eps": None,
        "bps": None,
        "roe": None,
        "equity_ratio": None,
        "forecast_eps": 40.0,
    }

    selected = select_preferred_financial_rows(pd.DataFrame([revision, statement]))

    assert len(selected) == 1
    assert selected.loc[0, "sales"] == 120
    assert selected.loc[0, "eps"] == 30
    assert selected.loc[0, "forecast_eps"] == 40
    assert selected.loc[0, "disclosure_number"] == "101"


def test_later_same_day_correction_wins_when_completeness_is_equal() -> None:
    previous_year = {
        **_financial_row("1111", "2025-05-10", "2025-03-31", 100, 10, 20, 200, 0.10),
        "disclosure_number": "001",
    }
    first = {
        **_financial_row("1111", "2026-05-10", "2026-03-31", 120, 15, 30, 240, 0.12),
        "disclosure_number": "100",
    }
    corrected = {
        **first,
        "disclosure_time": "17:00:00",
        "disclosure_number": "102",
        "sales": 130,
    }
    prices = pd.DataFrame(
        [{"canonical_code": "1111", "price_date": "2026-08-12", "latest_close": 1200.0}]
    )

    result = calculate_fundamental_features(
        pd.DataFrame([corrected, previous_year, first]), prices
    )
    shuffled = calculate_fundamental_features(
        pd.DataFrame([first, corrected, previous_year]), prices
    )

    assert result.loc[0, "sales_yoy"] == pytest.approx(0.3)
    pd.testing.assert_frame_equal(
        result.drop(columns="calculated_at"), shuffled.drop(columns="calculated_at")
    )


def test_qualitative_feature_is_confidence_weighted_and_missing_is_neutral() -> None:
    securities = pd.DataFrame(
        [
            {
                "canonical_code": "1111",
                "company_name": "Test",
                "sector33_name": "機械",
                "model_group": "general",
            }
        ]
    )
    prices = pd.DataFrame(
        [
            {
                "canonical_code": "1111",
                "snapshot_date": pd.Timestamp("2026-08-12").date(),
                "price_date": pd.Timestamp("2026-08-12").date(),
                "latest_close": 1000.0,
                "momentum_score": 50.0,
                "risk_score": 50.0,
            }
        ]
    )
    base = calculate_investment_ranks(securities, prices, pd.DataFrame()).iloc[0]
    qualitative = pd.DataFrame(
        [
            {
                "canonical_code": "1111",
                "disclosure_date": pd.Timestamp("2026-08-01").date(),
                "qualitative_score": 100.0,
                "qualitative_confidence": 0.8,
            }
        ]
    )
    enhanced = calculate_investment_ranks(securities, prices, pd.DataFrame(), qualitative).iloc[0]

    assert base["qualitative_confidence"] == 0
    assert base["qualitative_score"] == 50
    assert enhanced["score_6m"] - base["score_6m"] == pytest.approx(4.0)
    assert enhanced["score_12m"] - base["score_12m"] == pytest.approx(3.2)
