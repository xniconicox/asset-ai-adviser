from __future__ import annotations

import html
import json
import math
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from asset_poc.config import Settings
from asset_poc.model_inference import infer_latest_models

NAVY = colors.HexColor("#17233C")
BLUE = colors.HexColor("#2563A6")
LIGHT_GREY = colors.HexColor("#F5F7F9")
MID_GREY = colors.HexColor("#CBD3DC")
DARK_GREY = colors.HexColor("#46515C")
FONT_REGULAR = Path("/usr/share/fonts/opentype/ipaexfont-gothic/ipaexg.ttf")
FONT_BOLD = Path("/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf")


def _register_fonts() -> None:
    if not FONT_REGULAR.exists() or not FONT_BOLD.exists():
        raise FileNotFoundError("Japanese IPA fonts are required")
    if "Daily-Regular" not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont("Daily-Regular", str(FONT_REGULAR)))
        pdfmetrics.registerFont(TTFont("Daily-Bold", str(FONT_BOLD)))


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "DailyTitle",
            parent=base["Title"],
            fontName="Daily-Bold",
            fontSize=20,
            leading=26,
            textColor=NAVY,
            spaceAfter=2 * mm,
        ),
        "meta": ParagraphStyle(
            "DailyMeta",
            parent=base["Normal"],
            fontName="Daily-Regular",
            fontSize=8,
            leading=11,
            textColor=DARK_GREY,
            alignment=TA_CENTER,
        ),
        "section": ParagraphStyle(
            "DailySection",
            parent=base["Heading2"],
            fontName="Daily-Bold",
            fontSize=12,
            leading=15,
            textColor=BLUE,
            spaceBefore=3 * mm,
            spaceAfter=2 * mm,
        ),
        "head": ParagraphStyle(
            "DailyHead",
            parent=base["Normal"],
            fontName="Daily-Bold",
            fontSize=7.5,
            leading=9.5,
            textColor=colors.white,
            alignment=TA_CENTER,
        ),
        "cell": ParagraphStyle(
            "DailyCell",
            parent=base["Normal"],
            fontName="Daily-Regular",
            fontSize=7.7,
            leading=9.8,
            textColor=DARK_GREY,
        ),
        "number": ParagraphStyle(
            "DailyNumber",
            parent=base["Normal"],
            fontName="Daily-Regular",
            fontSize=7.7,
            leading=9.8,
            textColor=DARK_GREY,
            alignment=TA_RIGHT,
        ),
        "note": ParagraphStyle(
            "DailyNote",
            parent=base["Normal"],
            fontName="Daily-Regular",
            fontSize=7,
            leading=10,
            textColor=DARK_GREY,
        ),
    }


def _p(value: object, style: ParagraphStyle) -> Paragraph:
    return Paragraph(html.escape(str(value)), style)


def _number(value: object, digits: int = 1, percent: bool = False) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "-"
    if not math.isfinite(numeric):
        return "-"
    if percent:
        numeric *= 100
        return f"{numeric:.{digits}f}%"
    return f"{numeric:.{digits}f}"


def _text(value: object) -> str:
    return "-" if pd.isna(value) else str(value)


def _date_text(value: object) -> str:
    if pd.isna(value):
        return "-"
    return pd.Timestamp(value).date().isoformat()


def _ranking_table(
    ranking: pd.DataFrame,
    styles: dict[str, ParagraphStyle],
    limit: int,
) -> Table:
    rows: list[list[Paragraph]] = [
        [
            _p("モデル", styles["head"]),
            _p("Rule", styles["head"]),
            _p("コード", styles["head"]),
            _p("銘柄名", styles["head"]),
            _p("相対Score", styles["head"]),
            _p("PER", styles["head"]),
            _p("PBR", styles["head"]),
            _p("ROE", styles["head"]),
        ]
    ]
    selected = ranking.sort_values(["model_rank", "canonical_code"]).head(limit)
    for row in selected.itertuples():
        rows.append(
            [
                _p(int(row.model_rank), styles["number"]),
                _p(int(row.rule_rank), styles["number"]),
                _p(row.canonical_code, styles["cell"]),
                _p(_text(row.company_name), styles["cell"]),
                _p(_number(row.model_score, 2, percent=True), styles["number"]),
                _p(_number(row.per, 1), styles["number"]),
                _p(_number(row.pbr, 2), styles["number"]),
                _p(_number(row.roe, 1, percent=True), styles["number"]),
            ]
        )
    table = Table(
        rows,
        colWidths=[13 * mm, 13 * mm, 17 * mm, 49 * mm, 21 * mm, 17 * mm, 17 * mm, 19 * mm],
        repeatRows=1,
        hAlign="LEFT",
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("GRID", (0, 0), (-1, -1), 0.25, MID_GREY),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_GREY]),
            ]
        )
    )
    return table


def _footer(canvas, document) -> None:
    canvas.saveState()
    canvas.setStrokeColor(MID_GREY)
    canvas.line(18 * mm, 13 * mm, 192 * mm, 13 * mm)
    canvas.setFont("Daily-Regular", 6.8)
    canvas.setFillColor(DARK_GREY)
    canvas.drawString(18 * mm, 8 * mm, "個人研究用 - 売買推奨ではありません")
    canvas.drawRightString(192 * mm, 8 * mm, str(document.page))
    canvas.restoreState()


def generate_daily_ranking_report(
    settings: Settings,
    output: Path,
    limit: int = 10,
) -> dict[str, object]:
    """Create a one-page report of daily inference from the saved 6M/12M models."""
    _register_fonts()
    styles = _styles()
    inference = infer_latest_models(settings)
    manifest = json.loads(
        (settings.published_dir / "latest.json").read_text(encoding="utf-8")
    )
    published = settings.published_dir / str(manifest["run_dir"])
    quality = pd.read_parquet(published / "data_quality.parquet")
    ranking_6m = inference["6m"]
    ranking_12m = inference["12m"]
    snapshot_date = str(inference["snapshot_date"])
    price_date = _date_text(ranking_6m["price_date"].dropna().max())
    financial_date = _date_text(ranking_6m["disclosure_date"].dropna().max())
    model_version = str(ranking_6m["model_version"].iloc[0])
    model_run = str(inference["model_run"])
    financial_count = int(
        (ranking_6m["per"].notna() | ranking_6m["pbr"].notna() | ranking_6m["roe"].notna()).sum()
    )
    quality_status = (
        "-" if quality.empty else ("PASS" if not (quality["status"] == "FAIL").any() else "FAIL")
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    story: list[object] = [
        _p("日本株 モデル日次ランキング", styles["title"]),
        _p(
            f"推論基準日 {snapshot_date}　対象 {len(ranking_6m):,}社　"
            f"株価 {price_date}　決算反映 {financial_count:,}社　品質 {quality_status}",
            styles["meta"],
        ),
        Spacer(1, 3 * mm),
        _p("12M 上位10社", styles["section"]),
        _ranking_table(ranking_12m, styles, limit),
        _p("6M 上位10社", styles["section"]),
        _ranking_table(ranking_6m, styles, limit),
        Spacer(1, 3 * mm),
        _p(
            "相対Scoreは期待リターンではありません。Ruleは既存ルール順位。"
            f"最新決算開示日 {financial_date} / Model {model_run}",
            styles["note"],
        ),
    ]
    document = SimpleDocTemplate(
        str(output),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=14 * mm,
        bottomMargin=16 * mm,
        title="日本株 モデル日次ランキング",
        author="Asset AI Adviser",
    )
    document.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return {
        "output": str(output),
        "report_kind": "daily_model_inference",
        "snapshot_date": snapshot_date,
        "model_version": model_version,
        "model_run": model_run,
        "ranking_rows": len(ranking_6m),
        "top_rows_per_horizon": limit,
    }
