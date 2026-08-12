from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from asset_poc.config import Settings
from asset_poc.ranking import (
    QUALITATIVE_WEIGHT_6M,
    QUALITATIVE_WEIGHT_12M,
    WEIGHTS_6M,
    WEIGHTS_12M,
)

NAVY = colors.HexColor("#17233C")
BLUE = colors.HexColor("#2563A6")
TEAL = colors.HexColor("#148C8C")
PALE_BLUE = colors.HexColor("#EAF2FA")
PALE_TEAL = colors.HexColor("#E6F5F3")
LIGHT_GREY = colors.HexColor("#F3F5F7")
MID_GREY = colors.HexColor("#AAB4C0")
DARK_GREY = colors.HexColor("#46515C")
AMBER = colors.HexColor("#D98D1B")
RED = colors.HexColor("#B94A48")

FONT_REGULAR_PATH = Path("/usr/share/fonts/opentype/ipaexfont-gothic/ipaexg.ttf")
FONT_BOLD_PATH = Path("/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf")

FEATURE_LABELS = {
    "return_1m": "1Mリターン",
    "return_3m": "3Mリターン",
    "return_6m": "6Mリターン",
    "return_12m": "12Mリターン",
    "momentum_12_1": "12-1M Momentum",
    "volatility_20d": "20日Volatility",
    "volatility_60d": "60日Volatility",
    "downside_volatility_60d": "下方Volatility",
    "max_drawdown_252d": "252日最大DD",
    "high_52w_distance": "52週高値乖離",
    "per": "PER",
    "pbr": "PBR",
    "roe": "ROE",
    "equity_ratio": "自己資本比率",
    "operating_margin": "営業利益率",
    "sales_yoy": "売上YoY",
    "operating_profit_yoy": "営業利益YoY",
    "eps_yoy": "EPS YoY",
    "forecast_eps_revision": "予想EPS修正",
    "financial_completeness": "数値決算充足度",
    "qualitative_score": "定性総合",
    "qualitative_confidence": "定性信頼度",
}

FACTOR_COLUMNS = [
    "valuation_score",
    "quality_score",
    "growth_score",
    "earnings_score",
    "momentum_score",
    "risk_score",
]
FACTOR_LABELS = ["Valuation", "Quality", "Growth", "Earnings", "Momentum", "Risk"]


def _register_fonts() -> None:
    if not FONT_REGULAR_PATH.exists() or not FONT_BOLD_PATH.exists():
        raise FileNotFoundError("Japanese IPA fonts are required to generate the report")
    if "JP-Regular" not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont("JP-Regular", str(FONT_REGULAR_PATH)))
        pdfmetrics.registerFont(TTFont("JP-Bold", str(FONT_BOLD_PATH)))


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "IPAexGothic",
            "axes.unicode_minus": False,
            "axes.edgecolor": "#AAB4C0",
            "axes.labelcolor": "#46515C",
            "xtick.color": "#46515C",
            "ytick.color": "#46515C",
            "figure.facecolor": "white",
            "axes.facecolor": "#F8FAFC",
        }
    )


def _load_latest(settings: Settings) -> tuple[dict, Path, dict[str, pd.DataFrame]]:
    pointer = settings.published_dir / "latest.json"
    if not pointer.exists():
        raise FileNotFoundError("Published snapshot not found. Run `asset-poc publish` first.")
    manifest = json.loads(pointer.read_text(encoding="utf-8"))
    run_dir = (settings.published_dir / manifest["run_dir"]).resolve()
    if not run_dir.is_relative_to(settings.published_dir.resolve()):
        raise ValueError("Invalid published snapshot path")
    required = {
        "input": "model_input_latest.parquet",
        "coverage": "data_coverage.parquet",
        "prices": "price_history.parquet",
        "financials": "financial_history.parquet",
        "quality": "data_quality.parquet",
    }
    missing = [filename for filename in required.values() if not (run_dir / filename).exists()]
    if missing:
        raise FileNotFoundError(
            f"Snapshot does not contain EDA inputs: {missing}. Run `asset-poc publish` again."
        )
    frames = {name: pd.read_parquet(run_dir / filename) for name, filename in required.items()}
    return manifest, run_dir, frames


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "TitleJP",
            parent=base["Title"],
            fontName="JP-Bold",
            fontSize=25,
            leading=34,
            textColor=NAVY,
            alignment=TA_LEFT,
            spaceAfter=10 * mm,
            wordWrap="CJK",
        ),
        "subtitle": ParagraphStyle(
            "SubtitleJP",
            parent=base["Normal"],
            fontName="JP-Regular",
            fontSize=11,
            leading=18,
            textColor=DARK_GREY,
            wordWrap="CJK",
        ),
        "h1": ParagraphStyle(
            "H1JP",
            parent=base["Heading1"],
            fontName="JP-Bold",
            fontSize=17,
            leading=23,
            textColor=NAVY,
            spaceBefore=2 * mm,
            spaceAfter=5 * mm,
            wordWrap="CJK",
        ),
        "h2": ParagraphStyle(
            "H2JP",
            parent=base["Heading2"],
            fontName="JP-Bold",
            fontSize=12,
            leading=17,
            textColor=BLUE,
            spaceBefore=3 * mm,
            spaceAfter=2.5 * mm,
            wordWrap="CJK",
        ),
        "body": ParagraphStyle(
            "BodyJP",
            parent=base["BodyText"],
            fontName="JP-Regular",
            fontSize=9.2,
            leading=14.5,
            textColor=DARK_GREY,
            spaceAfter=2.5 * mm,
            wordWrap="CJK",
        ),
        "small": ParagraphStyle(
            "SmallJP",
            parent=base["BodyText"],
            fontName="JP-Regular",
            fontSize=7.5,
            leading=10.5,
            textColor=DARK_GREY,
            wordWrap="CJK",
        ),
        "table": ParagraphStyle(
            "TableJP",
            parent=base["BodyText"],
            fontName="JP-Regular",
            fontSize=7.3,
            leading=9.5,
            textColor=DARK_GREY,
            wordWrap="CJK",
        ),
        "table_head": ParagraphStyle(
            "TableHeadJP",
            parent=base["BodyText"],
            fontName="JP-Bold",
            fontSize=7.4,
            leading=9.5,
            textColor=colors.white,
            alignment=TA_CENTER,
            wordWrap="CJK",
        ),
        "callout": ParagraphStyle(
            "CalloutJP",
            parent=base["BodyText"],
            fontName="JP-Bold",
            fontSize=10,
            leading=15,
            textColor=NAVY,
            alignment=TA_CENTER,
            wordWrap="CJK",
        ),
    }


def _para(text: object, style: ParagraphStyle) -> Paragraph:
    return Paragraph(html.escape(str(text)).replace("\n", "<br/>"), style)


def _table(
    rows: list[list[object]],
    styles: dict[str, ParagraphStyle],
    widths: list[float] | None = None,
    align: str = "LEFT",
) -> Table:
    converted: list[list[object]] = []
    for row_index, row in enumerate(rows):
        style = styles["table_head"] if row_index == 0 else styles["table"]
        converted.append([cell if hasattr(cell, "wrap") else _para(cell, style) for cell in row])
    table = Table(converted, colWidths=widths, repeatRows=1, hAlign=align)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (0, 0), (-1, 0), "CENTER"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("GRID", (0, 0), (-1, -1), 0.25, MID_GREY),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_GREY]),
            ]
        )
    )
    return table


def _metric_cards(metrics: list[tuple[str, str]], styles: dict[str, ParagraphStyle]) -> Table:
    cells = []
    for label, value in metrics:
        cells.append(
            Table(
                [
                    [_para(value, styles["callout"])],
                    [_para(label, styles["small"])],
                ],
                colWidths=[32 * mm],
                rowHeights=[13 * mm, 8 * mm],
                style=TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, -1), PALE_BLUE),
                        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CAD8E8")),
                        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ]
                ),
            )
        )
    return Table([cells], colWidths=[34 * mm] * len(cells), hAlign="LEFT")


def _save_coverage_chart(coverage: pd.DataFrame, path: Path) -> None:
    categories = [
        "株価252日以上",
        "決算3回以上",
        "Core充足100%",
        "定性原文あり",
        "LLM構造化済",
    ]
    values = [
        int((coverage["price_rows"] >= 252).sum()),
        int((coverage["financial_periods"] >= 3).sum()),
        int((coverage["core_coverage_pct"] == 100).sum()),
        int((coverage["source_document_count"] > 0).sum()),
        int((coverage["analysis_count"] > 0).sum()),
    ]
    fig, ax = plt.subplots(figsize=(8.2, 3.2))
    bar_colors = ["#2563A6", "#2563A6", "#148C8C", "#D98D1B", "#D98D1B"]
    bars = ax.barh(categories[::-1], values[::-1], color=bar_colors[::-1], height=0.55)
    ax.set_xlim(0, len(coverage) * 1.08)
    ax.set_xlabel("銘柄数")
    ax.grid(axis="x", alpha=0.2)
    for bar, value in zip(bars, values[::-1]):
        ax.text(
            value + 5, bar.get_y() + bar.get_height() / 2, f"{value}/{len(coverage)}", va="center"
        )
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_distribution_chart(frame: pd.DataFrame, path: Path) -> None:
    columns = ["per", "pbr", "roe", "return_6m", "volatility_60d", "max_drawdown_252d"]
    fig, axes = plt.subplots(2, 3, figsize=(9.1, 5.5))
    for ax, column in zip(axes.flat, columns):
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        if values.empty:
            ax.text(0.5, 0.5, "データなし", ha="center", va="center")
            ax.set_axis_off()
            continue
        lower, upper = values.quantile([0.01, 0.99])
        clipped = values.clip(lower, upper)
        ax.hist(clipped, bins=24, color="#2563A6", alpha=0.82, edgecolor="white")
        ax.axvline(clipped.median(), color="#D98D1B", linewidth=1.6, linestyle="--")
        ax.set_title(FEATURE_LABELS[column], fontsize=10)
        ax.grid(axis="y", alpha=0.18)
    fig.suptitle("主要入力特徴量の分布（表示のみ1-99%へWinsorize）", fontsize=12, color="#17233C")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_factor_chart(frame: pd.DataFrame, path: Path) -> None:
    data = [pd.to_numeric(frame[column], errors="coerce").dropna() for column in FACTOR_COLUMNS]
    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    box = ax.boxplot(data, tick_labels=FACTOR_LABELS, patch_artist=True, showfliers=False)
    palette = ["#2563A6", "#148C8C", "#4A8D5C", "#D98D1B", "#7C5CBF", "#B94A48"]
    for patch, color in zip(box["boxes"], palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.78)
    ax.axhline(50, color="#46515C", linewidth=1, linestyle="--")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Factor Score")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_correlation_chart(frame: pd.DataFrame, path: Path) -> None:
    columns = FACTOR_COLUMNS + ["score_6m", "score_12m"]
    labels = FACTOR_LABELS + ["Score 6M", "Score 12M"]
    corr = frame[columns].apply(pd.to_numeric, errors="coerce").corr(method="spearman")
    cmap = LinearSegmentedColormap.from_list("eda_div", ["#B94A48", "#FFFFFF", "#2563A6"])
    fig, ax = plt.subplots(figsize=(7.4, 5.9))
    image = ax.imshow(corr, vmin=-1, vmax=1, cmap=cmap)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    for i in range(len(labels)):
        for j in range(len(labels)):
            value = corr.iloc[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=6.5)
    fig.colorbar(image, ax=ax, fraction=0.045, pad=0.04)
    ax.set_title("Factor/Score Spearman相関")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_sector_chart(frame: pd.DataFrame, path: Path) -> None:
    sector = (
        frame.groupby("sector33_name", dropna=False)
        .agg(companies=("canonical_code", "count"), median_score=("score_12m", "median"))
        .sort_values("companies", ascending=True)
    )
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 6.4), sharey=True)
    axes[0].barh(sector.index, sector["companies"], color="#2563A6")
    axes[0].set_title("業種別銘柄数")
    axes[0].set_xlabel("銘柄数")
    axes[1].barh(sector.index, sector["median_score"], color="#148C8C")
    axes[1].axvline(50, color="#D98D1B", linestyle="--", linewidth=1)
    axes[1].set_title("業種別12M Score中央値")
    axes[1].set_xlabel("Score")
    axes[1].set_xlim(35, 65)
    for ax in axes:
        ax.grid(axis="x", alpha=0.18)
        ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _summary_rows(frame: pd.DataFrame, columns: list[str]) -> list[list[object]]:
    rows: list[list[object]] = [["特徴量", "有効件数", "欠損率", "中央値", "P05", "P95"]]
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        valid = values.dropna()
        if valid.empty:
            stats = ["-", "-", "-"]
        else:
            stats = [
                f"{valid.median():.3g}",
                f"{valid.quantile(0.05):.3g}",
                f"{valid.quantile(0.95):.3g}",
            ]
        rows.append(
            [
                FEATURE_LABELS.get(column, column),
                f"{len(valid):,}",
                f"{values.isna().mean() * 100:.1f}%",
                *stats,
            ]
        )
    return rows


def _header_footer(canvas, doc) -> None:
    canvas.saveState()
    width, height = A4
    if doc.page > 1:
        canvas.setStrokeColor(colors.HexColor("#D6DCE3"))
        canvas.line(18 * mm, height - 14 * mm, width - 18 * mm, height - 14 * mm)
        canvas.setFont("JP-Regular", 7)
        canvas.setFillColor(DARK_GREY)
        canvas.drawString(18 * mm, height - 10.5 * mm, "Asset AI Adviser - Model & Input EDA")
    canvas.setFont("JP-Regular", 7)
    canvas.setFillColor(DARK_GREY)
    canvas.drawRightString(width - 18 * mm, 10 * mm, f"{doc.page}")
    canvas.restoreState()


def generate_model_eda_report(
    settings: Settings, output_path: Path | None = None
) -> dict[str, object]:
    _register_fonts()
    _configure_matplotlib()
    manifest, run_dir, frames = _load_latest(settings)
    model_input = frames["input"]
    coverage = frames["coverage"]
    prices = frames["prices"]
    financials = frames["financials"]
    quality = frames["quality"]

    output = output_path or settings.root / "output" / "reports" / "model_input_eda_report.pdf"
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    chart_dir = settings.root / "tmp" / "pdfs" / "model_eda"
    chart_dir.mkdir(parents=True, exist_ok=True)

    chart_paths = {
        "coverage": chart_dir / "coverage.png",
        "distribution": chart_dir / "distributions.png",
        "factor": chart_dir / "factors.png",
        "correlation": chart_dir / "correlation.png",
        "sector": chart_dir / "sectors.png",
    }
    _save_coverage_chart(coverage, chart_paths["coverage"])
    _save_distribution_chart(model_input, chart_paths["distribution"])
    _save_factor_chart(model_input, chart_paths["factor"])
    _save_correlation_chart(model_input, chart_paths["correlation"])
    _save_sector_chart(model_input, chart_paths["sector"])

    styles = _styles()
    story: list[object] = []
    generated_at = datetime.now(ZoneInfo("Asia/Tokyo"))
    snapshot_date = manifest.get("snapshot_date", "-")
    ranking_version = manifest.get("ranking_version", "-")
    price_min = pd.to_datetime(prices["trade_date"]).min().date()
    price_max = pd.to_datetime(prices["trade_date"]).max().date()
    financial_coverage = int((coverage["financial_periods"] > 0).sum())
    financial_3plus = int((coverage["financial_periods"] >= 3).sum())
    core_complete = int((coverage["core_coverage_pct"] == 100).sum())
    qualitative_count = int((coverage["analysis_count"] > 0).sum())

    # Cover
    story.extend(
        [
            Spacer(1, 18 * mm),
            _para("日本株ランキングモデル\n入力データEDAレポート", styles["title"]),
            _para(
                "学習開始前のモデル仕様、データ品質、分布、欠損、相関、業種構成、学習準備状況を記録する基準資料",
                styles["subtitle"],
            ),
            Spacer(1, 14 * mm),
            _metric_cards(
                [
                    ("対象銘柄", f"{len(model_input):,}"),
                    ("株価行", f"{len(prices):,}"),
                    ("決算行", f"{len(financials):,}"),
                    ("数値決算あり", f"{financial_coverage}/{len(model_input)}"),
                    ("定性分析あり", f"{qualitative_count}/{len(model_input)}"),
                ],
                styles,
            ),
            Spacer(1, 16 * mm),
            _table(
                [
                    ["項目", "値"],
                    ["Snapshot date", snapshot_date],
                    ["Ranking version", ranking_version],
                    ["Published batch", manifest.get("batch_run_id", "-")],
                    ["株価期間", f"{price_min} - {price_max}"],
                    ["生成日時 (JST)", generated_at.strftime("%Y-%m-%d %H:%M:%S")],
                ],
                styles,
                widths=[48 * mm, 118 * mm],
            ),
            Spacer(1, 16 * mm),
            Table(
                [
                    [
                        _para(
                            "重要: 本レポートはモデル開発・検証用であり、投資判断や売買推奨を目的としない。現在の定性ウェイトはバックテスト前の暫定値である。",
                            styles["body"],
                        )
                    ]
                ],
                colWidths=[166 * mm],
                style=TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FFF4DD")),
                        ("BOX", (0, 0), (-1, -1), 0.7, AMBER),
                        ("LEFTPADDING", (0, 0), (-1, -1), 10),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                        ("TOPPADDING", (0, 0), (-1, -1), 8),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                    ]
                ),
            ),
            PageBreak(),
        ]
    )

    # Model overview
    story.extend(
        [
            _para("1. モデル概要", styles["h1"]),
            _para(
                "対象はJPX規模区分のTOPIX Core30・Large70・Mid400相当。数値決算と株価からクロスセクションFactor Scoreを作り、6M・12Mの魅力度順位を計算する。金融業はmodel_groupを分離し、Factorの順位化は原則として同一model_group・業種内を重視する。",
                styles["body"],
            ),
            _para("処理フロー", styles["h2"]),
            _table(
                [
                    ["1. 原データ", "2. Point-in-time Feature", "3. Rule Score", "4. 公開"],
                    [
                        "JPX / Yahoo / J-Quants / 開示原文",
                        "Valuation / Quality / Growth / Earnings / Momentum / Risk / Qualitative",
                        "6M・12M重み、Confidence補正、定性補正",
                        "DQ Gate後にAtomic Snapshot",
                    ],
                ],
                styles,
                widths=[41.5 * mm] * 4,
            ),
            Spacer(1, 4 * mm),
            _para("基本Factorウェイト", styles["h2"]),
            _table(
                [
                    ["Factor", "6M", "12M", "主な入力"],
                    ["Valuation", "10%", "20%", "PER, PBR"],
                    ["Quality", "10%", "20%", "ROE, 自己資本比率, 営業利益率"],
                    ["Growth", "15%", "20%", "売上, 営業利益, EPSのYoY"],
                    ["Earnings", "25%", "20%", "予想EPS修正, 増益率"],
                    ["Momentum", "30%", "10%", "株価Momentum"],
                    ["Risk", "10%", "10%", "Volatility, Downside, Drawdown"],
                ],
                styles,
                widths=[32 * mm, 23 * mm, 23 * mm, 88 * mm],
            ),
            _para("スコア式", styles["h2"]),
            _para(
                "BaseScore = 50 + NumericConfidence × (WeightedFactorScore - 50)。数値決算が不足する銘柄はConfidenceが低下し、Scoreは中立値50へ縮小される。順位はScoreの降順で付与する。",
                styles["body"],
            ),
            _para("定性情報の統合", styles["h2"]),
            _para(
                "Qualitative = 見通し30% + 需要20% + 採算20% + リスク管理15% + 利益の質15%。6M補正は10% × LLM信頼度 × (Qualitative - 50)、12M補正は8% × LLM信頼度 × (Qualitative - 50)。定性情報がない場合は信頼度0のため補正は0となり、既存Rule Scoreを変えない。",
                styles["body"],
            ),
            _para(
                f"実装定数: Qualitative weight 6M={QUALITATIVE_WEIGHT_6M:.0%}, 12M={QUALITATIVE_WEIGHT_12M:.0%}。現Snapshotでは定性分析{qualitative_count}社に補正を適用し、未分析銘柄は補正0。",
                styles["small"],
            ),
            PageBreak(),
        ]
    )

    # Data inventory and coverage
    source_rows = [
        ["データ", "保存件数/社数", "期間・状態", "モデル用途"],
        [
            "株価日足",
            f"{len(prices):,}行 / {prices['canonical_code'].nunique()}社",
            f"{price_min} - {price_max}",
            "Momentum, Risk, Valuation時点価格",
        ],
        [
            "数値決算",
            f"{len(financials):,}行 / {financial_coverage}社",
            f"3回以上 {financial_3plus}社",
            "Valuation, Quality, Growth, Earnings",
        ],
        [
            "定性原文",
            f"{int(coverage['source_document_count'].sum())}文書",
            "未投入",
            "LLM構造化の根拠",
        ],
        ["定性分析", f"{int(coverage['analysis_count'].sum())}文書", "未実行", "Qualitative補正"],
    ]
    story.extend(
        [
            _para("2. 入力データと充足状況", styles["h1"]),
            _para(
                "EDAは画面と同じ公開Batchのmodel_input_latest.parquetを使用する。主DBの最新状態ではなく、品質ゲートを通過して実際にランキングへ使われた入力世代を分析対象とする。",
                styles["body"],
            ),
            _table(source_rows, styles, widths=[30 * mm, 41 * mm, 42 * mm, 53 * mm]),
            Spacer(1, 4 * mm),
            Image(str(chart_paths["coverage"]), width=166 * mm, height=65 * mm),
            Spacer(1, 2 * mm),
            _metric_cards(
                [
                    ("Core 100%", f"{core_complete}/{len(coverage)}"),
                    ("Core平均", f"{coverage['core_coverage_pct'].mean():.1f}%"),
                    ("拡張平均", f"{coverage['extended_coverage_pct'].mean():.1f}%"),
                    ("数値決算3+", f"{financial_3plus}/{len(coverage)}"),
                ],
                styles,
            ),
            Spacer(1, 5 * mm),
            _para("解釈", styles["h2"]),
            _para(
                f"株価カバレッジは{(coverage['price_rows'] > 0).mean() * 100:.1f}%。数値決算は{financial_coverage / len(coverage) * 100:.1f}%の銘柄に存在し、3回以上は{financial_3plus}社。Core充足100%は{core_complete}社であり、未充足銘柄は上場期間不足または決算履歴不足が中心。定性原文とLLM分析は{qualitative_count}社のため、拡張充足度は意図的に低い。",
                styles["body"],
            ),
            PageBreak(),
        ]
    )

    # Missingness and summary statistics
    summary_columns = [
        "return_6m",
        "return_12m",
        "volatility_60d",
        "max_drawdown_252d",
        "per",
        "pbr",
        "roe",
        "operating_margin",
        "sales_yoy",
        "operating_profit_yoy",
        "forecast_eps_revision",
        "qualitative_score",
    ]
    story.extend(
        [
            _para("3. 欠損と基本統計", styles["h1"]),
            _para(
                "欠損値は一律補完しない。数値決算Factorでは利用可能な指標の平均を取り、financial_completenessからConfidenceを計算して中立値へ縮小する。定性特徴量は欠損中立で、未取得時に補正しない。",
                styles["body"],
            ),
            _table(
                _summary_rows(model_input, summary_columns),
                styles,
                widths=[49 * mm, 22 * mm, 22 * mm, 24 * mm, 24 * mm, 24 * mm],
            ),
            Spacer(1, 5 * mm),
            _para("欠損上の注意", styles["h2"]),
            _para(
                "PERは予想EPSが正の場合に予想EPSを優先し、なければ最新通期EPSを利用する。赤字やBPS非正の場合は欠損となる。YoYは同じ会計期間種別の過去値が必要。したがって欠損はランダムではなく、赤字企業・新規上場・決算履歴不足へ偏る可能性がある。学習時には欠損フラグ自体の説明力とバイアスを評価する。",
                styles["body"],
            ),
            PageBreak(),
        ]
    )

    # Distributions
    story.extend(
        [
            _para("4. 特徴量分布", styles["h1"]),
            _para(
                "下図は外れ値による可読性低下を避けるため表示時のみ1-99%へWinsorizeしている。DB値とランキング計算値は変更していない。オレンジ破線は中央値。",
                styles["body"],
            ),
            Image(str(chart_paths["distribution"]), width=172 * mm, height=104 * mm),
            Spacer(1, 3 * mm),
            _para("観察事項", styles["h2"]),
            _para(
                "Valuation指標と成長率には長い裾があり、クロスセクション順位化前にmodel_group別2.5-97.5%でclipしている。ROE、リターン、Drawdownには符号があり、単純な平均より業種・企業群内の相対順位が適している。今後の学習では極端値処理を学習期間内だけで推定し、評価期間へ持ち越す必要がある。",
                styles["body"],
            ),
            PageBreak(),
        ]
    )

    # Factors and correlations
    story.extend(
        [
            _para("5. Factor分布と相関", styles["h1"]),
            _para(
                "Factor Scoreは原則0-100の相対尺度。中央値50付近を中心に比較できる一方、欠損が多い銘柄は利用可能指標の平均とConfidence縮小の影響を受ける。",
                styles["body"],
            ),
            Image(str(chart_paths["factor"]), width=166 * mm, height=76 * mm),
            Spacer(1, 3 * mm),
            Image(
                str(chart_paths["correlation"]), width=127 * mm, height=101 * mm, hAlign="CENTER"
            ),
            _para(
                "Scoreとの相関は現行ウェイトを反映した機械的な結果であり、将来リターンへの予測力を示すものではない。Factor間の高相関は情報の二重計上につながるため、学習前にVIF、正則化、Permutation Importance等で確認する。",
                styles["small"],
            ),
            PageBreak(),
        ]
    )

    # Sector and ranking
    top10 = model_input.nsmallest(10, "rank_12m")
    top_rows: list[list[object]] = [["12M", "Code", "銘柄", "業種", "Score", "PER", "PBR", "ROE"]]
    for row in top10.itertuples():
        top_rows.append(
            [
                int(row.rank_12m),
                row.canonical_code,
                row.company_name,
                row.sector33_name,
                f"{row.score_12m:.1f}",
                "-" if pd.isna(row.per) else f"{row.per:.1f}",
                "-" if pd.isna(row.pbr) else f"{row.pbr:.2f}",
                "-" if pd.isna(row.roe) else f"{row.roe * 100:.1f}%",
            ]
        )
    story.extend(
        [
            _para("6. 業種構成とランキング出力", styles["h1"]),
            _para(
                "Universeは業種ごとの銘柄数が不均衡である。現行モデルは業種内順位70%、model_group内順位30%を基本とし、業種内有効件数が5未満の場合はmodel_group順位へフォールバックする。",
                styles["body"],
            ),
            Image(str(chart_paths["sector"]), width=170 * mm, height=118 * mm),
            _para("12M上位10社（Snapshot時点）", styles["h2"]),
            _table(
                top_rows,
                styles,
                widths=[10 * mm, 15 * mm, 35 * mm, 29 * mm, 18 * mm, 18 * mm, 18 * mm, 18 * mm],
            ),
            _para(
                "上位表はモデル出力の動作確認であり、推奨リストではない。将来リターンで未検証のため、順位の経済的有効性はまだ評価できない。",
                styles["small"],
            ),
            PageBreak(),
        ]
    )

    # Learning readiness
    dq_fail = quality[quality["status"] == "FAIL"]
    readiness_rows = [
        ["項目", "状態", "学習開始前の対応"],
        ["現在時点の入力Snapshot", "準備済", "model_input_latest.parquetをversion管理"],
        ["株価・数値決算カバレッジ", "概ね準備済", "不足銘柄と新規上場の扱いを固定"],
        ["定性原文・LLM特徴量", "PoC検証中", "対象開示を拡大し、人手評価とprompt固定"],
        ["過去Point-in-Time Feature", "未準備", "各開示時点のSnapshotを再構築"],
        ["6M/12M将来リターンLabel", "未準備", "開示翌営業日起点、TOPIX/業種超過を計算"],
        ["Walk-forward評価", "未準備", "学習・検証・テスト期間を時間順に分離"],
        ["Model Registry/再学習", "未準備", "Champion/Challengerとロールバックを実装"],
    ]
    story.extend(
        [
            _para("7. 学習準備状況と次の実装", styles["h1"]),
            _para(
                "結論: 現SnapshotはEDAと推論の基準入力として利用可能だが、教師あり学習を開始する条件は未充足。最大の不足は過去時点Featureと6M/12M後のLabelである。現在の横断面だけで学習すると、将来情報漏洩や過学習を検出できない。",
                styles["body"],
            ),
            _table(readiness_rows, styles, widths=[42 * mm, 25 * mm, 99 * mm]),
            Spacer(1, 5 * mm),
            _para("推奨する目的変数", styles["h2"]),
            _table(
                [
                    ["Horizon", "起点", "Label", "補助評価"],
                    [
                        "6M",
                        "開示翌営業日終値",
                        "6M後のTOPIX/業種超過リターン",
                        "Hit率, IC, Turnover, Drawdown",
                    ],
                    [
                        "12M",
                        "開示翌営業日終値",
                        "12M後のTOPIX/業種超過リターン",
                        "Hit率, IC, Turnover, Drawdown",
                    ],
                ],
                styles,
                widths=[20 * mm, 42 * mm, 66 * mm, 38 * mm],
            ),
            _para("モデル比較順序", styles["h2"]),
            _para(
                "Rule Rank v3をBaselineとし、(1) ウェイトGrid/Random Search、(2) Ridge/Elastic Net、(3) LightGBM等の非線形モデルを同じWalk-forward splitで比較する。複雑モデルはIC、上位分位超過収益、安定性、Turnover、説明可能性の全てで改善した場合のみ採用する。",
                styles["body"],
            ),
            _para(
                f"現在のDQ Gate: FAIL {len(dq_fail)}件。qualitative_coverageは導入前WARNであり公開停止条件ではない。",
                styles["small"],
            ),
            PageBreak(),
        ]
    )

    # Governance / reproducibility
    weight_6 = ", ".join(f"{name}={weight:.0%}" for name, weight in WEIGHTS_6M.items())
    weight_12 = ", ".join(f"{name}={weight:.0%}" for name, weight in WEIGHTS_12M.items())
    story.extend(
        [
            _para("8. 再現性・制約・ガバナンス", styles["h1"]),
            _para("再現性情報", styles["h2"]),
            _table(
                [
                    ["項目", "値"],
                    ["Batch ID", manifest.get("batch_run_id", "-")],
                    ["Snapshot directory", run_dir.relative_to(settings.root)],
                    ["Ranking version", ranking_version],
                    ["6M weights", weight_6],
                    ["12M weights", weight_12],
                    ["Input rows", len(model_input)],
                    ["Quality checks", f"{len(quality)} checks / FAIL {len(dq_fail)}"],
                ],
                styles,
                widths=[42 * mm, 124 * mm],
            ),
            _para("主な制約", styles["h2"]),
            _para(
                "・Yahoo Financeは非公式二次ソースであり、本番利用では契約・公式データへの置換が必要。\n"
                "・J-Quants Freeの数値決算は遅延があり、昨日の開示を反映できない。\n"
                f"・定性特徴量は現在{qualitative_count}件で、暫定ウェイトの有効性評価には対象拡大が必要。\n"
                "・現在のEDAは1つの横断面Snapshotで、時系列安定性やRegime変化を表さない。\n"
                "・順位は予測力をまだバックテストしておらず、投資成果を保証しない。",
                styles["body"],
            ),
            _para("継続学習方針", styles["h2"]),
            _para(
                "日次はFeature計算とInferenceのみとし、モデルを自動更新しない。月次にData/Prediction Driftを監視し、6M/12M Labelが成熟した四半期または半年単位で候補モデルを再学習する。新モデルはChampion/Challengerで旧モデルと比較し、承認後に昇格、悪化時はversion単位でロールバックする。",
                styles["body"],
            ),
            Spacer(1, 8 * mm),
            Table(
                [
                    [
                        _para(
                            "次の開発ゲート: TDnet原文取得 → Point-in-Time履歴 → Forward Label → Walk-forward Backtest → 学習モデル比較",
                            styles["callout"],
                        )
                    ]
                ],
                colWidths=[166 * mm],
                style=TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, -1), PALE_TEAL),
                        ("BOX", (0, 0), (-1, -1), 0.8, TEAL),
                        ("LEFTPADDING", (0, 0), (-1, -1), 10),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                        ("TOPPADDING", (0, 0), (-1, -1), 10),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                    ]
                ),
            ),
        ]
    )

    document = SimpleDocTemplate(
        str(output),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=19 * mm,
        bottomMargin=16 * mm,
        title="日本株ランキングモデル 入力データEDAレポート",
        author="Asset AI Adviser PoC",
        subject="Model specification and input data exploratory analysis",
    )
    document.build(story, onFirstPage=_header_footer, onLaterPages=_header_footer)
    return {
        "output": str(output),
        "pages_expected": 9,
        "snapshot_date": snapshot_date,
        "ranking_version": ranking_version,
        "model_input_rows": len(model_input),
        "charts": [str(path) for path in chart_paths.values()],
    }
