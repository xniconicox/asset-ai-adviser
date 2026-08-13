from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from asset_poc.config import Settings

NAVY = colors.HexColor("#17233C")
BLUE = colors.HexColor("#2563A6")
TEAL = colors.HexColor("#148C8C")
PALE_BLUE = colors.HexColor("#EAF2FA")
PALE_TEAL = colors.HexColor("#E6F5F3")
LIGHT_GREY = colors.HexColor("#F3F5F7")
MID_GREY = colors.HexColor("#AAB4C0")
DARK_GREY = colors.HexColor("#46515C")

FONT_REGULAR = Path("/usr/share/fonts/opentype/ipaexfont-gothic/ipaexg.ttf")
FONT_BOLD = Path("/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf")


def _register_fonts() -> None:
    if not FONT_REGULAR.exists() or not FONT_BOLD.exists():
        raise FileNotFoundError("Japanese IPA fonts are required")
    if "Summary-Regular" not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont("Summary-Regular", str(FONT_REGULAR)))
        pdfmetrics.registerFont(TTFont("Summary-Bold", str(FONT_BOLD)))


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "SummaryTitle",
            parent=base["Title"],
            fontName="Summary-Bold",
            fontSize=24,
            leading=33,
            textColor=NAVY,
            spaceAfter=8 * mm,
        ),
        "subtitle": ParagraphStyle(
            "SummarySubtitle",
            parent=base["Normal"],
            fontName="Summary-Regular",
            fontSize=10,
            leading=16,
            textColor=DARK_GREY,
        ),
        "h1": ParagraphStyle(
            "SummaryH1",
            parent=base["Heading1"],
            fontName="Summary-Bold",
            fontSize=17,
            leading=23,
            textColor=NAVY,
            spaceAfter=4 * mm,
        ),
        "h2": ParagraphStyle(
            "SummaryH2",
            parent=base["Heading2"],
            fontName="Summary-Bold",
            fontSize=12,
            leading=17,
            textColor=BLUE,
            spaceBefore=3 * mm,
            spaceAfter=2 * mm,
        ),
        "body": ParagraphStyle(
            "SummaryBody",
            parent=base["BodyText"],
            fontName="Summary-Regular",
            fontSize=9,
            leading=14,
            textColor=DARK_GREY,
            spaceAfter=2.5 * mm,
        ),
        "small": ParagraphStyle(
            "SummarySmall",
            parent=base["BodyText"],
            fontName="Summary-Regular",
            fontSize=7.3,
            leading=10,
            textColor=DARK_GREY,
        ),
        "table": ParagraphStyle(
            "SummaryTable",
            parent=base["BodyText"],
            fontName="Summary-Regular",
            fontSize=7.1,
            leading=9.3,
            textColor=DARK_GREY,
        ),
        "table_head": ParagraphStyle(
            "SummaryTableHead",
            parent=base["BodyText"],
            fontName="Summary-Bold",
            fontSize=7.2,
            leading=9.3,
            textColor=colors.white,
            alignment=TA_CENTER,
        ),
        "callout": ParagraphStyle(
            "SummaryCallout",
            parent=base["BodyText"],
            fontName="Summary-Bold",
            fontSize=10,
            leading=15,
            textColor=NAVY,
            alignment=TA_CENTER,
        ),
    }


def _p(value: object, style: ParagraphStyle) -> Paragraph:
    return Paragraph(html.escape(str(value)).replace("\n", "<br/>"), style)


def _table(
    rows: list[list[object]],
    styles: dict[str, ParagraphStyle],
    widths: list[float],
) -> Table:
    body = []
    for row_index, row in enumerate(rows):
        style = styles["table_head"] if row_index == 0 else styles["table"]
        body.append([_p(cell, style) for cell in row])
    result = Table(body, colWidths=widths, repeatRows=1, hAlign="LEFT")
    result.setStyle(
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
    return result


def _cards(values: list[tuple[str, str]], styles: dict[str, ParagraphStyle]) -> Table:
    cards = []
    for label, value in values:
        cards.append(
            Table(
                [[_p(value, styles["callout"])], [_p(label, styles["small"])]],
                colWidths=[39 * mm],
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
    return Table([cards], colWidths=[41.5 * mm] * len(cards))


def _footer(canvas, document) -> None:
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#D9E0E8"))
    canvas.line(18 * mm, 13 * mm, 192 * mm, 13 * mm)
    canvas.setFont("Summary-Regular", 7)
    canvas.setFillColor(DARK_GREY)
    canvas.drawString(18 * mm, 8 * mm, "Asset AI Adviser - personal research")
    canvas.drawRightString(192 * mm, 8 * mm, f"{document.page}")
    canvas.restoreState()


def _latest_model_run(root: Path) -> Path:
    runs = [
        path
        for path in (root / "output" / "models").iterdir()
        if path.is_dir()
        and (path / "6m" / "model.json").exists()
        and (path / "12m" / "model.json").exists()
    ]
    if not runs:
        raise FileNotFoundError("Complete 6M/12M model run not found")
    return max(runs, key=lambda path: path.name)


def generate_system_summary(
    settings: Settings, output: Path | None = None
) -> dict[str, object]:
    _register_fonts()
    styles = _styles()
    output = output or settings.root / "output" / "pdf" / "investment_system_summary.pdf"
    output.parent.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(
        (settings.published_dir / "latest.json").read_text(encoding="utf-8")
    )
    published = settings.published_dir / manifest["run_dir"]
    ranking = pd.read_parquet(published / "ranking_latest.parquet")
    financials = pd.read_parquet(published / "financial_history.parquet")
    coverage = pd.read_parquet(published / "data_coverage.parquet")
    model_run = _latest_model_run(settings.root)

    models: dict[str, dict[str, object]] = {}
    for horizon in ("6m", "12m"):
        directory = model_run / horizon
        models[horizon] = {
            "document": json.loads((directory / "model.json").read_text(encoding="utf-8")),
            "metrics": json.loads((directory / "metrics.json").read_text(encoding="utf-8")),
            "coefficients": pd.read_csv(directory / "coefficients.csv"),
        }
    dataset_version = str(models["6m"]["document"]["dataset_version"])
    dataset = pd.read_parquet(
        settings.root / "output" / "training" / f"{dataset_version}.parquet"
    )
    ranking_version = str(ranking["ranking_version"].dropna().iloc[0])
    snapshot_date = str(manifest.get("snapshot_date", "-"))
    generated_at = datetime.now(ZoneInfo("Asia/Tokyo")).strftime("%Y-%m-%d %H:%M JST")

    story: list[object] = [
        _p("日本株投資支援システム\n現状・モデル評価サマリ", styles["title"]),
        _p(
            f"基準日 {snapshot_date} / 作成 {generated_at}\n"
            "個人利用向け。データ取得、ランキング、学習評価、定常運用を集約。",
            styles["subtitle"],
        ),
        Spacer(1, 8 * mm),
        _cards(
            [
                ("対象銘柄", f"{len(ranking):,}社"),
                ("学習行", f"{len(dataset):,}"),
                ("評価月", f"{dataset['evaluation_date'].nunique():,}"),
                ("財務履歴", f"{len(financials):,}行"),
            ],
            styles,
        ),
        Spacer(1, 8 * mm),
        _p("結論", styles["h1"]),
        Table(
            [[_p(
                "価格リークは修正済み。6M Ridgeは月次NAV検証へ進める候補。"
                "12M Ridgeは検証とテストの差が大きく、レジーム依存の確認が必要。"
                "表示値はCAGRや収益保証ではない。",
                styles["callout"],
            )]],
            colWidths=[166 * mm],
            style=TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), PALE_TEAL),
                    ("BOX", (0, 0), (-1, -1), 0.8, TEAL),
                    ("LEFTPADDING", (0, 0), (-1, -1), 10),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                    ("TOPPADDING", (0, 0), (-1, -1), 12),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
                ]
            ),
        ),
        Spacer(1, 6 * mm),
        _p("現在の構成", styles["h2"]),
        _table(
            [
                ["領域", "方式", "状態"],
                ["Universe", "TOPIX Core30・Large70・Mid400相当", f"{len(ranking)}社"],
                ["財務履歴", "J-Quants Standardを初月バックフィル", "ローカル保存"],
                ["日次開示", "TDnet PDF/XBRL/HTML", "Catch-up運用"],
                ["株価", "Yahoo Finance日足", "日次差分"],
                ["ランキング", "Rule Rank + Ridge Challenger", ranking_version],
                ["定性", "TDnet原文をLLMで証拠付き構造化", "補助特徴量"],
            ],
            styles,
            [32 * mm, 87 * mm, 47 * mm],
        ),
        PageBreak(),
        _p("1. データ取得と時点整合性", styles["h1"]),
        _p(
            "取得済みrawは変更しない。取得時刻、URL、ハッシュを残し、"
            "クリーニング・特徴量・学習表はVersion付き派生データとして再生成する。",
            styles["body"],
        ),
        _table(
            [
                ["データ", "初回取得", "定常取得", "用途"],
                ["財務", "J-Quants Standard 約10年", "Free遅延分を補完", "利益・財務・会社予想"],
                ["開示", "TDnet取得可能期間", "毎日Catch-up", "速報・原文・定性"],
                ["株価", "Yahoo period=max", "毎営業日差分", "特徴量・ラベル"],
                ["LLM", "既存未構造化文書", "週次・必要時", "根拠付き定性"],
            ],
            styles,
            [24 * mm, 48 * mm, 49 * mm, 45 * mm],
        ),
        _p("価格系列の分離", styles["h2"]),
        _table(
            [
                ["用途", "系列", "理由"],
                ["PER・PBR", "未調整終値 valuation_price", "将来配当の遡及調整を防ぐ"],
                ["Momentum・Return", "調整後終値 return_price", "配当・分割込み近似"],
                ["6M/12M Label", "調整後終値 return_price", "評価系列を統一"],
            ],
            styles,
            [35 * mm, 62 * mm, 69 * mm],
        ),
        _p("定常処理", styles["h2"]),
        _table(
            [
                ["頻度", "処理", "目的"],
                ["毎日 19:30", "TDnet・Yahoo・遅延財務・Rule Rank・品質検査・公開", "新情報を追記"],
                ["日次処理後", "モデル推論PDF・JSONをOneDriveへ転送", "履歴と最新版"],
                ["毎週", "DuckDBバックアップ・未構造化文書確認", "復旧・費用管理"],
                ["毎月", "Point-in-Time学習表・Challenger再学習", "新ラベル反映"],
                ["四半期", "モデル比較レビュー", "入替判断"],
            ],
            styles,
            [30 * mm, 91 * mm, 45 * mm],
        ),
        Spacer(1, 4 * mm),
        _p(
            f"財務カバレッジ: {int((coverage['financial_periods'] > 0).sum())}/"
            f"{len(coverage)}社。rawと派生データを分離し、再実行しても元データを変更しない。",
            styles["body"],
        ),
        PageBreak(),
        _p("2. モデルと評価", styles["h1"]),
        _p(
            "Ridge回帰は各月の銘柄横断パーセンタイルから、6M/12M先の"
            "Universe中央値に対する相対リターンを学習する。LLM定性は未投入。",
            styles["body"],
        ),
    ]

    evaluation_rows = [[
        "期間", "検証IC", "テストIC", "Rule IC", "上位超過", "勝率", "入替率", "月数", "判定"
    ]]
    for horizon, label in (("6m", "6M"), ("12m", "12M")):
        document = models[horizon]["document"]
        metrics = models[horizon]["metrics"]
        model = metrics["model"]
        rule = metrics["rule_baseline"]
        validation_ic = max(
            candidate["metrics"].get("mean_spearman_ic", float("-inf"))
            for candidate in document["alpha_candidates"]
        )
        evaluation_rows.append([
            label,
            f"{validation_ic:.3f}",
            f"{model['mean_spearman_ic']:.3f}",
            f"{rule['mean_spearman_ic']:.3f}",
            f"{model['top_decile_excess'] * 100:.1f}%",
            f"{model['top_decile_excess_win_rate'] * 100:.1f}%",
            f"{model['mean_top_decile_turnover'] * 100:.1f}%",
            model["months"],
            "NAV検証へ" if horizon == "6m" else "要安定性確認",
        ])
    story.extend([
        _table(
            evaluation_rows,
            styles,
            [13 * mm, 18 * mm, 18 * mm, 17 * mm, 21 * mm, 17 * mm, 17 * mm, 13 * mm, 32 * mm],
        ),
        _p(
            "上位超過は重複する将来リターン窓の月次平均で、年率収益・CAGR・"
            "コスト控除後NAVではない。12Mは検証IC約0.007、テストIC 0.319で期間依存を疑う。",
            styles["body"],
        ),
        _p("指標の意味と目安", styles["h2"]),
        _table(
            [
                ["指標", "意味", "合格目安"],
                ["Rank IC", "予想順位と実現順位の相関", "平均0.05以上、中央値も正"],
                ["ICプラス率", "ICが正だった月の割合", "55%以上を複数Regimeで維持"],
                ["上位10%超過", "上位と同月Universe平均の差", "コスト控除後も正"],
                ["勝率", "上位がUniverseを上回った割合", "55-60%以上を安定維持"],
                ["入替率", "上位銘柄の月次交代割合", "コストとの両立を確認"],
            ],
            styles,
            [34 * mm, 76 * mm, 56 * mm],
        ),
        _p("主要係数", styles["h2"]),
    ])
    coefficient_rows = [["期間", "影響が大きい特徴量（標準化係数）"]]
    for horizon, label in (("6m", "6M"), ("12m", "12M")):
        frame = models[horizon]["coefficients"].head(6)
        coefficient_rows.append([
            label,
            " / ".join(
                f"{row.feature} {row.coefficient:+.3f}" for row in frame.itertuples()
            ),
        ])
    story.extend([
        _table(coefficient_rows, styles, [24 * mm, 142 * mm]),
        PageBreak(),
        _p("3. 判定基準と次の作業", styles["h1"]),
        _p("モデル昇格条件", styles["h2"]),
        _table(
            [
                ["観点", "必要条件"],
                ["予測力", "平均IC 0.05以上、中央値正、Positive IC率55%以上"],
                ["安定性", "複数Regime・Walk-forwardで符号と成績を維持"],
                ["運用成績", "売買コスト控除後NAVがTOPIXを上回る"],
                ["集中リスク", "単一銘柄・業種・1年を除いても成績が残る"],
                ["再現性", "同じDataset/Model Versionから同じ結果を再生成"],
            ],
            styles,
            [35 * mm, 131 * mm],
        ),
        _p("現在の不足", styles["h2"]),
        _p(
            "過去Universe、上場廃止、月次実現NAV、売買コスト、TOPIX・業種・Size因子調整、"
            "複数Walk-forward、ROIC・FCF・Accrual・資本配分、LLM定性の増分評価。",
            styles["body"],
        ),
        _p("優先順位", styles["h2"]),
        _table(
            [
                ["順", "作業", "完了条件"],
                ["1", "月次NAV・翌営業日約定・コスト", "CAGR/DD/Sharpeを算出"],
                ["2", "TOPIX・業種・Size調整", "因子を除いた超過を確認"],
                ["3", "複数Walk-forward", "6M/12Mの安定性を再判定"],
                ["4", "過去Universe・上場廃止", "生存者バイアスを縮小"],
                ["5", "品質・資本配分・LLM特徴量", "Ablationで増分効果を確認"],
            ],
            styles,
            [12 * mm, 78 * mm, 76 * mm],
        ),
        Spacer(1, 8 * mm),
        Table(
            [[_p(
                "現時点: Rule Rankは日次スクリーナー。6M RidgeはChallenger。"
                "12M Ridgeは安定性確認まで採用しない。",
                styles["callout"],
            )]],
            colWidths=[166 * mm],
            style=TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), PALE_BLUE),
                    ("BOX", (0, 0), (-1, -1), 0.8, BLUE),
                    ("TOPPADDING", (0, 0), (-1, -1), 10),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                ]
            ),
        ),
        Spacer(1, 6 * mm),
        _p(
            f"Dataset: {dataset_version}\nModel run: {model_run.name}\n"
            f"Published run: {published.name}\nRanking: {ranking_version}",
            styles["small"],
        ),
    ])

    document = SimpleDocTemplate(
        str(output),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=19 * mm,
        bottomMargin=16 * mm,
        title="日本株投資支援システム 現状・モデル評価サマリ",
        author="Asset AI Adviser",
    )
    document.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return {
        "output": str(output),
        "snapshot_date": snapshot_date,
        "dataset_version": dataset_version,
        "model_run": model_run.name,
        "ranking_version": ranking_version,
    }
