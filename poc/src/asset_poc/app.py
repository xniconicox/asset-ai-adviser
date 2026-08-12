from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from asset_poc.config import Settings

LEARNING_FEATURE_LABELS = {
    "return_1m_pct": "直近1か月リターン",
    "return_3m_pct": "直近3か月リターン",
    "return_6m_pct": "直近6か月リターン",
    "return_12m_pct": "直近12か月リターン",
    "momentum_12_1_pct": "12か月−直近1か月モメンタム",
    "volatility_20d_pct": "20日価格変動",
    "volatility_60d_pct": "60日価格変動",
    "downside_volatility_60d_pct": "60日下方変動",
    "max_drawdown_252d_pct": "1年最大下落",
    "high_52w_distance_pct": "52週高値からの距離",
    "log_average_turnover_20d_pct": "売買代金",
    "per_pct": "PER（株価収益率）",
    "pbr_pct": "PBR（株価純資産倍率）",
    "roe_pct": "ROE（自己資本利益率）",
    "equity_ratio_pct": "自己資本比率",
    "operating_margin_pct": "営業利益率",
    "sales_yoy_pct": "売上成長率",
    "operating_profit_yoy_pct": "営業利益成長率",
    "eps_yoy_pct": "EPS成長率",
    "forecast_eps_revision_pct": "会社予想EPS修正",
    "financial_completeness_pct": "財務データ充足度",
}


def _reason_text(payload: object) -> str:
    try:
        reasons = json.loads(str(payload))
    except (TypeError, ValueError, json.JSONDecodeError):
        return "-"
    if not reasons:
        return "-"
    return " / ".join(f"{item['factor']} {float(item['contribution']):+.1f}" for item in reasons)


def _read_optional_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def _latest_learning_run(root: Path) -> Path | None:
    model_root = root / "output" / "models"
    if not model_root.exists():
        return None
    runs = [
        path
        for path in model_root.iterdir()
        if path.is_dir()
        and (path / "6m" / "model.json").exists()
        and (path / "12m" / "model.json").exists()
    ]
    return max(runs, key=lambda path: path.name) if runs else None


def _learning_cache_key(root: Path) -> int:
    run = _latest_learning_run(root)
    if run is None:
        return 0
    files = list(run.rglob("*"))
    files.extend((root / "output" / "training").glob("*.parquet"))
    return max((path.stat().st_mtime_ns for path in files if path.is_file()), default=0)


@st.cache_data(show_spinner=False)
def _load_learning_snapshot(root_value: str, cache_key: int) -> dict[str, object]:
    del cache_key
    root = Path(root_value).resolve()
    run = _latest_learning_run(root)
    if run is None:
        return {}
    if not run.resolve().is_relative_to((root / "output" / "models").resolve()):
        raise ValueError("invalid model artifact path")

    models: dict[str, object] = {}
    for horizon in ("6m", "12m"):
        model_dir = run / horizon
        model = json.loads((model_dir / "model.json").read_text(encoding="utf-8"))
        metrics = json.loads((model_dir / "metrics.json").read_text(encoding="utf-8"))
        models[horizon] = {
            "model": model,
            "metrics": metrics,
            "coefficients": pd.read_csv(model_dir / "coefficients.csv"),
            "predictions": pd.read_parquet(model_dir / "predictions.parquet"),
        }

    dataset_version = str(models["6m"]["model"]["dataset_version"])
    dataset_path = root / "output" / "training" / f"{dataset_version}.parquet"
    dataset = _read_optional_parquet(dataset_path)
    dataset_summary = {}
    if not dataset.empty:
        dataset_summary = {
            "rows": len(dataset),
            "codes": int(dataset["canonical_code"].nunique()),
            "months": int(dataset["evaluation_date"].nunique()),
            "start": str(dataset["evaluation_date"].min()),
            "end": str(dataset["evaluation_date"].max()),
            "labeled_6m": int(dataset["forward_return_6m"].notna().sum()),
            "labeled_12m": int(dataset["forward_return_12m"].notna().sum()),
            "financial_coverage": float(dataset["financial_completeness"].notna().mean()),
        }
    return {"run": run.name, "models": models, "dataset": dataset_summary}


@st.cache_data(show_spinner=False)
def _load_snapshot(published_dir_value: str, pointer_mtime_ns: int) -> dict[str, object]:
    del pointer_mtime_ns  # cache key only
    published_dir = Path(published_dir_value).resolve()
    manifest = json.loads((published_dir / "latest.json").read_text(encoding="utf-8"))
    run_dir = (published_dir / manifest["run_dir"]).resolve()
    if not run_dir.is_relative_to(published_dir):
        raise ValueError("invalid published snapshot path")
    return {
        "manifest": manifest,
        "ranking": pd.read_parquet(run_dir / "ranking_latest.parquet"),
        "prices": pd.read_parquet(run_dir / "price_history.parquet"),
        "financials": pd.read_parquet(run_dir / "financial_history.parquet"),
        "batches": pd.read_parquet(run_dir / "batch_runs.parquet"),
        "steps": pd.read_parquet(run_dir / "batch_steps.parquet"),
        "quality": pd.read_parquet(run_dir / "data_quality.parquet"),
        "price_quality": _read_optional_parquet(run_dir / "price_quality_events.parquet"),
        "coverage": _read_optional_parquet(run_dir / "data_coverage.parquet"),
        "qualitative": _read_optional_parquet(run_dir / "qualitative_latest.parquet"),
    }


settings = Settings()
st.set_page_config(page_title="日本株 魅力度ランキング", layout="wide")
st.title("日本株 魅力度ランキング PoC")
st.caption(
    "株価・決算による既存ルール順位と学習モデル順位。投資判断・売買推奨ではありません。"
)

pointer = settings.published_dir / "latest.json"
if not pointer.exists():
    st.warning("公開Snapshotがありません。先に `asset-poc publish` を実行してください。")
    st.stop()

try:
    snapshot = _load_snapshot(str(settings.published_dir), pointer.stat().st_mtime_ns)
except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
    st.error(f"公開Snapshotを読み込めません: {error}")
    st.stop()

try:
    learning = _load_learning_snapshot(
        str(settings.root), _learning_cache_key(settings.root)
    )
except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
    learning = {"error": str(error)}

manifest = snapshot["manifest"]
ranking = snapshot["ranking"].copy()
prices = snapshot["prices"]
financials = snapshot["financials"]
batches = snapshot["batches"]
steps = snapshot["steps"]
quality = snapshot["quality"]
price_quality = snapshot["price_quality"]
coverage = snapshot["coverage"]
qualitative = snapshot["qualitative"]

target_count = len(ranking)
price_count = prices["canonical_code"].nunique()
financial_count = financials["canonical_code"].nunique() if not financials.empty else 0
latest_snapshot = manifest.get("snapshot_date", "-")

metrics = st.columns(5)
for column, label, value in zip(
    metrics,
    ["対象銘柄", "株価取得", "決算取得", "ランキング", "基準日"],
    [target_count, price_count, financial_count, len(ranking), latest_snapshot],
):
    column.metric(label, f"{value:,}" if isinstance(value, int) else str(value))

if financial_count < target_count:
    st.warning(
        f"決算カバレッジは {financial_count}/{target_count} 社です。未取得銘柄は価格中心・"
        "低Confidenceで順位付けされています。全件バックフィル後に再計算してください。"
    )

tab_ranking, tab_learning, tab_detail, tab_coverage, tab_status = st.tabs(
    ["ランキング", "学習モデル", "銘柄詳細", "データ充足度", "データ状態"]
)

if not ranking.empty:
    ranking["roe_percent"] = ranking["roe"] * 100
    ranking["confidence_percent"] = ranking["confidence"] * 100
    ranking["プラス要因"] = ranking["positive_reasons"].map(_reason_text)
    ranking["マイナス要因"] = ranking["negative_reasons"].map(_reason_text)

with tab_ranking:
    if ranking.empty:
        st.info("ランキングデータがありません。日次処理を確認してください。")
    else:
        group_filter = st.segmented_control(
            "表示", ["すべて", "一般企業", "金融"], default="すべて"
        )
        display = ranking.copy()
        if group_filter != "すべて":
            wanted_group = "financial" if group_filter == "金融" else "general"
            display = display[display["model_group"] == wanted_group]
        display_columns = [
            "rank_12m",
            "rank_6m",
            "canonical_code",
            "company_name",
            "latest_price",
            "per",
            "pbr",
            "roe_percent",
            "confidence_percent",
            "プラス要因",
            "マイナス要因",
        ]
        st.dataframe(
            display[display_columns],
            width="stretch",
            hide_index=True,
            column_config={
                "rank_12m": "12M順位",
                "rank_6m": "6M順位",
                "canonical_code": "コード",
                "company_name": "銘柄名",
                "latest_price": st.column_config.NumberColumn("株価", format="%.1f"),
                "per": st.column_config.NumberColumn("PER", format="%.1f"),
                "pbr": st.column_config.NumberColumn("PBR", format="%.2f"),
                "roe_percent": st.column_config.NumberColumn("ROE", format="%.1f%%"),
                "confidence_percent": st.column_config.ProgressColumn(
                    "データ充足", min_value=0, max_value=100, format="%.0f%%"
                ),
            },
        )

with tab_learning:
    st.subheader("学習モデルの概要")
    st.caption(
        "過去の各月時点で入手できた株価・決算から、6M/12Mの銘柄間順位を予測します。"
        "LLM定性情報はまだ学習入力に含めていません。"
    )
    overview_columns = st.columns(3)
    with overview_columns[0], st.container(border=True):
        st.markdown("#### 1. 使用データ")
        st.write("**株価**: 値動き、価格変動、下落幅、売買代金")
        st.write("**決算**: PER、PBR、ROE、利益率、成長率、会社予想修正")
        st.caption("JPXの現在Universe、Yahoo Finance、J-Quantsを使用")
    with overview_columns[1], st.container(border=True):
        st.markdown("#### 2. 学習する答え")
        st.write("各月末の翌取引日から、**6か月後・12か月後**までの株価リターン")
        st.write("同じ月の銘柄中央値より、どれだけ上か下かを学習")
        st.caption("Yahoo調整後終値ベースの総収益近似")
    with overview_columns[2], st.container(border=True):
        st.markdown("#### 3. モデル")
        st.write("**Ridge回帰**という、係数を安定させた線形モデル")
        st.write("絶対株価ではなく、銘柄間の**相対的な魅力度順位**を出力")
        st.caption("LLM定性情報は未使用。既存Rule Rankとは別モデル")

    with st.expander("データと学習方法をもう少し詳しく見る"):
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "データ": "株価",
                        "取得元": "Yahoo Finance",
                        "主な入力": "1/3/6/12か月リターン、変動率、最大下落、売買代金",
                    },
                    {
                        "データ": "決算",
                        "取得元": "J-Quants",
                        "主な入力": "PER、PBR、ROE、利益率、前年比、予想EPS修正",
                    },
                    {
                        "データ": "対象銘柄",
                        "取得元": "JPX",
                        "主な入力": "現在のTOPIX 500相当492銘柄",
                    },
                ]
            ),
            width="stretch",
            hide_index=True,
        )
        st.markdown(
            """
            - 各月の最終取引日時点で公表済みの情報だけを使用
            - 学習・調整・最終テストを古い順に分割し、未来データを学習へ混ぜない
            - 銘柄ごとの数値は各月内の順位へ変換し、企業規模や単位の差を緩和
            - 正則化の強さは調整期間で選び、最後のテスト期間は成績確認だけに使用
            """
        )
    if learning.get("error"):
        st.error(f"学習成果物を読み込めません: {learning['error']}")
    elif not learning:
        st.info(
            "学習成果物がありません。asset-poc build-training-dataset と "
            "asset-poc train-model --horizon all を実行してください。"
        )
    else:
        dataset_summary = learning["dataset"]
        st.markdown("### 使用できたデータ量")
        if dataset_summary:
            dataset_metrics = st.columns(6)
            dataset_values = [
                ("月次×銘柄", f"{dataset_summary['rows']:,}行"),
                ("対象銘柄", f"{dataset_summary['codes']:,}社"),
                ("過去の評価月", f"{dataset_summary['months']:,}か月"),
                ("6M答えあり", f"{dataset_summary['labeled_6m']:,}行"),
                ("12M答えあり", f"{dataset_summary['labeled_12m']:,}行"),
                ("財務データあり", f"{dataset_summary['financial_coverage'] * 100:.1f}%"),
            ]
            for column, (label, value) in zip(dataset_metrics, dataset_values):
                column.metric(label, value)
            st.caption(
                f"Dataset: {dataset_summary['start']} 〜 {dataset_summary['end']} / "
                f"Model run: {learning['run']}"
            )

        st.warning(
            "現在の492銘柄が過去にも存在したものとして評価しているため、"
            "上場廃止銘柄などが含まれない偏りがあります。売買コスト、"
            "TOPIX・業種平均との差も未反映で、まだ運用判断には使いません。"
        )
        st.markdown("### 過去データでの成績確認")
        horizon_label = st.segmented_control(
            "予測期間", ["6M", "12M"], default="6M", key="learning_horizon"
        )
        horizon = horizon_label.lower()
        selected = learning["models"][horizon]
        model_document = selected["model"]
        model_metrics = selected["metrics"]["model"]
        baseline_metrics = selected["metrics"]["rule_baseline"]

        evaluation_metrics = st.columns(5)
        metric_values = [
            (
                "順位の一致度",
                f"{model_metrics['mean_spearman_ic']:.3f}",
                f"{model_metrics['mean_spearman_ic'] - baseline_metrics['mean_spearman_ic']:+.3f} 既存比",
                "normal",
                "Rank IC。1に近いほど予測順位と実際のリターン順位が一致します。",
            ),
            (
                "上位10%の平均との差",
                f"{model_metrics['top_decile_excess'] * 100:.1f}%",
                f"{(model_metrics['top_decile_excess'] - baseline_metrics['top_decile_excess']) * 100:+.1f}pt",
                "normal",
                "モデル上位10%のリターンから、同じ月の全銘柄平均を引いた値です。",
            ),
            (
                "上位10%の勝率",
                f"{model_metrics['top_decile_excess_win_rate'] * 100:.1f}%",
                f"{(model_metrics['top_decile_excess_win_rate'] - baseline_metrics['top_decile_excess_win_rate']) * 100:+.1f}pt",
                "normal",
                "上位10%が同じ月の全銘柄平均を上回った月の割合です。",
            ),
            (
                "上位と下位の差",
                f"{model_metrics['long_short_spread'] * 100:.1f}%",
                f"{(model_metrics['long_short_spread'] - baseline_metrics['long_short_spread']) * 100:+.1f}pt",
                "normal",
                "予測上位10%と下位10%の平均リターン差です。",
            ),
            (
                "毎月の入替率",
                f"{model_metrics['mean_top_decile_turnover'] * 100:.1f}%",
                f"{(model_metrics['mean_top_decile_turnover'] - baseline_metrics['mean_top_decile_turnover']) * 100:+.1f}pt",
                "inverse",
                "前月の上位10%から入れ替わった銘柄の割合です。低い方が安定的です。",
            ),
        ]
        for column, values in zip(evaluation_metrics, metric_values):
            label, value, delta, delta_color, help_text = values
            column.metric(
                label,
                value,
                delta=delta,
                delta_color=delta_color,
                help=help_text,
            )
        st.caption(
            f"最終テスト: {model_document['split']['test_start'][:10]}から"
            f"{model_metrics['months']}か月。比較は同じ銘柄・同じ期間で実施しています。"
        )
        st.caption(
            "上位10%リターンは重複する将来期間の月次平均で、年率収益ではありません。"
        )
        with st.expander("モデルの技術情報"):
            st.write("アルゴリズム: Ridge回帰")
            st.write(f"正則化係数 alpha: {model_document['selected_alpha']}")
            st.write(f"モデルID: {model_document['model_id']}")
            st.write(
                "入力変換: 毎月の銘柄横断パーセンタイル。"
                "欠損値は中立値0.5で補完。"
            )

        comparison = pd.DataFrame(
            [
                {
                    "指標": "順位の一致度",
                    "学習モデル": f"{model_metrics['mean_spearman_ic']:.3f}",
                    "既存ルール": f"{baseline_metrics['mean_spearman_ic']:.3f}",
                    "意味": "高い方がよい",
                },
                {
                    "指標": "上位10%の平均との差",
                    "学習モデル": f"{model_metrics['top_decile_excess'] * 100:.1f}%",
                    "既存ルール": f"{baseline_metrics['top_decile_excess'] * 100:.1f}%",
                    "意味": "高い方がよい",
                },
                {
                    "指標": "上位と下位の差",
                    "学習モデル": f"{model_metrics['long_short_spread'] * 100:.1f}%",
                    "既存ルール": f"{baseline_metrics['long_short_spread'] * 100:.1f}%",
                    "意味": "高い方がよい",
                },
                {
                    "指標": "毎月の入替率",
                    "学習モデル": (
                        f"{model_metrics['mean_top_decile_turnover'] * 100:.1f}%"
                    ),
                    "既存ルール": (
                        f"{baseline_metrics['mean_top_decile_turnover'] * 100:.1f}%"
                    ),
                    "意味": "低い方が安定",
                },
            ]
        )
        left, right = st.columns([1, 1])
        left.markdown("#### 学習モデルと既存ルールの比較")
        left.dataframe(
            comparison,
            width="stretch",
            hide_index=True,
        )
        left.info(
            "学習モデルは過去の関係から係数を決めます。"
            "既存ルールは人が決めた固定ウェイトです。"
        )

        coefficients = selected["coefficients"].head(12).copy()
        coefficients["方向"] = coefficients["coefficient"].map(
            lambda value: "プラス" if value >= 0 else "マイナス"
        )
        coefficients["要素"] = coefficients["feature"].map(
            LEARNING_FEATURE_LABELS
        ).fillna(coefficients["feature"])
        coefficients = coefficients.sort_values("coefficient")
        right.markdown("#### モデルが重視した要素")
        right.plotly_chart(
            px.bar(
                coefficients,
                x="coefficient",
                y="要素",
                color="方向",
                orientation="h",
                labels={"coefficient": "順位への影響（係数）"},
                color_discrete_map={"プラス": "#2E8B57", "マイナス": "#B22222"},
            ),
            width="stretch",
        )
        right.caption(
            "プラスは値が高いほど順位を上げ、マイナスは順位を下げる方向です。"
            "因果関係を示すものではありません。"
        )

        latest_predictions = selected["predictions"]
        latest_predictions = latest_predictions[
            latest_predictions["split"] == "latest"
        ].copy()
        if latest_predictions.empty:
            st.info("最新月の学習順位がありません。モデルを再学習してください。")
        else:
            published_rank = "rank_6m" if horizon == "6m" else "rank_12m"
            latest_predictions["model_output_percent"] = (
                latest_predictions["prediction"] * 100
            )
            latest_predictions = latest_predictions.merge(
                ranking[
                    [
                        "canonical_code",
                        "company_name",
                        published_rank,
                        "per",
                        "pbr",
                        "roe_percent",
                    ]
                ],
                on="canonical_code",
                how="left",
            ).sort_values("predicted_rank")
            st.markdown(f"### 現在の{horizon_label}候補順位")
            st.caption(
                "モデル順位は、過去に似た特徴を持つ銘柄の結果から算出した相対順位です。"
                "相対スコアは期待収益率や目標株価ではありません。"
            )
            display_count = st.selectbox(
                "表示件数",
                [20, 50, 100, len(latest_predictions)],
                key="learning_display_count",
            )
            prediction_display = latest_predictions.head(display_count)
            st.dataframe(
                prediction_display[
                    [
                        "predicted_rank",
                        published_rank,
                        "canonical_code",
                        "company_name",
                        "model_output_percent",
                        "per",
                        "pbr",
                        "roe_percent",
                    ]
                ],
                width="stretch",
                hide_index=True,
                column_config={
                    "predicted_rank": "モデル順位",
                    published_rank: "既存ルール順位",
                    "canonical_code": "コード",
                    "company_name": "銘柄名",
                    "model_output_percent": st.column_config.NumberColumn(
                        "相対スコア",
                        format="%+.2f%%",
                        help="同じ月の中央値を0としたモデル出力。期待収益率ではありません。",
                    ),
                    "per": st.column_config.NumberColumn("PER", format="%.1f"),
                    "pbr": st.column_config.NumberColumn("PBR", format="%.2f"),
                    "roe_percent": st.column_config.NumberColumn("ROE", format="%.1f%%"),
                },
            )

with tab_detail:
    if ranking.empty:
        st.info("ランキング計算後に銘柄詳細を表示できます。")
    else:
        options = {
            f"{row.canonical_code} {row.company_name}": row.canonical_code
            for row in ranking.itertuples()
        }
        selected_label = st.selectbox("銘柄", options)
        selected_code = options[selected_label]
        detail = ranking[ranking["canonical_code"] == selected_code].iloc[0]
        detail_metrics = st.columns(7)
        values = [
            ("12M順位", int(detail["rank_12m"])),
            ("6M順位", int(detail["rank_6m"])),
            ("株価", f"{detail['latest_price']:,.1f}"),
            ("PER", "-" if pd.isna(detail["per"]) else f"{detail['per']:.1f}"),
            ("PBR", "-" if pd.isna(detail["pbr"]) else f"{detail['pbr']:.2f}"),
            ("ROE", "-" if pd.isna(detail["roe"]) else f"{detail['roe'] * 100:.1f}%"),
            (
                "決算日",
                "-" if pd.isna(detail["disclosure_date"]) else str(detail["disclosure_date"]),
            ),
        ]
        for column, (label, value) in zip(detail_metrics, values):
            column.metric(label, value)

        factor_frame = pd.DataFrame(
            {
                "Factor": [
                    "Valuation",
                    "Quality",
                    "Growth",
                    "Earnings",
                    "Momentum",
                    "Risk",
                    "Qualitative",
                ],
                "Score": [
                    detail["valuation_score"],
                    detail["quality_score"],
                    detail["growth_score"],
                    detail["earnings_score"],
                    detail["momentum_score"],
                    detail["risk_score"],
                    detail.get("qualitative_score", 50.0),
                ],
            }
        )
        left, right = st.columns([1, 2])
        left.plotly_chart(
            px.bar(factor_frame, x="Score", y="Factor", orientation="h", range_x=[0, 100]),
            width="stretch",
        )
        selected_prices = prices[prices["canonical_code"] == selected_code]
        right.plotly_chart(
            px.line(selected_prices, x="trade_date", y="adjusted_close"), width="stretch"
        )
        st.write("プラス要因:", _reason_text(detail["positive_reasons"]))
        st.write("マイナス要因:", _reason_text(detail["negative_reasons"]))

        history_columns = [
            "disclosure_date",
            "current_period_type",
            "sales",
            "operating_profit",
            "net_income",
            "eps",
            "bps",
            "roe",
            "forecast_eps",
        ]
        history = financials[financials["canonical_code"] == selected_code]
        st.subheader("過去の決算")
        st.dataframe(history[history_columns], width="stretch", hide_index=True)

        selected_qualitative = (
            qualitative[qualitative["canonical_code"] == selected_code]
            if not qualitative.empty
            else pd.DataFrame()
        )
        st.subheader("定性分析（LLM構造化）")
        if selected_qualitative.empty:
            st.info("構造化済みの開示文書はありません。定性補正はランキングに未適用です。")
        else:
            latest_qualitative = selected_qualitative.sort_values(
                "disclosure_date", ascending=False
            ).iloc[0]
            qualitative_metrics = st.columns(6)
            qualitative_values = [
                ("総合", detail.get("qualitative_score", 50.0)),
                ("見通し", latest_qualitative["outlook_score"]),
                ("需要", latest_qualitative["demand_score"]),
                ("採算", latest_qualitative["profitability_score"]),
                ("リスク管理", latest_qualitative["risk_control_score"]),
                ("利益の質", latest_qualitative["earnings_quality_score"]),
            ]
            for column, (label, value) in zip(qualitative_metrics, qualitative_values):
                column.metric(label, f"{float(value):.1f}")
            st.write(latest_qualitative["summary"])
            st.caption(
                f"モデル: {latest_qualitative['model']} / "
                f"信頼度: {float(latest_qualitative['confidence']) * 100:.0f}% / "
                f"根拠: {latest_qualitative['source_url']}"
            )
            with st.expander("根拠と抽出結果"):
                st.json(json.loads(latest_qualitative["evidence"]))

with tab_coverage:
    st.subheader("銘柄別データ充足度")
    st.caption(
        "Coreは株価1年・決算3期・ランキングを評価します。拡張充足度は開示原文、"
        "LLM構造化、定性特徴量まで含みます。定性情報がなくても従来順位は維持されます。"
    )
    if coverage.empty:
        st.warning("充足度Snapshotがありません。`asset-poc publish`を再実行してください。")
    else:
        core_complete = int((coverage["core_coverage_pct"] == 100).sum())
        financial_ready = int((coverage["financial_periods"] >= 3).sum())
        source_ready = int((coverage["source_document_count"] > 0).sum())
        llm_ready = int((coverage["analysis_count"] > 0).sum())
        qualitative_used = int((coverage["qualitative_used_in_rank"] > 0).sum())
        coverage_metrics = st.columns(5)
        for column, label, value in zip(
            coverage_metrics,
            ["Core充足", "決算3期以上", "定性原文あり", "LLM構造化済", "順位へ定性反映"],
            [core_complete, financial_ready, source_ready, llm_ready, qualitative_used],
        ):
            column.metric(label, f"{value}/{target_count}")

        sector_coverage = (
            coverage.groupby("sector33_name", dropna=False)[
                ["core_coverage_pct", "extended_coverage_pct"]
            ]
            .mean()
            .reset_index()
            .sort_values("extended_coverage_pct")
        )
        st.plotly_chart(
            px.bar(
                sector_coverage,
                x=["core_coverage_pct", "extended_coverage_pct"],
                y="sector33_name",
                orientation="h",
                barmode="group",
                labels={"value": "充足度 (%)", "sector33_name": "業種", "variable": "区分"},
            ),
            width="stretch",
        )

        only_missing = st.checkbox("不足のある銘柄のみ表示", value=False)
        coverage_display = coverage.copy()
        if only_missing:
            coverage_display = coverage_display[
                (coverage_display["core_coverage_pct"] < 100)
                | (coverage_display["extended_coverage_pct"] < 100)
            ]
        coverage_columns = [
            "canonical_code",
            "company_name",
            "sector33_name",
            "core_coverage_pct",
            "extended_coverage_pct",
            "price_rows",
            "last_price_date",
            "financial_periods",
            "last_financial_date",
            "source_document_count",
            "analysis_count",
            "qualitative_used_in_rank",
        ]
        st.dataframe(
            coverage_display[coverage_columns],
            width="stretch",
            hide_index=True,
            column_config={
                "canonical_code": "コード",
                "company_name": "銘柄名",
                "sector33_name": "業種",
                "core_coverage_pct": st.column_config.ProgressColumn(
                    "Core", min_value=0, max_value=100, format="%d%%"
                ),
                "extended_coverage_pct": st.column_config.ProgressColumn(
                    "拡張", min_value=0, max_value=100, format="%d%%"
                ),
                "price_rows": "株価日数",
                "last_price_date": "最新株価日",
                "financial_periods": "決算回数",
                "last_financial_date": "最新決算日",
                "source_document_count": "定性原文",
                "analysis_count": "LLM構造化",
                "qualitative_used_in_rank": "定性反映",
            },
        )

with tab_status:
    st.caption(
        f"公開: {manifest.get('published_at', '-')} / Batch: {manifest.get('batch_run_id', '-')}"
    )
    st.subheader("品質チェック")
    st.dataframe(quality, width="stretch", hide_index=True)
    st.subheader("価格クリーニング監査")
    if price_quality.empty:
        st.info("価格品質イベントはありません。次回publishで生成されます。")
    else:
        excluded_rows = price_quality.loc[
            price_quality["action"] == "exclude_model_price",
            ["canonical_code", "trade_date", "source"],
        ].drop_duplicates()
        quality_metrics = st.columns(3)
        quality_metrics[0].metric("品質イベント", f"{len(price_quality):,}")
        quality_metrics[1].metric("モデル除外行", f"{len(excluded_rows):,}")
        quality_metrics[2].metric(
            "影響銘柄", f"{excluded_rows['canonical_code'].nunique():,}"
        )
        reason_summary = (
            price_quality.groupby(["reason_code", "severity", "action"])
            .size()
            .rename("rows")
            .reset_index()
            .sort_values("rows", ascending=False)
        )
        st.dataframe(reason_summary, width="stretch", hide_index=True)
        with st.expander("異常行と補正内容"):
            st.dataframe(price_quality, width="stretch", hide_index=True)

    st.subheader("定期実行")
    st.dataframe(batches, width="stretch", hide_index=True)
    st.subheader("Step実行")
    st.dataframe(steps, width="stretch", hide_index=True)
