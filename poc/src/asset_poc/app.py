from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from asset_poc.config import Settings
from asset_poc.model_inference import infer_latest_models

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
    return {
        "run": run.name,
        "models": models,
        "dataset": dataset_summary,
    }


def _availability_mask(frame: pd.DataFrame, layer: str) -> pd.Series:
    price_columns = [
        "return_1m", "return_3m", "return_6m", "return_12m",
        "momentum_12_1", "volatility_20d", "volatility_60d",
        "downside_volatility_60d", "max_drawdown_252d",
        "high_52w_distance", "log_average_turnover_20d",
    ]
    financial_columns = [
        "per", "pbr", "roe", "equity_ratio", "operating_margin",
        "sales_yoy", "operating_profit_yoy", "eps_yoy",
        "forecast_eps_revision", "financial_completeness",
    ]
    if layer == "株価日付":
        return frame["price_date"].notna()
    if layer == "決算日付":
        return frame["disclosure_date"].notna()
    if layer == "価格特徴量":
        return frame[price_columns].notna().sum(axis=1) >= 3
    if layer == "財務特徴量":
        return frame[financial_columns].notna().sum(axis=1) >= 1
    if layer == "学習入力":
        return (
            (frame[price_columns].notna().sum(axis=1) >= 3)
            & frame["financial_completeness"].notna()
        )
    if layer == "6Mラベル":
        return frame["forward_return_6m"].notna()
    if layer == "12Mラベル":
        return frame["forward_return_12m"].notna()
    raise ValueError(f"unknown availability layer: {layer}")


def _availability_heatmap(
    training_frame: pd.DataFrame,
    ranking: pd.DataFrame,
    daily_inference: dict[str, object],
    layer: str,
    limit: int,
) -> tuple[object, pd.DataFrame]:
    frame = training_frame.copy()
    frame["evaluation_date"] = pd.to_datetime(frame["evaluation_date"])
    frame["canonical_code"] = frame["canonical_code"].astype(str)
    frame["available"] = _availability_mask(frame, layer).astype(int)
    names = ranking[["canonical_code", "company_name"]].copy()
    names["canonical_code"] = names["canonical_code"].astype(str)
    names = names.drop_duplicates("canonical_code")
    frame = frame.merge(names, on="canonical_code", how="left")
    sort_frame = frame[["canonical_code", "company_name"]].drop_duplicates()
    if daily_inference and not daily_inference.get("error"):
        model_order = daily_inference["6m"][["canonical_code", "model_rank"]].copy()
        model_order["canonical_code"] = model_order["canonical_code"].astype(str)
        sort_frame = sort_frame.merge(model_order, on="canonical_code", how="left")
        sort_frame = sort_frame.sort_values(
            ["model_rank", "canonical_code"], na_position="last"
        )
    else:
        sort_frame = sort_frame.sort_values("canonical_code")
    selected_codes = sort_frame.head(limit)["canonical_code"].tolist()
    frame = frame[frame["canonical_code"].isin(selected_codes)]
    order = {code: index for index, code in enumerate(selected_codes)}
    frame["_row_order"] = frame["canonical_code"].map(order)
    name_map = (
        sort_frame.set_index("canonical_code")["company_name"]
        .fillna("-").astype(str).to_dict()
    )
    frame["ticker_label"] = frame["canonical_code"].map(
        lambda code: f"{code} {name_map.get(code, '-')}"
    )
    frame = frame.sort_values(["_row_order", "evaluation_date"])
    matrix = frame.pivot(
        index="ticker_label", columns="evaluation_date", values="available"
    )
    ordered_labels = [
        f"{code} {name_map.get(code, '-')}" for code in selected_codes
    ]
    matrix = matrix.reindex(index=ordered_labels)
    matrix.columns = [date.strftime("%Y-%m") for date in matrix.columns]
    matrix = matrix.fillna(0)
    figure = px.imshow(
        matrix,
        color_continuous_scale=[[0, "#F1F5F9"], [1, "#2563A6"]],
        zmin=0, zmax=1, aspect="auto",
        labels={"x": "評価月", "y": "銘柄", "color": "有無"},
    )
    figure.update_traces(
        hovertemplate="銘柄: %{y}<br>評価月: %{x}<br>データ: %{z}<extra></extra>",
        xgap=0.5, ygap=0.5,
    )
    figure.update_layout(
        height=max(520, min(1800, 280 + len(matrix.index) * 12)),
        margin={"l": 10, "r": 10, "t": 25, "b": 70},
        coloraxis_colorbar={"tickvals": [0, 1], "ticktext": ["なし", "あり"]},
        xaxis={"side": "bottom", "tickangle": -45,
               "dtick": max(1, len(matrix.columns) // 12)},
        yaxis={"categoryorder": "array", "categoryarray": list(matrix.index)},
    )
    return figure, matrix


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


@st.cache_data(show_spinner=False)
def _load_daily_model_inference(
    root_value: str,
    published_dir_value: str,
    pointer_mtime_ns: int,
    model_cache_key: int,
) -> dict[str, object]:
    del pointer_mtime_ns, model_cache_key
    inference_settings = Settings(
        root=Path(root_value).resolve(),
        published_dir=Path(published_dir_value).resolve(),
    )
    return infer_latest_models(inference_settings)


settings = Settings()
st.set_page_config(page_title="日本株 魅力度ランキング", layout="wide")
st.title("日本株 魅力度ランキング")
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

try:
    daily_inference = _load_daily_model_inference(
        str(settings.root),
        str(settings.published_dir),
        pointer.stat().st_mtime_ns,
        _learning_cache_key(settings.root),
    )
except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
    daily_inference = {"error": str(error)}

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

tab_ranking, tab_learning = st.tabs(["候補ランキング", "比較・評価"])

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

        if daily_inference and not daily_inference.get("error"):
            for horizon in ("6m", "12m"):
                model_rank_lookup = daily_inference[horizon][
                    ["canonical_code", "model_rank", "model_score"]
                ].rename(
                    columns={
                        "model_rank": f"model_rank_{horizon}",
                        "model_score": f"model_score_{horizon}",
                    }
                )
                display = display.merge(model_rank_lookup, on="canonical_code", how="left")

        st.subheader("予測起点と対象")
        info_cols = st.columns(4)
        data_ready = float((display["confidence_percent"] >= 80).mean() * 100) if not display.empty else 0.0
        info_cols[0].metric("起点日", str(latest_snapshot))
        info_cols[1].metric("予測期間", "6M / 12M")
        info_cols[2].metric("対象銘柄", f"{len(display):,}社")
        info_cols[3].metric("十分なデータ率", f"{data_ready:.0f}%")
        st.caption(
            "各銘柄の起点は上記の基準日までに取得済みの株価・決算情報を使い、"
            "その時点から6か月後・12か月後の相対順位を予測しています。"
        )

        st.dataframe(
            display[
                [
                    "rank_12m",
                    "rank_6m",
                    "model_rank_12m" if "model_rank_12m" in display.columns else None,
                    "model_rank_6m" if "model_rank_6m" in display.columns else None,
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
            ],
            width="stretch",
            hide_index=True,
            column_config={
                "rank_12m": "12M順位",
                "rank_6m": "6M順位",
                "model_rank_12m": "モデル12M順位",
                "model_rank_6m": "モデル6M順位",
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
    st.subheader("比較・評価")
    st.caption(
        "モデル評価は、候補ランキングの下に置き、過去テストでの予測性能と説明を確認します。"
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
        if dataset_summary:
            dataset_metrics = st.columns(4)
            dataset_values = [
                ("月次×銘柄", f"{dataset_summary['rows']:,}行"),
                ("対象銘柄", f"{dataset_summary['codes']:,}社"),
                ("6Mラベル", f"{dataset_summary['labeled_6m']:,}行"),
                ("12Mラベル", f"{dataset_summary['labeled_12m']:,}行"),
            ]
            for column, (label, value) in zip(dataset_metrics, dataset_values):
                column.metric(label, value)

        horizon_label = st.segmented_control(
            "予測期間", ["6M", "12M"], default="6M", key="learning_horizon"
        )
        horizon = horizon_label.lower()
        selected = learning["models"][horizon]
        model_metrics = selected["metrics"]["model"]
        baseline_metrics = selected["metrics"]["rule_baseline"]

        latest_predictions = (
            pd.DataFrame()
            if daily_inference.get("error")
            else daily_inference.get(horizon, pd.DataFrame()).copy()
        )
        if latest_predictions.empty:
            st.info("日次モデル推論がありません。公開Snapshotとモデル成果物を確認してください。")
        else:
            latest_predictions["model_output_percent"] = (
                latest_predictions["model_score"] * 100
            )
            latest_predictions = latest_predictions.sort_values("model_rank")
            st.markdown("### モデルの日次推論ランキング")
            display_count = st.selectbox(
                "表示件数",
                [20, 50, 100, len(latest_predictions)],
                key="learning_display_count",
            )
            st.dataframe(
                latest_predictions.head(display_count)[
                    [
                        "model_rank",
                        "rule_rank",
                        "canonical_code",
                        "company_name",
                        "model_output_percent",
                    ]
                ],
                width="stretch",
                hide_index=True,
                column_config={
                    "model_rank": "モデル順位",
                    "rule_rank": "既存ルール順位",
                    "canonical_code": "コード",
                    "company_name": "銘柄名",
                    "model_output_percent": st.column_config.NumberColumn(
                        "相対スコア",
                        format="%+.2f%%",
                        help="相対スコア。期待リターンではありません。",
                    ),
                },
            )

        st.markdown("### 過去データでの比較")
        eval_cols = st.columns(4)
        eval_values = [
            (
                "順位の一致度",
                f"{model_metrics['mean_spearman_ic']:.3f}",
                f"Rule {baseline_metrics['mean_spearman_ic']:.3f}",
            ),
            (
                "Top10%超過",
                f"{model_metrics['top_decile_excess'] * 100:.1f}%",
                f"Rule {baseline_metrics['top_decile_excess'] * 100:.1f}%",
            ),
            (
                "上位-下位差",
                f"{model_metrics['long_short_spread'] * 100:.1f}%",
                f"Rule {baseline_metrics['long_short_spread'] * 100:.1f}%",
            ),
            (
                "入替率",
                f"{model_metrics['mean_top_decile_turnover'] * 100:.1f}%",
                f"Rule {baseline_metrics['mean_top_decile_turnover'] * 100:.1f}%",
            ),
        ]
        for col, (label, value, rule_value) in zip(eval_cols, eval_values):
            col.metric(label, value, delta=rule_value)

        return_table = pd.DataFrame(
            [
                {
                    "指標": "Top10%平均リターン",
                    "モデル": f"{model_metrics.get('top_decile_return', 0.0) * 100:.1f}%",
                    "ルール": f"{baseline_metrics.get('top_decile_return', 0.0) * 100:.1f}%",
                    "差分": f"{(model_metrics.get('top_decile_return', 0.0) - baseline_metrics.get('top_decile_return', 0.0)) * 100:+.1f}pt",
                },
                {
                    "指標": "全体平均リターン",
                    "モデル": f"{model_metrics.get('universe_return', 0.0) * 100:.1f}%",
                    "ルール": f"{baseline_metrics.get('universe_return', 0.0) * 100:.1f}%",
                    "差分": f"{(model_metrics.get('universe_return', 0.0) - baseline_metrics.get('universe_return', 0.0)) * 100:+.1f}pt",
                },
                {
                    "指標": "Top10%超過",
                    "モデル": f"{model_metrics.get('top_decile_excess', 0.0) * 100:.1f}%",
                    "ルール": f"{baseline_metrics.get('top_decile_excess', 0.0) * 100:.1f}%",
                    "差分": f"{(model_metrics.get('top_decile_excess', 0.0) - baseline_metrics.get('top_decile_excess', 0.0)) * 100:+.1f}pt",
                },
                {
                    "指標": "上位-下位差",
                    "モデル": f"{model_metrics.get('long_short_spread', 0.0) * 100:.1f}%",
                    "ルール": f"{baseline_metrics.get('long_short_spread', 0.0) * 100:.1f}%",
                    "差分": f"{(model_metrics.get('long_short_spread', 0.0) - baseline_metrics.get('long_short_spread', 0.0)) * 100:+.1f}pt",
                },
            ]
        )
        st.markdown("### リターン比較")
        st.dataframe(return_table, width="stretch", hide_index=True)

        with st.expander("モデルの説明"):
            st.write(
                "モデルは、評価時点で利用可能だった価格・財務情報から、6M/12M後の相対順位を学習した比較用の予測モデルです。"
            )
            st.write(
                "本画面の比較指標は、過去のテスト期間での順位一致度・Top10%超過・長短差を確認するためのものです。"
            )
            st.write(
                "最新の候補ランキングと並べて見ることで、ルールベース順位とのズレや有望銘柄の違いを確認できます。"
            )

        st.caption(
            "評価対象の月次データは、各月末時点で利用可能だった情報だけを使っており、未来のデータはラベル作成に含めません。"
        )
