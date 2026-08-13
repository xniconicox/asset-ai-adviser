from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from asset_poc.config import Settings


def _configure_matplotlib() -> None:
    font_path = Path("/usr/share/fonts/opentype/ipaexfont-gothic/ipaexg.ttf")
    if font_path.exists():
        plt.rcParams["font.family"] = "IPAexGothic"
    plt.rcParams["axes.unicode_minus"] = False


def _latest_frames(settings: Settings) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pointer = settings.published_dir / "latest.json"
    if not pointer.exists():
        raise FileNotFoundError("Published snapshot not found. Run asset-poc publish first.")
    manifest = json.loads(pointer.read_text(encoding="utf-8"))
    run_dir = (settings.published_dir / manifest["run_dir"]).resolve()
    if not run_dir.is_relative_to(settings.published_dir.resolve()):
        raise ValueError("Invalid published snapshot path")
    return (
        pd.read_parquet(run_dir / "price_history.parquet"),
        pd.read_parquet(run_dir / "financial_history.parquet"),
        pd.read_parquet(run_dir / "model_input_latest.parquet"),
    )


def _monthly_grid(
    prices: pd.DataFrame,
    financials: pd.DataFrame,
    model_input: pd.DataFrame,
    limit: int,
    start: str | None,
    end: str | None,
) -> tuple[list[str], pd.DatetimeIndex, np.ndarray, pd.DataFrame]:
    prices = prices.copy()
    financials = financials.copy()
    model_input = model_input.copy()
    for frame in (prices, financials, model_input):
        frame["canonical_code"] = frame["canonical_code"].astype(str)
    prices["trade_date"] = pd.to_datetime(prices["trade_date"])
    financials["disclosure_date"] = pd.to_datetime(financials["disclosure_date"])

    first_date = pd.Timestamp(start) if start else prices["trade_date"].min()
    last_date = pd.Timestamp(end) if end else prices["trade_date"].max()
    months = pd.date_range(
        first_date.to_period("M").start_time,
        last_date.to_period("M").start_time,
        freq="MS",
    )
    model_input = model_input.sort_values(["rank_6m", "canonical_code"])
    codes = model_input["canonical_code"].drop_duplicates().head(limit).tolist()
    names = model_input.drop_duplicates("canonical_code").set_index("canonical_code")
    labels = [f"{code} {names.loc[code, 'company_name']}" for code in codes]

    prices = prices[
        prices["canonical_code"].isin(codes)
        & prices["trade_date"].between(first_date, last_date)
    ].copy()
    prices["month"] = prices["trade_date"].dt.to_period("M").dt.to_timestamp()
    counts = prices.groupby(["canonical_code", "month"]).size()
    price_grid = np.zeros((len(codes), len(months)), dtype=float)
    month_index = {month: index for index, month in enumerate(months)}
    code_index = {code: index for index, code in enumerate(codes)}
    for (code, month), count in counts.items():
        if month in month_index:
            price_grid[code_index[code], month_index[month]] = count

    financials = financials[
        financials["canonical_code"].isin(codes)
        & financials["disclosure_date"].between(first_date, last_date)
    ].copy()
    financials["month"] = financials["disclosure_date"].dt.to_period("M").dt.to_timestamp()
    events = financials[
        ["canonical_code", "month", "disclosure_date", "current_period_type", "document_type"]
    ].drop_duplicates()
    events = events[events["month"].isin(months)].copy()
    events["label"] = events["current_period_type"].fillna(
        events["document_type"]
    ).astype(str)
    events = events.sort_values(["canonical_code", "disclosure_date"])
    return labels, months, price_grid, events


def generate_data_coverage_chart(
    settings: Settings,
    output: Path | None = None,
    limit: int = 492,
    start: str | None = None,
    end: str | None = None,
) -> dict[str, object]:
    _configure_matplotlib()
    prices, financials, model_input = _latest_frames(settings)
    labels, months, price_grid, events = _monthly_grid(
        prices, financials, model_input, limit, start, end
    )
    if output is None:
        output = settings.root / "output" / "coverage" / "data-coverage-heatmap.png"
    output.parent.mkdir(parents=True, exist_ok=True)

    cmap = LinearSegmentedColormap.from_list(
        "price_days", ["#F4F7FB", "#C7DDF2", "#6FA8D4", "#2563A6"]
    )
    fig_height = max(16, min(36, 5 + len(labels) * 0.055))
    fig = plt.figure(
        figsize=(max(20, len(months) * 0.13), fig_height),
        constrained_layout=True,
    )
    axes = fig.subplot_mosaic(
        [["main", "coverage"]],
        width_ratios=[5.5, 1.4],
        gridspec_kw={"wspace": 0.08},
    )
    fig.patch.set_facecolor("white")

    max_count = max(float(price_grid.max()), 1.0)
    main = axes["main"]
    image = main.imshow(
        price_grid,
        aspect="auto",
        interpolation="none",
        cmap=cmap,
        vmin=0,
        vmax=max_count,
        origin="upper",
        extent=[-0.5, len(months) - 0.5, len(labels) - 0.5, -0.5],
    )
    main.set_title(
        "株価と決算を重ねたデータ充足（銘柄×月）",
        loc="left",
        fontsize=15,
        fontweight="bold",
        color="#17233C",
    )
    main.set_ylabel("銘柄（最新6M順位順）")
    main.set_yticks(np.arange(len(labels)))
    main.set_yticklabels(labels, fontsize=5.5)
    main.grid(False)
    colorbar = fig.colorbar(image, ax=main, fraction=0.012, pad=0.008)
    colorbar.set_label("株価日足の件数/月（青が濃いほど多い）", fontsize=9)

    month_index = {month: index for index, month in enumerate(months)}
    code_index = {label.split(" ", 1)[0]: index for index, label in enumerate(labels)}
    event_x = []
    event_y = []
    for row in events.itertuples():
        if row.month in month_index and row.canonical_code in code_index:
            event_x.append(month_index[row.month])
            event_y.append(code_index[row.canonical_code])
    main.scatter(
        event_x,
        event_y,
        s=12,
        c="#D92D20",
        alpha=0.9,
        marker="o",
        label="決算開示",
    )
    main.legend(loc="upper right", frameon=False, fontsize=9)
    main.set_xlim(-0.5, len(months) - 0.5)
    main.set_xlabel("月（青: 株価日足件数、赤丸: 決算開示イベント）")

    tick_step = max(1, len(months) // 18)
    tick_positions = np.arange(0, len(months), tick_step)
    tick_labels = [months[index].strftime("%Y-%m") for index in tick_positions]
    main.set_xticks(tick_positions)
    main.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)

    price_coverage = (price_grid > 0).mean(axis=1)
    financial_months = events[["canonical_code", "month"]].drop_duplicates()
    financial_counts = financial_months.groupby("canonical_code").size()
    financial_coverage = np.array(
        [financial_counts.get(code, 0) / len(months) for code in code_index]
    )
    coverage = axes["coverage"]
    y_values = np.arange(len(labels))
    coverage.barh(
        y_values, price_coverage, height=0.72, color="#6FA8D4",
        alpha=0.9, label="価格月",
    )
    coverage.scatter(
        financial_coverage, y_values, s=12, color="#D92D20",
        zorder=3, label="決算月",
    )
    coverage.set_xlim(0, 1.08)
    coverage.set_ylim(-0.5, len(labels) - 0.5)
    coverage.set_yticks(y_values)
    coverage.set_yticklabels([])
    coverage.set_xlabel("充足率", fontsize=9)
    coverage.set_title("銘柄別", fontsize=11, color="#17233C")
    coverage.xaxis.set_major_formatter(lambda value, pos: f"{value:.0%}")
    coverage.grid(axis="x", color="#D9E2EC", linewidth=0.5)
    coverage.legend(loc="upper right", frameon=False, fontsize=8)
    fig.text(
        0.01,
        0.005,
        "学習への対応: 青は価格特徴量の元となる日足の存在、赤丸は評価時点までに利用可能な決算開示を示す。"
        "元データは変更せず、公開Snapshotから生成。",
        fontsize=9,
        color="#46515C",
    )
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return {
        "status": "succeeded",
        "output": str(output),
        "tickers": len(labels),
        "months": len(months),
        "price_rows": int(price_grid.sum()),
        "disclosure_events": len(events),
        "date_start": months.min().strftime("%Y-%m") if len(months) else None,
        "date_end": months.max().strftime("%Y-%m") if len(months) else None,
    }
