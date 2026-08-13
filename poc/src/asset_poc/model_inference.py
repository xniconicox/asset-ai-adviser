from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from asset_poc.config import Settings
from asset_poc.learning import MODEL_FEATURE_COLUMNS


def _latest_complete_model_run(root: Path) -> Path:
    model_root = root / "output" / "models"
    if not model_root.exists():
        raise FileNotFoundError("Model artifact directory does not exist")
    runs = [
        path
        for path in model_root.iterdir()
        if path.is_dir()
        and (path / "6m" / "model.json").exists()
        and (path / "12m" / "model.json").exists()
    ]
    if not runs:
        raise FileNotFoundError("Complete 6M/12M model run not found")
    return max(runs, key=lambda path: path.name)


def _current_model_inputs(settings: Settings) -> tuple[pd.DataFrame, dict[str, object]]:
    manifest = json.loads(
        (settings.published_dir / "latest.json").read_text(encoding="utf-8")
    )
    published = settings.published_dir / str(manifest["run_dir"])
    frame = pd.read_parquet(published / "model_input_latest.parquet")
    if frame.empty:
        raise RuntimeError("Published model inputs are empty")
    return frame, manifest


def _rank_current_features(frame: pd.DataFrame) -> pd.DataFrame:
    values = frame.copy()
    turnover = pd.to_numeric(values["average_turnover_20d"], errors="coerce")
    values["log_average_turnover_20d"] = np.where(
        turnover >= 0, np.log1p(turnover), np.nan
    )
    numeric = values[MODEL_FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    ranked = numeric.rank(pct=True, method="average").fillna(0.5)
    ranked.columns = [f"{column}_pct" for column in ranked.columns]
    return ranked


def _predict_horizon(
    inputs: pd.DataFrame,
    ranked: pd.DataFrame,
    model_run: Path,
    horizon: str,
) -> pd.DataFrame:
    document = json.loads(
        (model_run / horizon / "model.json").read_text(encoding="utf-8")
    )
    fit = document["deployment_fit"]
    feature_names = list(fit["feature_names"])
    missing = sorted(set(feature_names) - set(ranked.columns))
    if missing:
        raise ValueError(f"Missing model features: {missing}")
    means = np.asarray(fit["means"], dtype=float)
    scales = np.asarray(fit["scales"], dtype=float)
    coefficients = np.asarray(fit["coefficients"], dtype=float)
    if len(feature_names) != len(means) or len(means) != len(coefficients):
        raise ValueError("Model artifact dimensions do not match")
    scales[scales < 1e-12] = 1.0
    matrix = ranked[feature_names].to_numpy(dtype=float)
    prediction = float(fit["intercept"]) + ((matrix - means) / scales) @ coefficients

    rule_rank = f"rank_{horizon}"
    result = inputs[
        [
            "canonical_code",
            "company_name",
            "sector33_name",
            "snapshot_date",
            "price_date",
            "disclosure_date",
            "latest_close",
            "per",
            "pbr",
            "roe",
            "confidence",
            rule_rank,
        ]
    ].copy()
    result = result.rename(columns={rule_rank: "rule_rank"})
    result["model_score"] = prediction
    result["model_rank"] = pd.Series(prediction, index=result.index).rank(
        method="min", ascending=False
    )
    result["horizon"] = horizon
    result["model_id"] = document["model_id"]
    result["model_version"] = document["model_version"]
    result["model_run"] = model_run.name
    return result.sort_values(["model_rank", "canonical_code"]).reset_index(drop=True)


def infer_latest_models(settings: Settings) -> dict[str, object]:
    """Apply the saved deployment models to the latest published daily features."""
    inputs, manifest = _current_model_inputs(settings)
    ranked = _rank_current_features(inputs)
    model_run = _latest_complete_model_run(settings.root)
    return {
        "snapshot_date": str(manifest.get("snapshot_date", inputs["snapshot_date"].max())),
        "published_run": str(manifest["run_dir"]),
        "model_run": model_run.name,
        "6m": _predict_horizon(inputs, ranked, model_run, "6m"),
        "12m": _predict_horizon(inputs, ranked, model_run, "12m"),
    }
