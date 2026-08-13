from __future__ import annotations

import json

import pandas as pd

from asset_poc.config import Settings
from asset_poc.learning import MODEL_FEATURE_COLUMNS
from asset_poc.model_inference import infer_latest_models


def test_daily_inference_uses_current_published_features(tmp_path) -> None:
    published = tmp_path / "data/published"
    run = published / "runs/run-1"
    run.mkdir(parents=True)
    (published / "latest.json").write_text(
        json.dumps({"run_dir": "runs/run-1", "snapshot_date": "2026-08-13"}),
        encoding="utf-8",
    )
    rows = []
    for code, per, rule_rank in (("1001", 10.0, 2), ("1002", 20.0, 1)):
        row = {column: 1.0 for column in MODEL_FEATURE_COLUMNS}
        row.pop("log_average_turnover_20d")
        row.update(
            {
                "canonical_code": code,
                "company_name": code,
                "sector33_name": "test",
                "snapshot_date": "2026-08-13",
                "price_date": "2026-08-13",
                "disclosure_date": "2026-08-12",
                "latest_close": 100.0,
                "average_turnover_20d": 1_000.0,
                "per": per,
                "pbr": 1.0,
                "roe": 0.1,
                "confidence": 1.0,
                "rank_6m": rule_rank,
                "rank_12m": rule_rank,
            }
        )
        rows.append(row)
    pd.DataFrame(rows).to_parquet(run / "model_input_latest.parquet", index=False)

    feature_names = [f"{column}_pct" for column in MODEL_FEATURE_COLUMNS]
    per_position = feature_names.index("per_pct")
    coefficients = [0.0] * len(feature_names)
    coefficients[per_position] = 1.0
    model_run = tmp_path / "output/models/20260813T000000Z"
    for horizon in ("6m", "12m"):
        directory = model_run / horizon
        directory.mkdir(parents=True)
        (directory / "model.json").write_text(
            json.dumps(
                {
                    "model_id": f"model-{horizon}",
                    "model_version": "ridge-v2",
                    "deployment_fit": {
                        "feature_names": feature_names,
                        "means": [0.0] * len(feature_names),
                        "scales": [1.0] * len(feature_names),
                        "coefficients": coefficients,
                        "intercept": 0.0,
                    },
                }
            ),
            encoding="utf-8",
        )

    result = infer_latest_models(
        Settings(root=tmp_path, published_dir=published)
    )

    assert result["snapshot_date"] == "2026-08-13"
    assert result["model_run"] == "20260813T000000Z"
    assert result["6m"].iloc[0]["canonical_code"] == "1002"
    assert result["12m"].iloc[0]["model_rank"] == 1
