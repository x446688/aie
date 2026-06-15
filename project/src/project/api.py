from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import joblib
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .config import load_config, resolve_path
from .core import clean_timeseries, predict as core_predict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/api.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("electricity_demand_forecast_api")

config = load_config()
model_cfg = config.get("model", {})
paths_cfg = config.get("paths", {})
service_cfg = config.get("service", {})

app = FastAPI(
    title=config.get("project", {}).get("name", "Electricity demand value forecaster"),
    version="0.3.0",
    description=config.get("project", {}).get(
        "description", "Time-series forecasting API"
    ),
    docs_url="/docs" if service_cfg.get("docs_enabled", True) else None,
    redoc_url=None,
)


class PredictionResponse(BaseModel):
    predicted_value: float = Field(..., description="Forecast in original scale")
    for_date: str = Field(..., description="Prediction date (ISO 8601)")
    model: str = Field(..., description="Model used: lstm/gru/ridge/naive")
    latency_ms: float = Field(..., description="Request processing time (ms)")
    input_rows: int = Field(..., description="Number of rows in input CSV")


class QualityResponse(BaseModel):
    ok_for_model: bool
    quality_score: float
    message: str
    latency_ms: float
    flags: dict[str, bool] | None = None
    dataset_shape: dict[str, int] | None = None


@app.get("/health", tags=["system"])
def health() -> dict[str, str | bool]:
    """Health check: verify artifacts exist."""
    artifacts_dir = resolve_path(
        config.get("artifacts", {}).get("dir", "artifacts"), config
    )
    best_model = artifacts_dir / config.get("artifacts", {}).get(
        "best_model", "best_model.joblib"
    )

    return {
        "status": "ok" if best_model.exists() else "model_missing",
        "version": "0.3.0",
        "model_available": best_model.exists(),
        "artifacts_dir": str(artifacts_dir),
    }


# Стандартная страница
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def ui() -> HTMLResponse:
    """Simple upload interface."""
    return HTMLResponse("""
<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="UTF-8">
  <title>Прогноз спроса электроэнергии (следующий час)</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 700px; margin: 40px auto; padding: 0 20px; }
    h1 { color: #1a365d; }
    form { background: #f7fafc; padding: 20px; border-radius: 8px; margin: 20px 0; }
    input[type="file"] { margin: 10px 0; }
    button { background: #2b6cb0; color: white; border: none; padding: 10px 20px; border-radius: 6px; cursor: pointer; }
    button:hover { background: #2c5282; }
    .hint { color: #718096; font-size: 14px; }
  </style>
</head>
<body>
  <h1>Прогноз значения спроса электроэнергии (следующий час)</h1>
  <p>Загрузите CSV с колонками <strong>дата</strong> и <strong>значение</strong>.</p>
  <form method="post" action="/predict" enctype="multipart/form-data">
    <label>CSV-файл:<br><input type="file" name="file" accept=".csv" required></label><br>
    <button type="submit">Прогнозировать</button>
  </form>
  <p class="hint">API docs: <a href="/docs">/docs</a> | Health: <a href="/health">/health</a></p>
</body>
</html>
    """)


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
async def predict(
    file: UploadFile = File(...),
) -> PredictionResponse:
    start = time.perf_counter()
    logger.info(f"Received prediction request: {file.filename}")

    try:
        df = pd.read_csv(file.file)
    except Exception as e:
        logger.error(f"Failed to read CSV: {e}")
        raise HTTPException(400, f"Failed to read CSV: {e}")

    if df.empty:
        raise HTTPException(400, "CSV file is empty")

    input_rows = len(df)
    logger.debug(f"Loaded {input_rows} rows")

    try:
        result = core_predict(df, config=config)
    except FileNotFoundError as e:
        logger.error(f"Model not found: {e}")
        raise HTTPException(503, "Model not found. Run training first.")
    except ValueError as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.exception(f"Unexpected prediction error: {e}")
        raise HTTPException(500, f"Prediction failed: {e}")

    latency_ms = (time.perf_counter() - start) * 1000
    logger.info(
        f"Prediction complete: {result['predicted_value']:.2f} "
        f"for {result['prediction_date']} (latency={latency_ms:.1f}ms)"
    )

    return PredictionResponse(
        predicted_value=result["predicted_value"],
        for_date=result["prediction_date"],
        model=result["model_used"],
        latency_ms=round(latency_ms, 1),
        input_rows=input_rows,
    )


@app.post("/quality-from-csv", response_model=QualityResponse, tags=["quality"])
async def quality_from_csv(file: UploadFile = File(...)) -> QualityResponse:
    from .core import compute_quality_flags, summarize_dataset, value_table

    start = time.perf_counter()

    if file.content_type not in (
        "text/csv",
        "application/vnd.ms-excel",
        "application/octet-stream",
    ):
        raise HTTPException(400, "Expected CSV file")

    try:
        df = pd.read_csv(file.file)
    except Exception as exc:
        raise HTTPException(400, f"Failed to read CSV: {exc}")

    if df.empty:
        raise HTTPException(400, "CSV file is empty")

    summary = summarize_dataset(df)
    missing_df = value_table(df, None)
    zeros_df = value_table(df, 0)
    flags_all = compute_quality_flags(
        summary,
        missing_df,
        zeros_df,
        min_missing_share=config.get("quality", {}).get("min_missing_share", 0.5),
        min_duplicates_share=config.get("quality", {}).get("min_duplicates_share", 0.2),
        min_zeros_share=config.get("quality", {}).get("min_zeros_share", 0.9),
    )

    score = float(flags_all.get("quality_score", 0.0))
    score = max(0.0, min(1.0, score))
    ok_for_model = score >= 0.7
    message = (
        "Данных достаточно для обучения."
        if ok_for_model
        else "Требуется доработка данных."
    )

    latency_ms = (time.perf_counter() - start) * 1000
    flags_bool = {
        k: bool(v) for k, v in flags_all.items() if isinstance(v, (bool, int, float))
    }

    return QualityResponse(
        ok_for_model=ok_for_model,
        quality_score=round(score, 3),
        message=message,
        latency_ms=round(latency_ms, 1),
        flags=flags_bool,
        dataset_shape={"n_rows": summary.n_rows, "n_cols": summary.n_cols},
    )


@app.get("/config", tags=["system"], include_in_schema=False)
def get_config() -> dict:
    """Return current config (for debugging)."""
    return {
        "project": config.get("project", {}),
        "model": model_cfg,
        "service": service_cfg,
        "artifacts": {
            k: str(resolve_path(v, config))
            for k, v in config.get("artifacts", {}).items()
        },
    }
