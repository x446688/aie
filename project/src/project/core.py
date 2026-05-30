from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import matplotlib
import numpy as np
import pandas as pd
import torch
from pandas.api import types as ptypes
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import load_config, resolve_path
from .flow import (
    TimeSeriesDataset,
    evaluate_loss,
    fit_model,
    inverse_scale,
    make_windows,
    naive_baseline_from_windows,
    predict_model,
    regression_metrics,
    set_seed,
    temporal_split,
    train_one_epoch,
)
from .models import GRUForecaster, LSTMForecaster
from logging.handlers import RotatingFileHandler

log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

if not logging.getLogger("project.core").handlers:
    handler = RotatingFileHandler(
        log_dir / "core.log", maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    logger = logging.getLogger("project.core")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

plt.style.use("seaborn-v0_8-darkgrid")
plt.rcParams["font.size"] = 9


def _to_python(obj):
    if isinstance(obj, dict):
        return {k: _to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_to_python(v) for v in obj]
    elif isinstance(obj, (np.floating, np.integer)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _plot_training_history(
    histories: Dict[str, Dict[str, List[float]]],
    output_path: Path,
    model_names: Optional[List[str]] = None,
) -> Path:
    if not histories:
        return output_path

    model_names = model_names or list(histories.keys())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax1 = axes[0]
    for name in model_names:
        if name in histories and "train_loss" in histories[name]:
            ax1.plot(
                histories[name]["train_loss"],
                label=f"{name.upper()} train",
                linewidth=1.5,
                alpha=0.8,
            )
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss (MSE)")
    ax1.set_title("Training Loss")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    for name in model_names:
        if name in histories and "val_loss" in histories[name]:
            ax2.plot(
                histories[name]["val_loss"],
                label=f"{name.upper()} val",
                linewidth=1.5,
                alpha=0.8,
            )
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss (MSE)")
    ax2.set_title("Validation Loss")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved training history plot to {output_path}")
    return output_path


def _plot_predictions_vs_actual(
    y_true: np.ndarray,
    y_pred: Dict[str, np.ndarray],
    output_path: Path,
    dates: Optional[pd.Series] = None,
    max_points: int = 500,
) -> Path:
    if len(y_true) > max_points:
        step = len(y_true) // max_points
        y_true = y_true[::step]
        y_pred = {k: v[::step] for k, v in y_pred.items()}
        if dates is not None:
            dates = dates.iloc[::step].reset_index(drop=True)

    n_models = len(y_pred)
    fig, axes = plt.subplots(n_models + 1, 1, figsize=(12, 3 * (n_models + 1)))

    ax0 = axes[0] if n_models > 0 else axes
    x_axis = dates if dates is not None else np.arange(len(y_true))
    ax0.plot(x_axis, y_true, label="Actual", color="black", linewidth=1, alpha=0.7)
    ax0.set_ylabel("Demand (MW)")
    ax0.set_title("Actual Values")
    ax0.grid(True, alpha=0.3)
    if dates is not None:
        ax0.tick_params(axis="x", rotation=45)

    for idx, (name, preds) in enumerate(y_pred.items(), start=1):
        ax = axes[idx] if n_models > 0 else axes
        ax.plot(x_axis, y_true, label="Actual", color="gray", linewidth=0.5, alpha=0.4)
        ax.plot(x_axis, preds, label=f"{name.upper()} prediction", linewidth=1)
        ax.set_ylabel("Demand (MW)")
        ax.set_title(f"{name.upper()} Model")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        if dates is not None:
            ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved predictions plot to {output_path}")
    return output_path


def _plot_residuals(
    y_true: np.ndarray,
    y_pred: Dict[str, np.ndarray],
    output_path: Path,
    max_points: int = 500,
) -> Path:
    if len(y_true) > max_points:
        step = len(y_true) // max_points
        y_true = y_true[::step]
        y_pred = {k: v[::step] for k, v in y_pred.items()}

    n_models = len(y_pred)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
    if n_models == 1:
        axes = [axes]

    for ax, (name, preds) in zip(axes, y_pred.items()):
        residuals = y_true - preds
        ax.hist(residuals, bins=30, edgecolor="black", alpha=0.7)
        ax.axvline(0, color="red", linestyle="--", linewidth=1, label="Zero error")
        ax.set_xlabel("Residual (Actual - Predicted)")
        ax.set_ylabel("Frequency")
        ax.set_title(f"{name.upper()} Residuals")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved residuals plot to {output_path}")
    return output_path


def _plot_model_comparison(
    metrics: Dict[str, Dict[str, float]],
    output_path: Path,
    metric: str = "RMSE",
) -> Path:
    models = list(metrics.keys())
    values = [metrics[m].get(metric, 0) for m in models]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#2b6cb0", "#2c5282", "#38a169", "#2f855a"]
    bars = ax.barh(models, values, color=colors[: len(models)])

    best_idx = np.argmin(values)
    bars[best_idx].set_color("#e53e3e")

    ax.set_xlabel(metric)
    ax.set_title(f"Model Comparison by {metric} (lower is better)")
    ax.grid(axis="x", alpha=0.3)

    for bar, val in zip(bars, values):
        ax.text(
            val + max(values) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}",
            va="center",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved model comparison plot to {output_path}")
    return output_path


@dataclass
class ColumnSummary:
    name: str
    dtype: str
    non_null: int
    zeros: int
    zeros_share: float
    missing: int
    missing_share: float
    unique: int
    example_values: List[Any]
    is_numeric: bool
    min: Optional[float] = None
    max: Optional[float] = None
    mean: Optional[float] = None
    std: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DatasetSummary:
    n_rows: int
    n_cols: int
    duplicates: int
    duplicates_share: float
    columns: List[ColumnSummary]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "columns": [c.to_dict() for c in self.columns],
        }


def clean_timeseries(
    df: pd.DataFrame, col_map: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    """Standardize dataframe to ['date', 'value'] columns."""
    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    df = df.copy()

    if col_map:
        df = df.rename(columns=col_map)

    if len(df.columns) != 2:
        raise ValueError(f"Expected exactly 2 columns, but got {len(df.columns)}.")

    cols = df.columns.tolist()
    date_col = None
    value_col = None

    if "date" in cols and "value" in cols:
        date_col, value_col = "date", "value"
    else:
        dt_candidates = df.select_dtypes(include=["datetime", "datetimetz"]).columns
        if len(dt_candidates) > 0:
            date_col = dt_candidates[0]
        else:
            for c in df.select_dtypes(include=["object", "string"]).columns:
                try:
                    pd.to_datetime(df[c].dropna().head(5))
                    date_col = c
                    break
                except Exception:
                    continue

        num_candidates = df.select_dtypes(include=["number"]).columns
        if len(num_candidates) > 0:
            value_col = num_candidates[0]

        if date_col is None or value_col is None:
            raise ValueError(
                "Could not auto-detect 'date' and 'value' columns. "
                "Provide an explicit `col_map` (e.g., {'timestamp': 'date', 'price': 'value'})."
            )

    df = df.rename(columns={date_col: "date", value_col: "value"})
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"])

    if df["date"].isna().any():
        raise ValueError("Unparseable dates found in the 'date' column.")
    if df["value"].isna().any():
        raise ValueError("Non-numeric or missing values found in the 'value' column.")

    return df.sort_values("date").reset_index(drop=True)[["date", "value"]]


def summarize_dataset(
    df: pd.DataFrame, example_values_per_column: int = 3
) -> DatasetSummary:
    """Generate dataset summary statistics."""
    n_rows, n_cols = df.shape
    columns: List[ColumnSummary] = []

    duplicates = int(df.duplicated().sum())
    duplicates_share = float(duplicates / n_rows) if n_rows > 0 else 0.0

    for name in df.columns:
        s = df[name]
        dtype_str = str(s.dtype)
        non_null = int(s.notna().sum())
        zeros = int(s.eq(0, fill_value=0).sum())
        zeros_share = float(zeros / n_rows) if n_rows > 0 else 0.0
        missing = n_rows - non_null
        missing_share = float(missing / n_rows) if n_rows > 0 else 0.0
        unique = int(s.nunique(dropna=True))
        examples = (
            s.dropna().astype(str).unique()[:example_values_per_column].tolist()
            if non_null > 0
            else []
        )

        is_numeric = bool(ptypes.is_numeric_dtype(s))
        min_val = max_val = mean_val = std_val = None
        if is_numeric and non_null > 0:
            min_val = float(s.min())
            max_val = float(s.max())
            mean_val = float(s.mean())
            std_val = float(s.std())

        columns.append(
            ColumnSummary(
                name=name,
                dtype=dtype_str,
                non_null=non_null,
                zeros=zeros,
                zeros_share=zeros_share,
                missing=missing,
                missing_share=missing_share,
                unique=unique,
                example_values=examples,
                is_numeric=is_numeric,
                min=min_val,
                max=max_val,
                mean=mean_val,
                std=std_val,
            )
        )

    return DatasetSummary(
        n_rows=n_rows,
        n_cols=n_cols,
        columns=columns,
        duplicates=duplicates,
        duplicates_share=duplicates_share,
    )


def value_table(df: pd.DataFrame, value) -> pd.DataFrame:
    """Count occurrences of a specific value (or None for missing) per column."""
    if df.empty:
        return pd.DataFrame(columns=["value_count", "value_share"])
    total = (
        df.isna().sum() if (value is None or np.isnan(value)) else df.eq(value).sum()
    )
    share = total / len(df)
    return pd.DataFrame({"value_count": total, "value_share": share}).sort_values(
        "value_share", ascending=False
    )


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation for numeric columns."""
    numeric_df = df.select_dtypes(include="number")
    return (
        numeric_df.corr(numeric_only=True) if not numeric_df.empty else pd.DataFrame()
    )


def compute_quality_flags(
    summary: DatasetSummary,
    missing_df: pd.DataFrame,
    zeros_df: pd.DataFrame,
    min_missing_share: int = 0.5,
    min_duplicates_share: int = 0.2,
    min_zeros_share: int = 0.9,
) -> Dict[str, Any]:
    """Simple heuristics for data quality assessment."""
    flags: Dict[str, Any] = {}
    flags["too_few_rows"] = summary.n_rows < 100
    flags["too_many_columns"] = summary.n_cols > 100

    max_missing_share = (
        float(missing_df["value_share"].max()) if not missing_df.empty else 0.0
    )
    max_zeros_share = (
        float(zeros_df["value_share"].max()) if not zeros_df.empty else 0.0
    )

    flags["max_missing_share"] = max_missing_share
    flags["max_zeros_share"] = max_zeros_share
    flags["too_many_zeros"] = max_zeros_share > min_zeros_share
    flags["too_many_missing"] = max_missing_share > min_missing_share
    flags["duplicates_share"] = summary.duplicates_share
    flags["too_many_duplicates"] = summary.duplicates_share > min_duplicates_share

    suspicious_id_duplicates = max(
        0, summary.n_rows - summary.columns[0].unique - summary.duplicates
    )
    flags["suspicious_id_duplicates"] = suspicious_id_duplicates

    score = 1.0
    score -= max_missing_share
    score -= max_zeros_share if max_zeros_share > min_zeros_share else 0
    score -= summary.duplicates_share
    score -= suspicious_id_duplicates * 0.01
    if summary.n_rows < 100:
        score -= 0.2
    if summary.n_cols > 100:
        score -= 0.1
    score = max(0.0, min(1.0, score))
    flags["quality_score"] = score
    return flags


def flatten_summary_for_print(summary: DatasetSummary) -> pd.DataFrame:
    """Convert DatasetSummary to DataFrame for printing."""
    rows: List[Dict[str, Any]] = []
    for col in summary.columns:
        rows.append(
            {
                "name": col.name,
                "dtype": col.dtype,
                "non_null": col.non_null,
                "missing": col.missing,
                "missing_share": col.missing_share,
                "zeros": col.zeros,
                "zeros_share": col.zeros_share,
                "unique": col.unique,
                "is_numeric": col.is_numeric,
                "min": col.min,
                "max": col.max,
                "mean": col.mean,
                "std": col.std,
            }
        )
    return pd.DataFrame(rows)


def top_categories(
    df: pd.DataFrame,
    max_columns: int = 5,
    top_k: int = 5,
) -> Dict[str, pd.DataFrame]:
    """Count top-k values for categorical/string columns."""
    result: Dict[str, pd.DataFrame] = {}
    candidate_cols: List[str] = []

    for name in df.columns:
        s = df[name]
        if ptypes.is_object_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
            candidate_cols.append(name)

    for name in candidate_cols[:max_columns]:
        s = df[name]
        vc = s.value_counts(dropna=True).head(top_k)
        if vc.empty:
            continue
        share = vc / vc.sum()
        table = pd.DataFrame(
            {
                "value": vc.index.astype(str),
                "count": vc.values,
                "share": share.values,
            }
        )
        result[name] = table

    return result


def train(
    df: pd.DataFrame,
    window_size: Optional[int] = None,
    dataset_name: str = "default",
    config: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Train all models, compare metrics, save best + visualizations."""
    cfg = config or load_config()
    model_cfg = cfg.get("model", {})
    split_cfg = cfg.get("split", {})
    train_cfg = cfg.get("training", {})
    artifacts_cfg = cfg.get("artifacts", {})

    set_seed(train_cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on device: {device}")

    df = clean_timeseries(df)
    logger.info(
        f"Data shape: {df.shape}, date range: {df['date'].iloc[0]} → {df['date'].iloc[-1]}"
    )

    train_df, val_df, test_df = temporal_split(
        df, split_cfg.get("train", 0.7), split_cfg.get("val", 0.15)
    )

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_df[["value"]]).astype(np.float32)
    val_scaled = scaler.transform(val_df[["value"]]).astype(np.float32)
    test_scaled = scaler.transform(test_df[["value"]]).astype(np.float32)

    logger.info(f"Scaler: mean={scaler.mean_[0]:.2f}, std={scaler.scale_[0]:.2f}")

    ws = window_size or model_cfg.get("window_size", 28)
    X_train, y_train = make_windows(train_scaled, ws)
    X_val, y_val = make_windows(val_scaled, ws)
    X_test, y_test = make_windows(test_scaled, ws)

    batch_size = model_cfg.get("lstm", {}).get("batch_size", 64)
    train_loader = DataLoader(
        TimeSeriesDataset(X_train, y_train), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(TimeSeriesDataset(X_val, y_val), batch_size=batch_size)
    test_loader = DataLoader(TimeSeriesDataset(X_test, y_test), batch_size=batch_size)

    results = {}
    artifacts_dir = resolve_path(artifacts_cfg.get("dir", "artifacts"), cfg)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Training naive baseline...")
    naive_pred = naive_baseline_from_windows(X_test)
    naive_pred_orig = inverse_scale(naive_pred, scaler)
    naive_true_orig = inverse_scale(y_test, scaler)
    results["naive"] = {
        "metrics": regression_metrics(naive_true_orig, naive_pred_orig),
        "predictions": naive_pred_orig.tolist()[:100],
    }
    logger.info(f"Naive metrics: {results['naive']['metrics']}")

    logger.info("Training Ridge baseline...")
    ridge_cfg = model_cfg.get("ridge", {})

    def make_lag_features(series: np.ndarray, lags: List[int]) -> np.ndarray:
        features = []
        for lag in lags:
            if lag < len(series):
                features.append(np.roll(series, lag, axis=0))
                features[-1][:lag] = np.nan
        return np.concatenate(features, axis=1)

    lags = [1, 7, 14, 21, 28]
    X_train_ridge = make_lag_features(train_scaled, lags)
    X_test_ridge = make_lag_features(test_scaled, lags)

    mask = ~np.isnan(X_train_ridge).any(axis=1)
    X_train_ridge = X_train_ridge[mask]
    y_train_ridge = train_scaled[mask].ravel()

    mask_test = ~np.isnan(X_test_ridge).any(axis=1)
    X_test_ridge = X_test_ridge[mask_test]
    y_test_ridge = test_scaled[mask_test].ravel()

    ridge = Ridge(
        alpha=ridge_cfg.get("alpha", 1.0), max_iter=ridge_cfg.get("max_iter", 1000)
    )
    ridge.fit(X_train_ridge, y_train_ridge)
    ridge_pred_scaled = ridge.predict(X_test_ridge)
    ridge_pred_orig = inverse_scale(ridge_pred_scaled.reshape(-1, 1), scaler)
    ridge_true_orig = inverse_scale(y_test_ridge.reshape(-1, 1), scaler)
    results["ridge"] = {
        "metrics": regression_metrics(ridge_true_orig, ridge_pred_orig),
        "model": ridge,
    }
    logger.info(f"Ridge metrics: {results['ridge']['metrics']}")

    logger.info("Training LSTM...")
    lstm_cfg = model_cfg.get("lstm", {})
    lstm_model = LSTMForecaster(
        input_size=int(model_cfg.get("input_size", 1)),
        hidden_size=int(lstm_cfg.get("hidden_size", 64)),
        num_layers=int(lstm_cfg.get("num_layers", 2)),
        dropout=float(lstm_cfg.get("dropout", 0.1)),
    ).to(device)

    lstm_history = fit_model(
        lstm_model,
        train_loader,
        val_loader,
        device=device,
        epochs=int(lstm_cfg.get("epochs", 50)),
        lr=float(lstm_cfg.get("lr", 5e-4)),
        patience=int(lstm_cfg.get("patience", 10)),
    )

    lstm_pred_scaled, _ = predict_model(lstm_model, test_loader, device)
    lstm_pred_orig = inverse_scale(lstm_pred_scaled, scaler)
    lstm_true_orig = inverse_scale(y_test, scaler)
    results["lstm"] = {
        "metrics": regression_metrics(lstm_true_orig, lstm_pred_orig),
        "history": {k: [float(v) for v in vals] for k, vals in lstm_history.items()},
        "model_state": lstm_model.state_dict(),
    }
    logger.info(f"LSTM metrics: {results['lstm']['metrics']}")

    logger.info("Training GRU...")
    gru_cfg = model_cfg.get("gru", {})
    gru_model = GRUForecaster(
        input_size=int(model_cfg.get("input_size", 1)),
        hidden_size=int(gru_cfg.get("hidden_size", 64)),
        num_layers=int(gru_cfg.get("num_layers", 2)),
        dropout=float(gru_cfg.get("dropout", 0.1)),
    ).to(device)

    gru_history = fit_model(
        gru_model,
        train_loader,
        val_loader,
        device=device,
        epochs=int(gru_cfg.get("epochs", 50)),
        lr=float(gru_cfg.get("lr", 5e-4)),
        patience=int(gru_cfg.get("patience", 10)),
    )

    gru_pred_scaled, _ = predict_model(gru_model, test_loader, device)
    gru_pred_orig = inverse_scale(gru_pred_scaled, scaler)
    gru_true_orig = inverse_scale(y_test, scaler)
    results["gru"] = {
        "metrics": regression_metrics(gru_true_orig, gru_pred_orig),
        "history": {k: [float(v) for v in vals] for k, vals in gru_history.items()},
        "model_state": gru_model.state_dict(),
    }
    logger.info(f"GRU metrics: {results['gru']['metrics']}")

    metric_key = train_cfg.get("best_metric", "RMSE")
    model_scores = {
        name: data["metrics"].get(metric_key, float("inf"))
        for name, data in results.items()
        if "metrics" in data
    }
    best_model_name = min(model_scores, key=model_scores.get)
    logger.info(
        f"Лучшая модель по метрике {metric_key}: {best_model_name} ({model_scores[best_model_name]:.4f})"
    )

    deploy_model_name = model_cfg.get("deploy", "best")
    final_model_name = (
        best_model_name if deploy_model_name == "best" else deploy_model_name
    )
    if final_model_name != best_model_name:
        logger.info(f"Using config-specified model: {final_model_name}")

    final_artifact = {
        "model_name": final_model_name,
        "scaler": scaler,
        "config": {
            "window_size": ws,
            "dataset_name": dataset_name,
            "trained_at": datetime.now().isoformat(),
            "all_metrics": {name: data["metrics"] for name, data in results.items()},
        },
        "metrics": results[final_model_name]["metrics"],
    }

    if final_model_name in ["lstm", "gru"]:
        final_artifact["model_state"] = results[final_model_name]["model_state"]
        final_artifact["model_cfg"] = model_cfg.get(final_model_name, {})
    elif final_model_name == "ridge":
        final_artifact["model"] = results["ridge"]["model"]

    artifact_path = artifacts_dir / artifacts_cfg.get("best_model", "best_model.joblib")
    joblib.dump(final_artifact, artifact_path)
    logger.info(f"Модель {artifact_path}")

    # Save metrics JSON
    metrics_summary = {
        "dataset": dataset_name,
        "best_model": best_model_name,
        "deployed_model": final_model_name,
        "metric_used": metric_key,
        "all_metrics": {name: data["metrics"] for name, data in results.items()},
        "trained_at": datetime.now().isoformat(),
    }
    metrics_path = artifacts_dir / artifacts_cfg.get("metrics", "metrics.json")

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.floating, np.integer)):
                return float(obj) if isinstance(obj, np.floating) else int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(metrics_path, "w") as f:
        json.dump(metrics_summary, f, indent=2, cls=NumpyEncoder)
    logger.info(f"Метрики {metrics_path}")

    history_data = {
        name: data.get("history", {})
        for name, data in results.items()
        if "history" in data
    }
    if history_data:
        history_path = artifacts_dir / artifacts_cfg.get(
            "history", "training_history.json"
        )
        with open(history_path, "w") as f:
            json.dump(history_data, f, indent=2)
        logger.info(f"Истрория обучения {history_path}")

    # Save scaler
    scaler_path = artifacts_dir / artifacts_cfg.get("scaler", "scaler.joblib")
    joblib.dump(scaler, scaler_path)
    deep_histories = {
        name: data["history"]
        for name, data in results.items()
        if name in ["lstm", "gru"] and "history" in data
    }
    if deep_histories:
        _plot_training_history(
            deep_histories,
            artifacts_dir / "training_loss_curves.png",
            model_names=list(deep_histories.keys()),
        )

    test_dates = test_df["date"].reset_index(drop=True)
    y_true_test = inverse_scale(y_test, scaler).ravel()
    y_pred_test = {}

    for name, data in results.items():
        if name == "naive":
            y_pred_test[name] = inverse_scale(naive_pred, scaler).ravel()
        elif name == "ridge":
            ridge_pred_scaled = results["ridge"]["model"].predict(X_test_ridge)
            y_pred_test[name] = inverse_scale(
                ridge_pred_scaled.reshape(-1, 1), scaler
            ).ravel()
        elif name in ["lstm", "gru"]:
            ModelClass = LSTMForecaster if name == "lstm" else GRUForecaster
            temp_model = ModelClass(
                input_size=1,
                hidden_size=model_cfg.get(name, {}).get("hidden_size", 64),
                num_layers=model_cfg.get(name, {}).get("num_layers", 2),
                dropout=model_cfg.get(name, {}).get("dropout", 0.1),
            ).to(device)
            temp_model.load_state_dict(data["model_state"])
            temp_model.eval()
            pred_scaled, _ = predict_model(temp_model, test_loader, device)
            y_pred_test[name] = inverse_scale(pred_scaled, scaler).ravel()

    min_len = min(
        len(y_true_test), *(len(v) for v in y_pred_test.values() if v is not None)
    )
    y_true_test = y_true_test[:min_len]
    y_pred_test = {k: v[:min_len] for k, v in y_pred_test.items() if v is not None}
    test_dates_plot = (
        test_dates.iloc[:min_len] if len(test_dates) >= min_len else test_dates
    )

    _plot_predictions_vs_actual(
        y_true_test,
        {k: v for k, v in y_pred_test.items() if k != "naive"},
        dates=test_dates_plot,
        output_path=artifacts_dir / "predictions_vs_actual.png",
    )

    # 3. Residuals
    _plot_residuals(
        y_true_test,
        {k: v for k, v in y_pred_test.items() if k != "naive"},
        output_path=artifacts_dir / "residuals_distribution.png",
    )

    _plot_model_comparison(
        {name: data["metrics"] for name, data in results.items()},
        output_path=artifacts_dir / "model_comparison.png",
        metric=metric_key,
    )

    logger.info(f"Графики сохранены в {artifacts_dir}")

    return {
        "best_model_name": best_model_name,
        "deployed_model": final_model_name,
        "all_metrics": {name: data["metrics"] for name, data in results.items()},
        "artifact_path": str(artifact_path),
        "metrics_path": str(metrics_path),
    }


def predict(df: pd.DataFrame, config: Optional[Dict] = None) -> Dict[str, Any]:
    cfg = config or load_config()
    artifacts_cfg = cfg.get("artifacts", {})

    artifacts_dir = resolve_path(artifacts_cfg.get("dir", "artifacts"), cfg)
    artifact_path = artifacts_dir / artifacts_cfg.get("best_model", "best_model.joblib")

    if not artifact_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {artifact_path}. Run training first."
        )

    artifact = joblib.load(artifact_path)
    model_name = artifact["model_name"]
    scaler = artifact["scaler"]
    model_cfg = artifact.get("model_cfg", {})

    df = clean_timeseries(df)
    ws = artifact["config"]["window_size"]

    if len(df) < ws:
        raise ValueError(f"Need at least {ws} rows for prediction, got {len(df)}")

    series_scaled = scaler.transform(df[["value"]]).astype(np.float32)
    X_last = series_scaled[-ws:].reshape(1, ws, 1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_name in ["lstm", "gru"]:
        ModelClass = LSTMForecaster if model_name == "lstm" else GRUForecaster
        model = ModelClass(
            input_size=1,
            hidden_size=model_cfg.get("hidden_size", 64),
            num_layers=model_cfg.get("num_layers", 2),
            dropout=model_cfg.get("dropout", 0.1),
        ).to(device)
        model.load_state_dict(artifact["model_state"])
        model.eval()

        X_tensor = torch.tensor(X_last, dtype=torch.float32).to(device)
        with torch.no_grad():
            output = model(X_tensor).cpu().numpy()
            pred_scaled = output[0, 0] if output.ndim == 2 else output[0]
    elif model_name == "ridge":

        def make_lag_features(series: np.ndarray, lags: List[int]) -> np.ndarray:
            features = []
            for lag in lags:
                if lag < len(series):
                    features.append(np.roll(series, lag, axis=0))
                    features[-1][:lag] = np.nan
            return np.concatenate(features, axis=1)

        lags = [1, 7, 14, 21, 28]
        X_ridge = make_lag_features(series_scaled, lags)[-1:]
        X_ridge = np.nan_to_num(X_ridge, nan=0.0)
        pred_scaled = artifact["model"].predict(X_ridge)[0]
    else:  # naive
        pred_scaled = series_scaled[-1, 0]

    pred_2d = np.array([[pred_scaled]], dtype=np.float32)
    pred_value = float(scaler.inverse_transform(pred_2d)[0, 0])

    last_date = df["date"].iloc[-1]
    next_date = last_date + pd.Timedelta(days=1)

    logger.info(
        f"Prediction: {pred_value:.2f} for {next_date.date()} "
        f"(model={model_name}, input_rows={len(df)})"
    )

    return {
        "predicted_value": round(pred_value, 2),
        "prediction_date": next_date.isoformat(),
        "model_used": model_name,
        "input_rows": len(df),
    }
