import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

plt.style.use("seaborn-v0_8-darkgrid")
plt.rcParams["font.size"] = 10


def plot_training_history(
    histories: Dict[str, Dict[str, List[float]]],
    output_path: Path,
    model_names: Optional[List[str]] = None,
) -> Path:
    """Plot training/validation loss curves for deep models."""
    if not histories:
        logger.warning("No training histories to plot")
        return output_path

    model_names = model_names or list(histories.keys())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Train loss
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

    # Val loss
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


def plot_predictions_vs_actual(
    y_true: np.ndarray,
    y_pred: Dict[str, np.ndarray],
    output_path: Path,
    dates: Optional[pd.Series] = None,
    max_points: int = 500,
) -> Path:
    """Plot actual vs predicted values for all models."""
    # Subsample if too many points
    if len(y_true) > max_points:
        step = len(y_true) // max_points
        y_true = y_true[::step]
        y_pred = {k: v[::step] for k, v in y_pred.items()}
        if dates is not None:
            dates = dates.iloc[::step].reset_index(drop=True)

    n_models = len(y_pred)
    fig, axes = plt.subplots(n_models + 1, 1, figsize=(12, 3 * (n_models + 1)))

    # Actual values
    ax0 = axes[0] if n_models > 0 else axes
    x_axis = dates if dates is not None else np.arange(len(y_true))
    ax0.plot(x_axis, y_true, label="Actual", color="black", linewidth=1, alpha=0.7)
    ax0.set_ylabel("Demand (MW)")
    ax0.set_title("Actual Values")
    ax0.grid(True, alpha=0.3)
    if dates is not None:
        ax0.tick_params(axis="x", rotation=45)

    # Each model's predictions
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


def plot_residuals(
    y_true: np.ndarray,
    y_pred: Dict[str, np.ndarray],
    output_path: Path,
    max_points: int = 500,
) -> Path:
    """Plot residual distributions for all models."""
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


def plot_model_comparison(
    metrics: Dict[str, Dict[str, float]],
    output_path: Path,
    metric: str = "RMSE",
) -> Path:
    """Bar chart comparing models by a specific metric."""
    models = list(metrics.keys())
    values = [metrics[m].get(metric, 0) for m in models]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(models, values, color=["#2b6cb0", "#2c5282", "#38a169", "#2f855a"])

    best_idx = np.argmin(values)  # Lower is better for RMSE/MAE/MAPE
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
