# tests/test_core.py
"""
Базовые тесты для проекта прогнозирования спроса на электроэнергию.
Запуск: pytest tests/ -v
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.project.core import clean_timeseries, predict, train
from src.project.flow import make_windows, regression_metrics, set_seed


@pytest.fixture
def sample_df():
    """Синтетический временной ряд для тестов."""
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    values = (
        50
        + np.cumsum(np.random.randn(100) * 0.5)
        + np.sin(np.arange(100) * 2 * np.pi / 7)
    )
    return pd.DataFrame({"timestamp": dates, "demand": values})


@pytest.fixture
def bad_df():
    """Некорректный DataFrame для проверки валидации."""
    return pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6], "col3": [7, 8, 9]})


class TestDataCleaning:
    """Тесты функции clean_timeseries()."""

    def test_auto_detect_columns(self, sample_df):
        """Автоматическое определение колонок даты и значения."""
        result = clean_timeseries(sample_df)
        assert list(result.columns) == ["date", "value"]
        assert pd.api.types.is_datetime64_any_dtype(result["date"])
        assert result["value"].dtype == "float64"
        assert len(result) == 100

    def test_explicit_column_mapping(self, sample_df):
        """Явное указание маппинга колонок."""
        result = clean_timeseries(
            sample_df.rename(columns={"timestamp": "ts", "demand": "val"}),
            col_map={"ts": "date", "val": "value"},
        )
        assert list(result.columns) == ["date", "value"]
        assert result["date"].is_monotonic_increasing

    def test_rejects_wrong_column_count(self, bad_df):
        """Отказ при неверном количестве колонок."""
        with pytest.raises(ValueError, match="Expected exactly 2 columns"):
            clean_timeseries(bad_df)

    def test_sorts_by_date(self, sample_df):
        """Сортировка по дате после очистки."""
        shuffled = sample_df.sample(frac=1, random_state=42).reset_index(drop=True)
        result = clean_timeseries(shuffled)
        assert result["date"].is_monotonic_increasing


class TestMetrics:
    """Тесты функций метрик."""

    def test_regression_metrics_output(self):
        """Проверка формата и типов возвращаемых метрик."""
        y_true = np.array([10, 20, 30, 40])
        y_pred = np.array([11, 19, 31, 39])
        metrics = regression_metrics(y_true, y_pred)

        assert isinstance(metrics, dict)
        assert all(k in metrics for k in ["MAE", "RMSE", "MAPE"])
        assert all(
            isinstance(v, (int, float)) for v in metrics.values()
        )  # Native Python types
        assert metrics["MAE"] == 1.0
        assert metrics["MAPE"] > 0  # Percentage should be positive

    def test_metrics_with_zeros(self):
        """Метрики не падают при нулевых значениях."""
        y_true = np.array([0, 10, 20])
        y_pred = np.array([1, 11, 19])
        metrics = regression_metrics(y_true, y_pred)
        assert all(v >= 0 for v in metrics.values())


class TestWindows:
    """Тесты создания окон для временных рядов."""

    def test_make_windows_shape(self):
        """Проверка формы выходных массивов."""
        series = np.random.randn(50, 1).astype(np.float32)
        X, y = make_windows(series, window_size=10)

        assert X.shape == (40, 10, 1)  # (samples, window, features)
        assert y.shape == (40,) or y.shape == (40, 1)  # Accept both
        # Проверка, что окно сдвигается правильно
        assert np.allclose(X[0, -1, 0], series[9, 0])
        assert np.allclose(X[1, 0, 0], series[1, 0])


class TestReproducibility:
    """Тесты воспроизводимости."""

    def test_set_seed_affects_numpy(self):
        """set_seed() влияет на numpy.random."""
        set_seed(42)
        a = np.random.randn(5)
        set_seed(42)
        b = np.random.randn(5)
        assert np.allclose(a, b)

    def test_set_seed_affects_torch(self):
        """set_seed() влияет на torch.random."""
        import torch

        set_seed(42)
        a = torch.randn(5)
        set_seed(42)
        b = torch.randn(5)
        assert torch.allclose(a, b)
