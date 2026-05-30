import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from pathlib import Path


def gen_base_series(
    n_days: int = 1100, start: str = "2021-01-01", k_sin: int = 2, seed: int = 42
) -> pd.DataFrame:
    """Генерирует реалистичный синтетический временной ряд с несколькими компонентами."""
    random.seed(seed)
    np.random.seed(seed)
    dates = pd.date_range(start=start, periods=n_days, freq="D")
    t = np.arange(n_days)

    # Компоненты временного ряда:
    trend = 60 + 0.04 * t  # Линейный тренд: начальное значение 60, рост 0.04 в день
    weekly = 7 * np.sin(k_sin * np.pi * t / 7) + 2.5 * np.cos(
        k_sin * np.pi * t / 7
    )  # Недельная сезонность
    long_wave = 9 * np.sin(
        k_sin * np.pi * t / 365.25
    )  # Долгосрочная (годовая) сезонность
    noise = np.random.normal(0, 2.7, size=n_days)  # Случайный шум

    # Бонус для выходных дней (суббота и воскресенье)
    dow = pd.Series(dates).dt.dayofweek.to_numpy()
    weekend_bonus = np.where(dow >= 5, 4.5, 0.0)

    # Случайные промо-скачки в спросе
    promo = np.zeros(n_days)
    promo_idx = np.random.choice(np.arange(20, n_days - 20), size=18, replace=False)
    promo[promo_idx] += np.random.uniform(8, 16, size=len(promo_idx))

    # Режимный сдвиг к концу ряда (например, открытие конкурирующего магазина)
    regime_shift = np.where(t > int(n_days * 0.78), 6.0, 0.0)

    # Суммируем все компоненты: тренд + сезонность + шум + события
    y = trend + weekly + long_wave + weekend_bonus + promo + regime_shift + noise
    y = np.maximum(y, 0.1)  # Убеждаемся, что значения положительные

    return pd.DataFrame({"date": dates, "value": y})


def display_series(df: pd.DataFrame, dir: Path):
    if not pd.DataFrame:
        print("DataFrame не должен быть пустым!")
        return
    fig, ax = plt.subplots()
    ax.plot(df["date"], df["value"], lw=1.5, label="value")
    ax.set_title("Синтетический временной ряд: ежедневный спрос")
    ax.set_xlabel("Дата")
    ax.set_ylabel("Значение")
    df.to_csv(dir / "example.csv")
    ax.legend()
    plt.show()
