"""
eda_cli – мини-утилита для EDA CSV-файлов.

Используется:
- на Семинаре 03 как CLI-приложение;
- на Семинаре 04 как библиотека для обёрток (HTTP-сервис и т.п.).
"""

from . import core, viz, datagen, test, flow, models

__all__ = ["core", "viz", "datagen", "test", "flow", "models"]
__version__ = "0.1.0"
