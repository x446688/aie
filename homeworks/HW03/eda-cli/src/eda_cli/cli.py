from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
import typer

from .core import (
    DatasetSummary,
    compute_quality_flags,
    correlation_matrix,
    flatten_summary_for_print,
    value_table,
    summarize_dataset,
    top_categories,
)
from .viz import (
    plot_correlation_heatmap,
    plot_missing_matrix,
    plot_zeros_matrix,
    plot_histograms_per_column,
    save_top_categories_tables,
)

app = typer.Typer(help="Мини-CLI для EDA CSV-файлов")


def _load_csv(
    path: Path,
    sep: str = ",",
    encoding: str = "utf-8",
) -> pd.DataFrame:
    if not path.exists():
        raise typer.BadParameter(f"Файл '{path}' не найден")
    try:
        return pd.read_csv(path, sep=sep, encoding=encoding)
    except Exception as exc:  # noqa: BLE001
        raise typer.BadParameter(f"Не удалось прочитать CSV: {exc}") from exc

@app.command()
def head(
    path: str = typer.Argument(..., help="Путь к CSV-файлу."),
    n: int = typer.Option(5, help="Количество выбираемых строк"),
    sep: str = typer.Option(",", help="Разделитель в CSV."),
    encoding: str = typer.Option("utf-8", help="Кодировка файла."),
) -> None:
    """
    Напечатать первые n строк датасета
    """
    df = _load_csv(Path(path), sep=sep, encoding=encoding)
    typer.echo(df.head(n).to_string(index=False))

@app.command()
def tail(
    path: str = typer.Argument(..., help="Путь к CSV-файлу."),
    n: int = typer.Option(5, help="Количество выбираемых строк"),
    sep: str = typer.Option(",", help="Разделитель в CSV."),
    encoding: str = typer.Option("utf-8", help="Кодировка файла."),
) -> None:
    """
    Напечатать последние n строк датасета
    """
    df = _load_csv(Path(path), sep=sep, encoding=encoding)
    typer.echo(df.tail(n).to_string(index=False))

@app.command()  
def overview(
    path: str = typer.Argument(..., help="Путь к CSV-файлу."),
    sep: str = typer.Option(",", help="Разделитель в CSV."),
    encoding: str = typer.Option("utf-8", help="Кодировка файла."),
) -> None:
    """
    Напечатать краткий обзор датасета:
    - размеры;
    - типы;
    - простая табличка по колонкам.
    """
    df = _load_csv(Path(path), sep=sep, encoding=encoding)
    summary: DatasetSummary = summarize_dataset(df)
    summary_df = flatten_summary_for_print(summary)

    typer.echo(f"Строк: {summary.n_rows}")
    typer.echo(f"Столбцов: {summary.n_cols}")
    typer.echo(f"Дубликаты строк: {summary.duplicates}")
    typer.echo("\nКолонки:")
    typer.echo(summary_df.to_string(index=False))


@app.command()
def report(
    path: str = typer.Argument(..., help="Путь к CSV-файлу."),
    title: str = typer.Option("EDA-отчёт", help="Заголовок отчёта."),
    out_dir: str = typer.Option("reports", help="Каталог для отчёта."),
    sep: str = typer.Option(",", help="Разделитель в CSV."),
    encoding: str = typer.Option("utf-8", help="Кодировка файла."),
    max_hist_columns: int = typer.Option(5, help="Максимум числовых колонок для гистограмм."),
    min_missing_share: float = typer.Option(0.5, help="Минимальная доля пропущенных значений."),
    min_duplicates_share: float = typer.Option(0.2, help="Минимальная доля дубликатов."),
    min_zeros_share: float = typer.Option(0.9, help="Минимальная доля нулевых значений."),
    max_columns: int = typer.Option(5, help="Максимум категориальных колонок для top-категорий."),
    top_k_categories: int = typer.Option(5, help="Количество top-значений для категориальных признаков.")
) -> None:
    """
    Сгенерировать полный EDA-отчёт:
    - текстовый overview и summary по колонкам (CSV/Markdown);
    - статистика пропусков;
    - корреляционная матрица;
    - top-k категорий по категориальным признакам;
    - картинки: гистограммы, матрица пропусков, heatmap корреляции.
    """
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    df = _load_csv(Path(path), sep=sep, encoding=encoding)

    # 1. Обзор
    summary = summarize_dataset(df)
    summary_df = flatten_summary_for_print(summary)
    missing_df = value_table(df,np.nan)
    zeros_df = value_table(df,0)
    corr_df = correlation_matrix(df)
    top_cats = top_categories(df,max_columns,top_k_categories)

    # 2. Качество в целом
    quality_flags = compute_quality_flags(summary, missing_df, zeros_df, min_missing_share, min_duplicates_share, min_zeros_share)

    # 3. Сохраняем табличные артефакты
    summary_df.to_csv(out_root / "summary.csv", index=False)
    if not missing_df.empty:
        missing_df.to_csv(out_root / "missing.csv", index=True)
    if not zeros_df.empty:
        zeros_df.to_csv(out_root / "zeros.csv", index=True)
    if not corr_df.empty:
        corr_df.to_csv(out_root / "correlation.csv", index=True)
    if summary.duplicates > 0:
        df[df.duplicated()].to_csv(out_root / "duplicates.csv", index=True)
    save_top_categories_tables(top_cats, out_root / "top_categories")

    # 4. Markdown-отчёт
    md_path = out_root / "report.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        f.write(f"Исходный файл: `{Path(path).name}`\n\n")
        f.write(f"Строк: **{summary.n_rows}**, столбцов: **{summary.n_cols}**\n\n")

        f.write("## Качество данных (эвристики)\n\n")
        f.write(f"- Оценка качества: **{quality_flags['quality_score']:.2f}**\n\n")
        f.write(f"- Слишком мало строк: **{quality_flags['too_few_rows']}**\n\n")
        f.write(f"- Слишком много колонок: **{quality_flags['too_many_columns']}**\n\n")
        f.write(f"- Макс. доля пропусков по колонке: **{quality_flags['max_missing_share']:.2%}**\n\n")
        f.write(f"- Доля дубликатов по строкам: **{quality_flags['duplicates_share']:.2%}**\n\n")
        f.write(f"- Макс. доля нулевых значений по колонке: **{quality_flags['max_zeros_share']:.2%}**\n\n")
        f.write(f"- Слишком много пропусков: **{quality_flags['too_many_missing']}**\n\n")
        f.write(f"- Слишком много дубликатов: **{quality_flags['too_many_duplicates']}**\n\n")
        f.write(f"- Слишком много нулевых значений: **{quality_flags['too_many_zeros']}**\n\n")

        f.write("## Заданные пороги для долей недопустимых данных\n\n")
        f.write(f"- Заданный порог для пропусков по колонке: **{min_missing_share:.2%}**\n\n")
        f.write(f"- Заданный порог для дубликатов по строкам: **{min_duplicates_share:.2%}**\n\n")
        f.write(f"- Заданный порог для нулевых значений по колонке: **{min_zeros_share:.2%}**\n\n")

        f.write("## Колонки\n\n")
        f.write("См. файл `summary.csv`.\n\n")

        f.write("## Пропуски\n\n")
        if missing_df.empty:
            f.write("Пропусков нет или датасет пуст.\n\n")
        else:
            f.write("См. файлы `missing.csv` и `missing_matrix.png`.\n\n")

        f.write("## Дубликаты\n\n")
        if summary.duplicates == 0:
            f.write("Дубликатов нет или датасет пуст.\n\n")
        else:
            f.write(f"Количество дубликатов по строкам: {summary.duplicates}.\n\nСм. файл `duplicates.csv`.\n\n")    
        
        f.write("## Нулевые значения\n\n")
        if zeros_df.empty:
            f.write("Нулевых значений нет или датасет пуст.\n\n")
        else:
            f.write("См. файлы `zeros.csv` и `zeros_matrix.png`.\n\n")
        
        f.write("## Корреляция числовых признаков\n\n")
        if corr_df.empty:
            f.write("Недостаточно числовых колонок для корреляции.\n\n")
        else:
            f.write("См. `correlation.csv` и `correlation_heatmap.png`.\n\n")

        f.write("## Категориальные признаки\n\n")
        if not top_cats:
            f.write("Категориальные/строковые признаки не найдены.\n\n")
        else:
            f.write("См. файлы в папке `top_categories/`.\n\n")

        f.write("## Гистограммы числовых колонок\n\n")
        f.write("См. файлы `hist_*.png`.\n")

    # 5. Картинки
    plot_histograms_per_column(df, out_root, max_columns=max_hist_columns)
    plot_missing_matrix(df, out_root / "missing_matrix.png")
    plot_zeros_matrix(df, out_root / "zeros_matrix.png")
    plot_correlation_heatmap(df, out_root / "correlation_heatmap.png")

    typer.echo(f"Отчёт сгенерирован в каталоге: {out_root}")
    typer.echo(f"- Основной markdown: {md_path}")
    typer.echo("- Табличные файлы: summary.csv, missing.csv, duplicates.csv, zeros.csv, correlation.csv, top_categories/*.csv")
    typer.echo("- Графики: hist_*.png, missing_matrix.png, zeros_matrix.png correlation_heatmap.png")


if __name__ == "__main__":
    app()
