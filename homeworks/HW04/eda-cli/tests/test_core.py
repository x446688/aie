from __future__ import annotations

import pandas as pd

import numpy as np

from eda_cli.core import (
    compute_quality_flags,
    correlation_matrix,
    flatten_summary_for_print,
    value_table,
    summarize_dataset,
    top_categories,
)

def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [10, 20, 30, None],
            "height": [140, 150, 160, 170],
            "city": ["A", "B", "A", None],
        }
    )

def _gross_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": [0,1,2,2,2,3,4],
            "purchases": [None,47,12,12,12,41,None],
            "money_spent":[0,600,0,0,0,144,537],
        }
    )

def _id_dup_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": [0,1,2,3,4,1,2,5,5],
            "purchases": [1,2,3,4,5,6,7,8,8],
            "money_spent":[10,200,3000,40000,5000,600,70,8,8],
        }
    )

def test_summarize_dataset_basic():
    df = _sample_df()
    summary = summarize_dataset(df)

    assert summary.n_rows == 4
    assert summary.n_cols == 3
    assert any(c.name == "age" for c in summary.columns)
    assert any(c.name == "city" for c in summary.columns)

    summary_df = flatten_summary_for_print(summary)
    assert "name" in summary_df.columns
    assert "missing_share" in summary_df.columns


def test_missing_table_and_quality_flags():
    df = _sample_df()
    missing_df = value_table(df,None)
    missing_df = value_table(df,np.nan)
    assert "value_count" in missing_df.columns
    assert missing_df.loc["age", "value_count"] == 1

    summary = summarize_dataset(df)
    flags = compute_quality_flags(summary, missing_df, pd.DataFrame({}))
    assert 0.0 <= flags["quality_score"] <= 1.0


def test_correlation_and_top_categories():
    df = _sample_df()
    corr = correlation_matrix(df)
    # корреляция между age и height существует
    assert "age" in corr.columns or corr.empty is False

    top_cats = top_categories(df, max_columns=5, top_k=2)
    assert "city" in top_cats
    city_table = top_cats["city"]
    assert "value" in city_table.columns
    assert len(city_table) <= 2

def test_zeros_and_duplicates():
    df = _gross_df()
    summary = summarize_dataset(df)
    assert summary.duplicates == 2
    summary_df = flatten_summary_for_print(summary)
    assert "zeros" in summary_df.columns
    assert "zeros_share" in summary_df.columns
    
def test_all_flags():
    df = _gross_df()
    missing_df = value_table(df,None)
    zeros_df = value_table(df,0)
    assert "value_count" in zeros_df.columns
    assert zeros_df.loc["money_spent", "value_count"] == 4
    summary = summarize_dataset(df)
    flags = compute_quality_flags(summary, missing_df, zeros_df, 0.5, 0.2, 0.5)
    assert flags["suspicious_id_duplicates"] == 0
    assert flags["too_many_zeros"] == True
    assert flags["too_many_duplicates"] == True
    assert flags["too_many_missing"] == False
    assert 0.0 <= flags["quality_score"] <= 1.0

def test_id_duplicates():
    df = _id_dup_df()
    summary = summarize_dataset(df)
    flags = compute_quality_flags(summary, pd.DataFrame(), pd.DataFrame())
    assert flags["suspicious_id_duplicates"] == 2
    assert summary.duplicates == 1
    assert 0.0 <= flags["quality_score"] <= 1.0