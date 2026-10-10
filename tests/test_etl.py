"""Unit tests for the ETL transforms: missing values, encoding, row counts."""

import sqlite3

import numpy as np
import pandas as pd
import pytest

from src.etl import RAW_PATH, clean, extract, feature_engineer, load, transform


@pytest.fixture
def sample_raw():
    """Small hand-built frame with the edge cases the pipeline must handle."""
    return pd.DataFrame({
        "RowNumber":       [1, 2, 3, 4, 5, 6],
        "CustomerId":      [101, 102, 103, 103, 105, 106],          # 103 duplicated
        "Surname":         ["A", "B", "C", "C", "E", "F"],
        "CreditScore":     [650, np.nan, 720, 720, 500, 1200],      # NaN + out-of-range
        "Geography":       ["France", " germany ", "Spain", "Spain", None, "France"],
        "Gender":          ["Female", "Male", "male", "male", "Female", "Male"],
        "Age":             [25, 45, 61, 61, 35, 40],
        "Tenure":          [2, 5, 8, 8, 0, 3],
        "Balance":         [0.0, 120000.0, 85000.0, 85000.0, np.nan, 5000.0],
        "NumOfProducts":   [1, 3, 2, 2, 1, 1],
        "HasCrCard":       [1, 0, 1, 1, np.nan, 1],
        "IsActiveMember":  [1, 0, 0, 0, 1, 1],
        "EstimatedSalary": [50000.0, 100000.0, 0.0, 0.0, 60000.0, 70000.0],
        "Exited":          [0, 1, 0, 0, 1, 0],
    })


# ---------------- cleaning ---------------- #
def test_clean_drops_pii_and_index_columns(sample_raw):
    out = clean(sample_raw)
    assert "Surname" not in out.columns
    assert "RowNumber" not in out.columns


def test_clean_removes_duplicate_customers(sample_raw):
    out = clean(sample_raw)
    assert out["CustomerId"].is_unique


def test_clean_leaves_no_missing_values(sample_raw):
    out = clean(sample_raw)
    assert out.isna().sum().sum() == 0


def test_clean_imputes_numeric_with_median(sample_raw):
    out = clean(sample_raw)
    # Balance NaN for customer 105 -> median of the deduped, non-null balances
    expected = sample_raw.drop_duplicates("CustomerId")["Balance"].median()
    assert out.loc[out["CustomerId"] == 105, "Balance"].iloc[0] == expected


def test_clean_normalises_categorical_text(sample_raw):
    out = clean(sample_raw)
    assert set(out["Geography"]) <= {"France", "Germany", "Spain"}
    assert set(out["Gender"]) == {"Female", "Male"}


def test_clean_drops_out_of_range_rows(sample_raw):
    out = clean(sample_raw)
    assert 106 not in out["CustomerId"].values      # CreditScore 1200
    assert out["CreditScore"].between(300, 900).all()


def test_clean_row_count(sample_raw):
    # 6 raw -> minus 1 duplicate -> minus 1 out-of-range = 4
    assert len(clean(sample_raw)) == 4


# ---------------- feature engineering / encoding ---------------- #
def test_gender_binary_encoding(sample_raw):
    out = feature_engineer(clean(sample_raw))
    assert set(out["GenderMale"].unique()) <= {0, 1}
    assert (out.loc[out["Gender"] == "Male", "GenderMale"] == 1).all()


def test_geography_one_hot_is_exclusive(sample_raw):
    out = feature_engineer(clean(sample_raw))
    geo_cols = [c for c in out.columns if c.startswith("Geo_")]
    assert len(geo_cols) >= 2
    assert (out[geo_cols].sum(axis=1) == 1).all()


def test_age_groups_assigned(sample_raw):
    out = feature_engineer(clean(sample_raw))
    row = out.loc[out["CustomerId"] == 103].iloc[0]
    assert row["AgeGroup"] == "60+"
    assert "nan" not in out["AgeGroup"].values


def test_balance_salary_ratio_handles_zero_salary(sample_raw):
    out = feature_engineer(clean(sample_raw))
    assert np.isfinite(out["BalanceSalaryRatio"]).all()
    assert out.loc[out["CustomerId"] == 103, "BalanceSalaryRatio"].iloc[0] == 0.0


def test_zero_balance_flag(sample_raw):
    out = feature_engineer(clean(sample_raw))
    assert out.loc[out["CustomerId"] == 101, "ZeroBalance"].iloc[0] == 1
    assert out.loc[out["CustomerId"] == 102, "ZeroBalance"].iloc[0] == 0


# ---------------- load ---------------- #
def test_load_row_count_matches(sample_raw, tmp_path):
    df = transform(sample_raw)
    db = tmp_path / "test.db"
    assert load(df, db_path=db) == len(df)
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM customers").fetchone()[0] == len(df)


# ---------------- full dataset ---------------- #
@pytest.mark.skipif(not RAW_PATH.exists(), reason="raw dataset not downloaded")
def test_full_dataset_keeps_all_10000_rows():
    df = transform(extract())
    assert len(df) == 10_000
    assert df.isna().sum().sum() == 0
