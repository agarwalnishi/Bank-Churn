"""
ETL pipeline for the Bank Customer Churn dataset.

Extract  -> read the raw CSV
Transform-> clean (dedupe, missing values, type/range checks) + feature engineering
Load     -> write the processed table into SQLite

Run:  python -m src.etl
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "Churn_Modelling.csv"
DB_PATH = ROOT / "data" / "churn.db"
TABLE = "customers"

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("etl")

RAW_COLUMNS = [
    "RowNumber", "CustomerId", "Surname", "CreditScore", "Geography", "Gender",
    "Age", "Tenure", "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember",
    "EstimatedSalary", "Exited",
]
NUMERIC_COLS = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "EstimatedSalary"]
BINARY_COLS = ["HasCrCard", "IsActiveMember"]
CATEGORICAL_COLS = ["Geography", "Gender"]


# --------------------------------------------------------------------------- #
# Extract
# --------------------------------------------------------------------------- #
def extract(path: Path = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = set(RAW_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Raw file is missing columns: {sorted(missing)}")
    log.info("Extracted %d rows from %s", len(df), path.name)
    return df


# --------------------------------------------------------------------------- #
# Transform: cleaning
# --------------------------------------------------------------------------- #
def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop non-analytic columns, dedupe, handle missing values, enforce types."""
    out = df.copy()

    # RowNumber is an index artifact; Surname is PII with no analytic value.
    out = out.drop(columns=["RowNumber", "Surname"], errors="ignore")

    # Text hygiene on categoricals
    for col in CATEGORICAL_COLS:
        out[col] = out[col].astype("string").str.strip().str.title()

    # Rows with no target or no ID are unusable
    before = len(out)
    out = out.dropna(subset=["CustomerId", "Exited"])
    out = out.drop_duplicates(subset="CustomerId", keep="first")
    dropped = before - len(out)
    if dropped:
        log.info("Dropped %d rows (missing ID/target or duplicate CustomerId)", dropped)

    # Impute: numeric -> median, binary/categorical -> mode
    for col in NUMERIC_COLS:
        if out[col].isna().any():
            out[col] = out[col].fillna(out[col].median())
    for col in BINARY_COLS + CATEGORICAL_COLS:
        if out[col].isna().any():
            out[col] = out[col].fillna(out[col].mode().iloc[0])

    # Types
    int_cols = ["CustomerId", "CreditScore", "Age", "Tenure", "NumOfProducts",
                "HasCrCard", "IsActiveMember", "Exited"]
    out[int_cols] = out[int_cols].astype("int64")
    out[["Balance", "EstimatedSalary"]] = out[["Balance", "EstimatedSalary"]].astype("float64")
    out[CATEGORICAL_COLS] = out[CATEGORICAL_COLS].astype(str)

    # Range sanity: drop impossible records rather than silently keeping them
    valid = (
        out["CreditScore"].between(300, 900)
        & out["Age"].between(18, 100)
        & out["Tenure"].between(0, 10)
        & out["NumOfProducts"].between(1, 4)
        & (out["Balance"] >= 0)
        & (out["EstimatedSalary"] >= 0)
        & out["Exited"].isin([0, 1])
    )
    if (~valid).any():
        log.info("Dropped %d rows failing range checks", int((~valid).sum()))
    out = out[valid].reset_index(drop=True)

    return out


# --------------------------------------------------------------------------- #
# Transform: feature engineering
# --------------------------------------------------------------------------- #
AGE_BINS = [17, 29, 39, 49, 59, 100]
AGE_LABELS = ["18-29", "30-39", "40-49", "50-59", "60+"]
CREDIT_BINS = [299, 579, 669, 739, 799, 900]
CREDIT_LABELS = ["Poor", "Fair", "Good", "Very Good", "Excellent"]


def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Readable segments for SQL GROUP BYs
    out["AgeGroup"] = pd.cut(out["Age"], bins=AGE_BINS, labels=AGE_LABELS).astype(str)
    out["CreditBand"] = pd.cut(out["CreditScore"], bins=CREDIT_BINS, labels=CREDIT_LABELS).astype(str)
    out["TenureGroup"] = pd.cut(out["Tenure"], bins=[-1, 2, 5, 10],
                                labels=["New (0-2)", "Mid (3-5)", "Loyal (6-10)"]).astype(str)

    # Numeric features
    out["ZeroBalance"] = (out["Balance"] == 0).astype("int64")
    out["BalanceSalaryRatio"] = np.where(
        out["EstimatedSalary"] > 0, out["Balance"] / out["EstimatedSalary"], 0.0
    ).round(4)
    out["ProductsPerTenure"] = (out["NumOfProducts"] / (out["Tenure"] + 1)).round(4)
    out["MultiProduct"] = (out["NumOfProducts"] >= 3).astype("int64")

    # Encoding: binary for Gender, one-hot for Geography (raw columns kept for SQL)
    out["GenderMale"] = (out["Gender"] == "Male").astype("int64")
    geo = pd.get_dummies(out["Geography"], prefix="Geo", dtype="int64")
    out = pd.concat([out, geo], axis=1)

    return out


def transform(df: pd.DataFrame) -> pd.DataFrame:
    return feature_engineer(clean(df))


# --------------------------------------------------------------------------- #
# Load
# --------------------------------------------------------------------------- #
def load(df: pd.DataFrame, db_path: Path = DB_PATH, table: str = TABLE) -> int:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        df.to_sql(table, conn, if_exists="replace", index=False)
        conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_geo ON {table}(Geography)")
        conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_exited ON {table}(Exited)")
        loaded = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    if loaded != len(df):
        raise RuntimeError(f"Row count mismatch: dataframe={len(df)} sqlite={loaded}")
    log.info("Loaded %d rows into %s:%s", loaded, db_path.name, table)
    return loaded


def run() -> pd.DataFrame:
    df = transform(extract())
    load(df)
    return df


if __name__ == "__main__":
    run()
