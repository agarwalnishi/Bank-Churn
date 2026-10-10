"""
Run every query in sql/analysis.sql against data/churn.db, save each result
to results/<name>.csv, and render the key charts to charts/.

Run:  python -m src.run_analysis
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "churn.db"
SQL_PATH = ROOT / "sql" / "analysis.sql"
RESULTS = ROOT / "results"
CHARTS = ROOT / "charts"

# Palette: one blue for data, a darker blue for the bar being called out,
# muted greys for text, axes and the reference line.
BLUE = "#86b6ef"
BLUE_HI = "#1c5cab"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK_2,
    "axes.titlecolor": INK,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": GRID,
    "grid.linewidth": 0.8,
})


def load_queries(path: Path = SQL_PATH) -> dict[str, str]:
    """Split the .sql file on '-- name: <id>' tags."""
    text = path.read_text()
    parts = re.split(r"^-- name:\s*(\S+)\s*$", text, flags=re.MULTILINE)
    # parts = [preamble, name1, body1, name2, body2, ...]
    return {parts[i]: parts[i + 1].strip() for i in range(1, len(parts), 2)}


def run_all() -> dict[str, pd.DataFrame]:
    RESULTS.mkdir(exist_ok=True)
    out: dict[str, pd.DataFrame] = {}
    with sqlite3.connect(DB_PATH) as conn:
        for name, sql in load_queries().items():
            df = pd.read_sql_query(sql, conn)
            df.to_csv(RESULTS / f"{name}.csv", index=False)
            out[name] = df
            print(f"\n=== {name} ===")
            print(df.to_string(index=False))
    return out


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
def _bar(ax, labels, values, highlight_idx, avg, fmt="{:.1f}%"):
    colors = [BLUE_HI if i in highlight_idx else BLUE for i in range(len(values))]
    bars = ax.bar(labels, values, color=colors, width=0.6, edgecolor=SURFACE, linewidth=2)
    ax.axhline(avg, color=INK_2, linewidth=1, linestyle=(0, (4, 3)),
               label=f"Overall churn {avg:.1f}%")
    ax.legend(loc="upper left", frameon=False, labelcolor=INK_2, fontsize=9,
              bbox_to_anchor=(0, 1.0))
    ax.set_ylim(0, max(values) * 1.18)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v, fmt.format(v), ha="center",
                va="bottom", color=INK, fontsize=9)
    ax.set_ylabel("Churn rate (%)")
    ax.set_axisbelow(True)


def make_charts(r: dict[str, pd.DataFrame]) -> None:
    CHARTS.mkdir(exist_ok=True)
    avg = float(r["01_overall_churn"]["churn_rate_pct"].iloc[0])

    # 1. Geography
    g = r["02_churn_by_geography"].sort_values("Geography")
    fig, ax = plt.subplots(figsize=(6, 4))
    _bar(ax, g["Geography"], g["churn_rate_pct"],
         {int(g["churn_rate_pct"].values.argmax())}, avg)
    ax.set_title("Germany churns at about 2x France and Spain")
    fig.tight_layout(); fig.savefig(CHARTS / "churn_by_geography.png", dpi=150); plt.close(fig)

    # 2. Products
    p = r["04_churn_by_products"]
    fig, ax = plt.subplots(figsize=(6, 4))
    _bar(ax, p["NumOfProducts"].astype(str), p["churn_rate_pct"],
         {i for i, n in enumerate(p["NumOfProducts"]) if n >= 3}, avg)
    ax.set_xlabel("Number of products held")
    ax.set_title("Customers with 3+ products almost all leave")
    fig.tight_layout(); fig.savefig(CHARTS / "churn_by_products.png", dpi=150); plt.close(fig)

    # 3. Activity
    a = r["03_churn_by_activity"].sort_values("member_status")
    fig, ax = plt.subplots(figsize=(5, 4))
    _bar(ax, a["member_status"], a["churn_rate_pct"],
         {int(a["churn_rate_pct"].values.argmax())}, avg)
    ax.set_title("Inactive members churn at about 2x")
    fig.tight_layout(); fig.savefig(CHARTS / "churn_by_activity.png", dpi=150); plt.close(fig)

    # 4. Age group
    ag = r["05_churn_by_age_group"]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    _bar(ax, ag["AgeGroup"], ag["churn_rate_pct"],
         {int(ag["churn_rate_pct"].values.argmax())}, avg)
    ax.set_xlabel("Age group")
    ax.set_title("Churn peaks in the 50-59 age band")
    fig.tight_layout(); fig.savefig(CHARTS / "churn_by_age_group.png", dpi=150); plt.close(fig)

    # 5. Risk segments: lost deposits (horizontal bars, in millions)
    s = r["11_risk_segment_matrix"].iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 4))
    vals = s["lost_deposits"] / 1e6
    bars = ax.barh(s["risk_segment"].str[3:], vals, color=BLUE, height=0.6,
                   edgecolor=SURFACE, linewidth=2)
    for b, v, rate in zip(bars, vals, s["churn_rate_pct"]):
        ax.text(v, b.get_y() + b.get_height() / 2, f"  {v:.1f}M  ({rate:.0f}% churn)",
                va="center", color=INK, fontsize=9)
    ax.grid(axis="y", visible=False); ax.grid(axis="x", visible=True); ax.set_axisbelow(True)
    ax.set_xlabel("Lost deposits (millions)")
    ax.set_xlim(0, vals.max() * 1.45)
    ax.set_title("Where the lost deposits came from")
    fig.tight_layout(); fig.savefig(CHARTS / "lost_deposits_by_segment.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    results = run_all()
    make_charts(results)
    print(f"\nSaved {len(results)} query results to results/ and charts to charts/")
