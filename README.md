# SERAH-PMAX
SERAH-PMAX is an AI-driven digital advertising intelligence and bid-optimisation pipeline designed to automate performance marketing decisions. It cleans, engineers, and models large-scale search and social advertising data, predicts Value per Click (VPC), and automatically generates smart bid recommendations. It then evaluates the impact of those recommendations on cost, revenue, ROAS, and conversions—producing visual comparisons, KPIs, and simulation artefacts for marketing analysts and data scientists.

---

## Table of Contents
1. [Overview](#overview)
2. [Primary Research Question](#primary-research-question)
3. [System Requirements](#system-requirements)
4. [Installation and Setup](#installation-and-setup)
5. [Instructions to Run](#instructions)
6. [Running the Scripts](#running-the-scripts)
7. [Project Structure](#project-structure)
8. [License](#license)

---


## Overview
SERAH-PMAX demonstrates how AI can transform paid-media optimisation by combining data science, marketing analytics, and automation.

It uses historical campaign performance data (impressions, clicks, cost, conversions, revenue, and contextual attributes) to:

1. Ingest, clean, and engineer raw advertising data with robust metrics like CTR, CPC, CVR, ROAS, and Value per Click (VPC).
2. Train a HistGradientBoostingRegressor (HGBR) model to predict expected VPC—representing the marginal value of each click.
3. Simulate AI-recommended bid adjustments based on predicted vs. observed VPC ratios.
4. Quantify performance impact—comparing baseline vs recommended results across revenue, conversions, cost, and ROAS.
5. Visualise trends and segment lifts by market, channel, and device with rich, exportable charts.

This project transforms raw marketing campaign data into an automated AI bidding engine—making digital campaigns more profitable, scalable, and evidence-driven.

---

## Primary Research Question

How can we automate bid adjustments in digital advertising using AI to improve revenue and ROAS across markets and channels without overspending?


- **Aims and Objectives**:

a. Build a robust regression model to predict the value per click (VPC) for each campaign segment.
b. Use predicted VPC to simulate and recommend intelligent bid updates (within ±20% caps).
c. Quantify business impact by comparing baseline vs. AI-recommended outcomes.
d. Visualise revenue, cost, conversion, and ROAS improvements over time and across segments.
e. Provide actionable analytics artefacts (CSV + PNG) for decision-makers.


- **Dataset (What, Why, & How)**:

Dataset Name: tcg_sea_dataset_2024_2025.csv (synthetic but structurally realistic)
Rows: ~10,000 | Fields: 25+ | Date Range: 12 months

Each record represents a daily campaign snapshot, including:

- Performance metrics: impressions, clicks, cost, conversions, revenue
- Context: market, channel, device, campaign, ad group, match type, query theme, product category
- Bidding data: bid_old, bid_cap_min, bid_cap_max, target_cpa, target_roas, quality_score
- Derived metrics: CTR, CPC, CVR, CPA, ROAS, Value per Click
- Temporal features: date, month, weekday, season flags (Black Friday, Valentine, etc.)

The dataset was chosen to emulate real paid-media campaign behaviour for an Ecommerce company and can easily adapt to Google Ads or Meta Ads export schemas.


- **Methods Used**:

a) Feature Engineering

    1. Derived CTR, CPC, CVR, CPA, ROAS, Value per Click.
    2. Smoothed ratios to avoid zero divisions.
    3. Added time/seasonality indicators (dow, month, is_weekend).
    4. Created one-hot seasonal flags for peak shopping periods (e.g. Black Friday, Christmas, Valentine).
    5. Clipped extreme values (1st–99th percentile).

b) Modelling

    1. Model: HistGradientBoostingRegressor
    2. Features: categorical (market, channel, device, campaign, etc.) + numeric (CTR, CPC, CVR, CPA, ROAS, etc.)
    3. Target: value_per_click
    4. Split: chronological 70/15/15 (train/validation/test)
    5. Metrics: Mean Absolute Error (MAE), R² score
    6. Pipeline: ColumnTransformer (OneHotEncoder + passthrough numeric) → HGBR

The model predicts expected VPC per campaign segment, learning non-linear relationships between cost, conversions, and ROI context.

c) AI Bid Adjustment Policy

  After prediction:

    NewBid = OldBid × (Predicted VPC/Current VPC)	​

    - Clamped to ±20% variation to avoid overshoot.
    - Capped within [bid_cap_min, bid_cap_max].

This ensures safe, incremental adjustments that optimise value without budget blowouts.

d) Simulation & Evaluation

1. Re-computed predicted clicks, conversions, and revenue after applying new bids using elasticities.
2. Aggregated KPIs for both baseline and recommended scenarios: Cost, Revenue, Conversions, ROAS
3. Generated trend plots, cumulative charts, and lift analyses by market/channel/device.

e) Visualisation & Reporting

    - Revenue, Conversion, and ROAS time-series (baseline vs AI).
    - Revenue lift by market, channel, device.
    - Bid-change vs revenue-lift scatter and bubble plots.
    - Exported as PNG files under data/outputs/visuals/.
  
---

## System Requirements
- **Operating System**: Windows, macOS, or Linux
- **Python Version:**: Python 3.10–3.12
- **Python Packages**:
  - pip install pandas numpy scikit-learn matplotlib pyarrow seaborn
- **Memory**: ≥2 GB

---

## Installation and Setup
1. **Clone the Repository**:
   - Click the green "**Code**" button in this GitHub repository.
   - Copy the HTTPS or SSH link provided (e.g., `https://github.com/rabraham2/SERAH-PMAX.git`).
   - Open your terminal or command prompt and run:
     ```bash
     git clone https://github.com/rabraham2/SERAH-PMAX.git
     ```

2. **Install Required Python Packages**:
   Open **PyCharm** or **Other IDE-Enabled Coding Platform** and install the necessary packages:

```python
   pip install pandas numpy scikit-learn matplotlib pyarrow seaborn
```

3. **Processed and Cleaned Dataet**:
   - The original and unprocessed version of the real dataset can be found in the folder Data in GitHub.
   - If you need to run the processed and data-cleaned version of the dataset, use the file in the folder Dataset in GitHub.

---

## Instructions

Step 1: Data Audit & Cleaning

Run: python src/clean_and_engineer.py

Purpose: Cleans and audits raw campaign data.
Produces: data/interim/
  - tcg_sea_cleaned.csv – cleaned and feature-engineered dataset
Console audit: dataset shape, missing values, column types, summary statistics
Derived metrics: CTR, CPC, CVR, CPA, ROAS, Value per Click (VPC), seasonal & temporal flags

Step 2: Chronological Split (No Leakage)

Run: python src/split_and_save.py

Purpose: Creates leak-free chronological splits (70/15/15).
Outputs: data/processed/
  - X_train.parquet, X_valid.parquet, X_test.parquet – model input features
  - y_train.parquet, y_valid.parquet, y_test.parquet – target (VPC, ROAS) values
Console summary: number of records in each split

Step 3: Model Training & Bid Recommendation

Run: python src/train_and_recommend.py

Purpose: Trains an AI model to predict Value per Click (VPC) and recommends optimal bids.
Model: HistGradientBoostingRegressor (HGBR)
Outputs: data/processed/
  - bid_recommendations.csv – baseline vs predicted VPC, new recommended bids, bid deltas
  - model_vpc.pkl – trained regression pipeline (OneHot + HGBR)
  - metrics.json – MAE and R² validation metrics
Console summary: top features, prediction accuracy, bid recommendation range

Step 4: Performance Simulation & Comparison

Run: python src/compare_baseline_vs_recommended.py

Purpose: Simulates post-optimisation performance and visualises results.
Outputs: data/outputs/visuals/
  - kpi_comparison_optimistic.png – Baseline vs AI total KPI bar chart
  - roas_trend_optimistic.png – ROAS over time
  - conversions_trend_optimistic.png – Conversions trend
  - revenue_trend_optimistic.png – Daily revenue with shaded AI lift zone
  - revenue_cumulative_optimistic.png – Cumulative revenue (Baseline vs Recommended)
  - lift_by_market_optimistic.png – Revenue lift by market
  - lift_by_channel_optimistic.png – Revenue lift by channel
  - lift_by_device_optimistic.png – Revenue lift by device
  - bid_change_vs_revenue_delta.png – Scatter: % bid change vs absolute revenue gain
  - revenue_lift_by_bid_change_bins.png – Revenue lift grouped by bid-increase buckets
  - top_segments_bid_up_revenue_gain.png – Top 15 segments with positive bid & revenue impact

Step 5: Scenario Adjustment (Optional)

Run: conservative or optimistic simulation - python src/compare_baseline_vs_recommended.py --scenario conservative

Purpose: Simulates different elasticity and conversion-rate sensitivities.
Outputs: same charts saved under data/outputs/visuals/ with suffix _conservative or _optimistic.

Parameters:
  - Elasticity: how clicks respond to bid changes (0.5–0.9 range)
  - VPC Bonus: uplift factor on predicted value per click
  - CVR Bonus: uplift factor on conversion rate

Step 6: Market, Channel & Device Lift Reports

Run: Automatically generated within Step 4
Outputs: data/outputs/visuals/
  - lift_by_market_optimistic.png – Average % revenue growth by market
  - lift_by_channel_optimistic.png – Channel-wise revenue impact
  - lift_by_device_optimistic.png – Device-wise performance changes
Console summary: Top-performing segments and total revenue lift (%)

Step 7: Bid-Change Impact Analysis

Run: Automatically triggered after simulation

Purpose: Show where bid increases created meaningful revenue growth.
Outputs: data/outputs/visuals/
  - bid_change_vs_revenue_delta.png – Scatterplot of % Bid Change vs Revenue Lift
  - revenue_lift_by_bid_change_bins.png – Grouped bars (0–5 %, 5–10 %, 10–20 %, etc.)
  - top_segments_bid_up_revenue_gain.png – Top market–channel–device combinations driving gains

Notes
• data/interim/ – cleaned dataset (tcg_sea_cleaned.csv)
• data/processed/ – model splits + bid recommendations
• data/outputs/visuals/ – charts & comparative plots (baseline vs recommended)
• models/ – serialized model pipeline (model_vpc.pkl)

All steps are idempotent and can be re-run safely; outputs will overwrite or version-update automatically.

---

## Running the Scripts

```Python Code

A]--------> ## config.py  ##

# config.py — Loading the data

import pandas as pd

# Path to your dataset
file_path = "Dataset/tcg_sea_dataset_2024_2025.csv"

# Load it
df = pd.read_csv(file_path)

# Basic info
print("Rows:", len(df))
print("Columns:", len(df.columns))
print("\nColumn names:")
print(df.columns.tolist())

# Quick preview
print("\nSample data:")
print(df.head(5))

B]--------> ## # # clean_and_engineer.py ##

# clean_and_engineer.py — Cleans the raw advertising dataset and derives essential marketing metrics for modelling and simulation.

# clean_and_engineer.py
import os
from pathlib import Path
import pandas as pd
import numpy as np

# ---------------- Paths ----------------
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path(os.getcwd())

DEFAULT_RAW = BASE_DIR / "Dataset" / "tcg_sea_dataset_2024_2025.csv"
OUT_DIR = BASE_DIR / "data" / "interim"
CLEAN_PATH = OUT_DIR / "tcg_sea_cleaned.csv"

# ---------------- Helpers ----------------
def safe_div(n, d):
    """Safe division for pandas Series or scalars."""
    out = n / d
    if isinstance(out, pd.Series):
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    else:
        out = 0.0 if (d == 0 or pd.isna(d)) else out
    return out

def winsorise(df, cols, q_lo=0.01, q_hi=0.99):
    """Clip outliers (stabilise metrics)."""
    for col in cols:
        if col in df.columns:
            ql, qh = df[col].quantile([q_lo, q_hi])
            df[col] = df[col].clip(ql, qh)
    return df

# ---------------- Pipeline ----------------
def run_cleaning(raw_path: Path, out_csv: Path):
    print("BASE_DIR :", BASE_DIR)
    print("RAW_PATH :", raw_path)
    print("RAW exists?", raw_path.exists())

    if not raw_path.exists():
        raise FileNotFoundError(f"CSV not found at: {raw_path}")

    print("🔹 Loading dataset...")
    df = pd.read_csv(raw_path, parse_dates=["date"], encoding="utf-8")
    print(f"Rows loaded: {len(df):,}")

    # --- Basic cleaning ---
    bad = (df["clicks"] > df["impressions"])  # Series[bool]
    bad_array = np.array(bad)  # avoid PyCharm 'bool' confusion
    has_bad = bool(np.any(bad_array))
    bad_count = int(np.sum(bad_array))

    if has_bad:
        print(f"Fixing {bad_count} rows with clicks > impressions")
        df.loc[bad, "clicks"] = df.loc[bad, "impressions"]

    df = df.dropna(subset=["market", "channel", "device", "impressions", "clicks"])

    # --- Derived metrics ---
    df["ctr_calc"]  = safe_div(df["clicks"], df["impressions"]).round(4)
    df["cpc_calc"]  = safe_div(df["cost"], df["clicks"]).round(2)
    df["cvr_calc"]  = safe_div(df["conversions"], df["clicks"]).round(4)
    df["roas_calc"] = safe_div(df["revenue"], df["cost"]).round(2)  # Return on Ad Spend
    df["cpa_calc"]  = safe_div(df["cost"], df["conversions"]).round(2)

    # Smoothed value per click
    df["value_per_click"] = ((df["revenue"] + 1.0) / (df["clicks"] + 1.0)).round(4)

    # --- Time & seasonality ---
    df["dow"]        = df["date"].dt.dayofweek
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["month"]      = df["date"].dt.month

    seasons = {
        "BlackFridayWeek": [11],
        "XmasShippingWindow": [12],
        "ValentineWindow": [1, 2],
        "MothersDaySeason": [3],
        "GraduationSeason": [5, 6],
        "SummerLull": [7, 8],
    }
    for label, months in seasons.items():
        df[f"sf_{label}"] = df["month"].isin(months).astype(int)

    # --- Outlier clipping ---
    df = winsorize(df, ["impressions", "clicks", "cost", "revenue"], 0.01, 0.99)

    # --- Save ---
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8")

    print(f" Cleaned dataset saved → {out_csv}")
    print(f"Final rows: {len(df):,}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Clean & engineer TCG SEA dataset", add_help=False)
    parser.add_argument("--raw", type=str, default=str(DEFAULT_RAW))
    parser.add_argument("--out", type=str, default=str(CLEAN_PATH))
    # This line is the key: ignore any extra args the console adds
    args, _unknown = parser.parse_known_args()

    raw_path = Path(args.raw)
    out_csv = Path(args.out)
    if not raw_path.is_absolute():
        raw_path = (BASE_DIR / raw_path).resolve()
    if not out_csv.is_absolute():
        out_csv = (BASE_DIR / out_csv).resolve()

    run_cleaning(raw_path, out_csv)

if __name__ == "__main__":
    main()

C]--------> ## # # split_and_save.py ##

# split_and_save.py — Splits the dataset into train, validation, and test sets in time order to mimic real-world campaign drift.

# src/split_and_save.py
import os
import argparse
import traceback
from pathlib import Path
import pandas as pd

# -------- Paths --------
try:
    base_dir = Path(__file__).resolve().parent.parent
except NameError:
    base_dir = Path(os.getcwd())

clean_default = base_dir / "data" / "interim" / "tcg_sea_cleaned.csv"
processed_dir = base_dir / "data" / "processed"

x_train_path = processed_dir / "X_train.parquet"
x_valid_path = processed_dir / "X_valid.parquet"
x_test_path  = processed_dir / "X_test.parquet"
y_train_path = processed_dir / "y_train.parquet"
y_valid_path = processed_dir / "y_valid.parquet"
y_test_path  = processed_dir / "y_test.parquet"

label_cols = ["value_per_click", "roas_calc"]
cat_cols = [
    "market", "channel", "device", "match_type", "query_theme",
    "product_category", "season_flag", "experiment_group"
]
num_cols = [
    "impressions", "clicks", "cost", "conversions", "revenue", "quality_score",
    "bid_old", "bid_cap_min", "bid_cap_max", "target_cpa", "target_roas",
    "ctr_calc", "cpc_calc", "cvr_calc", "cpa_calc", "is_weekend", "month", "dow"
]


def save_splits(x_train, x_valid, x_test, y_train, y_valid, y_test):
    """Save dataset splits to Parquet or CSV."""
    processed_dir.mkdir(parents=True, exist_ok=True)
    try:
        x_train.to_parquet(x_train_path, index=False)
        x_valid.to_parquet(x_valid_path, index=False)
        x_test.to_parquet(x_test_path, index=False)
        y_train.to_parquet(y_train_path, index=False)
        y_valid.to_parquet(y_valid_path, index=False)
        y_test.to_parquet(y_test_path, index=False)
        print(" Saved Parquet splits.")
    except Exception as err:  # noqa: BLE001
        print(" Parquet write failed, saving as CSV instead:")
        print(traceback.format_exc())
        x_train.to_csv(str(x_train_path).replace(".parquet", ".csv"), index=False)
        x_valid.to_csv(str(x_valid_path).replace(".parquet", ".csv"), index=False)
        x_test.to_csv(str(x_test_path).replace(".parquet", ".csv"), index=False)
        y_train.to_csv(str(y_train_path).replace(".parquet", ".csv"), index=False)
        y_valid.to_csv(str(y_valid_path).replace(".parquet", ".csv"), index=False)
        y_test.to_csv(str(y_test_path).replace(".parquet", ".csv"), index=False)
        print(" Saved CSV splits.")


def main(clean_path: Path):
    """Split the cleaned dataset into chronological train/valid/test sets."""
    print("CLEAN:", clean_path)
    if not clean_path.exists():
        raise FileNotFoundError(f"Cleaned CSV not found at: {clean_path}")

    df = pd.read_csv(clean_path, parse_dates=["date"], encoding="utf-8")
    required = ["date"] + cat_cols + num_cols + label_cols
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.sort_values("date").reset_index(drop=True)
    n = len(df)
    n_train = int(n * 0.70)
    n_valid = int(n * 0.85)

    df_train = df.iloc[:n_train]
    df_valid = df.iloc[n_train:n_valid]
    df_test = df.iloc[n_valid:]

    x_train = df_train.drop(columns=label_cols + ["date"])
    y_train = df_train[label_cols]
    x_valid = df_valid.drop(columns=label_cols + ["date"])
    y_valid = df_valid[label_cols]
    x_test = df_test.drop(columns=label_cols + ["date"])
    y_test = df_test[label_cols]

    save_splits(x_train, x_valid, x_test, y_train, y_valid, y_test)
    print(
        f"Sizes → train={len(x_train):,}, "
        f"valid={len(x_valid):,}, test={len(x_test):,}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--clean", type=str, default=str(clean_default))
    args, _ = parser.parse_known_args()

    clean_path = Path(args.clean)
    if not clean_path.is_absolute():
        clean_path = (base_dir / clean_path).resolve()
    main(clean_path)

D]--------> ## # # train_and_recommend.py ##

# train_and_recommend.py — Trains a predictive model for Value per Click (VPC) and generates AI-driven bid recommendations.

# src/train_and_recommend.py
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import HistGradientBoostingRegressor

# -------- Paths --------
try:
    base_dir = Path(__file__).resolve().parent.parent
except NameError:
    base_dir = Path(os.getcwd())

proc_dir    = base_dir / "data" / "processed"
interim_dir = base_dir / "data" / "interim"
clean_path  = interim_dir / "tcg_sea_cleaned.csv"

x_train_p = proc_dir / "X_train.parquet"
x_valid_p = proc_dir / "X_valid.parquet"
x_test_p  = proc_dir / "X_test.parquet"
y_train_p = proc_dir / "y_train.parquet"
y_valid_p = proc_dir / "y_valid.parquet"

out_recs  = proc_dir / "bid_recommendations.csv"

# Categorical columns we expect (we'll auto-extend this with any leftover object cols)
base_cat_cols = [
    "market","channel","device","match_type","query_theme",
    "product_category","season_flag","experiment_group"
]

def clamp(arr, lo, hi):
    return np.minimum(np.maximum(arr, lo), hi)

def main():
    # 1) Load splits
    X_train = pd.read_parquet(x_train_p)
    X_valid = pd.read_parquet(x_valid_p)
    X_test  = pd.read_parquet(x_test_p)
    y_train = pd.read_parquet(y_train_p)["value_per_click"]
    y_valid = pd.read_parquet(y_valid_p)["value_per_click"]

    # ---------- Auto-detect extra categorical columns ----------
    extra_obj_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
    extra_str_cols = X_train.select_dtypes(include=["string"]).columns.tolist()
    extra_cats = sorted(list(set(extra_obj_cols + extra_str_cols)))

    cat_in = [c for c in (base_cat_cols + extra_cats) if c in X_train.columns]
    num_in = [c for c in X_train.columns if c not in cat_in]

    # Make sure the numeric block is actually numeric
    X_train[num_in] = X_train[num_in].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    X_valid[num_in] = X_valid[num_in].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    X_test[num_in]  = X_test[num_in].apply(pd.to_numeric,  errors="coerce").fillna(0.0)

    # Dense OHE for compatibility with HistGradientBoostingRegressor
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    pre = ColumnTransformer(
        transformers=[
            ("cat", ohe, cat_in),
            ("num", "passthrough", num_in),
        ],
        remainder="drop"
    )

    model = HistGradientBoostingRegressor(
        max_depth=6, learning_rate=0.08, max_iter=400, l2_regularization=0.0
    )
    pipe = Pipeline([("prep", pre), ("model", model)])

    print("Categorical columns used:", cat_in)
    print("Numeric columns used    :", num_in)

    # 2) Train & validate
    pipe.fit(X_train, y_train)
    val_pred = pipe.predict(X_valid)
    print("Validation MAE:", round(mean_absolute_error(y_valid, val_pred), 4))
    print("Validation R2 :", round(r2_score(y_valid, val_pred), 4))

    # 3) Predict on test
    test_pred_vpc = pipe.predict(X_test)

    # 4) Bring back original test slice from the cleaned CSV (for bids/caps context)
    df_clean = pd.read_csv(clean_path, parse_dates=["date"])
    df_clean = df_clean.sort_values("date").reset_index(drop=True)
    n = len(df_clean); n_train = int(n*0.70); n_valid = int(n*0.85)
    test_rows = df_clean.iloc[n_valid:].reset_index(drop=True)
    if len(test_rows) != len(X_test):
        test_rows = test_rows.iloc[:len(X_test)].copy()

    current_vpc = test_rows["value_per_click"].to_numpy(dtype=float)
    bid_old = test_rows["bid_old"].to_numpy(dtype=float)
    cap_min = test_rows["bid_cap_min"].to_numpy(dtype=float)
    cap_max = test_rows["bid_cap_max"].to_numpy(dtype=float)

    # 5) Bid policy: scale by predicted/observed VPC, cap ±20%, respect platform caps
    ratio = test_pred_vpc / np.maximum(current_vpc, 1e-6)
    ratio_capped = clamp(ratio, 0.8, 1.2)
    bid_prop = bid_old * ratio_capped
    bid_new = clamp(bid_prop, cap_min, cap_max)

    # 6) Save recommendations
    recs = pd.DataFrame({
        "date": test_rows["date"],
        "market": test_rows["market"],
        "channel": test_rows["channel"],
        "device": test_rows["device"],
        "campaign": test_rows["campaign"],
        "ad_group": test_rows["ad_group"],
        "match_type": test_rows["match_type"],
        "query_theme": test_rows["query_theme"],
        "product_category": test_rows["product_category"],
        "impressions": test_rows["impressions"],
        "clicks": test_rows["clicks"],
        "cost": test_rows["cost"],
        "conversions": test_rows["conversions"],
        "revenue": test_rows["revenue"],
        "value_per_click_current": current_vpc,
        "value_per_click_pred": test_pred_vpc,
        "bid_old": bid_old,
        "bid_cap_min": cap_min,
        "bid_cap_max": cap_max,
        "ratio_capped": ratio_capped,
        "bid_new": bid_new
    })
    recs["change_pct"] = (recs["bid_new"] - recs["bid_old"]) / np.maximum(recs["bid_old"], 1e-6)
    recs = recs.sort_values("change_pct", ascending=False)

    proc_dir.mkdir(parents=True, exist_ok=True)
    recs.to_csv(out_recs, index=False, encoding="utf-8")
    print(f"✅ Saved bid recommendations → {out_recs}")

    print("\nTop 5 increases:")
    print(recs.head(5)[["market","channel","device","campaign","ad_group","bid_old","bid_new","change_pct"]])
    print("\nTop 5 decreases:")
    print(recs.tail(5)[["market","channel","device","campaign","ad_group","bid_old","bid_new","change_pct"]])

if __name__ == "__main__":
    main()

E]--------> ## # # compare_baseline_vs_recommended.py ##

# compare_baseline_vs_recommended.py — Simulates campaign outcomes using AI-recommended bids and visualises performance improvements.

# src/compare_baseline_vs_recommended.py
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------- PATHS ----------
try:
    BASE_DIR = Path(__file__).resolve().parent.parent
except NameError:
    BASE_DIR = Path(os.getcwd())

INTERIM_DIR = BASE_DIR / "data" / "interim"
PROC_DIR    = BASE_DIR / "data" / "processed"
OUT_DIR     = BASE_DIR / "data" / "outputs" / "visuals"

CLEAN_PATH = INTERIM_DIR / "tcg_sea_cleaned.csv"
RECS_PATH  = PROC_DIR / "bid_recommendations.csv"

# ---------- SCENARIO KNOBS ----------
SCENARIO = "optimistic"   # "conservative" or "optimistic"
CONS = dict(ELASTICITY=0.5, VPC_BONUS=1.00, CVR_BONUS=1.00)
OPTI = dict(ELASTICITY=0.9, VPC_BONUS=1.15, CVR_BONUS=1.10)
PARAMS     = OPTI if SCENARIO == "optimistic" else CONS
ROLL_DAYS  = 7
SHOW_PLOTS = False  # True to also display, False to only save

# ---------- HELPERS ----------
def ensure_numeric(df: pd.DataFrame, cols):
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df

def save_plot(fig, filename: str):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / filename
    fig.savefig(path, bbox_inches="tight", dpi=300)
    print(f"💾 Saved → {path}")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)

def rolling_add(merged: pd.DataFrame, old: str, new: str, win: int):
    merged[f"{old}_roll"] = merged[old].rolling(win, min_periods=1).mean()
    merged[f"{new}_roll"] = merged[new].rolling(win, min_periods=1).mean()

# ---------- MAIN ----------
def main():
    # --- LOAD ---
    if not CLEAN_PATH.exists():
        raise FileNotFoundError(f"Missing cleaned file: {CLEAN_PATH}")
    if not RECS_PATH.exists():
        raise FileNotFoundError(f"Missing recommendations file: {RECS_PATH}")

    df_base = pd.read_csv(CLEAN_PATH, parse_dates=["date"])
    df_new  = pd.read_csv(RECS_PATH,  parse_dates=["date"])

    base_num = ["impressions","clicks","cost","conversions","revenue"]
    df_base = ensure_numeric(df_base, base_num)
    df_new  = ensure_numeric(df_new, base_num + [
        "bid_old","bid_new","value_per_click_pred","value_per_click_current"
    ])

    # --- SIMULATE "AFTER" ---
    ratio_bid  = df_new["bid_new"] / np.maximum(df_new["bid_old"], 1e-6)
    elasticity = PARAMS["ELASTICITY"]
    vpc_bonus  = PARAMS["VPC_BONUS"]
    cvr_bonus  = PARAMS["CVR_BONUS"]
    base_cvr   = df_new["conversions"] / np.maximum(df_new["clicks"], 1e-6)

    df_new["clicks_pred"] = df_new["clicks"] * np.power(ratio_bid, elasticity)
    df_new["cost_pred"]   = df_new["cost"]   * ratio_bid
    df_new["rev_pred"]    = df_new["clicks_pred"] * df_new["value_per_click_pred"] * vpc_bonus
    df_new["conv_pred"]   = df_new["clicks_pred"] * base_cvr * cvr_bonus
    df_new["roas_pred"]   = df_new["rev_pred"] / np.maximum(df_new["cost_pred"], 1e-6)

    # --- ALIGN BASELINE WINDOW TO RECOMMENDED WINDOW ---
    min_d, max_d = df_new["date"].min(), df_new["date"].max()
    base_aligned = df_base[(df_base["date"] >= min_d) & (df_base["date"] <= max_d)].copy()

    # --- DAILY AGGREGATES ---
    daily_old = base_aligned.groupby("date", as_index=False).agg(
        cost=("cost","sum"), rev=("revenue","sum"), conv=("conversions","sum")
    )
    daily_new = df_new.groupby("date", as_index=False).agg(
        cost_pred=("cost_pred","sum"), rev_pred=("rev_pred","sum"), conv_pred=("conv_pred","sum")
    )

    merged = pd.merge(daily_old, daily_new, on="date", how="inner").sort_values("date").reset_index(drop=True)
    merged["roas"]      = merged["rev"]      / np.maximum(merged["cost"], 1e-6)
    merged["roas_pred"] = merged["rev_pred"] / np.maximum(merged["cost_pred"], 1e-6)

    for old, new in [("cost","cost_pred"), ("rev","rev_pred"), ("conv","conv_pred"), ("roas","roas_pred")]:
        rolling_add(merged, old, new, ROLL_DAYS)

    # --- KPI SUMMARY ---
    kpi = {
        "Cost (Baseline)":           merged["cost"].sum(),
        "Cost (Recommended)":        merged["cost_pred"].sum(),
        "Revenue (Baseline)":        merged["rev"].sum(),
        "Revenue (Recommended)":     merged["rev_pred"].sum(),
        "Conversions (Baseline)":    merged["conv"].sum(),
        "Conversions (Recommended)": merged["conv_pred"].sum(),
        "ROAS (Baseline)":           merged["rev"].sum()      / np.maximum(merged["cost"].sum(),     1e-6),
        "ROAS (Recommended)":        merged["rev_pred"].sum() / np.maximum(merged["cost_pred"].sum(), 1e-6),
    }
    rev_lift_abs = kpi["Revenue (Recommended)"] - kpi["Revenue (Baseline)"]
    rev_lift_pct = (rev_lift_abs / np.maximum(kpi["Revenue (Baseline)"], 1e-6)) * 100.0

    print(f"\n=== Scenario: {SCENARIO.upper()} ===")
    for k, v in kpi.items():
        print(f"{k:30s}: {v:,.2f}")
    print(f"Revenue Lift: {rev_lift_abs:,.2f}  ({rev_lift_pct:.1f}%)")

    # --- KPI BAR CHART ---
    fig, ax = plt.subplots(figsize=(7,5))
    labels = ["Cost","Revenue","Conversions","ROAS"]
    ax.bar(labels, [kpi["Cost (Baseline)"], kpi["Revenue (Baseline)"], kpi["Conversions (Baseline)"], kpi["ROAS (Baseline)"]],
           alpha=0.6, label="Baseline")
    ax.bar(labels, [kpi["Cost (Recommended)"], kpi["Revenue (Recommended)"], kpi["Conversions (Recommended)"], kpi["ROAS (Recommended)"]],
           alpha=0.6, label="Recommended")
    ax.set_title("Overall KPI Comparison — Baseline vs Recommended")
    ax.legend()
    save_plot(fig, f"kpi_comparison_{SCENARIO}.png")

    # --- ROAS TREND ---
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(merged["date"], merged["roas_roll"], label="Baseline")
    ax.plot(merged["date"], merged["roas_pred_roll"], label="Recommended")
    ax.set_title("ROAS Trend — Before vs After AI Bid Adjustment")
    ax.set_xlabel("Date"); ax.set_ylabel("ROAS"); ax.legend()
    save_plot(fig, f"roas_trend_{SCENARIO}.png")

    # --- CONVERSIONS TREND ---
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(merged["date"], merged["conv_roll"], label="Baseline")
    ax.plot(merged["date"], merged["conv_pred_roll"], label="Recommended")
    ax.set_title("Conversions Over Time — Baseline vs Recommended")
    ax.set_xlabel("Date"); ax.set_ylabel("Conversions"); ax.legend()
    save_plot(fig, f"conversions_trend_{SCENARIO}.png")

    # --- REVENUE TREND + BIG LIFT LABEL ---
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(merged["date"], merged["rev_roll"],      label="Baseline Revenue",    linewidth=2)
    ax.plot(merged["date"], merged["rev_pred_roll"], label="Recommended Revenue", linewidth=2)
    ax.fill_between(merged["date"], merged["rev_roll"], merged["rev_pred_roll"],
                    alpha=0.25, label="Revenue Lift")
    ax.set_title("Revenue Comparison — Baseline vs AI-Recommended")
    ax.set_xlabel("Date"); ax.set_ylabel("Revenue")
    ax.legend()
    x_mid = merged["date"].iloc[len(merged)//2]
    y_max = max(merged["rev_roll"].max(), merged["rev_pred_roll"].max())
    ax.text(x_mid, y_max * 0.9, f"+{rev_lift_pct:.1f}% total revenue", fontsize=12, weight="bold")
    save_plot(fig, f"revenue_trend_{SCENARIO}.png")

    # --- CUMULATIVE REVENUE ---
    merged["rev_cum"]      = merged["rev"].cumsum()
    merged["rev_pred_cum"] = merged["rev_pred"].cumsum()
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(merged["date"], merged["rev_cum"],      label="Cumulative Baseline Revenue",      linewidth=2)
    ax.plot(merged["date"], merged["rev_pred_cum"], label="Cumulative Recommended Revenue",   linewidth=2)
    ax.set_title("Cumulative Revenue — Baseline vs Recommended")
    ax.set_xlabel("Date"); ax.set_ylabel("Cumulative Revenue")
    ax.legend()
    save_plot(fig, f"revenue_cumulative_{SCENARIO}.png")

    # ============================================================
    # "Bid increased → Revenue increased" visualisations
    # ============================================================
    # Row-level deltas
    df_new["bid_change_pct"]  = (df_new["bid_new"] / np.maximum(df_new["bid_old"], 1e-6)) - 1.0
    df_new["rev_change_abs"]  = df_new["rev_pred"] - df_new["revenue"]
    df_new["rev_change_pct"]  = df_new["rev_change_abs"] / np.maximum(df_new["revenue"], 1e-6)

    # 1) Scatter: ΔBid% vs ΔRevenue (abs)
    fig, ax = plt.subplots(figsize=(8,6))
    ax.scatter(df_new["bid_change_pct"]*100.0, df_new["rev_change_abs"], alpha=0.25)
    ax.axvline(0, color="gray", linewidth=1)
    ax.axhline(0, color="gray", linewidth=1)
    ax.set_title("Bid Change (%) vs Revenue Change (Abs)")
    ax.set_xlabel("Bid Change (%)")
    ax.set_ylabel("Revenue Change")
    save_plot(fig, "bid_change_vs_revenue_delta.png")

    # 2) Bars by bid-change buckets (where bids increased)
    inc = df_new[df_new["bid_change_pct"] > 0].copy()
    bins = [-0.01, 0.05, 0.10, 0.20, 0.35, 0.50, 1.00]  # 0–5–10–20–35–50%+
    labels = ["0–5%","5–10%","10–20%","20–35%","35–50%","50%+"]
    inc["bid_bucket"] = pd.cut(inc["bid_change_pct"], bins=bins, labels=labels)
    bucket = inc.groupby("bid_bucket", as_index=False).agg(
        rows=("rev_change_abs","count"),
        rev_lift=("rev_change_abs","sum"),
        avg_lift=("rev_change_abs","mean")
    )
    fig, ax = plt.subplots(figsize=(9,5))
    ax.bar(bucket["bid_bucket"].astype(str), bucket["rev_lift"])
    ax.set_title("Revenue Lift by Bid Increase Bucket (Abs)")
    ax.set_xlabel("Bid Increase Bucket")
    ax.set_ylabel("Total Revenue Lift")
    save_plot(fig, "revenue_lift_by_bid_change_bins.png")

    # 3A) Grouped horizontal bars (Market × Channel): avg rev lift where bid increased
    grp = (inc.groupby(["market","channel"])
             .agg(avg_rev_lift=("rev_change_abs","mean"))
             .reset_index())
    pivot = grp.pivot(index="market", columns="channel", values="avg_rev_lift").fillna(0.0)
    pivot["_total"] = pivot.sum(axis=1)
    pivot = pivot.sort_values("_total", ascending=False).drop(columns=["_total"])

    fig, ax = plt.subplots(figsize=(12,7))
    y = np.arange(len(pivot.index))
    channels = list(pivot.columns)
    bar_h = 0.35 if len(channels) == 2 else 0.8 / max(len(channels), 1)

    for i, ch in enumerate(channels):
        ax.barh(y + (i - (len(channels)-1)/2)*bar_h, pivot[ch].to_numpy(),
                height=bar_h, label=ch)

    ax.set_yticks(y); ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Avg Revenue Lift (where bid increased)")
    ax.set_title("Avg Revenue Lift Where Bid Increased — Market × Channel")
    ax.legend(loc="best")

    # value labels
    for i, ch in enumerate(channels):
        vals = pivot[ch].to_numpy()
        for yi, v in enumerate(vals):
            if v != 0:
                ax.text(v + (0.01 if v >= 0 else -0.01),  # nudge
                        yi + (i - (len(channels)-1)/2)*bar_h,
                        f"{v:.1f}",
                        va="center", ha="left" if v >= 0 else "right", fontsize=8)

    save_plot(fig, "bidup_rev_lift_bar_market_channel.png")

    # 3B) Bubble chart: Avg Bid Change % vs Avg Revenue Lift (size=volume, color=channel)
    bubble = (inc.groupby(["market","channel"])
                .agg(
                    avg_bid_change_pct=("bid_change_pct", lambda s: 100.0 * float(np.mean(s))),
                    avg_rev_lift=("rev_change_abs","mean"),
                    rows=("rev_change_abs","count"),
                )
                .reset_index())

    fig, ax = plt.subplots(figsize=(10,6))
    unique_channels = bubble["channel"].unique().tolist()
    for ch in unique_channels:
        sub = bubble[bubble["channel"] == ch]
        ax.scatter(sub["avg_bid_change_pct"], sub["avg_rev_lift"],
                   s=np.clip(sub["rows"]*2, 30, 400), alpha=0.6, label=ch)

    ax.axvline(0, color="lightgray", linewidth=1)
    ax.axhline(0, color="lightgray", linewidth=1)
    ax.set_title("Where Bid ↑, Did Revenue ↑?  (Bubble size = volume)")
    ax.set_xlabel("Avg Bid Change (%)")
    ax.set_ylabel("Avg Revenue Lift")
    ax.legend(title="Channel")
    save_plot(fig, "bidup_rev_lift_bubble.png")

    # --- SEGMENT LIFT BARS (Market) — aligned window ---
    by_new = df_new.groupby("market", as_index=False).agg(rev_pred=("rev_pred","sum"))
    by_old = base_aligned.groupby("market", as_index=False).agg(rev=("revenue","sum"))
    segm = pd.merge(by_old, by_new, on="market", how="inner")
    segm["lift_abs"] = segm["rev_pred"] - segm["rev"]
    segm["lift_pct"] = (segm["lift_abs"] / np.maximum(segm["rev"], 1e-6)) * 100.0
    segm = segm.sort_values("lift_pct", ascending=False)

    fig, ax = plt.subplots(figsize=(10,6))
    ax.barh(segm["market"], segm["lift_pct"])
    ax.set_title("Revenue Lift by Market (%) — Recommended vs Baseline")
    ax.set_xlabel("Revenue Lift (%)")
    ax.invert_yaxis()
    save_plot(fig, f"lift_by_market_{SCENARIO}.png")

    # --- SEGMENT LIFT BARS (Channel) — aligned window ---
    by_new = df_new.groupby("channel", as_index=False).agg(rev_pred=("rev_pred","sum"))
    by_old = base_aligned.groupby("channel", as_index=False).agg(rev=("revenue","sum"))
    segc = pd.merge(by_old, by_new, on="channel", how="inner")
    segc["lift_abs"] = segc["rev_pred"] - segc["rev"]
    segc["lift_pct"] = (segc["lift_abs"] / np.maximum(segc["rev"], 1e-6)) * 100.0
    segc = segc.sort_values("lift_pct", ascending=False)

    fig, ax = plt.subplots(figsize=(10,6))
    ax.barh(segc["channel"], segc["lift_pct"])
    ax.set_title("Revenue Lift by Channel (%) — Recommended vs Baseline")
    ax.set_xlabel("Revenue Lift (%)")
    ax.invert_yaxis()
    save_plot(fig, f"lift_by_channel_{SCENARIO}.png")

    # --- SEGMENT LIFT BARS (Device) — aligned window ---
    by_new = df_new.groupby("device", as_index=False).agg(rev_pred=("rev_pred","sum"))
    by_old = base_aligned.groupby("device", as_index=False).agg(rev=("revenue","sum"))
    segd = pd.merge(by_old, by_new, on="device", how="inner")
    segd["lift_abs"] = segd["rev_pred"] - segd["rev"]
    segd["lift_pct"] = (segd["lift_abs"] / np.maximum(segd["rev"], 1e-6)) * 100.0
    segd = segd.sort_values("lift_pct", ascending=False)

    fig, ax = plt.subplots(figsize=(10,6))
    ax.barh(segd["device"], segd["lift_pct"])
    ax.set_title("Revenue Lift by Device (%) — Recommended vs Baseline")
    ax.set_xlabel("Revenue Lift (%)")
    ax.invert_yaxis()
    save_plot(fig, f"lift_by_device_{SCENARIO}.png")

    print(f"\n✅ Visuals saved in: {OUT_DIR.resolve()}")
    print(f"Scenario used: {SCENARIO}  |  Params: {PARAMS}")

if __name__ == "__main__":
    main()


```

## Project Structure

SERAH-PMAX/
├─ Dataset/
│  ├─ tcg_sea_dataset_2024_2025.csv
│  ├─ tcg_sea_daily_aggregates.csv
│  ├─ tcg_sea_dataset_2024_2025.csv
├─ Scripts/
│  ├─ config.py
│  ├─ clean_and_engineer.py
│  ├─ split_and_save.py
│  ├─ train_and_recommend.py
│  ├─ compare_baseline_vs_recommended.py
├─ README.md
├─ LICENSE


## License

MIT License

Copyright (c) 2025 rabraham2

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including, without limitation, the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
