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

def winsorize(df, cols, q_lo=0.01, q_hi=0.99):
    """Clip outliers (stabilize metrics)."""
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

    print(f"✅ Cleaned dataset saved → {out_csv}")
    print(f"Final rows: {len(df):,}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Clean & engineer TCG SEA dataset", add_help=False)
    parser.add_argument("--raw", type=str, default=str(DEFAULT_RAW))
    parser.add_argument("--out", type=str, default=str(CLEAN_PATH))
    # 👇 This line is the key: ignore any extra args the console adds
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
