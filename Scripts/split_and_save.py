# src/split_chrono.py
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
        print("✅ Saved Parquet splits.")
    except Exception as err:  # noqa: BLE001
        print("⚠️ Parquet write failed, saving as CSV instead:")
        print(traceback.format_exc())
        x_train.to_csv(str(x_train_path).replace(".parquet", ".csv"), index=False)
        x_valid.to_csv(str(x_valid_path).replace(".parquet", ".csv"), index=False)
        x_test.to_csv(str(x_test_path).replace(".parquet", ".csv"), index=False)
        y_train.to_csv(str(y_train_path).replace(".parquet", ".csv"), index=False)
        y_valid.to_csv(str(y_valid_path).replace(".parquet", ".csv"), index=False)
        y_test.to_csv(str(y_test_path).replace(".parquet", ".csv"), index=False)
        print("✅ Saved CSV splits.")


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
