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

    # Make sure numeric block is actually numeric
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
