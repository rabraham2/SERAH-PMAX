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
