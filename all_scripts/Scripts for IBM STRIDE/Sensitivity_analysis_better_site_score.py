# -*- coding: utf-8 -*-
"""
Modified on Sat May  2 12:01:09 2026
Author: Jannatul

Sensitivity analysis for MIN_BETTER_SITE_SCORE
Uses the finished IBM as-is, without modifying the model code.
"""

import os
import sys
import time
import shutil
import importlib.util
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# USER SETTINGS
# =========================================================
MODEL_PATH = r"[path]/STRIDE v2.3.py"
SENS_OUT_DIR = r"[path]/Sensitivity_MIN_BETTER_SITE_SCORE"
THRESHOLD_VALUES = [0.00]
SEEDS = list(range(1,4))
YEARS = [2000, 2001, 2012, 2018, 2022]

os.makedirs(SENS_OUT_DIR, exist_ok=True)


# =========================================================
# LOAD MODEL MODULE FRESH EACH TIME
# =========================================================
def load_model_module():
    module_name = "mdl"

    # remove old copy if already loaded
    if module_name in sys.modules:
        del sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, MODEL_PATH)
    mdl = importlib.util.module_from_spec(spec)

    # critical: register before executing
    sys.modules[module_name] = mdl
    spec.loader.exec_module(mdl)

    return mdl


# =========================================================
# HELPERS
# =========================================================
def run_one_case(threshold, seed):
    mdl = load_model_module()

    mdl.MIN_BETTER_SITE_SCORE = threshold

    current_birds = mdl.load_birds_from_points(
        mdl.POINTS_PATH,
        mdl.FIXED_RSF_PATH,
        mdl.RANDOM_RSF_PATH,
        mdl.STEP_MINUTES_DEFAULT,
    )

    last_summary = None
    last_shift_csv = None
    last_completed_year = None

    for year in YEARS:
        current_birds = [b for b in current_birds if b.alive]
        if len(current_birds) == 0:
            break

        lulc = f"[path]/LC_{year}.tif"
        ndvi = f"[path]/NDVI_{year}_JulAug.tif"
        ndwi = f"[path]/MNDWI_{year}_JulAug.tif"

        frag_paths_year = {
            "SHEI": f"[path]/SHEI_{year}.tif",
            "ED": f"[path]/ED_{year}.tif",
        }

        all_rasters = [lulc, ndvi, ndwi] + list(frag_paths_year.values())
        missing = [f for f in all_rasters if not os.path.exists(f)]
        if missing:
            print(f"    Skipping year {year}; missing rasters.")
            continue

        env = mdl.Environment(lulc, ndvi, ndwi, frag_paths_year)
        cfg = mdl.SimulationConfig(start_year=year, end_year=year, rng_seed=seed)
        scenario = mdl.Scenario()
        ibm = mdl.IBM(env, current_birds, cfg, scenario)
        ibm.apply_scenario()

        out_dir_year = os.path.join(
            SENS_OUT_DIR,
            "_tmp_runs",
            f"thr_{threshold:.2f}",
            f"seed_{seed}",
            str(year),
        )
        os.makedirs(out_dir_year, exist_ok=True)

        ibm.run([year], out_dir_year)

        summary_path = os.path.join(out_dir_year, "summaries", f"summary_{year}.csv")
        shift_path = os.path.join(out_dir_year, "summaries", f"home_range_shifts_{year}.csv")

        if os.path.exists(summary_path):
            last_summary = summary_path
            last_shift_csv = shift_path if os.path.exists(shift_path) else None
            last_completed_year = year

    if last_summary is None:
        return {
            "threshold": threshold,
            "seed": seed,
            "final_year": np.nan,
            "n_alive": np.nan,
            "n_successful_breeders": np.nan,
            "n_failed_breeders": np.nan,
            "n_non_breeders": np.nan,
            "n_shifters": np.nan,
            "n_attempted_breeding": np.nan,
            "mean_energy_cumulative": np.nan,
            "mean_loyalty": np.nan,
            "mean_shift_distance_m": np.nan,
        }

    df = pd.read_csv(last_summary)

    n_alive = int(df["alive"].sum())
    n_successful_breeders = int((df["final_state"] == "successful_breeder").sum())
    n_failed_breeders = int((df["final_state"] == "failed_breeder").sum())
    n_non_breeders = int((df["final_state"] == "non_breeder").sum())
    n_shifters = int((df["final_state"] == "shifter").sum())
    n_attempted_breeding = int(df["attempted_breeding"].sum())
    mean_energy_cumulative = float(df["energy_cumulative"].mean())
    mean_loyalty = float(df["breeding_site_loyalty"].mean())

    if last_shift_csv is not None and os.path.exists(last_shift_csv):
        df_shift = pd.read_csv(last_shift_csv)
        mean_shift_distance_m = float(df_shift["shift_distance_m"].mean()) if not df_shift.empty else np.nan
    else:
        mean_shift_distance_m = np.nan

    return {
        "threshold": threshold,
        "seed": seed,
        "final_year": last_completed_year,
        "n_alive": n_alive,
        "n_successful_breeders": n_successful_breeders,
        "n_failed_breeders": n_failed_breeders,
        "n_non_breeders": n_non_breeders,
        "n_shifters": n_shifters,
        "n_attempted_breeding": n_attempted_breeding,
        "mean_energy_cumulative": mean_energy_cumulative,
        "mean_loyalty": mean_loyalty,
        "mean_shift_distance_m": mean_shift_distance_m,
    }


def summarize_results(df):
    metric_cols = [
        "n_alive",
        "n_successful_breeders",
        "n_failed_breeders",
        "n_non_breeders",
        "n_shifters",
        "n_attempted_breeding",
        "mean_energy_cumulative",
        "mean_loyalty",
        "mean_shift_distance_m",
    ]

    rows = []
    for thr, sub in df.groupby("threshold", dropna=False):
        row = {"threshold": thr, "n_runs": len(sub)}
        for col in metric_cols:
            row[f"{col}_mean"] = float(np.nanmean(sub[col]))
            row[f"{col}_sd"] = float(np.nanstd(sub[col], ddof=1)) if len(sub) > 1 else 0.0
        rows.append(row)

    return pd.DataFrame(rows).sort_values("threshold")


def make_plot(summary_df, ycol, ylabel, fname):
    plt.figure(figsize=(5.3, 4.0))
    plt.errorbar(
        summary_df["threshold"],
        summary_df[f"{ycol}_mean"],
        yerr=summary_df[f"{ycol}_sd"],
        marker="o",
        capsize=4,
        linewidth=1.2,
    )
    plt.xlabel("MIN_BETTER_SITE_SCORE")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(os.path.join(SENS_OUT_DIR, fname), dpi=300)
    plt.close()


# =========================================================
# MAIN
# =========================================================
def main():
    all_rows = []

    print("\nRunning sensitivity analysis for MIN_BETTER_SITE_SCORE")
    print("Model file:", MODEL_PATH)
    print("Output dir :", SENS_OUT_DIR)

    for thr in THRESHOLD_VALUES:
        print(f"\n### Threshold = {thr:.2f}")
        for seed in SEEDS:
            print(f"  Seed {seed}")
            row = run_one_case(thr, seed)
            all_rows.append(row)

    df_results = pd.DataFrame(all_rows)
    df_summary = summarize_results(df_results)

    detailed_csv = os.path.join(SENS_OUT_DIR, "sensitivity_min_better_site_score_detailed.csv")
    summary_csv = os.path.join(SENS_OUT_DIR, "sensitivity_min_better_site_score_summary.csv")

    df_results.to_csv(detailed_csv, index=False)
    df_summary.to_csv(summary_csv, index=False)

    make_plot(df_summary, "n_shifters", "Number of shifters", "shifters_vs_threshold.png")
    make_plot(df_summary, "n_successful_breeders", "Number of successful breeders", "successful_breeders_vs_threshold.png")
    make_plot(df_summary, "n_alive", "Number of alive birds", "alive_vs_threshold.png")

    print("\nDone.")
    print("Detailed:", detailed_csv)
    print("Summary :", summary_csv)

    tmp_dir = os.path.join(SENS_OUT_DIR, "_tmp_runs")

    if os.path.exists(tmp_dir):
        try:
            shutil.rmtree(tmp_dir)
        except PermissionError:
            print("Temp folder could not be deleted now. File is locked.")
            print("You can delete it manually later.")
            print("Temporary run folders deleted.")


if __name__ == "__main__":
    main()