# -*- coding: utf-8 -*-
"""
Modified on Sat May  2 17:09:27 2026

@author: jannatul

Sensitivity validation for IBM:
1. MIN_SHIFT_DISTANCE_M
2. BREED_MIN_EXCESS

Run-safe version:
- in case of interruptions and rerun, skips completed parameter/seed cases
- saves detailed CSV after each case
- creates summary CSV and plots
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

SENS_OUT_DIR = r"[path]/Sensitivity_validation_final"

YEARS = [2000, 2001, 2012, 2018, 2022]
SEEDS = [1, 2, 3]

PARAMETER_TESTS = {
    "MIN_SHIFT_DISTANCE_M": [1000, 3000, 10000],
    "BREED_MIN_EXCESS": [0.10, 0.30, 0.50],
}

os.makedirs(SENS_OUT_DIR, exist_ok=True)

DETAILED_CSV = os.path.join(SENS_OUT_DIR, "sensitivity_validation_detailed.csv")
SUMMARY_CSV = os.path.join(SENS_OUT_DIR, "sensitivity_validation_summary.csv")


# =========================================================
# LOAD MODEL MODULE FRESH EACH TIME
# =========================================================
def load_model_module():
    module_name = "mdl"

    if module_name in sys.modules:
        del sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, MODEL_PATH)
    mdl = importlib.util.module_from_spec(spec)

    sys.modules[module_name] = mdl
    spec.loader.exec_module(mdl)

    return mdl


# =========================================================
# RUN ONE CASE
# =========================================================
def run_one_case(parameter_name, parameter_value, seed):
    mdl = load_model_module()

    # overwrite selected parameter
    setattr(mdl, parameter_name, parameter_value)

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
            print(f"    Skipping year {year}; missing rasters:")
            for f in missing:
                print("     ", f)
            continue

        env = mdl.Environment(lulc, ndvi, ndwi, frag_paths_year)
        cfg = mdl.SimulationConfig(start_year=year, end_year=year, rng_seed=seed)
        scenario = mdl.Scenario()
        ibm = mdl.IBM(env, current_birds, cfg, scenario)
        ibm.apply_scenario()

        out_dir_year = os.path.join(
            SENS_OUT_DIR,
            "_tmp_runs",
            parameter_name,
            f"value_{parameter_value}",
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
            "parameter": parameter_name,
            "value": parameter_value,
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
            "mean_center_score": np.nan,
            "mean_best_site_score": np.nan,
            "mean_site_improvement_raw": np.nan,
            "mean_site_improvement_effective": np.nan,
            "mean_shift_distance_m": np.nan,
        }

    df = pd.read_csv(last_summary)

    n_alive = int(df["alive"].sum())
    n_successful_breeders = int((df["final_state"] == "successful_breeder").sum())
    n_failed_breeders = int((df["final_state"] == "failed_breeder").sum())
    n_non_breeders = int((df["final_state"] == "non_breeder").sum())
    n_shifters = int((df["final_state"] == "shifter").sum())
    n_attempted_breeding = int(df["attempted_breeding"].sum())

    mean_energy_cumulative = float(df["energy_cum"].mean())
    mean_loyalty = float(df["breeding_site_loyalty"].mean())

    mean_center_score = float(df["current_center_score"].mean())
    mean_best_site_score = float(df["best_site_score"].mean())

    df["site_improvement_raw"] = df["best_site_score"] - df["current_center_score"]
    mean_site_improvement_raw = float(df["site_improvement_raw"].mean())

    # effective improvement is not saved directly in summary,
    # approximate using loyalty subtraction, consistent with model logic
    df["site_improvement_effective"] = df["site_improvement_raw"] - df["breeding_site_loyalty"]
    mean_site_improvement_effective = float(df["site_improvement_effective"].mean())

    if last_shift_csv is not None and os.path.exists(last_shift_csv):
        df_shift = pd.read_csv(last_shift_csv)
        mean_shift_distance_m = float(df_shift["shift_distance_m"].mean()) if not df_shift.empty else np.nan
    else:
        mean_shift_distance_m = np.nan

    return {
        "parameter": parameter_name,
        "value": parameter_value,
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
        "mean_center_score": mean_center_score,
        "mean_best_site_score": mean_best_site_score,
        "mean_site_improvement_raw": mean_site_improvement_raw,
        "mean_site_improvement_effective": mean_site_improvement_effective,
        "mean_shift_distance_m": mean_shift_distance_m,
    }


# =========================================================
# SAVE / RESUME HELPERS
# =========================================================
def load_existing_results():
    if os.path.exists(DETAILED_CSV):
        return pd.read_csv(DETAILED_CSV)
    return pd.DataFrame()


def case_already_done(existing_df, parameter_name, parameter_value, seed):
    if existing_df.empty:
        return False

    sub = existing_df[
        (existing_df["parameter"] == parameter_name)
        & (existing_df["value"].astype(float) == float(parameter_value))
        & (existing_df["seed"].astype(int) == int(seed))
    ]

    return len(sub) > 0


def append_and_save(row):
    existing = load_existing_results()
    new = pd.DataFrame([row])

    combined = pd.concat([existing, new], ignore_index=True)

    combined = combined.drop_duplicates(
        subset=["parameter", "value", "seed"],
        keep="last"
    )

    combined.to_csv(DETAILED_CSV, index=False)


# =========================================================
# SUMMARIZE
# =========================================================
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
        "mean_center_score",
        "mean_best_site_score",
        "mean_site_improvement_raw",
        "mean_site_improvement_effective",
        "mean_shift_distance_m",
    ]

    rows = []

    for (param, val), sub in df.groupby(["parameter", "value"], dropna=False):
        row = {
            "parameter": param,
            "value": val,
            "n_runs": len(sub),
        }

        for col in metric_cols:
            row[f"{col}_mean"] = float(np.nanmean(sub[col]))
            row[f"{col}_sd"] = float(np.nanstd(sub[col], ddof=1)) if len(sub) > 1 else 0.0

        rows.append(row)

    return pd.DataFrame(rows).sort_values(["parameter", "value"])


# =========================================================
# PLOTS
# =========================================================
def make_plot(summary_df, parameter_name, ycol, ylabel, fname):
    sub = summary_df[summary_df["parameter"] == parameter_name].copy()

    if sub.empty:
        return

    sub = sub.sort_values("value")

    plt.figure(figsize=(5.3, 4.0))

    plt.errorbar(
        sub["value"],
        sub[f"{ycol}_mean"],
        yerr=sub[f"{ycol}_sd"],
        marker="o",
        capsize=4,
        linewidth=1.2,
    )

    plt.xlabel(parameter_name)
    plt.ylabel(ylabel)
    plt.tight_layout()

    plt.savefig(os.path.join(SENS_OUT_DIR, fname), dpi=300)
    plt.close()


def make_all_plots(summary_df):
    for param in PARAMETER_TESTS.keys():
        safe_param = param.lower()

        make_plot(
            summary_df,
            param,
            "n_alive",
            "Number of alive birds",
            f"{safe_param}_alive.png",
        )

        make_plot(
            summary_df,
            param,
            "n_successful_breeders",
            "Number of successful breeders",
            f"{safe_param}_successful_breeders.png",
        )

        make_plot(
            summary_df,
            param,
            "n_failed_breeders",
            "Number of failed breeders",
            f"{safe_param}_failed_breeders.png",
        )

        make_plot(
            summary_df,
            param,
            "n_non_breeders",
            "Number of non-breeders",
            f"{safe_param}_non_breeders.png",
        )

        make_plot(
            summary_df,
            param,
            "n_shifters",
            "Number of shifters",
            f"{safe_param}_shifters.png",
        )

        make_plot(
            summary_df,
            param,
            "mean_energy_cumulative",
            "Mean cumulative energy",
            f"{safe_param}_mean_energy.png",
        )

        make_plot(
            summary_df,
            param,
            "mean_site_improvement_raw",
            "Mean raw site improvement",
            f"{safe_param}_raw_site_improvement.png",
        )

        make_plot(
            summary_df,
            param,
            "mean_site_improvement_effective",
            "Mean effective site improvement",
            f"{safe_param}_effective_site_improvement.png",
        )


# =========================================================
# MAIN
# =========================================================
def main():
    print("\nRunning IBM sensitivity validation")
    print("Model file:", MODEL_PATH)
    print("Output dir :", SENS_OUT_DIR)
    print("Years      :", YEARS)
    print("Seeds      :", SEEDS)

    start_time = time.time()

    existing_df = load_existing_results()

    for parameter_name, values in PARAMETER_TESTS.items():
        print(f"\n=================================================")
        print(f"Testing parameter: {parameter_name}")
        print(f"Values: {values}")
        print(f"=================================================")

        for value in values:
            print(f"\n### {parameter_name} = {value}")

            for seed in SEEDS:
                existing_df = load_existing_results()

                if case_already_done(existing_df, parameter_name, value, seed):
                    print(f"  Seed {seed}: already completed, skipping.")
                    continue

                print(f"  Seed {seed}: running...")

                case_start = time.time()

                row = run_one_case(parameter_name, value, seed)

                elapsed = (time.time() - case_start) / 60
                print(f"  Seed {seed}: completed in {elapsed:.1f} min")

                append_and_save(row)

                # update summary after every completed case
                df_results = load_existing_results()
                df_summary = summarize_results(df_results)
                df_summary.to_csv(SUMMARY_CSV, index=False)
                make_all_plots(df_summary)

    df_results = load_existing_results()
    df_summary = summarize_results(df_results)

    df_results.to_csv(DETAILED_CSV, index=False)
    df_summary.to_csv(SUMMARY_CSV, index=False)
    make_all_plots(df_summary)

    total_elapsed = (time.time() - start_time) / 3600

    print("\nDone.")
    print(f"Total elapsed time: {total_elapsed:.2f} hours")
    print("Detailed:", DETAILED_CSV)
    print("Summary :", SUMMARY_CSV)

    tmp_dir = os.path.join(SENS_OUT_DIR, "_tmp_runs")

    # keep temp folders for inspection by default
    print("\nTemporary run folders were kept for inspection:")
    print(tmp_dir)
    print("Delete manually later if disk space becomes an issue.")


if __name__ == "__main__":
    main()