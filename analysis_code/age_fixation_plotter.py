from pathlib import Path
from typing import Optional, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- config
ROOT = Path("/Users/apple/git/large_eye_tracking")
INTERMEDIATE_DIR = ROOT / "data" / "intermediate_data"
OUTPUT_DIR = ROOT / "analysis_results" / "plots" / "fixation_duration_trajectory"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_NAME_FIX = "fixation_duration.png"
OUTPUT_NAME_COMBO = "fixdur_saccamp.png"

WINDOW_SPAN_YEARS = 7          # local mean window width (inclusive, integer ages)
WINDOW_STEP_YEARS = 2          # step between consecutive window centers
LOW_DURATION_THRESHOLD_MS = 125.0
POLY_ORDER = 5
MIN_PARTICIPANTS_PER_POINT = 3

FIX_COLS = {"onset", "offset", "duration", "avg_x", "avg_y", "ID", "label"}
INFO_COLS = {"ID", "Age", "Valid demographics", "Valid task"}


def generate_plot(
    task: str,
    *,
    include_saccade: bool = False,
    min_age: Optional[float] = None,
    max_age: Optional[float] = 70, # too large error bar due to insufficient data after 70 year
    enable_polynomial: bool = True,
    suffix: str = "",
) -> Optional[Path]:
    """Create a trajectory plot for the given task."""
    data_dir = INTERMEDIATE_DIR / task
    fix = cast(pd.DataFrame, pd.read_csv(data_dir / "compiled_fixations.csv", usecols=lambda c: c in FIX_COLS))
    info = cast(pd.DataFrame, pd.read_csv(data_dir / "participant_info.csv", usecols=lambda c: c in INFO_COLS))

    merged = cast(pd.DataFrame, fix.merge(info, on="ID", how="left"))
    merged = merged[(merged["Valid demographics"]) & (merged["Valid task"])].copy()
    merged["Age"] = pd.to_numeric(merged["Age"], errors="coerce")
    age_series = cast(pd.Series, merged["Age"])
    duration_series = cast(pd.Series, merged["duration"])
    merged = merged[age_series.notna() & duration_series.notna()].copy()
    if min_age is not None:
        merged = merged[merged["Age"] >= min_age].copy()
    if max_age is not None:
        merged = merged[merged["Age"] <= max_age].copy()
    merged = cast(pd.DataFrame, merged)
    if merged.empty:
        print(f"[{task}] skipped: no data after filters.")
        return None

    merged = cast(pd.DataFrame, merged.sort_values(by=["ID", "onset"]))
    merged["duration_ms"] = merged["duration"] * 1000.0
    merged["lt_threshold"] = merged["duration_ms"] < LOW_DURATION_THRESHOLD_MS
    merged["dx"] = merged.groupby("ID")["avg_x"].diff()
    merged["dy"] = merged.groupby("ID")["avg_y"].diff()
    merged["sacc_amp_px"] = np.sqrt(merged["dx"] ** 2 + merged["dy"] ** 2)

    grouped = merged.groupby("ID")
    per_id = pd.DataFrame(
        {
            "Age": grouped["Age"].first(),
            "mean_duration_ms": grouped["duration_ms"].mean(),
            "prop_lt_threshold": grouped["lt_threshold"].mean(),
            "mean_sacc_amp_px": grouped["sacc_amp_px"].mean(),
        }
    ).dropna(subset=["Age"])
    if per_id.empty:
        print(f"[{task}] skipped: no per-participant stats.")
        return None

    ages = per_id["Age"].to_numpy()
    age_min = np.floor(ages.min())
    age_max = np.ceil(ages.max())
    centers = np.arange(age_min + (WINDOW_SPAN_YEARS - 1) / 2, age_max, WINDOW_STEP_YEARS)

    rows = []
    for center in centers:
        lower = center - (WINDOW_SPAN_YEARS - 1) / 2
        upper = center + (WINDOW_SPAN_YEARS - 1) / 2
        window = per_id[(per_id["Age"] >= lower) & (per_id["Age"] <= upper)]
        if len(window) < MIN_PARTICIPANTS_PER_POINT:
            continue
        dur = window["mean_duration_ms"]
        sac = cast(pd.Series, window["mean_sacc_amp_px"]).dropna()
        if sac.empty:
            sac_mean = np.nan
            sac_sem = np.nan
        elif len(sac) == 1:
            sac_mean = float(sac.iloc[0])
            sac_sem = 0.0
        else:
            sac_mean = float(sac.mean())
            sac_sem = float(sac.std(ddof=1) / np.sqrt(len(sac)))
        rows.append(
            {
                "age_center": center,
                "duration_mean": dur.mean(),
                "duration_sem": dur.std(ddof=1) / np.sqrt(len(dur)) if len(dur) > 1 else 0.0,
                "prop_lt": window["prop_lt_threshold"].mean() * 100.0,
                "sacc_amp_mean": sac_mean,
                "sacc_amp_sem": sac_sem,
            }
        )

    local = pd.DataFrame(rows)
    if local.empty:
        print(f"[{task}] skipped: no local means.")
        return None

    fig, ax_left = plt.subplots(figsize=(7, 4))
    ax_left.errorbar(
        local["age_center"],
        local["duration_mean"],
        yerr=local["duration_sem"],
        fmt="-o",
        color="tab:blue",
        capsize=3,
        linewidth=1.8,
        markersize=4.5,
        label="FixDur (ms)",
    )

    if enable_polynomial and per_id["Age"].nunique() > POLY_ORDER:
        coeffs = np.polyfit(per_id["Age"], per_id["mean_duration_ms"], POLY_ORDER)
        age_fit = np.linspace(local["age_center"].min(), local["age_center"].max(), 400)
        ax_left.plot(age_fit, np.polyval(coeffs, age_fit), color="tab:red", linewidth=2.0, label="5th-order fit")

    dataset_label = "Free-viewing" if task == "freeviewing" else "Search"
    title_suffix = ""
    if min_age is not None and max_age is not None:
        title_suffix = f" (ages {min_age:g}–{max_age:g})"
    elif min_age is not None:
        title_suffix = f" (ages ≥{min_age:g})"
    elif max_age is not None:
        title_suffix = f" (≤{max_age:g})"

    if include_saccade:
        ax_right = ax_left.twinx()
        ax_right.errorbar(
            local["age_center"],
            local["sacc_amp_mean"],
            yerr=local["sacc_amp_sem"],
            fmt="-s",
            color="tab:orange",
            capsize=3,
            linewidth=1.6,
            markersize=4,
            label="SaccAmp (px)",
        )
        ax_right.set_ylabel("Saccade Amplitude (pixels)")
        handles_left, labels_left = ax_left.get_legend_handles_labels()
        handles_right, labels_right = ax_right.get_legend_handles_labels()
        ax_left.legend(handles_left + handles_right, labels_left + labels_right)
        output_name = OUTPUT_NAME_COMBO.format(task=task, suffix=suffix)
        ax_left.set_title(f"{dataset_label} dataset: FixDur & SaccAmp{title_suffix}")
    else:
        ax_right = ax_left.twinx()
        ax_right.plot(
            local["age_center"],
            local["prop_lt"],
            linestyle="--",
            color="limegreen",
            linewidth=1.6,
            label=f"<{int(LOW_DURATION_THRESHOLD_MS)} ms proportion",
        )
        ax_right.set_ylabel(f"Proportion of FixDur < {int(LOW_DURATION_THRESHOLD_MS)} ms (%)")
        handles_left, labels_left = ax_left.get_legend_handles_labels()
        handles_right, labels_right = ax_right.get_legend_handles_labels()
        ax_left.legend(handles_left + handles_right, labels_left + labels_right)
        output_name = OUTPUT_NAME_FIX.format(task=task, suffix=suffix)
        ax_left.set_title(f"{dataset_label} dataset: Developmental Trajectory of Fixation Duration{title_suffix}")

    ax_left.set_xlabel("Age (years)")
    ax_left.set_ylabel("Mean Fixation Duration (ms)")
    ax_left.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax_left.spines["top"].set_visible(False)
    ax_right.spines["top"].set_visible(False)
    plt.tight_layout()

    out_path = OUTPUT_DIR / f"{task}_{output_name}"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[{task}] saved {out_path}")
    return out_path


def main() -> None:
    generate_plot("freeviewing", include_saccade=False, suffix="")
    generate_plot("freeviewing", include_saccade=True, suffix="_free_poly_off")
    generate_plot("search", include_saccade=False, suffix="")
    generate_plot("search", include_saccade=True, suffix="_search_poly_off")


if __name__ == "__main__":
    main()

