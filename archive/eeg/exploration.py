"""
Group-Level Behavioral Exploration

Coordinates the generation of group-level plots and metrics
from synchronized valence–arousal data across subjects.

This module:
    - Receives the combined behavioral DataFrame
    - Calls visualization utilities for each analysis type
    - Saves all figures and metrics in a structured output folder

Required packages:
    - pandas
    - numpy
    - matplotlib
    - scipy

Author:
Created: 10.08.2025
"""

# --------------------------------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------------------------------

from pathlib import Path
from typing import Optional, Dict

import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis, iqr, pearsonr


from eeg.visualization import (
    plot_group_histograms,
    plot_group_histograms_originallabels,
    plot_2d_density,
    plot_grand_mean_timeseries,
    plot_slope_histograms,
    plot_subject_boxplots,
    plot_correlation_matrices,
    plot_subject_value_boxplots,
    plot_individual_subject_timeseries
)

# --------------------------------------------------------------------------------------------
# Functions
# --------------------------------------------------------------------------------------------
def save_subject_summary_stats(df: pd.DataFrame, output_path: Path):
    """
    Compute and save per-subject summary statistics for valence and arousal.
    
    Args:
        df (pd.DataFrame): DataFrame containing columns ['subject_id', 'valence', 'arousal']
        output_path (Path): Path to save the CSV file
    """
    agg_funcs = {
        "valence": ['mean', 'max', 'median', 'min', 'std'],
        "arousal": ['mean', 'max', 'median', 'min', 'std']
    }
    
    summary_df = df.groupby('subject_id').agg(agg_funcs)
    # Flatten multi-level columns
    summary_df.columns = ['_'.join(col) for col in summary_df.columns]
    summary_df.reset_index(inplace=True)
    
    summary_df.to_csv(output_path, index=False)


def compute_group_metrics(df: pd.DataFrame) -> dict:
    """
    Compute group-level metrics for valence and arousal.

    Args:
        df (pd.DataFrame): Must contain columns 'valence' and 'arousal'.

    Returns:
        dict: Metrics dictionary with descriptive stats and quadrant coverage.
    """
    val = df["valence"].to_numpy()
    aro = df["arousal"].to_numpy()

    metrics = {}

    for label, arr in [("valence", val), ("arousal", aro)]:
        metrics[f"{label}_mean"] = np.nanmean(arr)
        metrics[f"{label}_median"] = np.nanmedian(arr)
        metrics[f"{label}_skewness"] = skew(arr, nan_policy="omit")
        metrics[f"{label}_kurtosis"] = kurtosis(arr, nan_policy="omit")
        metrics[f"{label}_variance"] = np.nanvar(arr)
        metrics[f"{label}_iqr"] = iqr(arr, nan_policy="omit")

    # Correlation between valence and arousal
    valid_mask = ~np.isnan(val) & ~np.isnan(aro)
    if valid_mask.sum() > 2:
        metrics["valence_arousal_corr"], _ = pearsonr(val[valid_mask], aro[valid_mask])
    else:
        metrics["valence_arousal_corr"] = np.nan

    # Quadrant coverage
    q1 = np.sum((val > 0) & (aro > 0))  # High valence, high arousal
    q2 = np.sum((val < 0) & (aro > 0))  # Low valence, high arousal
    q3 = np.sum((val < 0) & (aro < 0))  # Low valence, low arousal
    q4 = np.sum((val > 0) & (aro < 0))  # High valence, low arousal
    total = len(val)

    metrics["quadrant1_pct"] = 100 * q1 / total
    metrics["quadrant2_pct"] = 100 * q2 / total
    metrics["quadrant3_pct"] = 100 * q3 / total
    metrics["quadrant4_pct"] = 100 * q4 / total

    return metrics

def run_group_exploration(
    df: pd.DataFrame,
    output_root: Path,
    save_metrics: bool = True
) -> Optional[Dict[str, float]]:
    output_root.mkdir(parents=True, exist_ok=True)

    # Metrics
    metrics = compute_group_metrics(df)

    # ---- Plots ----
    plot_group_histograms(df, output_root / "group_histograms.png")
    plot_group_histograms_originallabels(df, output_root / "original_group_histograms.png")
    plot_2d_density(df, output_root / "2d_density_valence_arousal.png")
    plot_grand_mean_timeseries(df, output_root / "grand_mean_timeseries.png")
    plot_slope_histograms(df, output_root / "slope_distributions.png")
    plot_subject_boxplots(df, output_root / "subject_boxplots.png")
    plot_correlation_matrices(df, output_root)
    plot_individual_subject_timeseries(df, output_root / "individual_subject_timeseries.html")
    plot_subject_value_boxplots(df, output_root / "valence_arousal_subject_boxplots.png")

    # ---- Save and plot subject summary stats ----
    summary_csv_path = output_root / "subject_summary_stats.csv"
    save_subject_summary_stats(df, summary_csv_path)

    summary_df = pd.read_csv(summary_csv_path)

    from eeg.visualization import plot_subject_summary_stats
    summary_plot_path = output_root / "subject_summary_stats_plot.png"
    plot_subject_summary_stats(summary_df, summary_plot_path)

    # ---- Save metrics ----
    if save_metrics:
        metrics_path = output_root / "exploration_metrics.csv"
        pd.DataFrame([metrics]).to_csv(metrics_path, index=False)

    return metrics

