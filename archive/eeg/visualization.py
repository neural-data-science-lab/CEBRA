"""
Visualization Utilities for Emotion and EEG Embeddings

Contains utility functions to visualize valence-arousal distributions,
histograms, and the emotion color wheel based on affective angle mapping.

Required packages:
    - numpy
    - matplotlib
    - pathlib

Author:
Created: 10.06.2025
Last updated: 21.07.2025
"""

# --------------------------------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------------------------------

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from eeg.colors import compute_angle_vector_length, valence_arousal_emotion_color

# --------------------------------------------------------------------------------------------
# Functions
# --------------------------------------------------------------------------------------------

def debug_valence_arousal_distribution(
    valence: np.ndarray,
    arousal: np.ndarray,
    subject_key: Optional[str] = None,
    output_root: Optional[Path] = None,
    config_str: Optional[str] = None
) -> None:
    """
    Visualize valence-arousal distribution and save plots if output directory is provided.

    Args:
        valence (np.ndarray): Valence values array.
        arousal (np.ndarray): Arousal values array.
        subject_key (str, optional): Subject identifier for saving.
        output_root (Path, optional): Root directory path to save plots.
        config_str (str, optional): Configuration string for filenames.

    Returns:
        None
    """
    angle, vector_length = compute_angle_vector_length(valence, arousal)
    colors = valence_arousal_emotion_color(valence, arousal)

    save_dir = None
    if output_root and subject_key and config_str:
        save_dir = Path(output_root) / "exploration" / subject_key
        save_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: Scatter Valence vs Arousal
    plt.figure(figsize=(6, 6))
    plt.scatter(valence, arousal, c=colors, s=8)
    plt.axhline(0, color='gray', linestyle='--')
    plt.axvline(0, color='gray', linestyle='--')
    plt.xlabel("Valence")
    plt.ylabel("Arousal")
    plt.title("Valence-Arousal Distribution")
    plt.gca().set_aspect('equal')
    if save_dir:
        plt.savefig(save_dir / f"VA_distribution_{config_str}.png", dpi=150)
        plt.close()
    else:
        plt.show()

    # Plot 2: Histogram of Angles
    plt.figure()
    plt.hist(angle, bins=36)
    plt.title("Histogram of Valence-Arousal Angles")
    plt.xlabel("Angle (°)")
    plt.ylabel("Frequency")
    if save_dir:
        plt.savefig(save_dir / f"VA_angle_histogram_{config_str}.png", dpi=150)
        plt.close()
    else:
        plt.show()

    # Plot 3: Histogram of Range (Vector Length)
    plt.figure()
    plt.hist(vector_length, bins=36)
    plt.title("Histogram of Intensity (Vector Length)")
    plt.xlabel("Vector Length")
    plt.ylabel("Frequency")
    if save_dir:
        plt.savefig(save_dir / f"VA_vector_length_histogram_{config_str}.png", dpi=150)
        plt.close()
    else:
        plt.show()


def plot_valence_arousal_color_wheel(
    valence: np.ndarray,
    arousal: np.ndarray,
    res: int = 300,
    desaturate_color: Tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> None:
    """
    Plot the custom valence-arousal emotional color wheel with radial saturation.

    Args:
        res (int): Resolution of the grid for plotting.
        desaturate_color (Tuple[float, float, float]): RGB color to mix toward for desaturation.

    Returns:
        None
    """
    v_min, v_max = np.min(valence), np.max(valence)
    a_min, a_max = np.min(arousal), np.max(arousal)

    val_grid, aro_grid = np.meshgrid(
        np.linspace(v_min, v_max, res),
        np.linspace(a_min, a_max, res)
    )

    val_flat, aro_flat = val_grid.flatten(), aro_grid.flatten()
    colors = valence_arousal_emotion_color(val_flat, aro_flat, desaturate_color)
    image = colors.reshape(res, res, 3)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image, extent=(v_min, v_max, a_min, a_max), origin="lower")
    ax.set_xlabel("Valence")
    ax.set_ylabel("Arousal")
    ax.set_title("Valence-Arousal Color Wheel\n(Hue by Emotion, Saturation by Intensity)")

    # Annotate key angles 
    labels = {
        (0.6, 0.6): "Joyfull\n(45°)",
        (0.0, 0.8): "Tense/Excited\n(90°)",
        (-0.6, 0.6): "Angry\n(135°)",
        (-0.8, 0.0): "Frustrated/Depressed\n(180°)",
        (-0.6, -0.6): "Sad\n(225°)",
        (0.0, -0.8): "Tired/Calm\n(270°)",
        (0.6, -0.6): "Relaxed\n(315°)",
        (0.8, 0.0): "Content/Happy\n(0°)",
        (0.0, 0.0): "Neutral\n"
    }

    for (x, y), label in labels.items():
        ax.text(x, y, label, ha='center', va='center', fontsize=8,
                color='black', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    plt.grid(False)
    plt.tight_layout()
    plt.show()


def plot_group_histograms(df: pd.DataFrame, save_path: Path) -> None:
    """Plot histograms for valence and arousal distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, col in zip(axes, ["valence", "arousal"]):
        arr = df[col].to_numpy()
        ax.hist(arr, bins=30, alpha=0.7, color="skyblue", edgecolor="black")
        ax.axvline(np.nanmean(arr), color="red", linestyle="--", label="Mean")
        ax.axvline(np.nanmedian(arr), color="green", linestyle=":", label="Median")
        ax.set_title(f"{col.capitalize()} Histogram")
        ax.set_xlabel(col.capitalize())
        ax.set_ylabel("Frequency")
        ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_group_histograms_originallabels(df: pd.DataFrame, save_path: Path) -> None:
    """Plot histograms for valence and arousal distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, col in zip(axes, ["flubber_frequency", "flubber_amplitude"]):
        arr = df[col].to_numpy()
        ax.hist(arr, bins=30, alpha=0.7, color="skyblue", edgecolor="black")
        ax.axvline(np.nanmean(arr), color="red", linestyle="--", label="Mean")
        ax.axvline(np.nanmedian(arr), color="green", linestyle=":", label="Median")
        ax.set_title(f"{col.capitalize()} Histogram")
        ax.set_xlabel(col.capitalize())
        ax.set_ylabel("Frequency")
        ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()



def plot_2d_density(df: pd.DataFrame, save_path: Path) -> None:
    """Plot 2D density heatmap for valence vs arousal."""
    plt.figure(figsize=(6, 6))
    plt.hist2d(df["valence"], df["arousal"], bins=50, cmap="viridis")
    plt.colorbar(label="Count")
    plt.axhline(0, color="gray", linestyle="--")
    plt.axvline(0, color="gray", linestyle="--")
    plt.xlabel("Valence")
    plt.ylabel("Arousal")
    plt.title("Valence–Arousal Density Heatmap")
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_grand_mean_timeseries(df: pd.DataFrame, save_path: Path) -> None:
    """Plot grand mean ± SD time series for valence and arousal."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    for ax, col in zip(axes, ["valence", "arousal"]):
        grouped = df.groupby("timestamp")[col]
        mean = grouped.mean()
        std = grouped.std()
        t = mean.index

        ax.plot(t, mean, label="Mean", color="blue")
        ax.fill_between(t, mean - std, mean + std, alpha=0.3, label="±1 SD")
        ax.set_ylabel(col.capitalize())
        ax.set_title(f"Grand Mean {col.capitalize()} ± SD")
        ax.legend()

    axes[-1].set_xlabel("Time")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_slope_histograms(df: pd.DataFrame, save_path: Path) -> None:
    """Plot histograms of first differences (slopes) for valence and arousal."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, col in zip(axes, ["valence", "arousal"]):
        slopes = df.groupby("subject_id")[col].diff()
        ax.hist(slopes.dropna(), bins=30, alpha=0.7, color="orange", edgecolor="black")
        ax.set_title(f"{col.capitalize()} Δ Histogram")
        ax.set_xlabel(f"Δ {col.capitalize()}")
        ax.set_ylabel("Frequency")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_subject_boxplots(df: pd.DataFrame, save_path: Path) -> None:
    """Plot boxplots of per-subject mean and SD for valence and arousal."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    for row, col in enumerate(["valence", "arousal"]):
        grouped = df.groupby("subject_id")[col]
        means = grouped.mean()
        sds = grouped.std()

        axes[row, 0].boxplot(means.dropna())
        axes[row, 0].set_title(f"{col.capitalize()} Mean per Subject")
        axes[row, 0].set_ylabel(col.capitalize())

        axes[row, 1].boxplot(sds.dropna())
        axes[row, 1].set_title(f"{col.capitalize()} SD per Subject")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_correlation_matrices(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot subject–subject correlation matrices for valence and arousal."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for col in ["valence", "arousal"]:
        pivot = df.pivot(index="timestamp", columns="subject_id", values=col)
        corr = pivot.corr()

        plt.figure(figsize=(8, 6))
        im = plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        plt.colorbar(im, label="Correlation")
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
        plt.yticks(range(len(corr.index)), corr.index)
        plt.title(f"{col.capitalize()} Correlation Matrix")
        plt.tight_layout()
        plt.savefig(output_dir / f"{col}_correlation_matrix.png", dpi=150)
        plt.close()


def plot_subject_summary_stats(summary_df: pd.DataFrame, save_path: Path):
    """
    Plot subject-level summary statistics (mean vs std) for valence and arousal.
    Annotate subjects that are potential outliers (e.g., beyond 2 std deviations from mean).

    Args:
        summary_df (pd.DataFrame): DataFrame with columns like
            ['subject_id', 'valence_mean', 'valence_std', 'arousal_mean', 'arousal_std']
        save_path (Path): Path to save the plot image.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot valence mean vs std
    axes[0].scatter(summary_df['valence_mean'], summary_df['valence_std'], color='blue')
    axes[0].set_xlabel('Valence Mean')
    axes[0].set_ylabel('Valence Std Dev')
    axes[0].set_title('Subject Valence Mean vs Std Dev')

    # Plot arousal mean vs std
    axes[1].scatter(summary_df['arousal_mean'], summary_df['arousal_std'], color='green')
    axes[1].set_xlabel('Arousal Mean')
    axes[1].set_ylabel('Arousal Std Dev')
    axes[1].set_title('Subject Arousal Mean vs Std Dev')

    # Annotate subjects that are outside typical range (e.g., 2 std deviations from mean)
    def annotate_outliers(ax, x, y, labels):
        mean_x, std_x = x.mean(), x.std()
        mean_y, std_y = y.mean(), y.std()
        for i, (xx, yy) in enumerate(zip(x, y)):
            if (abs(xx - mean_x) > 2*std_x) or (abs(yy - mean_y) > 2*std_y):
                ax.annotate(labels[i], (xx, yy), textcoords="offset points", xytext=(5,5), fontsize=8)

    annotate_outliers(axes[0], summary_df['valence_mean'], summary_df['valence_std'], summary_df['subject_id'])
    annotate_outliers(axes[1], summary_df['arousal_mean'], summary_df['arousal_std'], summary_df['subject_id'])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_subject_value_boxplots(df: pd.DataFrame, save_path: Path) -> None:
    """
    Plot boxplots of valence and arousal values per subject to spot outliers.
    """
    subjects = df['subject_id'].unique()
    valence_data = [df[df['subject_id'] == s]['valence'].dropna() for s in subjects]
    arousal_data = [df[df['subject_id'] == s]['arousal'].dropna() for s in subjects]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    axes[0].boxplot(valence_data, labels=subjects, patch_artist=True,
                    boxprops=dict(facecolor="lightblue"))
    axes[0].set_title("Valence per Subject")
    axes[0].set_ylabel("Valence")
    axes[0].grid(True, axis='y')

    axes[1].boxplot(arousal_data, labels=subjects, patch_artist=True,
                    boxprops=dict(facecolor="lightgreen"))
    axes[1].set_title("Arousal per Subject")
    axes[1].set_ylabel("Arousal")
    axes[1].set_xlabel("Subject ID")
    axes[1].grid(True, axis='y')

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_individual_subject_timeseries(df: pd.DataFrame, save_html_path: Path):
    subjects = df['subject_id'].unique()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=("Valence", "Arousal"))

    for subject in subjects:
        subj_df = df[df['subject_id'] == subject]
        
        # Valence line
        fig.add_trace(
            go.Scatter(
                x=subj_df['timestamp'],
                y=subj_df['valence'],
                mode='lines',
                name=str(subject),       # Legend name
                legendgroup=str(subject), # Group for toggling together
                hoverinfo='name+x+y',
                line=dict(width=1),
                showlegend=True           # Show legend for all subjects
            ),
            row=1, col=1
        )

        # Arousal line, same legend group but hide legend to avoid duplicates
        fig.add_trace(
            go.Scatter(
                x=subj_df['timestamp'],
                y=subj_df['arousal'],
                mode='lines',
                name=str(subject),
                legendgroup=str(subject),
                hoverinfo='name+x+y',
                line=dict(width=1),
                showlegend=False          # Only show legend once per group
            ),
            row=2, col=1
        )

    fig.update_layout(height=600, width=900,
                      title_text="Individual Subject Time Series (Valence and Arousal)",
                      hovermode="x unified")

    fig.update_xaxes(title_text="Timestamp", row=2, col=1)
    fig.update_yaxes(title_text="Valence", row=1, col=1)
    fig.update_yaxes(title_text="Arousal", row=2, col=1)

    fig.write_html(save_html_path)
    print(f"Interactive plot saved to {save_html_path}")



