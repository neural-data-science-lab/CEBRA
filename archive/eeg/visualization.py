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
import matplotlib.pyplot as plt

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
    val_grid, aro_grid = np.meshgrid(
        np.linspace(-1, 1, res),  # Valence: horizontal axis
        np.linspace(-1, 1, res)   # Arousal: vertical axis
    )
    val_flat, aro_flat = val_grid.flatten(), aro_grid.flatten()
    colors = valence_arousal_emotion_color(val_flat, aro_flat, desaturate_color)
    image = colors.reshape(res, res, 3)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image, extent=(-1, 1, -1, 1), origin="lower")
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
