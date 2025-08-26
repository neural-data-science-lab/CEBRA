"""
CEBRA Embedding Pipeline

This script contains logic for fitting the CEBRA model on EEG data, transforming it into embeddings,
and preparing interactive plots with emotion-based color coding.

This module hides CEBRA-specific details behind a clean function interface.

Required packages:
    - numpy
    - matplotlib
    - plotly
    - scipy
    - mne
    - cebra

Author:
Created: 10.06.2025
Last updated: 21.07.2025
"""

# --------------------------------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------------------------------

from pathlib import Path
from typing import Optional, Dict, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly.io as pio
from scipy.interpolate import interp1d
import mne

from cebra import CEBRA
from cebra.integrations.plotly import plot_embedding_interactive

from data import eeg_dataloader
from eeg.colors import valence_arousal_emotion_color
from eeg.visualization import debug_valence_arousal_distribution



pio.renderers.default = "browser"

# --------------------------------------------------------------------------------------------
# Globals
# --------------------------------------------------------------------------------------------

cebra_model = CEBRA(
    model_architecture="offset50-model",  # Alternatives: "offset1-model", "offset10-model"
    batch_size=512,
    learning_rate=3e-4,
    temperature_mode='constant',
    temperature=1.12,
    max_iterations=5000,
    conditional=None,  # Unsupervised mode
    output_dimension=3,
    distance='cosine',  # Try "euclidean" for different effect
    device="cuda_if_available",
    verbose=True,
    time_offsets=500, # Alternatives 5000 = 10s, 100 ~ 0.1 s
)

# --------------------------------------------------------------------------------------------
# Functions
# --------------------------------------------------------------------------------------------

def quick_run_cebra(
    X: np.ndarray,
    valence_aligned: np.ndarray,
    arousal_aligned: np.ndarray,
    title: str = "CEBRA Embedding"
) -> Tuple[object, np.ndarray]:
    """
    Fit the CEBRA model on EEG data and generate an interactive embedding plot.

    Args:
        X (np.ndarray): EEG data matrix, shape (samples, features).
        valence_aligned (np.ndarray): Aligned valence labels per sample.
        arousal_aligned (np.ndarray): Aligned arousal labels per sample.
        title (str): Title for the embedding plot.

    Returns:
        Tuple[object, np.ndarray]: Plotly figure object and embedding array.
    """
    model = cebra_model.fit(X)
    embedding = model.transform(X)

    # [TODO] Automatically determine sampling frequency instead of hardcoded 500 Hz
    times = np.arange(X.shape[0]) / 500.0

    # Downsample for plotting (limit points to ~15000)
    n_points = 15000
    step = max(1, len(X) // n_points)
    idx = np.arange(0, len(X), step)

    embedding_small = embedding[idx]
    valence_small = valence_aligned[idx]
    arousal_small = arousal_aligned[idx]

    valid_mask = ~(np.isnan(valence_small) | np.isnan(arousal_small))
    embedding_small = embedding_small[valid_mask]
    valence_small = valence_small[valid_mask]
    arousal_small = arousal_small[valid_mask]

    # Map valence/arousal to RGB colors
    colors = valence_arousal_emotion_color(valence_small, arousal_small)
    colors_hex = np.array([mcolors.to_hex(c) for c in colors])

    fig = plot_embedding_interactive(
        embedding_small,
        embedding_labels=colors_hex,
        title=title,
        markersize=2,
    )
    plt.close('all')  # Clean up matplotlib figures
    return fig, embedding


def prepare_subject_data(
    subject_key: str,
    raw: mne.io.Raw,
    t_start: int,
    t_end: int,
    band: Union[str, int, float],
    channels: list,
    channels_label: str,
    root: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, str]:
    """
    Preprocess EEG and align behavioral labels.
    Returns: (X, valence_aligned, arousal_aligned, sfreq, config_str)
    """


    sfreq = raw.info["sfreq"]
    X = raw.get_data().T  # shape: (samples, channels)
    t_eeg = np.arange(X.shape[0]) / sfreq

    # Load behavioral labels
    subject_folder = root / subject_key
    beh_df = eeg_dataloader.load_behavioral_labels(subject_folder)

    valence = beh_df["valence"].values
    arousal = beh_df["arousal"].values
    t_behavior = beh_df["timestamp"].values

    # Align
    if len(valence) != X.shape[0]:
        interp_val = interp1d(t_behavior, valence, kind="linear", bounds_error=False, fill_value="extrapolate")
        interp_aro = interp1d(t_behavior, arousal, kind="linear", bounds_error=False, fill_value="extrapolate")
        valence_aligned = interp_val(t_eeg)
        arousal_aligned = interp_aro(t_eeg)
    else:
        valence_aligned = valence
        arousal_aligned = arousal

    config_str = f"T{t_start}-{t_end}_B{band}_CH{channels_label}"
    return X, valence, arousal, valence_aligned, arousal_aligned, sfreq, config_str

def run_cebra_embedding(
    X: np.ndarray,
    valence: np.ndarray,
    arousal: np.ndarray,
    subject_key: str,
    config_str: str,
    output_root: Path,
    save_html: bool = True,
    save_embedding: bool = True
) -> None:
    fig, embedding = quick_run_cebra(X, valence, arousal, title=f"CEBRA - {subject_key} - {config_str}")

    subject_folder = output_root / subject_key
    subject_folder.mkdir(parents=True, exist_ok=True)

    if save_html:
        fig.write_html(str(subject_folder / f"Moffset_50_DT500{subject_key}_{config_str}.html"))

    if save_embedding:
        np.save(subject_folder / f"Moffset50_DT500{subject_key}_{config_str}.npy", embedding)

def run_subject_pipeline(
    subject_key: str,
    root: Path,
    raw: mne.io.Raw,
    t_start: int,
    t_end: int,
    band: Union[str, int, float],
    channels: list,
    channels_label: str,
    output_root: Path,
    save_html: bool = True,
    save_embedding: bool = True,
    explore_behavior_data: bool = True,
    train_model: bool = True,
):
    X, valence, arousal,  valence_aligned, arousal_aligned, sfreq, config_str = prepare_subject_data(
        subject_key, raw, t_start, t_end, band, channels, channels_label, root
    )

    if np.isnan(X).any():
        print(f"[WARNING] NaNs detected in EEG data for {subject_key}, config {config_str}. Replacing NaNs with zero.")
        # replace NaNs with zero
        X = np.nan_to_num(X)

    if explore_behavior_data:
        debug_valence_arousal_distribution(
            valence,
            arousal,
            subject_key=subject_key,
            output_root=output_root,
            config_str=config_str,
        )

    if train_model: 
        run_cebra_embedding(
            X,
            valence_aligned,
            arousal_aligned,
            subject_key,
            config_str,
            output_root,
            save_html=True,
            save_embedding=True,
        )
    
    #[TODO] if validate embedding, analyze embeddings
