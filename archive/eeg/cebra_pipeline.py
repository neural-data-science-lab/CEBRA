"""
CEBRA Embedding Pipeline

This script contains logic for fitting the CEBRA model on EEG data, transforming it into embeddings.

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
from typing import List, Tuple, Optional, Union
import numpy as np
import mne
from cebra import CEBRA

# --------------------------------------------------------------------------------------------
# Globals
# --------------------------------------------------------------------------------------------

cebra_model = CEBRA(
    model_architecture="offset10-model",  # Alternatives: "offset1-model", "offset10-model"
    batch_size=512,
    learning_rate=3e-4,
    temperature_mode='constant',
    temperature=1.12,
    max_iterations=5000,
    conditional=None,  # Unsupervised mode
    output_dimension=16,
    distance='cosine',  # Try "euclidean" for different effect
    device="cpu",
    verbose=True,
    time_offsets=500, # Alternatives 5000 = 10s, 100 ~ 0.1 s
)

# --------------------------------------------------------------------------------------------
# Functions
# --------------------------------------------------------------------------------------------

def quick_run_cebra(
    X: np.ndarray,
    output_root: Path,
    subject_key: str,
    config_str: str,

) -> np.ndarray:
    """
    Fit the CEBRA model on EEG data and generate an interactive embedding plot.

    Args:
        X (np.ndarray): EEG data matrix, shape (samples, features).
       output_root (Path): Root folder to save the embedding files.
        subject_key (str): Identifier for the subject.
        config_str (str): Configuration string used for naming the output file.

    Returns:
        np.ndarray: Embedding array of shape (samples, embedding_dimension).
    """
    model = cebra_model.fit(X)
    embedding = model.transform(X)

    subject_folder = output_root / subject_key
    subject_folder.mkdir(parents=True, exist_ok=True)

    np.save(subject_folder / f"{subject_key}_{config_str}.npy", embedding)

    return embedding


def prepare_subject_data(
    subject_key: str,
    raw: mne.io.Raw,
    t_start: int,
    t_end: int,
    band: Union[str, int, float],
    channels: list[str],
    channels_label: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, str]:
    """
    Preprocess EEG and align behavioral labels.
    Args:
        subject_key (str): Subject identifier.
        raw (mne.io.Raw): Raw EEG data object.
        t_start (int): Start time in seconds for the segment.
        t_end (int): End time in seconds for the segment.
        band (Union[str, int, float]): Frequency band or identifier for preprocessing.
        channels (list[str]): List of channel names to include.
        channels_label (str): Short label representing selected channels.
       

    Returns:
        Tuple containing:
            - X (np.ndarray): EEG data matrix, shape (samples, channels).
            - sfreq (float): Sampling frequency of the EEG data.
            - config_str (str): Configuration string for the current preprocessing setup.
    """

    sfreq = raw.info["sfreq"]
    # Load eeg datas
    X = raw.get_data().T  # shape: (samples, channels)

    output_dim = cebra_model.output_dimension
    time_offset = cebra_model.time_offsets
    model_arch = cebra_model.model_architecture

    config_str = f"D{output_dim}_TO{time_offset}_M{model_arch}_T{t_start}-{t_end}_B{band}_CH{channels_label}"
    return X, sfreq, config_str

#full pipeline: preprocess EEG and embed with CEBRA
def run_subject_pipeline(
    subject_key: str,
    raw: mne.io.Raw,
    t_start: int,
    t_end: int,
    band: Union[str, int, float],
    channels: list[str],
    channels_label: str,
    output_root: Path,
)-> None:
    """
    Complete pipeline for processing EEG data for a single subject: preprocess, visualize, and embed.

    Args:
        subject_key (str): Subject identifier.
        root (Path): Root folder containing behavioral data.
        raw (mne.io.Raw): Raw EEG data object.
        t_start (int): Start time in seconds.
        t_end (int): End time in seconds.
        band (Union[str, int, float]): Frequency band or preprocessing identifier.
        channels (list[str]): Channels to include in preprocessing.
        channels_label (str): Short label representing selected channels.
        output_root (Path): Root folder to save outputs (plots, embeddings).
    Returns:
        None
    """

    X, sfreq, config_str = prepare_subject_data(
        subject_key, raw, t_start, t_end, band, channels, channels_label, root
    )

    quick_run_cebra(X, output_root, subject_key, config_str)

    #[TODO] validate embedding, analyze embeddings
