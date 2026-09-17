"""
EEG Preprocessing Functions

Encapsulates logic for filtering, cropping, and channel selection of raw EEG data.

This module is designed to simplify and standardize preprocessing steps
used across training and exploratory workflows.

Required packages:
    - mne

Author:
Created: 10.06.2025
Last updated: 21.07.2025
"""

# --------------------------------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------------------------------

from typing import List, Tuple, Optional, Union
import numpy as np
import mne

# --------------------------------------------------------------------------------------------
# Functions
# --------------------------------------------------------------------------------------------
def compute_angle_vector_length(
    valence: Union[np.ndarray, float],
    arousal: Union[np.ndarray, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute angle in degrees and normalized vector length from valence-arousal values.

    Args:
        valence (Union[np.ndarray, float]): Valence values, expected range [-1, 1].
        arousal (Union[np.ndarray, float]): Arousal values, expected range [-1, 1].

    Returns:
        Tuple[np.ndarray, np.ndarray]: 
            - angle_deg: Angles in degrees [0, 360).
            - vector_length: Normalized vector lengths [0, 1].
    """
    valence = np.asarray(valence, dtype=float)
    arousal = np.asarray(arousal, dtype=float)

    # Angle in degrees using raw scales
    angle_rad = np.arctan2(arousal, valence)
    angle_deg = (np.degrees(angle_rad) + 360) % 360

    # Vector length = Euclidean distance from origin
    vector_length = np.sqrt(valence**2 + arousal**2)
    max_length = np.sqrt(2)
    vector_length = vector_length / max_length
    
    return angle_deg, vector_length


def filter_crop_data(
    raw: mne.io.Raw,
    t_start: Optional[float] = None,
    t_end: Optional[float] = None,
    filter_frequency_band: Optional[Tuple[float, float]] = None,
    pick_channels: Optional[List[str]] = None
) -> mne.io.Raw:
    """
    Apply time cropping, frequency filtering, and channel selection on a Raw EEG object.

    Args:
        raw (mne.io.Raw): Raw EEG data (loaded full).
        t_start (Optional[float]): Start time in seconds for cropping.
        t_end (Optional[float]): End time in seconds for cropping.
        filter_frequency_band (Optional[Tuple[float, float]]): Frequency band (fmin, fmax) to apply filtering.
        pick_channels (Optional[List[str]]): List of channel names to select.

    Returns:
        mne.io.Raw: Filtered, cropped, and channel-selected Raw EEG object.
    """
    raw = raw.copy()

    if pick_channels is not None:
        raw.pick_channels(pick_channels)

    if filter_frequency_band is not None:
        fmin, fmax = filter_frequency_band
        raw.filter(fmin, fmax, fir_design='firwin', verbose=False)
        
    if t_start is not None and t_end is not None:
        raw.crop(tmin=t_start, tmax=t_end)

    return raw
