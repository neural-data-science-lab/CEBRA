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

from typing import List, Tuple, Optional
import mne

# --------------------------------------------------------------------------------------------
# Functions
# --------------------------------------------------------------------------------------------

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

    if t_start is not None and t_end is not None:
        raw.crop(tmin=t_start, tmax=t_end)

    if filter_frequency_band is not None:
        fmin, fmax = filter_frequency_band
        raw.filter(fmin, fmax, fir_design='firwin', verbose=False)

    if pick_channels is not None:
        raw.pick_channels(pick_channels)

    return raw
