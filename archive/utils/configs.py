"""
Configuration Generator

Generates experimental configurations for EEG data processing
by varying time windows, frequency bands, and EEG channel selections.

Used to systematically explore the preprocessing space.

Author:
Created: 07.08.2025
Last updated: 07.08.2025
"""

# --------------------------------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------------------------------

from typing import List, Tuple, Optional

# --------------------------------------------------------------------------------------------
# Functions
# --------------------------------------------------------------------------------------------

def generate_configurations(
    time_configs: List[Tuple[Optional[int], Optional[int]]],
    filter_bands: List[Optional[Tuple[int, int]]],
    channel_configs: List[Tuple[Optional[Tuple[str, ...]], str]],
) -> List[Tuple[Tuple[Optional[int], Optional[int]], Optional[Tuple[int, int]], Tuple[Optional[Tuple[str, ...]], str]]]:
    """
    Generate a list of unique configuration tuples combining time windows,
    frequency bands, and channel selections.

    Args:
        time_configs (List[Tuple[Optional[int], Optional[int]]]): List of time window tuples (start, end).
        filter_bands (List[Optional[Tuple[int, int]]]): List of frequency band tuples (fmin, fmax) or None.
        channel_configs (List[Tuple[Optional[Tuple[str, ...]], str]]): List of channel configuration tuples.

    Returns:
        List[Tuple]: List of unique configuration tuples.
    """
    baseline_time = (None, None)
    baseline_band = None
    baseline_channels = (None, "all")

    configs = set()
    configs.add((baseline_time, baseline_band, baseline_channels))

    # Add configurations varying only one parameter at a time
    for t in time_configs:
        if t != baseline_time:
            configs.add((t, baseline_band, baseline_channels))
    for b in filter_bands:
        if b != baseline_band:
            configs.add((baseline_time, b, baseline_channels))
    for ch in channel_configs:
        if ch != baseline_channels:
            configs.add((baseline_time, baseline_band, ch))

    return list(configs)
