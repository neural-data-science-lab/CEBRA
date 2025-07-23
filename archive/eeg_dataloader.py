
"""
Data loading

This script provides functions to manage and load EEG data files stored in zipped archives
containing either EDF or preprocessed FIF formats. It handles unzipping, file detection, and
data loading using the MNE toolbox.

Required packages:
    - mne
    - pathlib
    - zipfile

Author: 
Created: 10.06.2025
Last updated: 21.07.2025
"""
# --------------------------------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------------------------------
from pathlib import Path
import mne 
import numpy as np
from typing import Optional, List, Dict, Tuple
from scipy.signal import hilbert
from mne import create_info
from mne.io import RawArray
from scipy.ndimage import maximum_filter, uniform_filter1d
# --------------------------------------------------------------------------------------------
#  Functions
# --------------------------------------------------------------------------------------------
def load_subject(subject_folder: Path, data_type: str = "preproc") -> mne.io.Raw:
    """
    Load EEG data for a single subject from either preprocessed or raw format.

    Args:
        subject_folder (Path): Path to a subject directory (e.g., sub-001).
        data_type (str): Type of data to load. 
                         - 'preproc': Loads MNE FIF files from the 'eeg' folder.
                         - 'rawdata': Loads EDF files from the 'eeg' folder.

    Returns:
        mne.io.Raw: Loaded EEG data as an MNE Raw object.

    Raises:
        FileNotFoundError: If no valid EEG file is found.
        ValueError: If an invalid data_type is provided.
    """
    eeg_folder = subject_folder / "eeg"
    if not eeg_folder.exists():
        raise FileNotFoundError(f"[ERROR] EEG folder not found: {eeg_folder}")

    # Try loading FIF first (preprocessed)
    if data_type == "preproc":
        files = list(eeg_folder.glob("*before_ica.fif"))
        if not files:
            raise FileNotFoundError(f"[INFO] No FIF files found in {eeg_folder}")
        raw = mne.io.read_raw_fif(files[0], preload=True, verbose=False)
        fif_file = files[0]
        print(f"[INFO] Loaded preprocessed FIF file: {fif_file}")
    elif data_type == "rawdata":
        files = list(eeg_folder.glob("*.edf"))
        if not files:
            raise FileNotFoundError(f"[INFO] No EDF files found in {eeg_folder}")
        raw = mne.io.read_raw_edf(files[0], preload=True, verbose=False)
        edf_file = files[0]
        print(f"[INFO] Loaded raw EDF file: {edf_file}")
    else:
        raise FileNotFoundError(f"[ERROR] No EDF or FIF file found in {eeg_folder}")
    
    print(f"[INFO] Loaded {files[0].name} for {subject_folder.name}")
    return raw


def load_all_subjects(
    data_dir: str = "data",
    data_type: str = "preproc",
    pick_channels: Optional[List[str]] = None,
    subjects_to_load: Optional[List[str]] = None,
    t_start: Optional[float] = None,
    t_end: Optional[float] = None,
    filter_frequency_band:  Optional[Tuple[float, float]] = None
) -> Dict[str, Dict]:
    """
    Load EEG data for specified subjects, optionally extracting a snippet and reporting metadata.

    Returns:
        dict: Keys are subject IDs, values are dictionaries with:
              - 'raw': Raw object
              - 'sfreq': Sampling frequency
              - 'duration_sec': Total duration of full data
              - 'snippet_duration_sec': Duration of the snippet (if used)
    """
    data_dir = Path(data_dir)
    loaded_data = {}

    for subject_folder in sorted(data_dir.glob("sub-*")):
        subject_id = subject_folder.name
        if subjects_to_load is not None and subject_id not in subjects_to_load:
            continue

        print(f"[INFO] Loading subject: {subject_id}")

        try:
            raw = load_subject(subject_folder, data_type=data_type)
            original_raw = raw.copy()  # Keep full for duration
            if pick_channels is not None:
                raw.pick(pick_channels)

            # Duration of full dataset
            full_duration = original_raw.times[-1] - original_raw.times[0]
            sfreq = raw.info['sfreq']

            snippet_duration = None
            if t_start is not None and t_end is not None:
                raw = raw.copy().crop(tmin=t_start, tmax=t_end)
                snippet_duration = raw.times[-1] - raw.times[0]
            
            if filter_frequency_band is not None:
                fmin, fmax = filter_frequency_band
                raw.filter(fmin, fmax, fir_design='firwin', verbose=False)

            loaded_data[subject_id] = {
                "raw": raw,
                "sfreq": sfreq,
                "duration_sec": full_duration,
                "snippet_duration_sec": snippet_duration if snippet_duration else full_duration
            }

        except Exception as e:
            print(f"[ERROR] Failed to load {subject_id}: {e}")

    print(f"[INFO] Total loaded subjects: {len(loaded_data)}")
    return loaded_data

def remove_oscillation(raw: mne.io.Raw, window_size: int = 100) -> mne.io.Raw:
    """
    Remove oscillations by replacing the signal with non-overlapping max within windows.

    Args:
        raw (mne.io.Raw): EEG data (must be preloaded).
        window_size (int): Window size in samples for non-overlapping max filter.

    Returns:
        mne.io.Raw: New Raw object containing piecewise max-filtered signals.
    """

    data = raw.get_data()
    envelopes = []

    def non_overlapping_max(signal, wsize):
        n = len(signal)
        max_values = []
        for start in range(0, n, wsize):
            end = min(start + wsize, n)
            max_val = np.max(signal[start:end])
            max_values.extend([max_val] * (end - start))
        return np.array(max_values)

    for ch_idx in range(data.shape[0]):
        # Directly apply non-overlapping max on raw EEG data 
        max_filtered = non_overlapping_max(data[ch_idx], window_size)
        envelopes.append(max_filtered)

    envelopes = np.array(envelopes)  # shape: (n_channels, n_times)
    info = raw.info.copy()
    raw_filtered = RawArray(envelopes, info)

    return raw_filtered
