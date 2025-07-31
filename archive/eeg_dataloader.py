
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
        files = list(eeg_folder.glob("*after_ica.fif"))
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
    subjects_to_load: Optional[List[str]] = None,
) -> Dict[str, mne.io.Raw]:
    """
    Load full raw data for all specified subjects without cropping or filtering.

    Args:
        data_dir (str): Path to data directory.
        data_type (str): 'preproc' or 'rawdata'.
        subjects_to_load (List[str], optional): List of subject IDs to load. Loads all if None.

    Returns:
        Dict[str, mne.io.Raw]: Dictionary of subject_id to raw MNE objects.
    """
    data_dir = Path(data_dir)
    loaded_raws = {}

    for subject_folder in sorted(data_dir.glob("sub-*")):
        subject_id = subject_folder.name
        if subjects_to_load is not None and subject_id not in subjects_to_load:
            continue
        try:
            raw = load_subject(subject_folder, data_type=data_type)
            loaded_raws[subject_id] = raw
            print(f"[INFO] Loaded full data for {subject_id}")
        except Exception as e:
            print(f"[ERROR] Failed to load {subject_id}: {e}")

    print(f"[INFO] Total subjects fully loaded: {len(loaded_raws)}")
    return loaded_raws


def filter_crop_data(
    raw: mne.io.Raw,
    t_start: Optional[float] = None,
    t_end: Optional[float] = None,
    filter_frequency_band: Optional[Tuple[float, float]] = None,
    pick_channels: Optional[List[str]] = None
) -> mne.io.Raw:
    """
    Apply time cropping, frequency filtering, and channel selection on a Raw object.

    Args:
        raw (mne.io.Raw): Raw EEG data (loaded full).
        t_start (float, optional): Start time in seconds for cropping.
        t_end (float, optional): End time in seconds for cropping.
        filter_frequency_band (Tuple[float, float], optional): Frequency band (fmin, fmax) to filter.
        pick_channels (List[str], optional): List of channel names to select.

    Returns:
        mne.io.Raw: Processed Raw object with filters applied.
    """
    processed = raw.copy()

    if t_start is not None and t_end is not None:
        processed.crop(tmin=t_start, tmax=t_end)

    if filter_frequency_band is not None:
        fmin, fmax = filter_frequency_band
        processed.filter(fmin, fmax, fir_design='firwin', verbose=False)

    if pick_channels is not None:
        processed.pick_channels(pick_channels)

    return processed



