"""
Data loading

This script provides functions to manage and load EEG data files stored in zipped archives
containing either EDF or preprocessed FIF formats. It handles unzipping, file detection, and
data loading using the MNE toolbox.

Required packages:
    - mne
    - pathlib
    - zipfile
    - pandas

Author: 
Created: 10.06.2025
Last updated: 21.07.2025
"""

# --------------------------------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------------------------------

from pathlib import Path
from typing import Optional, List, Dict
import mne
import pandas as pd

# --------------------------------------------------------------------------------------------
# Functions
# --------------------------------------------------------------------------------------------

def load_subject(subject_folder: Path, data_type: str = "preproc") -> mne.io.Raw:
    """
    Load EEG data for a single subject from either preprocessed FIF or raw EDF format.

    Args:
        subject_folder (Path): Path to a subject directory (e.g., sub-001).
        data_type (str): Type of data to load. 
                         Options:
                            - 'preproc': Loads MNE FIF files from the 'eeg' folder.
                            - 'rawdata': Loads EDF files from the 'eeg' folder.

    Returns:
        mne.io.Raw: Loaded EEG data as an MNE Raw object.

    Raises:
        FileNotFoundError: If no valid EEG file is found or the EEG folder does not exist.
        ValueError: If the data_type is not supported.
    """
    eeg_folder = subject_folder / "eeg"
    if not eeg_folder.exists():
        raise FileNotFoundError(f"[ERROR] EEG folder not found: {eeg_folder}")

    if data_type == "preproc":
        files = list(eeg_folder.glob("*after_ica.fif"))
        if not files:
            raise FileNotFoundError(f"[INFO] No FIF files found in {eeg_folder}")
        raw = mne.io.read_raw_fif(files[0], preload=True, verbose=False)
        print(f"[INFO] Loaded preprocessed FIF file: {files[0].name}")

    elif data_type == "rawdata":
        files = list(eeg_folder.glob("*.edf"))
        if not files:
            raise FileNotFoundError(f"[INFO] No EDF files found in {eeg_folder}")
        raw = mne.io.read_raw_edf(files[0], preload=True, verbose=False)
        print(f"[INFO] Loaded raw EDF file: {files[0].name}")

    else:
        raise ValueError(f"[ERROR] Invalid data_type '{data_type}'. Use 'preproc' or 'rawdata'.")

    print(f"[INFO] Loaded {files[0].name} for {subject_folder.name}")
    return raw


def load_all_subjects(
    data_dir: str = "data",
    data_type: str = "preproc",
    subjects_to_load: Optional[List[str]] = None,
) -> Dict[str, mne.io.Raw]:
    """
    Load full EEG data for all specified subjects from the given directory.

    Args:
        data_dir (str): Path to the data directory containing subject folders.
        data_type (str): 'preproc' for preprocessed FIF files, or 'rawdata' for raw EDF files.
        subjects_to_load (Optional[List[str]]): List of subject IDs to load. If None, all subjects are loaded.

    Returns:
        Dict[str, mne.io.Raw]: A dictionary mapping subject IDs to loaded MNE Raw objects.
    """
    data_path = Path(data_dir)
    loaded_raws: Dict[str, mne.io.Raw] = {}

    for subject_folder in sorted(data_path.glob("sub-*")):
        subject_id = subject_folder.name
        if subjects_to_load and subject_id not in subjects_to_load:
            continue

        try:
            raw = load_subject(subject_folder, data_type=data_type)
            loaded_raws[subject_id] = raw
            print(f"[INFO] Loaded full data for {subject_id}")
        except Exception as e:
            print(f"[ERROR] Failed to load {subject_id}: {e}")

    print(f"[INFO] Total subjects fully loaded: {len(loaded_raws)}")
    return loaded_raws


def load_behavioral_labels(subject_folder: Path) -> pd.DataFrame:
    """
    Load behavioral data (e.g., valence/arousal ratings) from TSV file in the subject's 'beh' folder.

    Args:
        subject_folder (Path): Path to the subject directory (e.g., sub-001).

    Returns:
        pd.DataFrame: DataFrame containing behavioral labels like timestamp, valence, arousal, etc.

    Raises:
        FileNotFoundError: If no .tsv file is found in the behavioral folder.
    """
    beh_folder = subject_folder / "beh"
    tsv_files = list(beh_folder.glob("*.tsv"))

    if not tsv_files:
        raise FileNotFoundError(f"[ERROR] No .tsv behavioral files found in {beh_folder}")

    df = pd.read_csv(tsv_files[0], sep="\t")
    print(f"[INFO] Loaded behavioral data: {tsv_files[0].name}")
    return df
