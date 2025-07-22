
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
from typing import Optional, List, Dict

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


from pathlib import Path
from typing import Optional, List, Dict
import mne

def load_all_subjects(
    data_dir: str = "data",
    data_type: str = "preproc",
    pick_channels: Optional[List[str]] = None,
    subjects_to_load: Optional[List[str]] = None
) -> Dict[str, mne.io.Raw]:
    """
    Load EEG data for specified subjects from data_dir using consistent style.

    Args:
        data_dir (str): Path to data directory containing subject folders.
        data_type (str): Type of data to load ("preproc" or "rawdata").
        pick_channels (list, optional): Channels to pick from each raw object.
        subjects_to_load (list, optional): List of subject IDs to load (e.g., ["sub-001"]).

    Returns:
        dict: Keys are subject IDs, values are MNE Raw objects.
    """
    data_dir = Path(data_dir)
    loaded_data = {}

    # Use glob for consistent folder matching
    for subject_folder in sorted(data_dir.glob("sub-*")):
        subject_id = subject_folder.name

        # Filter subjects exactly like in load_subjects
        if subjects_to_load is not None and subject_id not in subjects_to_load:
            continue

        print(f"[INFO] Loading subject: {subject_id}")

        try:
            raw = load_subject(subject_folder, data_type=data_type)

            if pick_channels is not None:
                raw.pick(pick_channels)

            loaded_data[subject_id] = raw

        except Exception as e:
            print(f"[ERROR] Failed to load {subject_id}: {e}")

    print(f"[INFO] Total loaded subjects: {len(loaded_data)}")
    return loaded_data
