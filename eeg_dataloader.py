
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
Last updated: 19.06.2025
"""
# --------------------------------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------------------------------
from pathlib import Path
import zipfile
import mne 

# --------------------------------------------------------------------------------------------
#  Functions
# --------------------------------------------------------------------------------------------
def unzip_subject(zip_path: Path, subject_folder: Path) -> None:
    """
    Unzip the subject data if the subject folder does not already exist.
    

    Parameters:
        zip_path (Path): Path to the subject zip file.
        subject_folder (Path): Folder where the subject data should be extracted.
    

    Returns: 
        None
    """
    if not subject_folder.exists():
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(subject_folder.parent)
        print(f"[INFO] Extracted {zip_path} to {subject_folder.parent}")
    else:
        print(f"[INFO] {subject_folder} already exists. Skipping unzip.")


def load_edf_file(edf_path: Path) -> mne.io.Raw:
    """
    Load the EDF file using MNE.

    Parameters:
        edf_path (Path): Path to the EDF file.

    Returns:
        raw (mne.io.Raw): Loaded raw EEG data.
    """
    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
    print(f"Loaded EDF file: {edf_path}")
    return raw

def load_subject(subject_id: str, data_dir: str = "data") -> mne.io.Raw:
    """
    Load EEG data for a given subject from either a FIF or EDF file.
    
    Parameters:
        subject_id (str): Identifier of the subject e.g., 'sub-001' or 'sub-001_pp'
        data_dir (str): optional // directory where the zipped subject data is stored // default is "data".
    
    Returns:
        raw (mne.io.Raw): EEG data loaded into an MNE Raw object.
    """
    data_dir = Path(data_dir)
    zip_path = data_dir / f"{subject_id}.zip"
    subject_folder = data_dir / subject_id

    unzip_subject(zip_path, subject_folder)

    eeg_folder = subject_folder / "eeg"

    # Try loading FIF first
    fif_files = list(eeg_folder.glob("*after_ica.fif"))
    if fif_files:
        fif_file = fif_files[0]
        print({fif_file})
        #raw = mne.io.read_raw_fif(fif_file, preload=True, verbose=False, allow_maxshield=True)
        raw = mne.io.read_raw_fif(fif_file, preload=True, verbose=False)
        print(f"[INFO] Loaded preprocessed FIF file: {fif_file}")
        return raw

    # Fallback: Load EDF
    edf_files = list(eeg_folder.glob("*.edf"))
    if edf_files:
        edf_file = edf_files[0]
        raw = mne.io.read_raw_edf(edf_file, preload=True, verbose=False)
        print(f"Loaded raw EDF file: {edf_file}")
        return raw

    raise FileNotFoundError(f"[ERROR] No EDF or FIF file found in {eeg_folder}")


def load_all_subjects(data_dir: str = "data") -> Dict[str, mne.io.Raw]:
    """
    Load EEG data for all subjects (all ZIP files named sub-*.zip in data_dir).

    Parameters: 
        data_dir(str): optional // directory containing subject .zip archives. Default is "data".
    
    Returns:
        dict: Keys are subject IDs (e.g) 'sub-001'), values are MNE Raw objects.
    """
    data_dir = Path(data_dir)
    loaded_data = {}
    zip_files = sorted(data_dir.glob("sub-*.zip"))
    if not zip_files:
        print("[WARNING] No subject .zip files found in the directory.")
        return loaded_data

    for zip_file in zip_files:
        subject_id = zip_file.stem  # e.g., 'sub-001' from 'sub-001.zip'
        print(f"[INFO] Loading subject: {subject_id}")
        try:
            raw = load_subject(subject_id, data_dir=str(data_dir))
            loaded_data[subject_id] = raw
        except Exception as e:
            print(f"[ERROR] Failed to load {subject_id}: {e}")

    return loaded_data

