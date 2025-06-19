from pathlib import Path
import zipfile
import mne

def unzip_subject(zip_path: Path, subject_folder: Path):
    """
    Unzip the subject data if the subject folder does not already exist.
    
    Parameters:
        zip_path (Path): Path to the subject zip file.
        subject_folder (Path): Folder where the subject data should be extracted.
    """
    if not subject_folder.exists():
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(subject_folder.parent)
        print(f"Extracted {zip_path} to {subject_folder.parent}")
    else:
        print(f"{subject_folder} already exists. Skipping unzip.")

def find_edf_file(subject_folder: Path) -> Path:
    """
    Find the first EDF file in the eeg subfolder of the subject folder.

    Parameters:
        subject_folder (Path): Path to the unzipped subject folder.

    Returns:
        Path to the EDF file.
    """
    eeg_folder = subject_folder / "eeg"
    edf_files = list(eeg_folder.glob("*.edf"))
    if not edf_files:
        raise FileNotFoundError(f"No EDF file found in {eeg_folder}")
    return edf_files[0]

def load_edf_file(edf_path: Path) -> mne.io.Raw:
    """
    Load the EDF file using MNE.

    Parameters:
        edf_path (Path): Path to the EDF file.

    Returns:
        mne.io.Raw: Loaded raw EEG data.
    """
    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
    print(f"Loaded EDF file: {edf_path}")
    return raw

def load_subject(subject_id: str, data_dir: str = "data") -> mne.io.Raw:
    """
    Load EEG data for a given subject.

    Parameters:
        subject_id (str): Subject folder/zip name (e.g., 'sub-001').
        data_dir (str): Directory where subject zip files are stored.

    Returns:
        mne.io.Raw: Loaded raw EEG data.
    """
    data_dir = Path(data_dir)
    zip_path = data_dir / f"{subject_id}.zip"
    subject_folder = data_dir / subject_id  # e.g. data/sub-001

    unzip_subject(zip_path, subject_folder)

    edf_file = find_edf_file(subject_folder)
    raw = load_edf_file(edf_file)

    return raw

def load_all_subjects(data_dir: str = "data") -> dict:
    """
    Load EEG data for all subjects (all ZIP files named sub-*.zip in data_dir).

    Returns:
        dict: Keys are subject IDs (like 'sub-001'), values are MNE Raw objects.
    """
    data_dir = Path(data_dir)
    loaded_data = {}
    for zip_file in data_dir.glob("sub-*.zip"):
        subject_id = zip_file.stem  # 'sub-001' from 'sub-001.zip'
        print(f"Loading {subject_id} ...")
        raw = load_subject(subject_id, data_dir=str(data_dir))
        loaded_data[subject_id] = raw
    return loaded_data

