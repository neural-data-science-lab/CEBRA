from pathlib import Path
from typing import Tuple, Union
import numpy as np
from scipy.interpolate import interp1d
import mne
import plotly.io as pio

from cebra import CEBRA
from data import eeg_dataloader
from eeg.visualization import debug_valence_arousal_distribution

pio.renderers.default = "browser"

# --------------------------------------------------------------------------------------------
# Globals: Define CEBRA models
# --------------------------------------------------------------------------------------------

cebra_model_labels = CEBRA(
    model_architecture="offset10-model",
    batch_size=512,
    learning_rate=3e-4,
    temperature_mode='constant',
    temperature=1.12,
    max_iterations=5000,
    conditional="labels",
    output_dimension=8,
    distance='cosine',
    device="cpu",
    verbose=True,
    time_offsets=500,
)

cebra_model_continuous = CEBRA(
    model_architecture="offset10-model",
    batch_size=512,
    learning_rate=3e-4,
    temperature_mode='constant',
    temperature=1.12,
    max_iterations=5000,
    conditional="continuous",
    output_dimension=8,
    distance='cosine',
    device="cpu",
    verbose=True,
    time_offsets=500,
)

# --------------------------------------------------------------------------------------------
# Functions
# --------------------------------------------------------------------------------------------

def prepare_subject_data(
    subject_key: str,
    raw: mne.io.Raw,
    t_start: int,
    t_end: int,
    band: Union[str, int, float],
    channels: list[str],
    channels_label: str,
    root: Path,
    cebra_model: CEBRA,   # Pass model here to generate config string
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, str]:
    """
    Preprocess EEG and align behavioral labels.
    Returns EEG matrix, valence_aligned, arousal_aligned, sampling frequency, config string.
    """
    sfreq = raw.info["sfreq"]
    X = raw.get_data().T
    t_eeg = np.arange(X.shape[0]) / sfreq

    subject_folder = root / subject_key
    beh_df = eeg_dataloader.load_behavioral_labels(subject_folder)

    valence = beh_df["valence"].values
    arousal = beh_df["arousal"].values
    t_behavior = beh_df["timestamp"].values

    if len(valence) != X.shape[0]:
        interp_val = interp1d(t_behavior, valence, kind="linear", bounds_error=False, fill_value="extrapolate")
        interp_aro = interp1d(t_behavior, arousal, kind="linear", bounds_error=False, fill_value="extrapolate")
        valence_aligned = interp_val(t_eeg)
        arousal_aligned = interp_aro(t_eeg)
    else:
        valence_aligned = valence
        arousal_aligned = arousal

    config_str = f"D{cebra_model.output_dimension}_TO{cebra_model.time_offsets}_M{cebra_model.model_architecture}_T{t_start}-{t_end}_B{band}_CH{channels_label}"

    return X, valence_aligned, arousal_aligned, sfreq, config_str


def compute_population_median(subject_ids, data_dir):
    """
    Compute global median values for valence and arousal across all subjects.
    Subjects without behavioral data are skipped.
    """
    all_val, all_aro = [], []

    for subj in subject_ids:
        beh_df = eeg_dataloader.load_behavioral_labels(Path(data_dir) / subj)
        if beh_df is None:
            continue
        all_val.append(beh_df["valence"].values)
        all_aro.append(beh_df["arousal"].values)

    if not all_val or not all_aro:
        raise RuntimeError("[ERROR] No valid behavioral data found for any subjects!")

    val_median = np.median(np.concatenate(all_val))
    aro_median = np.median(np.concatenate(all_aro))

    return val_median, aro_median


def quick_run_cebra_supervised(model, X, Y, output_root, subject_key, config_str, run_name):
    """
    Fit CEBRA in supervised mode and save embeddings.
    """
    model.fit(X, Y)
    embedding = model.transform(X)

    subject_folder = output_root / subject_key
    subject_folder.mkdir(parents=True, exist_ok=True)

    np.save(subject_folder / f"{subject_key}_{config_str}_{run_name}.npy", embedding)
    return embedding


def run_subject_pipeline(subject_key, root, raw, t_start, t_end, band, channels, channels_label, output_root, val_median, aro_median):
    """
    Run all 3 supervised runs for a subject:
    1) Interpolated valence & arousal
    2) Binary valence
    3) Binary arousal
    """
    # Use continuous model for config string
    X, val_aligned, aro_aligned, sfreq, config_str = prepare_subject_data(
        subject_key, raw, t_start, t_end, band, channels, channels_label, root, cebra_model_continuous
    )

    if np.isnan(X).any():
        X = np.nan_to_num(X)

    # # Run 1: Continuous valence & arousal
    # labels_run1 = np.stack([val_aligned, aro_aligned], axis=1)
    # assert X.shape[0] == labels_run1.shape[0], f"X vs labels mismatch: {X.shape[0]} vs {labels_run1.shape[0]}"

    # quick_run_cebra_supervised(cebra_model_continuous, X, labels_run1, output_root, subject_key, config_str, run_name="Lbeh_")

    # Run 2: Binary valence
    val_bin = (val_aligned > val_median).astype(int)
    labels_run2 = val_bin[:, None]
    quick_run_cebra_supervised(cebra_model_labels, X, labels_run2, output_root, subject_key, config_str, run_name="Lbinaryvalence_")

    # Run 3: Binary arousal
    aro_bin = (aro_aligned > aro_median).astype(int)
    labels_run3 = aro_bin[:, None]
    quick_run_cebra_supervised(cebra_model_labels, X, labels_run3, output_root, subject_key, config_str, run_name="Lbinaryarousal_")
