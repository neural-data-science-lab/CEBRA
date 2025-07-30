"""
The script:

- Loads preprocessed EEG data from each subjects.
- Applies different preprocessing configurations:
    - Time cropping
    - Frequency filtering (theta, alpha, beta, gamma bands)
    - EEG channel selection
- Runs the CEBRA model (unsupervised embedding) on the EEG data.
- Plots and saves the resulting embeddings under results/sub-XXX/ as interactive HTML plots. Filename includes:
    - Subject ID
    - Time range
    - Frequency band
    - Channel configuration
"""
import numpy as np
import matplotlib.pyplot as plt
import mne
from pathlib import Path
import logging
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"

from cebra import CEBRA
from cebra.integrations.plotly import plot_embedding_interactive
import eeg_dataloader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========== Variables ==========

# Path to your EEG data
data_dir = Path(r"E:\.Cris Work\preproc_cleaned\preproc")
output_root = Path("results")
output_root.mkdir(exist_ok=True)

# Configurations
subject_range = (1, 6)
subject_ids_to_load = [f"sub-{i:03d}" for i in range(subject_range[0], subject_range[1])]

# time cropping
time_configs = [(None,None),
                (0, 300), 
                (300, 600), 
                (600, 900), 
                (900, 1200),
                (1200, 1391)] # seconds

# Filter by frequency band
theta = (4, 8)
alpha = (8, 12)
beta = (13, 30)
filter_bands = [None, theta, alpha, beta] #Hz

# Select specific EEG channels
frontal_channels = ['Fz', 'F3', 'F4', 'F7', 'F8', 'AF3', 'AF4', 'AFz']
occipital_channels = ['O1', 'O2', 'Oz']
parietal_channels = ['Pz', 'P3', 'P4']
temporal_channels = ['T7', 'T8']
channel_configs = [(None, "all"),
                    (tuple(frontal_channels), "frontal"),
                    (tuple(occipital_channels), "occipital"), 
                    (tuple(parietal_channels), "parietal"),
                    (tuple(temporal_channels), "temporal"),]

# Generate configurations: baseline + one parameter changed
# Define baseline configuration
baseline_time = (None, None)
baseline_band = None
baseline_channels = (None,"all")
configurations = []

# 1. Baseline configuration
configurations.append((baseline_time, baseline_band, baseline_channels))

# 2. Vary time only (keep band and channels at baseline)
for time_config in time_configs:
    if time_config != baseline_time:
        configurations.append((time_config, baseline_band, baseline_channels))

# 3. Vary band only (keep time and channels at baseline)  
for band in filter_bands:
    if band != baseline_band:
        configurations.append((baseline_time, band, baseline_channels))

# 4. Vary channels only (keep time and band at baseline)
for channel_config in channel_configs:
    if channel_config != baseline_channels:
        configurations.append((baseline_time, baseline_band, channel_config))

# Remove duplicates (in case baseline appears in other lists)
configurations = list(set(configurations))

logger.info(f"Total configurations to run: {len(configurations)}")
for i, (time_cfg, band_cfg, ch_cfg) in enumerate(configurations):
    ch_label = ch_cfg[1] if ch_cfg[1] is not None else "all"
    logger.info(f"  {i+1}. Time: {time_cfg}, Band: {band_cfg}, Channels: {ch_label}")


# Model
cebra_model = CEBRA(
    model_architecture="offset10-model",  # Alternatives: "offset1-model", "offset50-model"
    batch_size=512,
    learning_rate=3e-4,
    temperature_mode='constant',
    temperature=1.12,
    max_iterations=5000,
    conditional=None,  # Unsupervised mode
    output_dimension=3,
    distance='cosine',  # Try "euclidean" for different effect
    device="cuda_if_available",
    verbose=True,
    time_offsets=10,
)

# ========== Helper Functions ==========

def plot_data(X, title="Data Example"):
    plt.figure(figsize=(12, 6))
    for ch in range(X.shape[1]):
        plt.plot(X[:, ch] + ch*2, label=f'Channel {ch}')  # offset vertically 
    plt.title(title)
    plt.xlabel("Samples (time)")
    plt.ylabel("Amplitude + offset")
    plt.legend()
    plt.show(block=False)

def quick_run_cebra(X, title="CEBRA Embedding"):
    model = cebra_model.fit(X)
    embedding = model.transform(X)
    times = np.arange(X.shape[0]) / 500.0  #  500 Hz sampling
    # Downsample for plotting
    n_points=15000
    step = max(1, len(X) // n_points)
    idx = np.arange(0, len(X), step)
    embedding_small = embedding[idx]
    times_small = times[idx]

    fig = plot_embedding_interactive(
        embedding_small,
        embedding_labels=times_small,
        title=title,
        markersize=2,
        cmap="rainbow"
    )
    return fig, embedding

def run_subject_pipeline(subject_key, info, t_start, t_end, band, ch_label, output_root):
    raw = info["raw"].copy()
    raw.apply_function(lambda x: x * 1e6, picks='eeg')  # Scale to µV

    sfreq = raw.info["sfreq"]
    picks = mne.pick_types(raw.info, eeg=True, eog=False)

# Optional: to check for correctness
    # # Plot full raw data using MNE's built-in plot 
    # raw.plot(scalings='auto', title=f"Raw EEG - {subject_key}")
    # # Plot PSD 
    # desired_window_sec = 0.5
    # n_per_seg = int(sfreq * desired_window_sec)
    # psd = raw.compute_psd(fmin=1, fmax=50, picks=picks, n_per_seg=n_per_seg)
    # psds = psd.get_data()
    # psd.plot()
 
    logger.info(f"Subject: {subject_key}")
    logger.info(f"  Sampling Frequency: {sfreq} Hz")
    logger.info(f"  Duration: {info['duration_sec']:.2f} sec")

    X = raw.get_data(picks=picks).T

# Optional: to check for correctness
    # # Plot picked & transposed data 
    # plot_data(X, title=f"EEG After Picks & Transpose (µV) - {subject_key}")

    config_str = f"T{t_start}-{t_end}_B{band}_CH{ch_label}"
    fig_title = f"CEBRA - {subject_key} - {config_str}"
    fig, embedding = quick_run_cebra(X, title=fig_title)
    if SAVE_HTML:
        subject_folder = output_root / subject_key
        subject_folder.mkdir(parents=True, exist_ok=True)
        output_file = subject_folder / f"{subject_key}_{config_str}_embedding.html"
        fig.write_html(str(output_file), auto_open=False)
        logger.info(f"Saved: {output_file}")
    else:
        fig.show() 

# ===== Main Loop =====
SAVE_HTML = False # change to True to export files

# loop through all config combinations
all_subjects_raws = eeg_dataloader.load_all_subjects(
    data_dir=str(data_dir),
    data_type="preproc",
    subjects_to_load=subject_ids_to_load
)

for time_config, band_config, (channels, ch_label) in configurations:
    t_start, t_end = time_config
    
    logger.info(f"Loading data for config: time={t_start}-{t_end}, band={band_config}, channels={ch_label}")
    for subject_key, raw_full in all_subjects_raws.items():
        # Apply filters and cropping on the fly
        raw_processed = eeg_dataloader.filter_crop_data(
            raw_full,
            t_start=t_start,
            t_end=t_end,
            filter_frequency_band=band_config,
            pick_channels=channels
        )
        
        info = {
            "raw": raw_processed,
            "sfreq": raw_processed.info['sfreq'],
            "duration_sec": raw_full.times[-1] - raw_full.times[0],
            "snippet_duration_sec": raw_processed.times[-1] - raw_processed.times[0]
        }
       
        run_subject_pipeline(subject_key, info, t_start, t_end, band_config, ch_label, output_root)


