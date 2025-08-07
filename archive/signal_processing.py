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


- added colour system for encoding valence affect labels insto colour system
"""
import numpy as np
import matplotlib.pyplot as plt
import mne
from pathlib import Path
import logging
import plotly.io as pio
pio.renderers.default = "browser"
import matplotlib.colors as mcolors
from typing import Optional, List, Dict, Tuple

from cebra import CEBRA
from cebra.integrations.plotly import plot_embedding_interactive
import archive.data.eeg_dataloader as eeg_dataloader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


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



# Global dict to hold embeddings and figures for dashboard
# Key structure: {subject: {config_str: fig}}

def run_subject_pipeline(subject_key, info, t_start, t_end, band, ch_label, output_root):


# Optional: to check for correctness
    # # Plot full raw data using MNE's built-in plot 
    # raw.plot(scalings='auto', title=f"Raw EEG - {subject_key}")
    # # Plot PSD 
    # desired_window_sec = 0.5
    # n_per_seg = int(sfreq * desired_window_sec)
    # psd = raw.compute_psd(fmin=1, fmax=50, picks=picks, n_per_seg=n_per_seg)
    # psds = psd.get_data()
    # psd.plot()

# Optional: to check for correctness
    # # Plot picked & transposed data 
    # plot_data(X, title=f"EEG After Picks & Transpose - {subject_key}")


# ===== Main Loop =====

if __name__ == "__main__":
    # plot_valence_arousal_color_wheel()
    SAVE_HTML = True  # Change to True to save HTML files instead of showing
    
    all_subjects_raws = eeg_dataloader.load_all_subjects(
        data_dir=str(data_dir),
        data_type="preproc",
        subjects_to_load=subject_ids_to_load
    )
    
    for time_config, band_config, (channels, ch_label) in configurations:
        t_start, t_end = time_config
        
        logger.info(f"Loading data for config: time={t_start}-{t_end}, band={band_config}, channels={ch_label}")
        for subject_key, raw_full in all_subjects_raws.items():
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
    
