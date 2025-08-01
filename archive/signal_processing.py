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
import eeg_dataloader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ========== Variables ==========

# Path to your EEG data
data_dir = Path(r"E:\Cris_Work\preproc")
output_root = Path("results")
output_root.mkdir(exist_ok=True)

# Configurations
subject_range = (0, 4)
subject_ids_to_load = [f"sub-{i:03d}" for i in range(subject_range[0], subject_range[1])]

# time cropping
time_configs = [(None,None), # Full session
                (0, 300),  # Baseline / Start
                (600, 900),  # Stimulus segment
                (600, 660),] # single moment
                #(1200, 1391)] # Ending segment

# Filter by frequency band
#theta = (4, 8)
#alpha = (8, 12)
#beta = (13, 30)
filter_bands = [None] # theta, alpha, beta Hz

# Select specific EEG channels
frontal_channels = [
    'F3', 'F4', 'F7', 'F8',
    'AF3', 'AF4', 'AF7', 'AF8',
    'Fz', 'AFz'
]
central_parietal_channels = [
    'Cz', 'Pz', 'CPz',
    'P3', 'P4', 'P7', 'P8',
    'POz', 'FC1', 'FC2',
    'CP1', 'CP2'
]
combined = [
    'Cz', 'Pz', 'CPz',
    'P3', 'P4', 'P7', 'P8',
    'POz', 'FC1', 'FC2',
    'CP1', 'CP2', 'F3', 'F4', 'F7', 'F8',
    'AF3', 'AF4', 'AF7', 'AF8',
    'Fz', 'AFz'
]
#CP and F, tremporal,  insula 
channel_configs = [(None, "all"),]
                    #(tuple(frontal_channels), "frontal"),
                    #(tuple(central_parietal_channels), "central_parietal"), 
                    #(tuple(combined), "frontal_central_parietal")]


# Generate configurations: baseline + one parameter changed
# Define baseline configuration
baseline_time = (None, None)
baseline_band = None
baseline_channels = (None, "all")
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

#Define custom emotional color wheel (angle → color)
angle_degrees = np.array([0, 45, 90, 135, 180, 225, 270, 315, 360])
color_hex = ['#90ee90',  # 0°  light green
             '#ffff00',  # 45° yellow
             '#ff9900',  # 90° orange
             '#ff0000',  # 135° red
             '#800080',  # 180° purple
             '#0000ff',  # 225° blue
             '#00ffff',  # 270° cyan
             '#00ff00',  # 315° green
             '#90ee90']  # 360° repeat light green
rgb_colors = np.array([mcolors.to_rgb(c) for c in color_hex])

# Compute angle and vector_length from centered VA
def compute_angle_vector_length(valence, arousal):
    val_clipped = np.clip(valence, -1, 1)
    aro_clipped = np.clip(arousal, -1, 1)

    x = val_clipped
    y = aro_clipped

    angle_rad = np.arctan2(y, x)
    angle_deg = (np.degrees(angle_rad) + 360) % 360

    vector_length = np.sqrt(x**2 + y**2) / np.sqrt(2)  # Normalize to [0, 1]
    return angle_deg, np.clip(vector_length, 0, 1)

#Interpolate color by angle
def interpolate_rgb_from_angle(angle_deg):
    angle_deg = np.asarray(angle_deg)
    interpolated_rgb = np.zeros((len(angle_deg), 3))

    for i, angle in enumerate(angle_deg):
        idx = np.searchsorted(angle_degrees, angle) - 1
        idx = np.clip(idx, 0, len(angle_degrees) - 2)

        angle1 = angle_degrees[idx]
        angle2 = angle_degrees[idx + 1]
        color1 = rgb_colors[idx]
        color2 = rgb_colors[idx + 1]

        t = (angle - angle1) / (angle2 - angle1)
        interpolated_rgb[i] = (1 - t) * color1 + t * color2

    return interpolated_rgb

def valence_arousal_emotion_color(valence, arousal, desaturate_color=(1.0, 1.0, 1.0)):
    angle, vector_length = compute_angle_vector_length(valence, arousal)

    base_rgb = interpolate_rgb_from_angle(angle)

    # Apply radial saturation: mix toward white
    final_rgb = (1 - vector_length[:, None]) * desaturate_color + vector_length[:, None] * base_rgb
    return np.clip(final_rgb, 0, 1)

 
def quick_run_cebra(X, valence_aligned, arousal_aligned, title="CEBRA Embedding by Valence-Arousal"):
    model = cebra_model.fit(X)
    embedding = model.transform(X)
    times = np.arange(X.shape[0]) / 500.0  #  500 Hz sampling
    # Downsample for plotting
    n_points=15000
    step = max(1, len(X) // n_points)
    idx = np.arange(0, len(X), step)
    embedding_small = embedding[idx]
    times_small = times[idx]

    # Get joint RGB colors
    valence_small = valence_aligned[idx]
    arousal_small = arousal_aligned[idx]
    colors = valence_arousal_emotion_color(valence_small, arousal_small)
    colors_hex = np.array([mcolors.to_hex(c) for c in colors])

    fig = plot_embedding_interactive(
        embedding_small,
        embedding_labels= colors_hex,
        title=title,
        markersize=2,
    )
    plt.close('all') # clean up hidden matplotlib figures
    return fig, embedding

# Global dict to hold embeddings and figures for dashboard
# Key structure: {subject: {config_str: fig}}

def run_subject_pipeline(subject_key, info, t_start, t_end, band, ch_label, output_root):
    raw = info["raw"].copy()
    all_channels = [ch for ch in raw.info['ch_names'] if ch in mne.pick_info(raw.info, mne.pick_types(raw.info, eeg=True))['ch_names']]
    logger.info(f"Available EEG channels: {all_channels}")

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
    # plot_data(X, title=f"EEG After Picks & Transpose - {subject_key}")

    # Load behavioral labels 
    subject_folder = data_dir / subject_key
    beh_df = eeg_dataloader.load_behavioral_labels(subject_folder)

    # 'valence' has one value per timestamp (sample)
    valence = beh_df["valence"].values
    arousal = beh_df["arousal"].values

    t_behavior = beh_df["timestamp"].values
    t_eeg = np.arange(X.shape[0]) / raw.info["sfreq"]  # EEG timestamps

    if len(valence) != X.shape[0]:
        print("[INFO] Interpolating behavioral values to match EEG sample rate...")
        from scipy.interpolate import interp1d

        interp_val = interp1d(t_behavior, valence, kind="linear", bounds_error=False, fill_value="extrapolate")
        interp_aro = interp1d(t_behavior,   arousal  , kind="linear", bounds_error=False, fill_value="extrapolate")

        valence_aligned = interp_val(t_eeg)
        arousal_aligned = interp_aro(t_eeg)
    else:
        valence_aligned = valence
        arousal_aligned = arousal
    
    # Downsample for plotting
    config_str = f"T{t_start}-{t_end}_B{band}_CH{ch_label}"
    debug_valence_arousal_distribution(
        valence_aligned, 
        arousal_aligned,
        subject_key=subject_key,
        output_root=output_root,
        config_str=config_str
    )

    fig_title = f"CEBRA - {subject_key} - {config_str}"
    fig, embedding = quick_run_cebra(X, valence_aligned, arousal_aligned, title=fig_title)



    if SAVE_HTML:
        subject_folder = output_root / subject_key
        subject_folder.mkdir(parents=True, exist_ok=True)
        output_file = subject_folder / f"VA_{subject_key}_{config_str}_embedding.html"
        fig.write_html(str(output_file), auto_open=False)
        logger.info(f"Saved: {output_file}")


def debug_valence_arousal_distribution(valence, arousal, subject_key=None, output_root=None, config_str=None):
    angle, _ = compute_angle_vector_length(valence, arousal)
    colors = valence_arousal_emotion_color(valence, arousal)

    # Ordner zum Speichern anlegen, wenn Pfad und Subject Key übergeben wurden
    if output_root is not None and subject_key is not None and config_str is not None:
        save_dir = Path(output_root) / "exploration" / subject_key
        save_dir.mkdir(parents=True, exist_ok=True)
    else:
        save_dir = None

    # Plot 1: Scatter Valence vs Arousal
    plt.figure(figsize=(6, 6))
    plt.scatter(valence, arousal, c=colors, s=8)
    plt.axhline(0, color='gray', linestyle='--')
    plt.axvline(0, color='gray', linestyle='--')
    plt.xlabel("Valence")
    plt.ylabel("Arousal")
    plt.title("Valence-Arousal Distribution Colored by Angle")
    plt.grid(True)
    plt.gca().set_aspect('equal')

    if save_dir:
        plt.savefig(save_dir / f"VA_distribution_{config_str}.png", dpi=150)
        plt.close()
    else:
        plt.show()

    # Plot 2: Histogram of Angles
    plt.figure()
    plt.hist(angle, bins=36)
    plt.title("Histogram of Valence-Arousal Angles")
    plt.xlabel("Angle (°)")
    plt.ylabel("Frequency")

    if save_dir:
        plt.savefig(save_dir / f"VA_angle_histogram_{config_str}.png", dpi=150)
        plt.close()
    else:
        plt.show()



def plot_valence_arousal_color_wheel(res=300, desaturate_color=(1.0, 1.0, 1.0)):
    """
    Plot the custom valence-arousal emotional color wheel with radial saturation.
    """
    val_grid, aro_grid = np.meshgrid(
        np.linspace(-1, 1, res),  # Valence: horizontal axis
        np.linspace(-1, 1, res)   # Arousal: vertical axis
    )

    val_flat = val_grid.flatten()
    aro_flat = aro_grid.flatten()

    colors = valence_arousal_emotion_color(val_flat, aro_flat, desaturate_color)
    image = colors.reshape(res, res, 3)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image, extent=(-1, 1, -1, 1), origin='lower')

    ax.set_xlabel("Valence")
    ax.set_ylabel("Arousal")
    ax.set_title("Valence-Arousal Color Wheel\n(Hue by Emotion, Saturation by Intensity)")

    # Annotate key angles (optional)
    labels = {
        (0.6, 0.6): "Joyfull\n(45°)",
        (0.0, 0.8): "Tense/Excited\n(90°)",
        (-0.6, 0.6): "Angry\n(135°)",
        (-0.8, 0.0): "Frustrated/Depressed\n(180°)",
        (-0.6, -0.6): "Sad\n(225°)",
        (0.0, -0.8): "Tired/Calm\n(270°)",
        (0.6, -0.6): "Relaxed\n(315°)",
        (0.8, 0.0): "Content/Happy\n(0°)",
        (0.0, 0.0): "Neutral\n"
    }

    for (x, y), label in labels.items():
        ax.text(x, y, label, ha='center', va='center', fontsize=8, color='black', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    plt.grid(False)
    plt.tight_layout()
    plt.show()

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
    
