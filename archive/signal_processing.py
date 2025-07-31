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
import dash
from dash import dcc, html, Input, Output, State
import plotly.subplots as sp
import re

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
subject_range = (0, 48)
subject_ids_to_load = [f"sub-{i:03d}" for i in range(subject_range[0], subject_range[1])]

# time cropping
time_configs = [(None,None), # Full session
                (0, 300),  # Baseline / Start
                (600, 900),  # Stimulus segment
                (600, 660), # single moment
                (0, 1390),  # Entire session
                (1200, 1391)] # Ending segment

# Filter by frequency band
theta = (4, 8)
alpha = (8, 12)
beta = (13, 30)
filter_bands = [None, theta, alpha, beta] #Hz

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
channel_configs = [(None, "all"),
                    (tuple(frontal_channels), "frontal"),
                    (tuple(central_parietal_channels), "central_parietal"), 
                    (tuple(combined), "frontal_central_parietal")]


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

    config_str = f"T{t_start}-{t_end}_B{band}_CH{ch_label}"
    fig_title = f"CEBRA - {subject_key} - {config_str}"
    fig, embedding = quick_run_cebra(X, title=fig_title)

    if SAVE_HTML:
        subject_folder = output_root / subject_key
        subject_folder.mkdir(parents=True, exist_ok=True)
        output_file = subject_folder / f"{subject_key}_{config_str}_embedding.html"
        fig.write_html(str(output_file), auto_open=False)
        logger.info(f"Saved: {output_file}")


# ===== Dash APP =====
def get_available_embeddings(output_root):
    subject_dirs = list(output_root.glob("sub-*"))
    subject_config_map = {}

    for subject_dir in subject_dirs:
        subject_key = subject_dir.name
        html_files = subject_dir.glob("*.html")
        subject_config_map[subject_key] = []
        for html_file in html_files:
            match = re.search(rf"{subject_key}_(.+?)_embedding\.html", html_file.name)
            if match:
                config_str = match.group(1)
                subject_config_map[subject_key].append((config_str, html_file))
    return subject_config_map

def make_grid_of_figures(figs, rows, cols, subplot_titles):
    # Create a specs grid with 3D scene type in every cell
    specs = [[{'type': 'scene'} for _ in range(cols)] for _ in range(rows)]
    fig = sp.make_subplots(rows=rows, cols=cols, specs=specs, subplot_titles=subplot_titles)

    
    for i, plotly_fig in enumerate(figs):
        r = i // cols + 1
        c = i % cols + 1
        for trace in plotly_fig.data:
            fig.add_trace(trace, row=r, col=c)
    
    fig.update_layout(height=300*rows, width=400*cols, showlegend=False, title_text="CEBRA Embeddings Grid")
    return fig


def launch_dashboard():
    app = dash.Dash(__name__)
    
    subject_config_map = get_available_embeddings(output_root)
    subjects = sorted(subject_config_map.keys())

    # Collect all config_strs from available files
    all_configs = sorted(set(
        config for configs in subject_config_map.values() for config, _ in configs
    ))

    app.layout = html.Div([
        html.H1("Live EEG CEBRA Embeddings Dashboard"),
        
        html.Div([
            html.Label("Filter by Subject:"),
            dcc.Dropdown(
                id='subject-dropdown',
                options=[{'label': s, 'value': s} for s in subjects],
                placeholder="Select a subject"
            ),
        ], style={'width': '48%', 'display': 'inline-block'}),
        
        html.Div([
            html.Label("Filter by Configuration:"),
            dcc.Dropdown(
                id='config-dropdown',
                options=[{'label': c, 'value': c} for c in all_configs],
                placeholder="Select a configuration"
            ),
        ], style={'width': '48%', 'display': 'inline-block'}),

        html.Button("Refresh", id="refresh-button", n_clicks=0),
        
        html.Div(id='plots-container')
    ])
    
    @app.callback(
        Output('plots-container', 'children'),
        Input('subject-dropdown', 'value'),
        Input('config-dropdown', 'value'),
        Input("refresh-button", "n_clicks")
    )
    def update_iframes(selected_subject, selected_config, n_clicks):
        subject_config_map = get_available_embeddings(output_root)
        figs_to_show = []

        if selected_subject and selected_config:
            return html.Div("Please select either Subject OR Configuration, not both.")

        elif selected_subject:
            configs = subject_config_map.get(selected_subject, [])
            for config_str, filepath in configs:
                figs_to_show.append(html.Div([
                    html.H4(f"{selected_subject} - {config_str}"),
                    html.Iframe(src=filepath.as_uri(), width="100%", height="600px")
                ]))

        elif selected_config:
            for subject, configs in subject_config_map.items():
                for config_str, filepath in configs:
                    if config_str == selected_config:
                        figs_to_show.append(html.Div([
                            html.H4(f"{subject} - {config_str}"),
                            html.Iframe(src=filepath.as_uri(), width="100%", height="600px")
                        ]))

        else:
            return html.Div("Please select a Subject or Configuration to display plots.")

        return figs_to_show if figs_to_show else html.Div("No HTML files found.")

    app.run(debug=False, port=8050)


# ===== Main Loop =====

if __name__ == "__main__":
    # Your original main loop with SAVE_HTML toggle
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
    
    # Launch Dash dashboard after processing all embeddings
    launch_dashboard()