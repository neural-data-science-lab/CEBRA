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

from cebra import CEBRA
from cebra.integrations.plotly import plot_embedding_interactive
import eeg_dataloader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ========== Variables ==========

# Path to your EEG data
data_dir = Path(r"E:\.Cris Work\preproc_cleaned\preproc")
output_root = Path("results")
output_root.mkdir(exist_ok=True)

# Configurations
subject_range = (0, 48)
subject_ids_to_load = [f"sub-{i:03d}" for i in range(subject_range[0], subject_range[1])]

# time cropping
time_configs = [(None,None), # Full session
                (0, 300),  # Baseline / Start
                (600, 900),  # Stimulus A
                (600, 660), # Peak emotional moment
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
embeddings_dict = {}

def store_embedding_for_dashboard(subject_key, config_str, fig):
    if subject_key not in embeddings_dict:
        embeddings_dict[subject_key] = {}
    embeddings_dict[subject_key][config_str] = fig


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
    # Store for dashboard
    store_embedding_for_dashboard(subject_key, config_str, fig)
    logger.info(f"Embeddings dict keys: {list(embeddings_dict.keys())}")
    for subj, configs in embeddings_dict.items():
        logger.info(f"Subject {subj} configs: {list(configs.keys())}")

    if SAVE_HTML:
        subject_folder = output_root / subject_key
        subject_folder.mkdir(parents=True, exist_ok=True)
        output_file = subject_folder / f"{subject_key}_{config_str}_embedding.html"
        fig.write_html(str(output_file), auto_open=False)
        logger.info(f"Saved: {output_file}")


# ===== Dash APP =====
def make_grid_of_figures(figs, rows, cols, subplot_titles):
    # Create a specs grid with 3D scene type in every cell
    specs = [[{'type': 'scene'} for _ in range(cols)] for _ in range(rows)]
    fig = sp.make_subplots(rows=rows, cols=cols, specs=specs, subplot_titles=subplot_titles)

    
    for i, plotly_fig in enumerate(figs):
        r = i // cols + 1
        c = i % cols + 1
        for trace in plotly_fig.data:
            fig.add_trace(trace, row=r, col=c)
        # For 3D subplots, axes titles are part of the 'scene'
        # So if you want to customize axes labels, you must update scene layout, e.g.:
        # scene_key = f'scene{(i+1) if (i>0) else ""}'
        # fig['layout'][scene_key].update(xaxis_title='X', yaxis_title='Y', zaxis_title='Z')
    
    fig.update_layout(height=300*rows, width=400*cols, showlegend=False, title_text="CEBRA Embeddings Grid")
    return fig


# ==== Dash app ====

def launch_dashboard():
    app = dash.Dash(__name__)
    
    subjects = sorted(embeddings_dict.keys())
    # Collect all config_strs from any subject
    all_configs = set()
    for subject in embeddings_dict:
        all_configs.update(embeddings_dict[subject].keys())
    all_configs = sorted(all_configs)

    app.layout = html.Div([
        html.H1("EEG CEBRA Embeddings Dashboard"),
        
        html.Div([
            html.Label("Filter by Subject:"),
            dcc.Dropdown(
                id='subject-dropdown',
                options=[{'label': s, 'value': s} for s in subjects],
                multi=False,
                placeholder="Select a subject"
            ),
        ], style={'width': '48%', 'display': 'inline-block'}),
        
        html.Div([
            html.Label("Filter by Configuration:"),
            dcc.Dropdown(
                id='config-dropdown',
                options=[{'label': c, 'value': c} for c in all_configs],
                multi=False,
                placeholder="Select a configuration"
            ),
        ], style={'width': '48%', 'display': 'inline-block'}),
        
        html.Div(id='plots-container')
    ])
    
    @app.callback(
        Output('plots-container', 'children'),
        Input('subject-dropdown', 'value'),
        Input('config-dropdown', 'value'),
    )
    def update_plots(selected_subject, selected_config):
        if selected_subject and selected_config:
            return html.Div("Please select either Subject OR Configuration, not both.")
        
        figs = []
        subplot_titles = []
        rows, cols = 1, 1
        
        if selected_subject:
            # Show all configurations for this subject in grid
            subject_figs = embeddings_dict.get(selected_subject, {})
            figs = [subject_figs[c] for c in sorted(subject_figs.keys())]
            subplot_titles = sorted(subject_figs.keys())
            n = len(figs)
            cols = 4
            rows = (n // cols) + int(n % cols != 0)
        elif selected_config:
            # Show all subjects for this configuration side by side
            figs = []
            available_subjects = []
            for subj in sorted(embeddings_dict.keys()):
                subj_figs = embeddings_dict[subj]
                if selected_config in subj_figs:
                    figs.append(subj_figs[selected_config])
                    available_subjects.append(subj)
            subplot_titles = available_subjects
            n = len(figs)
            cols = min(4, n)
            rows = (n // cols) + int(n % cols != 0)
        else:
            return html.Div("Please select a Subject or Configuration to display plots.")
        
        if len(figs) == 0:
            return html.Div("No plots found for selected filter.")
        
        grid_fig = make_grid_of_figures(figs, rows, cols, subplot_titles)
        
        return dcc.Graph(figure=grid_fig, style={"height": f"{rows*350}px"})
    
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