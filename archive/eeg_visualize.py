import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def get_eeg_summary_df(raw, subject_id=None):
    info = raw.info
    n_channels = info['nchan']
    ch_names = info['ch_names']
    sfreq = info['sfreq']
    duration_sec = raw.n_times / sfreq

    summary_dict = {
        'Subject': subject_id or "Unknown",
        'Number of channels': n_channels,
        'Sampling frequency (Hz)': sfreq,
        'Recording duration (s)': duration_sec,
        'Channels (first 5)': ", ".join(ch_names[:5]) + ("..." if n_channels > 5 else "")
    }
    return pd.DataFrame([summary_dict])

def plot_eeg_overview(raw, segment_duration=10.0, sfreq=None):
    """
    Plot an overview of EEG signals and a scatter plot of one channel amplitude.
    """
    if sfreq is None:
        sfreq = raw.info['sfreq']
    
    n_samples = int(segment_duration * sfreq)
    data, times = raw[:, :n_samples]

    fig = make_subplots(rows=1, cols=2, subplot_titles=("EEG Signal Overview", "Channel Amplitude Scatter"))

    fig.add_trace(
        go.Heatmap(
            z=data,
            x=times,
            y=raw.ch_names,
            colorscale='Blues',
            colorbar=dict(title='Amplitude (µV)'),
            zsmooth='best',
        ),
        row=1, col=1
    )

    channel_idx = 0
    fig.add_trace(
        go.Scatter(
            x=times,
            y=data[channel_idx],
            mode='markers',
            marker=dict(
                color=data[channel_idx],
                colorscale='Rainbow',
                size=4,
                colorbar=dict(title='Amplitude (µV)'),
                showscale=True
            ),
            name=raw.ch_names[channel_idx]
        ),
        row=1, col=2
    )

    fig.update_layout(
        height=500,
        width=900,
        title_text=f"EEG Data Overview ({segment_duration} seconds)",
        showlegend=False
    )

    fig.update_xaxes(title_text="Time (s)", row=1, col=1)
    fig.update_yaxes(title_text="Channel", row=1, col=1)
    fig.update_xaxes(title_text="Time (s)", row=1, col=2)
    fig.update_yaxes(title_text="Amplitude (µV)", row=1, col=2)

    fig.show()


def plot_selected_channels(raw, channels, segment_duration=10.0, sfreq=None):
    """
    Plot time series line plots of selected EEG channels over a segment.
    """
    if sfreq is None:
        sfreq = raw.info['sfreq']

    n_samples = int(segment_duration * sfreq)

    channel_indices = [raw.ch_names.index(ch) for ch in channels]
    data, times = raw[channel_indices, :n_samples]

    fig = go.Figure()

    for i, ch in enumerate(channels):
        fig.add_trace(go.Scatter(
            x=times,
            y=data[i],
            mode='lines',
            name=ch
        ))

    fig.update_layout(
        title=f"Time Series of Selected Channels ({segment_duration}s)",
        xaxis_title="Time (s)",
        yaxis_title="Amplitude (µV)",
        height=400,
        width=900
    )
    fig.show()


def overview_loaded_data(data, channels=None, segment_duration=10):
    """
    Helper to handle visualization and summary for loaded data.
    data: single mne.Raw or dict of {subject_id: mne.Raw}
    channels: list of channels for time series overlay plot
    """
    if channels is None:
        channels = ['Fz', 'Cz', 'Oz']

    if isinstance(data, dict):
        for subject_id, raw in data.items():
            print(f"\n--- Subject: {subject_id} ---")
            print_eeg_summary(raw, subject_id=subject_id)  # Make sure this is called!
            plot_eeg_overview(raw, segment_duration=segment_duration)
            try:
                plot_selected_channels(raw, channels, segment_duration=segment_duration)
            except ValueError as e:
                print(f"Error plotting channels for {subject_id}: {e}")
    else:
        print_eeg_summary(data)
        plot_eeg_overview(data, segment_duration=segment_duration)
        try:
            plot_selected_channels(data, channels, segment_duration=segment_duration)
        except ValueError as e:
            print(f"Error plotting channels: {e}")