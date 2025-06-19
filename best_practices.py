import sys
import numpy as np
import matplotlib.pyplot as plt
import mne # processing EEG, MEG, and other neurophysiological data.
from pathlib import Path
from sklearn.model_selection import train_test_split
import cebra
from cebra import CEBRA
import os
import tempfile
from pathlib import Path

import eeg_visualize 
import eeg_dataloader

'''
CEBRA is a self-supervised model that learns low-dimensional latent representations (embedding) from high-dimensional neural or behavioral time series data
'''

# 0. Define a CEBRA model with defaults
cebra_model = CEBRA(
    model_architecture="offset10-model", #consider: "offset10-model-mse" if Euclidean
    batch_size=512,
    learning_rate=3e-4,
    temperature_mode='constant',
    temperature=1.12,
    max_iterations=5000, #we will sweep later; start with default
    #max_adapt_iterations = 10, # TODO(user): use and to change to ~100-500 if adapting
    conditional='time', #unsupervised approach; for supervised, put 'time_delta', or 'delta'
    output_dimension=3,
    distance='cosine', #consider 'euclidean'; if you set this, output_dimension min=2
    device="cuda_if_available",
    verbose=True,
    time_offsets=10, 

)

#2. Load Data
# Define the path to the 'data' directory
data_dir = Path("data")
data_dir.mkdir(parents=True, exist_ok=True)

# Load EEG data
# For a single subject:
subject_id = "sub-001"
raw = eeg_dataloader.load_subject(subject_id, data_dir=str(data_dir))
# For all subjects: 
# data_dict = eeg_dataloader.load_all_subjects(data_dir=str(data_dir))

# 2.1 Visualize the Data
df_summary = eeg_visualize.get_eeg_summary_df(raw, subject_id)
print(df_summary)

eeg_visualize.plot_eeg_overview(raw, segment_duration=10)

channels_to_plot = ['Fz', 'Cz', 'Oz']  # Adjust channels to your dataset
try:
    eeg_visualize.plot_selected_channels(raw, channels_to_plot, segment_duration=10)
except ValueError as e:
    print(f"Visualization error: {e}")


# 3. Quick test
# Pick EEG channels
picks = mne.pick_types(raw.info, eeg=True, eog=False)
#transpose to (n_times, n_channels)
data = raw.get_data(picks=picks).T
print("Data shape:", data.shape)

# fit rhe mmodel to the data
# Fit model
cebra_time_full_model = cebra_model.fit(data)
cebra_time_full = cebra_model.transform(data)
#GoF
gof_full = cebra.sklearn.metrics.goodness_of_fit_score(cebra_time_full_model, data)
print("GoF in bits - full:", gof_full)

times = np.arange(data.shape[0]) / 500.0  # in seconds
# plot embedding
# fig = cebra.integrations.plotly.plot_embedding_interactive(cebra_time_full, embedding_labels=times, title = "CEBRA-Time (full)", markersize=3, cmap = "rainbow")
# fig.write_html("embedding.html", auto_open=True)

# downsample first
idx = np.linspace(0, cebra_time_full.shape[0] - 1, 10000).astype(int)
embedding_small = cebra_time_full[idx]
times_small = times[idx]

# Updated plot with larger dots
fig = cebra.integrations.plotly.plot_embedding_interactive(
    embedding_small,
    embedding_labels=times_small,
    title="CEBRA-Time (Downsampled to 10K)",
    markersize=5,  # ← bigger so they're visible
    cmap="rainbow"
)
fig.show()

# plot the loss curve
ax = cebra.plot_loss(cebra_time_full_model)

print("Embedding shape:", cebra_time_full.shape)
print("Embedding min:", np.min(cebra_time_full))
print("Embedding max:", np.max(cebra_time_full))
print("Unique rows:", np.unique(cebra_time_full, axis=0).shape[0])

print("Label shape:", times.shape)
print("Label min/max:", np.min(times), np.max(times))
print("Any NaNs?", np.isnan(times).any())
