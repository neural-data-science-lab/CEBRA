
import numpy as np
import matplotlib.pyplot as plt
import mne 
from pathlib import Path
from sklearn.model_selection import train_test_split
import cebra
import tempfile
from cebra import CEBRA
from pathlib import Path
from cebra.integrations.plotly import plot_embedding_interactive
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
    #max_adapt_iterations = 10, #TODO(user): use and to change to ~100-500 if adapting
    conditional=None, #unsupervised approach; for supervised, put 'time_delta', or 'delta'
    output_dimension=3,
    distance='cosine', #consider 'euclidean'; if you set this, output_dimension min=2
    device="cuda_if_available",
    verbose=True,
    time_offsets=10, 

)

#2. Load Data
# Define the path to the 'data' directory

data_dir = Path(r"E:\.Cris Work\preproc_cleaned\preproc")

# Load EEG data
# For a single subject: 
subject_ids_to_load = ["sub-020"]

# Cshnnel selection
all_channels = ['VEOG_up', 'Fz', 'F3', 'F7', 'HEOG_left', 'FC5', 'FC1', 'C3', 'T7', 'TP9', 'CP5', 'CP1', 'Pz', 'P3', 'P7', 'O1', 'Oz', 'O2', 'P4', 'P8', 'TP10', 'CP6', 'CP2', 'Cz', 'C4', 'T8', 'HEOG_right', 'FC6', 'FC2', 'F4', 'F8', 'VEOG_down', 'AF7', 'AF3', 'AFz', 'F1', 'F5', 'FT7', 'FC3', 'C1', 'C5', 'TP7', 'CP3', 'P1', 'P5', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'P6', 'P2', 'CPz', 'CP4', 'TP8', 'C6', 'C2', 'FC4', 'FT8', 'F6', 'F2', 'AF8', 'AF4', 'Iz', 'ECG']
frontal_channels = ['Fz', 'F3', 'F4', 'F7', 'F8', 'AF3', 'AF4', 'AFz']
frontotemporal_channels = ['Fz', 'F3', 'F4', 'F7', 'F8', 'T7', 'T8']
parietal_channels = ['Pz', 'P3', 'P4']


data_dict = eeg_dataloader.load_all_subjects(
    data_dir=str(data_dir),
    subjects_to_load=subject_ids_to_load,
    pick_channels=all_channels,
    t_start=900,  # seconds
    t_end=1020)

for subj, info in data_dict.items():
    print(f"Subject: {subj}")
    print(f"  Sampling Frequency: {info['sfreq']} Hz")
    print(f"  Total Duration: {info['duration_sec']:.2f} sec")
    print(f"  Snippet Duration: {info['snippet_duration_sec']:.2f} sec")

# 3. Quick test
all_data = []

for subject_key, raw in data_dict.items():
    raw = info["raw"]
    raw.plot(n_channels=30, duration=10, block=True, title=f"EEG: {subject_key}")
    
    # # Bandpass filter (optional but good for state analysis)
    # raw.filter(1., 40., fir_design='firwin', verbose=False)py

    # Pick EEG channels only
    print("Available channel names:", raw.info['ch_names'])

    picks = mne.pick_types(raw.info, eeg=True, eog=False)

    data = raw.get_data(picks=picks).T  # shape: (n_times, n_channels)
    all_data.append(data)

# Combine into a single array
X = np.concatenate(all_data, axis=0)  # (n_times, n_channels)
# Fit model
cebra_model.fit(X)
embedding = cebra_model.transform(X)


times = np.arange(data.shape[0]) / 500.0  # in seconds
# [ERROR]: plot embedding: Problem rendering, too many points --> solution downsampled visualisation
# fig = cebra.integrations.plotly.plot_embedding_interactive(cebra_time_full, embedding_labels=times, title = "CEBRA-Time (full)", markersize=3, cmap = "rainbow")
# fig.write_html("embedding.html", auto_open=True)

# downsample first
# Create time axis
fs = 500  # change to your actual sampling rate
times = np.arange(X.shape[0]) / fs  # seconds

# Downsample for plotting
idx = np.linspace(0, len(times) - 1, 10000).astype(int)
embedding_small = embedding[idx]
times_small = times[idx]

# Interactive plot
fig = plot_embedding_interactive(
    embedding_small,
    embedding_labels=times_small,
    title="CEBRA Embedding (Brain States)",
    markersize=5,
    cmap="rainbow"
)
fig.write_html("embedding.html", auto_open=True)

# Optional: Plot training loss
cebra.plot_loss(embedding)

# Overview information
print("Embedding shape:", embedding.shape)
print("Embedding min:", np.min(embedding))
print("Embedding max:", np.max(embedding))
print("Unique rows:", np.unique(embedding, axis=0).shape[0])
print("Label shape:", times.shape)
print("Label min/max:", np.min(times), np.max(times))
print("Any NaNs?", np.isnan(times).any())


# 3. Create a Train/Validation Split
split_idx = int(0.8 * len(X))  # you can adjust this (e.g., 0.95 for large data)

train_data = X[:split_idx]
valid_data = X[split_idx:]

# Use time as continuous label for later (like hippocampus_pos.continuous_index)
train_continuous_label = times[:split_idx]
valid_continuous_label = times[split_idx:]

print("Train data shape:", train_data.shape)
print("Validation data shape:", valid_data.shape)
print("Train label shape:", train_continuous_label.shape)
print("Validation label shape:", valid_continuous_label.shape)

# 4. Train the CEBRA model on training data
cebra_train_model = cebra_model.fit(train_data)

# Save the trained model to a temporary file
tmp_file = Path(tempfile.gettempdir(), 'cebra.pt')
cebra_train_model.save(tmp_file)

# Reload the model (optional but good for testing)
cebra_train_model = cebra.CEBRA.load(tmp_file)

# 5. Transform the training and validation data into embeddings
train_embedding = cebra_train_model.transform(train_data)
valid_embedding = cebra_train_model.transform(valid_data)

# 6. Plot and Save Train Embedding
fig = plot_embedding_interactive(
    train_embedding,
    embedding_labels=train_continuous_label,
    title="CEBRA-Time Train",
    markersize=3,
    cmap="rainbow"
)
fig.write_html("train_embedding.html", auto_open=False)  # Save plot

# 7. Plot and Save Validation Embedding
fig = plot_embedding_interactive(
    valid_embedding,
    embedding_labels=valid_continuous_label,
    title="CEBRA-Time Validation",
    markersize=3,
    cmap="rainbow"
)
fig.write_html("fit_embedding.html", auto_open=False)  # Save plot