import sys
import numpy as np
import matplotlib.pyplot as plt
import mne # processing EEG, MEG, and other neurophysiological data.
from pathlib import Path
from sklearn.model_selection import train_test_split
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


