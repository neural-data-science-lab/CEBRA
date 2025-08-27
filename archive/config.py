"""
Variables and Configuration for EEG Preprocessing Pipeline

Defines constants and configuration parameters such as data paths,
subject IDs, time cropping windows, frequency bands, and EEG channel selections.

These variables drive systematic preprocessing and analysis workflows.
"""

# =====================================================================
# Variables and Configuration
# =====================================================================

from pathlib import Path

# Path to your EEG data
DATA_DIR = Path(r"E:\Cris_Work\preproc")
OUTPUT_ROOT = Path("results")
OUTPUT_ROOT.mkdir(exist_ok=True)

# Configurations
# List of time offsets to test (in samples)
TIME_OFFSETS_TO_TEST = [100, 500, 1000, 5000]  # e.g., 0.1s, 0.5s, 1s, 10s

# Subjects
subject_range = (0, 2)
SUBJECT_IDS_TO_LOAD = [f"sub-{i:03d}" for i in range(subject_range[0], subject_range[1])]

# Time cropping windows (in seconds)
TIME_CONFIGS = [
    (None, None),   # Full session
    #(0, 300),       # Baseline / Start
    #(600, 900),     # Stimulus segment
    #(600, 660),     # Single moment
]

# Frequency bands for filtering (Hz)
theta = (4, 8)
alpha = (8, 12)
beta = (13, 30)
FILTER_BANDS = [None,
                 #theta,
                 #  alpha,
                 #  beta
                 ]

# EEG channel selections
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
    'CP1', 'CP2',
    'F3', 'F4', 'F7', 'F8',
    'AF3', 'AF4', 'AF7', 'AF8',
    'Fz', 'AFz'
]

# Channel configuration tuples (channels, label)
CHANNEL_CONFIGS = [
    (None, "all"),
    #(tuple(frontal_channels), "frontal"),
    #(tuple(central_parietal_channels), "central_parietal"),
    #(tuple(combined), "frontal_central_parietal"),
]
