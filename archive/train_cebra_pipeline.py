"""
train_and_analyze.py

Full EEG embedding pipeline:
- Preprocess EEG
- Align behavioral data
- Train CEBRA
- Save embeddings (.npy), interactive HTML plots, and behavioral distribution histograms

Author: You
Created: 2025-08-07
"""

# --------------------------------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------------------------------
from pathlib import Path

from config import (
    DATA_DIR,
    SUBJECT_IDS_TO_LOAD,
    TIME_CONFIGS,
    FILTER_BANDS,
    CHANNEL_CONFIGS,
    SAVE_HTML,
    OUTPUT_ROOT,
)
from data import eeg_dataloader
from eeg.perprocessing import filter_crop_data
from eeg.cebra_pipeline import run_subject_pipeline
from utils.configs import generate_configurations
from utils.logger import setup_logger

logger = setup_logger()
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# --------------------------------------------------------------------------------------------
# Main Execution
# --------------------------------------------------------------------------------------------

def main():
    configurations = generate_configurations(
        time_configs=TIME_CONFIGS,
        filter_bands=FILTER_BANDS,
        channel_configs=CHANNEL_CONFIGS,
    )

    logger.info(f"[INFO] Total configurations to run: {len(configurations)}")

    all_subjects = eeg_dataloader.load_all_subjects(
        data_dir=str(DATA_DIR),
        data_type="preproc",
        subjects_to_load=SUBJECT_IDS_TO_LOAD,
    )

    for time_cfg, band_cfg, (channels, ch_label) in configurations:
        t_start, t_end = time_cfg
        logger.info(f"[INFO]Running config: Time={time_cfg}, Band={band_cfg}, Channels={ch_label}")

        for subject_key, raw in all_subjects.items():
            logger.info(f"[INFO] Subject: {subject_key}")

            raw_processed = filter_crop_data(
                raw,
                t_start=t_start,
                t_end=t_end,
                filter_frequency_band=band_cfg,
                pick_channels=channels,
            )

            run_subject_pipeline(
                subject_key=subject_key,
                root = DATA_DIR,
                raw=raw_processed,
                t_start=t_start,
                t_end=t_end,
                band=band_cfg,
                channels=channels,
                channels_label=ch_label,
                output_root=OUTPUT_ROOT,
                save_html=SAVE_HTML,
                explore_behavior_data = False, #set to false if only training
                train_model  = True, # set to false if only explorative analyses
            )

    
if __name__ == "__main__":
    main()
