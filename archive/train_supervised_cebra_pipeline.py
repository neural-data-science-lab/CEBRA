"""
train_supervised_cebra_main.py

Full EEG supervised embedding pipeline:
- Preprocess EEG
- Align behavioral data
- Train CEBRA (supervised, 3 runs)
- Save embeddings (.npy) and behavioral distribution plots

Author: You
Created: 2025-08-27
"""

from pathlib import Path
import warnings

from config import (
    DATA_DIR,
    SUBJECT_IDS_TO_LOAD,
    TIME_CONFIGS,
    FILTER_BANDS,
    CHANNEL_CONFIGS,
    OUTPUT_ROOT,
)
from data import eeg_dataloader
from eeg.perprocessing import filter_crop_data
from eeg.cebra_supervise import run_subject_pipeline, compute_population_median
from utils.configs import generate_configurations
from utils.logger import setup_logger

warnings.filterwarnings("ignore", category=RuntimeWarning)
logger = setup_logger()


def main():
    # Generate all time/band/channel configurations
    configurations = generate_configurations(
        time_configs=TIME_CONFIGS,
        filter_bands=FILTER_BANDS,
        channel_configs=CHANNEL_CONFIGS,
    )
    logger.info(f"[INFO] Total configurations to run: {len(configurations)}")

    # Load raw EEG for all subjects
    all_subjects = eeg_dataloader.load_all_subjects(
        data_dir=str(DATA_DIR),
        data_type="preproc",
        subjects_to_load=SUBJECT_IDS_TO_LOAD,
    )
    if not all_subjects:
        logger.error("[ERROR] No subjects found!")
        return

    # Compute global medians for binary valence/arousal
    val_median, aro_median = compute_population_median(SUBJECT_IDS_TO_LOAD, DATA_DIR)
    logger.info(f"[INFO] Global medians - Valence: {val_median}, Arousal: {aro_median}")

    # Loop through all configurations
    for time_cfg, band_cfg, (channels, ch_label) in configurations:
        t_start, t_end = time_cfg
        logger.info(f"[INFO] Running config: Time={time_cfg}, Band={band_cfg}, Channels={ch_label}")

        for subject_key, raw in all_subjects.items():
            logger.info(f"[INFO] Subject: {subject_key}")

            # Preprocess EEG (filter + crop)
            raw_processed = filter_crop_data(
                raw,
                t_start=t_start,
                t_end=t_end,
                filter_frequency_band=band_cfg,
                pick_channels=channels,
            )

            # Run supervised CEBRA pipeline (all 3 runs)
            run_subject_pipeline(
                subject_key=subject_key,
                root=DATA_DIR,
                raw=raw_processed,
                t_start=t_start,
                t_end=t_end,
                band=band_cfg,
                channels=channels,
                channels_label=ch_label,
                output_root=OUTPUT_ROOT,
                val_median=val_median,
                aro_median=aro_median
            )


if __name__ == "__main__":
    main()
