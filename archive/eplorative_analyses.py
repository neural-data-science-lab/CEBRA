"""
Full Behavioral Exploration Script

This script:
- Loads behavioral .tsv files for each subject
- Aligns them to the full EEG time axis (no cropping/filtering)
- Saves aligned CSVs to OUTPUT_ROOT/behavior_data/
- Runs group exploration plots across all subjects

Author: You
"""

# --------------------------------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------------------------------

from pathlib import Path
import pandas as pd
from config import DATA_DIR, SUBJECT_IDS_TO_LOAD, OUTPUT_ROOT
from data.eeg_dataloader import load_behavioral_labels
from eeg.exploration import run_group_exploration
from utils.logger import setup_logger

logger = setup_logger()

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
# --------------------------------------------------------------------------------------------
# Main Execution
# --------------------------------------------------------------------------------------------

def main():
    all_behavioral_dfs = []
    for subject_id in SUBJECT_IDS_TO_LOAD:
        subject_folder = DATA_DIR / subject_id
        try:
            beh_df = load_behavioral_labels(subject_folder) 
            logger.info(f"{subject_id}: loaded behavioral data shape {beh_df.shape}")
            nan_counts = beh_df[['valence', 'arousal']].isna().sum().to_dict()
            logger.info(f"{subject_id}: NaN counts before combining - {nan_counts}")
            
            beh_df["subject_id"] = subject_id 
            all_behavioral_dfs.append(beh_df)
            logger.info(f"Loaded behavioral data for {subject_id}")
        except FileNotFoundError as e:
            logger.error(str(e))

    if not all_behavioral_dfs:
        logger.error("No behavioral data loaded. Exiting.")
        return

    combined_df = pd.concat(all_behavioral_dfs, ignore_index=True)
    combined_nan_counts = combined_df[['valence', 'arousal']].isna().sum().to_dict()
    logger.info(f"Combined data NaN counts before dropping: {combined_nan_counts}")
    combined_df = combined_df.dropna(subset=['valence', 'arousal'])

    group_output_dir = OUTPUT_ROOT / "overview_exploration"
    metrics = run_group_exploration(combined_df, group_output_dir)

    logger.info(f"Group exploration complete. Metrics:\n{metrics}")

if __name__ == "__main__":
    main()