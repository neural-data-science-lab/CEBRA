# EEG Data Overview

This descirption was taken from ARV Project. 

The dataset consists of 54 subjects each with a measurement of 20 minutes with a 64 channel EEG. 

## Data Processing Pipeline

The EEG data processing follows a comprehensive pipeline that transforms raw LSL-output format (`.xdf`) files into analysis-ready datasets using BIDS-compatible structure. 

## EEG Signal Processing

### Preprocessing Steps

The EEG preprocessing pipeline implements a robust multi-stage approach to ensure high-quality neural signal extraction:

**Initial Processing:** Raw EEG data undergoes cropping of 2.5 seconds from both start and end to eliminate edge artifacts that commonly occur during recording initiation and termination.

**Signal Cleaning:** The cleaning process involves several critical steps. Power line interference is removed at 50 Hz, followed by re-referencing to the average across all channels to reduce common noise. Band-pass filtering between 0.1 and 45 Hz isolates the neurophysiologically relevant frequency range while removing low-frequency drift and high-frequency noise.

**Epoch Processing:** The continuous EEG signal is segmented into 10-second epochs for systematic analysis. Bad channels and epochs are identified and either rejected or interpolated using the `autoreject` algorithm applied to 1 Hz-filtered data, ensuring optimal data quality while preserving maximum usable data.

**Artifact Removal:** Independent Component Analysis (ICA) is employed to identify and remove physiological artifacts including eye movements, cardiac interference, and muscle activity that can contaminate neural signals.

**Quality Control:** A final threshold check excludes participants with more than 30% noisy epochs remaining after preprocessing, ensuring only high-quality datasets proceed to analysis.

### Feature Extraction

The feature extraction process transforms preprocessed EEG signals into frequency-domain representations suitable for analysis:

**Signal Preparation:** Data is resampled to 100 Hz for computational efficiency while preserving essential frequency information. Symmetric padding of 90 seconds of mirrored data at signal boundaries prevents edge artifacts during time-frequency decomposition.

**Time-Frequency Analysis:** Continuous Wavelet Transform (CWT) using the `fCWT` library generates Time Frequency Representations (TFR), providing detailed information about spectral power changes over time.

**Temporal Averaging:** Power values are averaged over 2-second windows with 50% overlap, balancing temporal resolution with statistical stability.

**Frequency Band Integration:** The TFR is integrated across standard neurophysiological frequency bands (delta, theta, alpha, beta, gamma) to extract band-specific power measures that correspond to different cognitive and physiological states.

**Spatial Averaging:** Finally, power values are averaged across anatomically defined regions of interest including posterior, frontal, and whole-brain regions, providing both localized and global measures of neural activity.

This comprehensive preprocessing and feature extraction pipeline ensures that the EEG data maintains high signal quality while providing meaningful neurophysiological measures for subsequent analysis.

## VR-Specific Challenges for EEG

**Novel Artifact Sources**:

- **Head Movement Artifacts**: VR headset weight and balance create constant micro-movements
- **Visual-Induced Artifacts**: Rapid eye movements tracking VR content differ from natural viewing
- **Electromagnetic Interference**: VR displays and tracking systems introduce novel noise patterns
- **Comfort Artifacts**: Progressive muscle tension and sweating under headset affect electrode impedance

**Technical Integration Issues**:

- Electrode placement complicated by VR headset
- Potential electromagnetic interference between systems
- Synchronization challenges between VR events and EEG timing
- Limited head movement freedom affects natural viewing behavior

**Participant Experience Effects**:

- VR sickness can confound emotional responses
- Novelty effects may overshadow target emotional content
- Individual differences in VR adaptation and comfort
- Dual-task interference from continuous emotional labeling