# Data Loading and Preprocessing

### Dataset Overview

This study utilized electroencephalography (EEG) data from multiple subjects stored in a standardized BIDS-compatible format. The dataset consists of preprocessed EEG recordings from individual subjects, with each subject's data stored in separate directories following the naming convention `sub-XXX`.

### EEG Data Loading

#### Subject-Level Data Loading

A custom data loading pipeline was implemented to handle both preprocessed (.fif) and raw (.edf) EEG file formats. The `load_subject()` function serves as the core loading mechanism, accepting a subject directory path and data type specification. For this analysis, preprocessed data in MNE-Python FIF format was utilized, specifically targeting files with the naming pattern `*before_ica.fif`, indicating data that had undergone preprocessing but prior to independent component analysis.

#### Multi-Subject Data Integration

The `load_subjects()` function orchestrates the loading of multiple subjects with several key optimizations:

- **Channel Selection**: To reduce computational complexity and focus on regions of interest, analysis was restricted to four midline electrodes: Fz (frontal), Cz (central), Pz (parietal), and Oz (occipital). This electrode selection provides coverage across major cortical regions while maintaining computational efficiency.

- **Temporal Downsampling**: To further optimize memory usage and processing speed, EEG signals were downsampled by a factor of 2, reducing the effective sampling rate while preserving essential signal characteristics.

- **Memory Management**: The pipeline incorporates explicit memory management through garbage collection and data type optimization (float32 precision) to handle large multi-subject datasets efficiently.

### Temporal Alignment and Indexing

#### Continuous Time Index Creation

To enable temporal alignment across multiple recording sessions, a continuous time indexing system was implemented through the `create_time_index()` function. This approach:

- Generates sequential time indices for each subject's recording session
- Assumes a default sampling rate of 500 Hz (adjustable based on actual recording parameters)
- Creates time vectors in seconds, formatted as (n_samples, 1) arrays for compatibility with downstream analysis

#### Multi-Session Dataset Construction

The final step in data preparation involves creating a unified multi-session dataset using the CEBRA framework's specialized data structures:

1. **TensorDataset Construction**: Each subject's neural data is encapsulated within a `TensorDataset` object, which serves as CEBRA's core single-session data container. The TensorDataset class provides several key features:
    - **Neural Data Storage**: Accepts neural activity arrays of shape (N, D) where N represents time points and D represents the number of channels
    - **Continuous Indexing**: Time indices are stored as continuous behavioral variables of shape (N, d), enabling temporal contrastive learning
    - **Data Type Management**: Automatically converts numpy arrays to PyTorch tensors and ensures appropriate data types (float32 for neural data, float for continuous indices)
    - **Device Compatibility**: Supports GPU acceleration through device specification
    - **Flexible Indexing**: Supports both discrete and continuous auxiliary variables for different types of behavioral correlates

2. **DatasetCollection Architecture**: Individual TensorDataset objects are aggregated into a `DatasetCollection`, which implements CEBRA's multi-session dataset interface:
    - **Session Management**: Maintains ordered collections of single-session datasets while preserving individual session characteristics
    - **Cross-Session Consistency**: Validates that all sessions use consistent indexing schemes (all continuous or all discrete)
    - **Unified Access**: Provides consolidated access to neural data and behavioral indices across all sessions through concatenated index tensors
    - **Device Synchronization**: Ensures all constituent datasets reside on the same computational device
    - **Dimension Compatibility**: Validates input dimensions across sessions for consistent model training

### Data Structure and Format

The resulting dataset architecture leverages CEBRA's specialized data containers:

- **Neural Data**: Multi-dimensional tensors with dimensions (time_points, channels) for each subject, stored as float32 PyTorch tensors within TensorDataset objects
- **Temporal Indices**: Continuous time vectors stored as behavioral variables, enabling cross-session temporal alignment and contrastive learning
- **Multi-Session Format**: A DatasetCollection containing multiple TensorDataset instances, each representing an individual subject's recording session
- **Index Concatenation**: The DatasetCollection automatically concatenates continuous indices across all sessions, creating unified temporal representations for multi-session analysis
- **Session Preservation**: While enabling cross-session analysis, the architecture maintains individual session boundaries and characteristics through the underlying TensorDataset structure
