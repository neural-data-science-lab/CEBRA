# CEBRA: latent embeddings neural dynamics

`[Last update: June 10, 2025]`

    Period:     2025-06 -  
    Status:     Active   

## Overview

This repository contains a pipeline for training a **CEBRA (Contrastive Embeddings for Behavioral Representation Analysis)** model on multi-subject EEG data. The pipeline supports loading raw or preprocessed EEG signals, preparing them as multi-session datasets, and training a shared neural network embedding model using contrastive learning.

---

## Table of Contents

- [Project Structure](#project-structure) 
- [Replicating CEBRA Best Practices](#replicating-cebra-best-practices)  
- [Datasets](#datasets)  
- [Usage](#usage)  
- [Installation](#installation)  
- [References](#references)  
- [Contributing](#contributing)  
- [Contact](#contact)

---
## Project Structure


## What is CEBRA?

**CEBRA** is a framework for learning *contrastive embeddings* that capture latent behavioral and neural states across sessions and subjects. It leverages contrastive learning to generate low-dimensional embeddings that align temporally close neural states (positive pairs) while pushing apart unrelated states (negative pairs).

Key features:

- Learns one shared model from multi-subject datasets.
    - Supports continuous temporal indexing for aligning neural activity.
- Flexible neural architectures initialized via CEBRA's `torch API`.
    - Provides dataset handling with `TensorDataset`, `DatasetCollection`, `MultiSeessionDataLoader`.
    - Uses `Solver` for efficient training loops and optimization.

---

## Datasets
 - *Affective VR* (AVR)
 - 

## Usage

## Installation

### Prerequisites
 
- Python 3.12 (managed via conda environment)  

### Conda Environment Setup

This project uses **conda** with the `conda-forge` channel to manage packages, as it ensures better compatibility and fewer dependency conflicts compared to `pip`.

#### Create the environment

```bash
conda create -n cebra python==3.9
conda activate cebra
conda install -c conda-forge pytables==3.8.0
conda install pytorch cudatoolkit=11.3 -c pytorch
# in jupyter notebook: conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -c conda-forge ipykernel mne mkdocs mkdocs-material plotly                         
```

### Install dependencies
```bash
pip install cebra
pip install '.[dev,docs,integrations,demos,datasets]'
```

Use pip only if necessary after conda installs.

CEBRA must be installed via pip, as it's currently only published on PyPI

### Jupyter Notebook

python -m ipykernel install --user --name


###  Clone the repository
```
git clone https://github.com/username/project-name.git
cd project-name
```
change the cebra code according to the described [modifications](modifications.md). 

## Usage

- Basic usage example (Demo)
    - First Steps: Create project and run with example and check output 
- configuration options
- command line

## Workflow

### 1. Data Loading

- **`load_subject`**: Loads a single subject’s EEG data from either preprocessed FIF or raw EDF files using [MNE](https://mne.tools/stable/index.html).  
- **`load_subjects`**: Loads multiple subjects, optionally picking specific EEG channels and downsampling the data for memory efficiency. Returns a dictionary mapping subject IDs to numpy arrays `(samples × channels)`.
  
### 2. Data Preprocessing

- **`create_time_index`**: Generates a continuous time vector per subject for temporal alignment during training.
- **`create_combined_cebra_dataset`**: Converts all subject data and time indices into `TensorDataset`s, then combines them into a `DatasetCollection`. This creates a multi-session dataset compatible with CEBRA’s training pipeline.

### 3. Model Definition

- **`create_model`**: Initializes a shared CEBRA neural network model based on input dimensionality and desired embedding output size.
- The model architecture is flexible but typically includes dense layers producing an embedding space optimized with a contrastive loss.

### 4. Training Pipeline

- **`create_dataloader`**: Constructs a multi-session `MultiSessionLoader` that samples anchor, positive, and negative examples across subjects, preserving temporal continuity.
- **`create_solver`**: Constructs a multi-session `Solver` that coordinates the training (optimizer + loss) 
- **`train_cebra_model`**: Coordinates training by setting up the model, configuring datasets, dataloader, and solver
    - Uses `FixedCosineInfoNCE` loss to contrast embeddings based on cosine similarity with temperature scaling.
---

## Functions & Components

### Core functions

### Main components

## Technical Notes

- GPU usage is detected automatically; fallback to CPU is supported.

---

## Expected Results and Usage



## Referencesy
### Primary CEBRA Best Practices Notebook
- [CEBRA GitHub Repository](https://github.com/robince/cebra)
- [MNE-Python Documentation](https://mne.tools/stable/index.html)

### Projects
- [AffectiveVR (AVR) Project](https://github.com/lucyroe/AVR)

## Contributors
- Cristina Bayer
- (insert others)

## Contact

For questions or contributions, please contact the author or open an issue on the repository.

## Changelog

## Licence
This project is licensed under the MIT License. 