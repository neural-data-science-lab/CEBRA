# CEBRA: latent embeddings neural dynamics

`[Last update: June 10, 2025]`

    Period:     2025-06 -  
    Status:     Active   

## Overview

This repository is a codebase aimed at  **applying and extending the CEBRA (Contrastive Embedding for Behavioral and Neural Analysis) method** across multiple datasets. The primary goal is to set up and reproduce the [CEBRA best practices notebook](https://colab.research.google.com/github/AdaptiveMotorControlLab/CEBRA-demos/blob/main/CEBRA_best_practices.ipynb#scrollTo=8wexciDCXx79) to enable analyzing neural and behavioral data using CEBRA embeddings. 

The main goal is to replicate and extend the  workflow from the Google Colab notebook in a local environment that is stable, reproducible, and aligned with community best practices.

---

## Table of Contents

- [Project Structure](#project-structure) 
- [Replicating CEBRA Best Practices](#replicating-cebra-best-practices)  
- [Datasets](#datasets)  
- [Usage](#usage)  
- [Installation](#installation)  
- [Conda Environment Setup](#conda-environment-setup) 
- [References](#references)  
- [Contributing](#contributing)  
- [Contact](#contact)

---
## Project Structure


## Replicating CEBRA Best Practices

This project follows the workflow demonstrated in the  [CEBRA best practices colab notebook](https://colab.research.google.com/github/AdaptiveMotorControlLab/CEBRA-demos/blob/main/CEBRA_best_practices.ipynb#scrollTo=8wexciDCXx79).

## Datasets
 - *Affective VR* (AVR)
 - 

## Usage

## Installation

### Prerequisites
 
- Python 3.10 (managed via conda environment)  

### Create and activate environment

```bash
conda create -n cebra_venv python=3.10 -c conda-forge

conda activate cebra_venv
```

## Conda Environment Setup

This project uses **conda** with the `conda-forge` channel to manage packages, as it ensures better compatibility and fewer dependency conflicts compared to `pip`.

### Create the environment

```bash
conda create -n cebra_venv python=3.10 -c conda-forge
```

### Install dependencies
```bash
conda install --file requirements.txt -c conda-forge
```

Use pip only if necessary after conda installs.

## References
### Primary CEBRA Best Practices Notebook
- [CEBRA best practices colab notebook](https://colab.research.google.com/github/AdaptiveMotorControlLab/CEBRA-demos/blob/main/CEBRA_best_practices.ipynb#scrollTo=8wexciDCXx79)

### Projects
- [AffectiveVR (AVR) Project](https://github.com/lucyroe/AVR)

## Contributors
- Cristina Bayer
- (insert others)