# Initial Exploration of EEG Embeddings

The initial phase of this project focused on exploring neural embeddings learned from **preprocessed EEG data** using the [CEBRA](https://github.com/AdaptiveMotorControlLab/cebra) (Contrastive Embedding for Behavioral and neural Representation Analysis) framework. This step served as an unsupervised, exploratory analysis to evaluate the latent structure of neural activity and its variability depending on the selected EEG channels and subject identity.

### Objectives

* Test the CEBRA embedding framework on preprocessed EEG recordings.
* Investigate how the resulting embeddings vary across different channel subsets.
* Visualize the temporal dynamics of EEG representations in a reduced latent space.
* Establish a foundation for future extension and comparisons across subjects or conditions.

### 

* **Channel Effects**: Preliminary visual differences were examined when using different sets of EEG channels (e.g., frontal vs. parietal)
* **Subject-Specific Embedding**: Only one subject was analyzed in this stage. Generalization across subjects will be tested in subsequent experiments.

#### Data Preparation

Preprocessed EEG data were loaded for a single subject (e.g `sub-020`) from a cleaned dataset directory. The data had already undergone necessary preprocessing (e.g., filtering, artifact removal, ICA), and therefore represent neural activity in a relatively clean and interpretable form.

Three channel configurations were defined to examine the spatial dependence of the embeddings:

* **All Channels**: Full set of EEG, EOG, and ECG channels.
* **Frontal and Frontotemporal Channels**: To isolate activity from cognitive and executive function regions.
* **Parietal Channels**: Targeting posterior activity, often associated with visual and sensorimotor processing.

Only EEG channels were used during model fitting. Auxiliary channels (e.g., EOG, ECG) were excluded from the embedding process.

```python
picks = mne.pick_types(raw.info, eeg=True, eog=False)
data = raw.get_data(picks=picks).T  # shape: (n_times, n_channels)
```

The time series data were transposed and concatenated to form a matrix of shape `(n_timepoints, n_channels)`, suitable for input to the CEBRA model.

---

#### Embedding via CEBRA

The CEBRA model was initialized with the following configuration:

* **Model architecture**: Offset-based temporal contrastive model (`offset10-model`)
* **Training mode**: Unsupervised (`conditional=None`)
* **Distance metric**: Cosine similarity
* **Latent dimensionality**: 3 (for visualization)
* **Hardware**: GPU used if available

```python
cebra_model = CEBRA(
    model_architecture="offset10-model",
    batch_size=512,
    learning_rate=3e-4,
    temperature_mode='constant',
    temperature=1.12,
    max_iterations=5000,
    conditional=None,
    output_dimension=3,
    distance='cosine',
    device="cuda_if_available",
)
```

After training, the latent representation of the EEG time series was computed:

```python
embedding = cebra_model.transform(X)
```


#### Visualization

The embeddings were visualized using an interactive 3D scatter plot, with time used as the color gradient to illustrate temporal progression of brain states.

Due to the large number of timepoints, a **downsampling** step was applied before visualization to improve rendering performance and clarity:

```python
idx = np.linspace(0, len(times) - 1, 10000).astype(int)
embedding_small = embedding[idx]
times_small = times[idx]
```

The resulting plot displays the trajectory of brain states over time in the learned latent space:

```python
fig = plot_embedding_interactive(
    embedding_small,
    embedding_labels=times_small,
    title="CEBRA Embedding (Brain States)",
    markersize=5,
    cmap="rainbow"
)
```


### Next Step:

- **Multi-session shared embedding** Include additional subjects to assess cross-subject variability and potential alignment in the embedding space.
- **Multimodal embedding**Incorporate behavioral or event-based labels to train supervised embeddings using time-aligned contrasts.
* Quantify embedding quality using metrics like clustering, decoding accuracy, or neighborhood preservation.

# Multi-Session shared embedding

Great! Here’s a clean, organized draft section for your **Multi-Session Shared Embedding** part based on what you described. It fits well as a continuation of your initial exploration and sets the stage for cross-subject analysis using CEBRA.

---

# Multi-Session Shared Embedding

### Overview

Building upon the initial single-subject analysis, this phase extends the embedding framework to a **multi-subject, multi-session** setting. The goal is to learn a shared neural embedding space that generalizes across subjects, capturing common latent brain state dynamics while accounting for individual variability.

---

### Data Loading and Preprocessing
analog to Initial Exploration

### Temporal Alignment and Dataset Construction

* A continuous time indexing system (`create_time_index()`) aligned EEG samples across sessions, generating sequential time vectors (in seconds) compatible with downstream analysis.
* Each subject's data was encapsulated within a **`TensorDataset`**, CEBRA’s single-session data container holding neural data `(N, D)` and continuous time indices `(N, 1)`.
* These datasets were combined into a **`DatasetCollection`**, which manages multi-session data by maintaining session boundaries and ensuring consistent indexing formats.
* The multi-session structure enables unified access to all subjects’ data while preserving session-specific features.


### Model Architecture and Training Setup

* The input dimension was determined by the number of channels (4 electrodes), ensuring proper model-data compatibility.
* A shared neural network was initialized with:

  * An input layer matching the EEG channel count.
  * A hidden layer with 256 units.
  * An output embedding layer with a user-defined dimensionality (`output_dim`).

```python
model = cebra.models.init(
    name="offset10-model",
    num_neurons=input_dim,
    num_units=256,
    num_output=output_dim
).to(device)
```

* A **multi-session dataloader** (`MultiSessionLoader`) was configured to sample contrastive triplets uniformly across sessions, including:

  * **Anchor samples:** reference EEG time points.
  * **Positive samples:** temporally nearby points within the same session.
  * **Negative samples:** points from other sessions/subjects.
*   embeddings that capture both within-subject temporal continuity and across-subject variability.


### Loss Function and Optimization

* Training leveraged CEBRA’s **MultiSessionSolver**, which integrates multi-session batch processing, model inference, and gradient updates.
* The **FixedCosineInfoNCE** loss encouraged high similarity between positive pairs and low similarity with negative pairs, using a cosine distance metric and temperature scaling.
* Optimization was performed using Adam with a predefined learning rate.

```python
solver = cebra.solver.init(
    name="multi-session",
    model=model,
    criterion=FixedCosineInfoNCE(temperature=1.0),
    optimizer=torch.optim.Adam(model.parameters(), lr=learning_rate),
    tqdm_on=True
)
```

### Training Execution and Output

* The training loop was run via `solver.fit()`, managing data batching, forward passes, loss computation, and backpropagation.
* Upon completion, the method returned:

  * The **trained embedding model** representing shared latent neural dynamics.
  * The **solver instance** for tracking training progress and enabling further analysis or fine-tuning.



### Summary

This multi-session shared embedding approach enables cross-subject comparison of EEG brain states within a unified latent space, paving the way for analyses of inter-subject variability, group-level clustering, and transfer learning across recording sessions.


