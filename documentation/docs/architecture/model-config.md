# Model Training

### Overview

The core objective of this step is to train a shared neural embedding model using CEBRA’s multi-session framework. The model learns to represent neural activity from multiple subjects/sessions in a shared low-dimensional embedding space, enabling cross-subject generalization.

### Model Creation

#### Input Dimension Estimation

The model input dimension is determined from the first session in the dataset collection, ensuring the model architecture matches the data dimensionality. This input dimension corresponds to the number of neural features (e.g., channels or neurons) in each time step.

#### Architecture

A shared multi-layer neural network model is instantiated using CEBRA’s model initialization utility. The model contains:

* An input layer matching the input dimension (number of neurons/channels)
* A hidden layer with 256 units
* An output embedding layer with a user-specified dimension (`output_dim`), representing the learned low-dimensional neural embedding space

```python
model = cebra.models.init(
    name="offset10-model",
    num_neurons=input_dim,
    num_units=256,
    num_output=output_dim
).to(device)
```

### Dataset Configuration

To ensure compatibility between the dataset and model, the dataset collection is configured to align with the model’s input expectations. This step prepares the dataset for effective training by sampling appropriate positive and negative examples across sessions.

### Dataloader Setup

The training process begins by creating a multi-session dataloader, MultiSessionLoader, which orchestrates sampling of contrastive data pairs (anchor, positive, and negative examples) uniformly across all sessions contained in the DatasetCollection. This dataloader facilitates contrastive learning by:

* **Anchor samples:** reference points from neural activity
* **Positive samples:** temporally nearby activity from the same subject/session
* **Negative samples:** activity from other subjects/sessions to provide contrast

The loader continuously samples batches with these triplets during training, ensuring uniform representation of subjects to avoid bias toward any particular session.

The sampled indices are returned as a BatchIndex containing:
- reference, positive, and negative indices for contrastive training.
- index and index_reversed tensors used to align and mix positive pairs correctly during inference.

This setup allows efficient construction of contrastive batches across sessions, supporting temporal and subject variability. 

### Loss Function and Optimization

At the core of training is the MultiSessionSolver, a specialized training engine registered under the "multi-session" variant in CEBRA. This solver integrates model inference, loss computation, and parameter updates across all sessions in a unified framework.  
Core Responsibilities of the Solver
- Parameter Management: The solver iterates over all model and criterion parameters for gradient updates. Since a shared model is used across sessions, parameters correspond to this single unified network.

- Batch Processing & Inference: For each batch of sampled triplets (reference, positive, negative), the solver computes feature embeddings by forwarding the samples through the shared model. It applies a mixing operation (_mix) to reorder positive samples according to indices, aligning reference-positive pairs correctly.

The training uses the **FixedCosineInfoNCE** loss function, a cosine similarity-based contrastive loss tailored for embedding learning. This loss encourages the model to maximize similarity between positive pairs while minimizing similarity with negative pairs, controlled by a temperature hyperparameter that sharpens or smooths the distribution of similarities.

The optimizer of choice is Adam, configured with the specified learning rate, to update model weights based on gradient descent.

```python
solver = cebra.solver.init(
    name="multi-session",
    model=model,
    criterion=FixedCosineInfoNCE(temperature=1.0),
    optimizer=torch.optim.Adam(model.parameters(), lr=learning_rate),
    tqdm_on=True
)
```

### Training Execution

The training loop is executed by calling the solver’s `.fit()` method, which internally manages batch retrieval, forward passes through the model, loss computation, and weight updates. 

### Output

Upon successful training, the function returns a dictionary containing:

* **Trained model:** The learned embedding model instance
* **Solver instance:** Contains training state and parameters for further analysis or fine-tuning


