# Introduction to CEBRA

## What is CEBRA?

**CEBRA** (Consistent EmBeddings of high-dimensional Recordings using Auxiliary variables) is a machine learning framework designed to discover meaningful low-dimensional representations of high-dimensional neural data. Unlike traditional dimensionality reduction techniques, CEBRA leverages auxiliary variables—such as behavioral states, task conditions, or temporal information—to guide the learning process toward biologically meaningful embeddings.

## Core Principles

### 1. Contrastive Learning Framework

CEBRA operates on the principle of **contrastive learning**, where the algorithm learns to:

- **Bring together** neural activity patterns that share similar auxiliary variables (positive pairs)
- **Separate** neural activity patterns with different auxiliary variables (negative pairs)
- **Preserve** the underlying structure of the data while reducing dimensionality

This approach ensures that the learned representations capture the aspects of neural activity that are most relevant to the auxiliary variables of interest.

### 2. Consistency Across Sessions

The "Consistent" in CEBRA refers to the framework's ability to learn representations that are stable and comparable across different recording sessions, subjects, or experimental conditions. This is achieved through:

- **Shared embedding spaces**: Multiple datasets can be mapped to the same low-dimensional space
- **Aligned representations**: Similar neural states across sessions occupy similar positions in the embedding space
- **Robust feature extraction**: The learned features generalize across different data sources

### 3. Auxiliary Variable Integration

CEBRA's key innovation lies in its use of **auxiliary variables** to guide the learning process:

- **Behavioral variables**: Movement trajectories, task performance, decision states
- **Temporal variables**: Time stamps, sequence information, event markers
- **Experimental variables**: Task conditions, stimuli, environmental factors

## Mathematical Foundation

### Embedding Objective

CEBRA learns a mapping function $f: \mathbb{R}^D \rightarrow \mathbb{R}^d$ where $D$ is the high-dimensional neural data space and $d$ is the low-dimensional embedding space ($d << D$).

The objective function combines:

$$\mathcal{L} = \mathcal{L}_{contrastive} + \lambda \mathcal{L}_{consistency}$$

Where:
- $\mathcal{L}_{contrastive}$ ensures that similar auxiliary variables lead to similar embeddings
- $\mathcal{L}_{consistency}$ maintains stability across different sessions or conditions
- $\lambda$ balances the two objectives

### Contrastive Loss

The contrastive loss for a triplet $(x_i, x_i^+, x_i^-)$ is:

$$\mathcal{L}_{contrastive} = -\log \frac{\exp(f(x_i) \cdot f(x_i^+) / \tau)}{\exp(f(x_i) \cdot f(x_i^+) / \tau) + \sum_{j} \exp(f(x_i) \cdot f(x_j^-) / \tau)}$$

Where:
- $x_i$ is the reference neural activity
- $x_i^+$ is neural activity with similar auxiliary variables (positive)
- $x_j^-$ are neural activities with different auxiliary variables (negative)
- $\tau$ is the temperature parameter
- $\cdot$ denotes the dot product

## Why CEBRA for EEG Analysis?

### 1. Temporal Structure Preservation

EEG data has rich temporal dynamics that traditional methods often struggle to capture. CEBRA's ability to use temporal auxiliary variables makes it particularly suitable for:

- **State transition analysis**: Understanding how brain states evolve over time
- **Sequence learning**: Capturing temporal dependencies in neural activity
- **Event-related dynamics**: Linking neural responses to specific task events

### 2. Multi-Subject Generalization

In your multi-session EEG analysis, CEBRA's consistency principle enables:

- **Cross-subject comparisons**: Identifying shared neural patterns across individuals
- **Population-level insights**: Understanding common brain dynamics
- **Individual difference analysis**: Quantifying how subjects differ in their neural responses

### 3. High-Dimensional Data Handling

EEG recordings typically involve:
- **Many channels**: 64, 128, or more electrodes
- **High sampling rates**: Thousands of samples per second
- **Complex spatial patterns**: Interactions between different brain regions

CEBRA efficiently handles this complexity by learning compact representations that preserve the most important information for your research questions.

## Advantages Over Traditional Methods

### Compared to PCA/ICA:
- **Guided dimensionality reduction**: Uses auxiliary variables instead of just variance
- **Nonlinear mappings**: Can capture complex, nonlinear relationships
- **Cross-session consistency**: Maintains alignment across different recordings

### Compared to t-SNE/UMAP:
- **Auxiliary variable integration**: Uses task-relevant information for embedding
- **Consistency across datasets**: Enables meaningful comparisons between sessions
- **Interpretable dimensions**: Embedding dimensions relate to auxiliary variables

### Compared to Autoencoders:
- **Contrastive learning**: Focuses on behaviorally relevant features
- **Multi-session support**: Built-in mechanisms for handling multiple datasets
- **Theoretical grounding**: Principled approach to representation learning

## Applications in Neuroscience

CEBRA has been successfully applied to:

- **Motor control studies**: Understanding movement-related brain activity
- **Cognitive neuroscience**: Mapping decision-making and attention processes
- **Clinical applications**: Analyzing pathological brain states
- **Cross-species comparisons**: Identifying conserved neural mechanisms

## Biological Interpretability

The embeddings learned by CEBRA often correspond to:

- **Neural population states**: Coherent patterns of activity across neurons
- **Behavioral modes**: Different phases of task execution
- **Cognitive states**: Attention, memory, decision-making processes
- **Temporal dynamics**: State transitions and sequence information

This interpretability makes CEBRA particularly valuable for neuroscientific research, where understanding the biological meaning of the results is crucial.

## Limitations and Considerations

### Data Requirements:
- **Auxiliary variables**: Requires well-defined behavioral or temporal variables
- **Sample size**: Needs sufficient data for robust contrastive learning
- **Data quality**: Sensitive to noise in both neural and auxiliary data

### Computational Considerations:
- **Training time**: Can be computationally intensive for large datasets
- **Hyperparameter sensitivity**: Requires careful tuning of parameters
- **Memory requirements**: High-dimensional data can be memory-intensive

### Interpretation Challenges:
- **Embedding dimensions**: Understanding what each dimension represents
- **Nonlinear relationships**: Complex mappings can be difficult to interpret
- **Auxiliary variable choice**: Results depend heavily on the quality of auxiliary variables

## Future Directions

Current research in CEBRA focuses on:

- **Improved consistency**: Better methods for aligning representations across sessions
- **Temporal modeling**: Enhanced handling of complex temporal dynamics
- **Interpretability**: Tools for understanding embedding dimensions
- **Scalability**: Methods for handling larger datasets and more complex auxiliary variables

This theoretical foundation provides the basis for understanding how CEBRA can reveal shared neural dynamics across your multi-session EEG recordings, enabling insights into the common patterns of brain activity that underlie cognitive and behavioral processes.