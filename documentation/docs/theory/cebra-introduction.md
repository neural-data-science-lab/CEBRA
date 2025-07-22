# Introduction to CEBRA

## What is CEBRA?

**CEBRA** (Consistent EmBeddings of high-dimensional Recordings using Auxiliary variables) is a machine learning framework designed to discover meaningful low-dimensional representations of high-dimensional neural data. Unlike traditional dimensionality reduction techniques, CEBRA leverages auxiliary variables—such as behavioral states, task conditions, or temporal information—to guide the learning process toward biologically meaningful embeddings.

## Core Principles
The contrastive objective maximizes mutual information between neural embeddings and auxiliary variables.The InfoNCE loss provides a lower bound on this mutual information, making CEBRA a principled approach to information-maximizing representation learning.
The learned embedding space exhibits specific geometric properties:

1. **Manifold alignment**: Neural manifolds across sessions become aligned in the embedding space
2. **Clustering properties**: Neural states with similar auxiliary variables form coherent clusters
3. **Smoothness**: Continuous auxiliary variables induce smooth trajectories in embedding space

### Theoretical foundation

Given high-dimensional neural recordings $X \in \mathbb{R}^{N \times D}$ where $N$ is the number of time points and $D$ is the dimensionality (e.g., number of channels), and auxiliary variables $Y \in \mathbb{R}^{N \times A}$ encoding behavioral, temporal, or experimental information. 

CEBRA seeks to learn a mapping function:
$f_\theta: \mathbb{R}^D \rightarrow \mathbb{R}^d$
where $d \ll D$ and $\theta$ represents learnable parameters. The objective is to construct embeddings $Z = f_\theta(X)$ that preserve relationships defined by the auxiliary variables while maintaining consistency across different recording sessions or experimental conditions.

### 1. Auxiliary Variable-Guided Representation Learning

Traditional unsupervised dimensionality reduction techniques (PCA, t-SNE, UMAP) optimize for variance preservation or local neighborhood structure without consideration of task-relevant information. CEBRA fundamentally differs by incorporating auxiliary variables $y_i$ that encode biologically or behaviorally meaningful information about each neural observation $x_i$.

The auxiliary variables can take various forms:

- Continuous behavioral variables: Position trajectories, velocity, acceleration
- Discrete state variables: Task phases, decision outcomes, stimulus categories
- Temporal variables: Time indices, sequence positions, event markers
- Multi-modal auxiliary data: Simultaneously recorded behavioral measurements 

### 1. Contrastive Learning Framework

CEBRA operates on the principle of **contrastive learning**, where the algorithm learns to:

- **Positive Pairs**: Bring together neural activity patterns that share similar auxiliary variables 
- **Negative Pairs**: Seperate neural activity patterns with different auxiliary variables 
- **Preserve** the underlying structure of the data while reducing dimensionality

This ensures that the learned representations capture the aspects of neural activity that are most relevant to the auxiliary variables of interest. (EXPLAIN FURTHER)

### 2. Consistency Across Sessions (EXPLAIN BETTER)

The "Consistent" in CEBRA refers to the framework's ability to learn representations that are stable and comparable across different recording sessions, subjects, or experimental conditions. This is achieved through:

- **Shared embedding spaces**: Multiple datasets can be mapped to the same low-dimensional space
- **Aligned representations**: Similar neural states across sessions occupy similar positions in the embedding space
- **Robust feature extraction**: The learned features generalize across different data sources

## Mathematical Foundation

### Embedding Objective

CEBRA learns a mapping function $f: \mathbb{R}^D \rightarrow \mathbb{R}^d$ where $D$ is the high-dimensional neural data space and $d$ is the low-dimensional embedding space ($d << D$).

The objective function combines:

$$\mathcal{L} = \mathcal{L}_{contrastive} + \lambda_{cons} \mathcal{L}_{consistency} + \lambda_{reg} \mathcal{L}_{regularization} $$


Where:
- $\mathcal{L}_{contrastive}$ ensures that similar auxiliary variables lead to similar embeddings
- $\mathcal{L}_{consistency}$ maintains stability across different sessions or conditions
- $\mathcal{L}_{regularization}$ INSERT WHAT
and  $\lambda_{cons}$ and $\lambda_{reg}$ are weighting hyperparameters.

### Contrastive Loss Formulation

The contrastive loss is based on the InfoNCE (Noise Contrastive Estimation) framework. For a reference sample $x_i$ with auxiliary variable $y_i$, the loss is:

$$\mathcal{L}_{contrastive} = -\mathbb{E}_{i} \left[ \log \frac{\exp(s(z_i, z_i^+) / \tau)}{\exp(s(z_i, z_i^+) / \tau) + \sum_{k=1}^{K} \exp(s(z_i, z_k^-) / \tau)} \right]$$

where:
- $z_i = f_\theta(x_i)$ is the embedding of the reference sample
- $z_i^+ = f_\theta(x_j)$ where $x_j$ is a positive sample (similar auxiliary variable)
- $z_k^-$ are embeddings of $K$ negative samples (dissimilar auxiliary variables)
- $s(\cdot, \cdot)$ is a similarity function (typically cosine similarity or dot product)
- $\tau > 0$ is the temperature parameter controlling the concentration of the distribution. 

### Positive and Negative Sampling Strategies

The effectiveness of contrastive learning critically depends on the sampling strategy:

- **Positive Sampling**: 

For continuous auxiliary variables, positive samples are selected within a neighborhood:
$$\mathcal{N}_\epsilon(y_i) = \{y_j : \|y_i - y_j\| \leq \epsilon \}$$

For discrete variables, positive samples share the same categorical label.

- **Negative Sampling**: 
Negative samples are selected to ensure diversity
    - **Hard negative mining**: Selecting challenging negative samples that are close in neural space but distant in auxiliary space
    - **Balanced sampling**: Ensuring representation across different auxiliary variable values
    - **Temporal separation**: For time-series data, maintaining sufficient temporal distance

### Consistency Loss

The consistency term ensures alignment across different recording sessions or subjects. Let $\{X^{(s)}, Y^{(s)}\}_{s=1}^S$ represent data from $S$ different sessions. The consistency loss can be formulated as:

$$\mathcal{L}_{consistency} = \sum_{s,s'=1}^S \mathbb{E}_{i,j} \left[ \|f_\theta(x_i^{(s)}) - f_\theta(x_j^{(s')})\|^2 \cdot \mathbf{1}[d(y_i^{(s)}, y_j^{(s')}) < \delta] \right]$$

where $\mathbf{1}[\cdot]$ is the indicator function and $\delta$ defines the threshold for auxiliary variable similarity across sessions.

### Neural Network Architecture

CEBRA typically employs deep neural networks as the encoder stability

## Limitations and Theoretical Challenges

### Auxiliary Variable Dependence

CEBRA's performance critically depends on:
- **Quality of auxiliary variables**: Noisy or irrelevant auxiliary data degrades performance
- **Completeness**: Missing auxiliary variables can lead to suboptimal embeddings
- **Temporal alignment**: Misalignment between neural and auxiliary data affects learning

### Data Requirements:
- **Auxiliary variables**: Requires well-defined behavioral or temporal variables
- **Sample size**: Needs sufficient data for robust contrastive learning
- **Data quality**: Sensitive to noise in both neural and auxiliary data

### Computational Considerations:
- **Training time**: Can be computationally intensive for large datasets
- **Memory requirements**: High-dimensional data can be memory-intensive

### Interpretation Challenges:
- **Embedding dimensions**: Understanding what each dimension represents
- **Nonlinear relationships**: Complex mappings can be difficult to interpret
- **Auxiliary variable choice**: Results depend heavily on the quality of auxiliary variables


## Conclusion

CEBRA represents a theoretically grounded approach to neural representation learning that addresses fundamental challenges in neuroscience data analysis. By combining contrastive learning with auxiliary variable integration and consistency constraints, CEBRA provides a principled framework for discovering meaningful low-dimensional representations of complex neural dynamics. The mathematical foundation ensures both theoretical rigor and practical effectiveness, making CEBRA a powerful tool for understanding neural computation and behavior.