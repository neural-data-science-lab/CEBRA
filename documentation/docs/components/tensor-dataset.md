# TensorDataset Design

## Conceptual Role

The **TensorDataset** serves as the fundamental building block for organizing and accessing individual EEG recording sessions in your multi-session analysis. It acts as a bridge between raw neural data and the sophisticated machine learning framework of CEBRA, providing a structured way to handle the temporal and spatial complexity of EEG recordings.

## Core Design Philosophy

### 1. Unified Data Representation

TensorDataset creates a **unified interface** for neural data by:

- **Standardizing formats**: Converting various data formats (NumPy arrays, PyTorch tensors) into a consistent representation
- **Handling heterogeneity**: Accommodating different data types (continuous behavioral variables, discrete task labels)
- **Maintaining relationships**: Preserving the temporal alignment between neural activity and auxiliary variables

### 2. Auxiliary Variable Integration

The dataset design explicitly supports **auxiliary variables** that guide the learning process:

```python
# Conceptual structure
TensorDataset(
    neural=eeg_data,        # Shape: (time_points, channels)
    continuous=behavior,    # Shape: (time_points, behavioral_dims)
    discrete=task_labels   # Shape: (time_points,)
)
```

This tri-partite structure ensures that:
- **Neural data** provides the high-dimensional observations
- **Continuous variables** capture smooth behavioral changes (e.g., attention levels, arousal)
- **Discrete variables** encode categorical information (e.g., task conditions, stimulus types)

## Data Structure and Organization

### Temporal Alignment

The dataset maintains **strict temporal alignment** across all data modalities:

$$\mathbf{X}(t) = \begin{bmatrix} \mathbf{x}_{neural}(t) \\ \mathbf{x}_{continuous}(t) \\ \mathbf{x}_{discrete}(t) \end{bmatrix}$$

Where:
- $\mathbf{x}_{neural}(t) \in \mathbb{R}^{N_{channels}}$ is the EEG activity at time $t$
- $\mathbf{x}_{continuous}(t) \in \mathbb{R}^{N_{behavioral}}$ are continuous behavioral variables
- $\mathbf{x}_{discrete}(t) \in \mathbb{Z}^{N_{categorical}}$ are discrete task variables

### Offset Mechanism

The **offset** property enables temporal context extraction:

```python
offset = Offset(left=5, right=5)  # 10 time points total
```

This creates **temporal windows** around each reference time point:

$$\mathbf{X}_{window}(t) = [\mathbf{X}(t-5), \mathbf{X}(t-4), \ldots, \mathbf{X}(t+4), \mathbf{X}(t+5)]$$

The offset mechanism serves multiple purposes:

1. **Temporal context**: Captures neural dynamics leading up to and following events
2. **Noise reduction**: Smooths out transient fluctuations
3. **Feature enrichment**: Provides more information for pattern recognition
4. **Causal modeling**: Enables analysis of cause-effect relationships

## Functional Capabilities

### 1. Dynamic Indexing

The `expand_index` method transforms simple time indices into **rich temporal windows**:

```python
# Simple index
index = [100, 150, 200]

# Expanded with offset
expanded = dataset.expand_index(index)
# Shape: (3, 11) for offset=5 left and right
```

This expansion enables:
- **Multi-scale analysis**: Examining neural activity at different temporal scales
- **Context preservation**: Maintaining information about surrounding time points
- **Flexible sampling**: Adapting to different analysis requirements

### 2. Data Type Validation

The dataset enforces **data type consistency**:

- **Neural data**: Must be floating-point (typically float32 or float64)
- **Continuous variables**: Must be floating-point for smooth interpolation
- **Discrete variables**: Must be integer types for categorical encoding

This validation prevents common errors and ensures compatibility with downstream analysis tools.

### 3. Device Management

The dataset handles **device placement** for efficient computation:

```python
dataset = TensorDataset(neural_data, device="cuda")
```

This enables:
- **GPU acceleration**: Faster computation for large datasets
- **Memory management**: Efficient use of device memory
- **Seamless integration**: Compatibility with PyTorch-based models

## Data Access Patterns

### Temporal Window Retrieval

When accessing data, the dataset returns **temporal windows** rather than single time points:

```python
# Access single time point
data = dataset[t]
# Returns shape: (channels, temporal_window)
```

This access pattern:
- **Preserves temporal structure**: Maintains the sequential nature of neural data
- **Enables convolution**: Compatible with convolutional neural networks
- **Supports temporal modeling**: Facilitates analysis of temporal dependencies

### Batch Processing

The dataset supports **efficient batch processing**:

```python
# Batch access
batch_data = dataset[batch_indices]
# Returns shape: (batch_size, channels, temporal_window)
```

Batch processing enables:
- **Parallel computation**: Simultaneous processing of multiple samples
- **Memory efficiency**: Optimal use of computational resources
- **Scalability**: Handling large datasets efficiently

## Integration with CEBRA Framework

### Auxiliary Variable Utilization

The dataset's auxiliary variables directly inform CEBRA's contrastive learning:

1. **Positive pairs**: Samples with similar auxiliary variables
2. **Negative pairs**: Samples with different auxiliary variables
3. **Reference samples**: Anchor points for comparison

### Temporal Consistency

The offset mechanism ensures **temporal consistency** in the learned representations:

- **Smooth transitions**: Nearby time points have similar representations
- **Temporal invariance**: Representations are stable across small time shifts
- **Causal relationships**: Past context informs current representations

## Advantages for EEG Analysis

### 1. Temporal Richness

EEG data has complex temporal structure that the TensorDataset captures:

- **Fast dynamics**: Capturing millisecond-scale neural events
- **Slow trends**: Preserving longer-term changes in brain state
- **Rhythmic patterns**: Maintaining oscillatory components

### 2. Multimodal Integration

The dataset naturally handles **multiple data streams**:

- **Neural activity**: High-dimensional EEG recordings
- **Behavioral measures**: Continuous performance metrics
- **Task structure**: Discrete experimental conditions

### 3. Preprocessing Integration

The dataset can incorporate **preprocessing steps**:

- **Artifact removal**: Filtering out noise and artifacts
- **Normalization**: Standardizing signal amplitudes
- **Feature extraction**: Computing derived measures

## Memory and Computational Considerations

### Memory Optimization

The dataset implements several **memory optimization strategies**:

1. **Lazy loading**: Loading data only when accessed
2. **Efficient storage**: Using appropriate data types
3. **Device management**: Optimal placement of data in memory

### Computational Efficiency

The design prioritizes **computational efficiency**:

1. **Vectorized operations**: Using efficient tensor operations
2. **Batch processing**: Minimizing loop overhead
3. **GPU compatibility**: Leveraging parallel processing

## Limitations and Design Trade-offs

### Memory Requirements

The temporal window approach requires more memory:

$$\text{Memory} = N_{samples} \times N_{channels} \times N_{timepoints} \times \text{dtype\_size}$$

Where $N_{timepoints}$ is determined by the offset size.

### Temporal Resolution

The offset mechanism imposes constraints on temporal resolution:

- **Fixed windows**: All samples use the same temporal context
- **Boundary effects**: Samples near recording boundaries may be excluded
- **Resolution trade-offs**: Larger offsets capture more context but reduce temporal precision

## Future Enhancements

### Adaptive Offsets

Future versions might support **adaptive offset sizes**:

- **Task-dependent**: Different offsets for different task conditions
- **Data-driven**: Automatically determining optimal offset sizes
- **Hierarchical**: Multi-scale temporal analysis

### Advanced Preprocessing

Enhanced preprocessing capabilities could include:

- **Real-time processing**: Online artifact removal and normalization
- **Adaptive filtering**: Context-dependent signal processing
- **Quality assessment**: Automatic data quality evaluation

The TensorDataset design provides a robust foundation for handling the complexity of EEG data while maintaining the flexibility needed for sophisticated neural state analysis. Its integration with auxiliary variables and temporal windowing makes it particularly well-suited for CEBRA's contrastive learning approach, enabling the discovery of meaningful patterns in neural activity across different subjects and experimental conditions.
