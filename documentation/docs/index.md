# CEBRA neural latent embeddings

`[Last update: September 16, 2024]`

    Period:     2025-06 - 2025-08
    Status:     in progress

    Author(s):  Cristina Maria Bayer
    Contact:    bayer-cristina@t-online.de


## Links

### GitHub Repository

The GitHub Repository with all the code can be assessed here: [github.com]()

### Documentation
The Documentation can be assessed here: [mkdocs]()


## Project description

This project focuses on applying CEBRA (Contrastive Embedding for Behavior and Representation Analysis) to EEG data collected during a 20-minute immersive VR experience from the AVR Project. The dataset includes neural responses to a sequence of different videos shown in VR. The primary goal is to implement and explore the application of CEBRA for generating neural embeddings from EEG signals to better understand patterns and representations associated with different video stimuli. Preprocessed EEG data was provided by the AVR Project, allowing for direct experimentation with the CEBRA framework.

### Project-Specific Challenges

**Complex Stimulus Structure**:

- **Habituation Effects**: Neural responses to 'b' may change across repetitions
- **Context Effects**: Same 'b' video may evoke different responses depending on preceding content
- **Memory Interference**: Later 'b' presentations may activate memory traces from earlier ones
- **Expectation Effects**: Participants may learn the pattern and anticipate repeated videos

**Continuous Labeling Interference**:

- **Cognitive Load**: Dual-task demands affect natural emotional responses
- **Motor Artifacts**: Hand movements for labeling create systematic EEG noise
- **Attention Division**: Split attention between content and introspective monitoring
- **Response Bias**: The labeling process may alter the emotional experience itself

**Temporal Alignment Complexity**:

- Multiple overlapping temporal structures (video timing, emotional dynamics, labeling timing, neural timing)
- Different frequency bands have different temporal characteristics
- Emotional responses may lag or persist beyond stimulus boundaries
- Individual differences in emotional response timing

### Experimental Design Strengths

**Built-in Validation Structure**:

- Repeated 'b' videos provide natural ground truth for embedding consistency
- Enable assessment of habituation vs. consistency in neural responses
- Allow detection of context effects (same stimulus, different preceding context)
- Serve as quality control for artifact removal effectiveness

**Rich Temporal Dynamics**:

- Continuous emotional experience rather than discrete trials
- Natural emotional flow and transition patterns
- Captures emotional persistence and carryover effects
- Realistic emotional dynamics with overlapping responses

**Multi-dimensional Emotional Space**:

- Different videos likely span various emotional dimensions
- Enables exploration of emotional transition dynamics
- Captures individual differences in emotional response patterns
- Provides diverse content for manifold discovery

