import numpy as np
import cebra
from cebra import CEBRA
from cebra.integrations.plotly import plot_embedding_interactive
import plotly.io as pio
import matplotlib.pyplot as plt


# Synthetic data generators

def generate_cluster_dataset(fs=500, T=120,  block_sec=10):
    #Goal CEBRA needs to rely on signal features (frequency, phase differences) to cluster.
    N = fs * T
    t = np.linspace(0, T, N)
    # 5 channels
    channels = np.zeros((N, 5))
    #low-frequency sine waves  and high-frequency sine wave
    cluster1 = np.array([np.sin(2 * np.pi * 0.1 * t + ch) for ch in range(5)]).T  # shape (N, 5)
    cluster2 = np.array([np.sin(2 * np.pi * 0.3 * t + ch * 2) for ch in range(5)]).T  # shape (N, 5)

    #Intermixing of timepoints and cluster:
    block_size = int(fs * block_sec)
    num_blocks = N // block_size
    # random cluster assignment for each time point
    block_labels = np.random.choice([0, 1], size=num_blocks , p=[0.5, 0.5])
    # Mix signals with 1 s blcoks belinging to each cluster
    for i in range(num_blocks):
        start = i * block_size
        end = start + block_size
        if block_labels[i] == 0:
            channels[start:end, :] = cluster1[start:end, :]
        else:
            channels[start:end, :] = cluster2[start:end, :]
    
    # Add noise
    channels += 0.1 * np.random.randn(*channels.shape)
    
    return channels


def generate_loop_dataset(fs=500, T=120):
    N = fs * T
    t = np.linspace(0, T, N)
    # 5 channels
    channels = np.zeros((N, 5))
    # same frequency for all 20s period
    base_freq = 0.05
    # phase shift
    for ch in range(5):
        phase_shift = (2 * np.pi / 5) * ch
        channels[:, ch] = np.sin(2 * np.pi * base_freq * t + phase_shift)
     # Add noise
    channels += 0.05 * np.random.randn(*channels.shape)
    return channels

def plot_data(X, title="Data Example"):
    plt.figure(figsize=(12, 6))
    for ch in range(X.shape[1]):
        plt.plot(X[:, ch] + ch*2, label=f'Channel {ch}')  # offset vertically for clarity
    plt.title(title)
    plt.xlabel("Samples (time)")
    plt.ylabel("Amplitude + offset")
    plt.legend()
    plt.show()

# Define CEBRA model
cebra_model = CEBRA(
    model_architecture="offset10-model",
    batch_size=512,
    learning_rate=3e-4,
    max_iterations=2000,
    output_dimension=3,
    distance='cosine',
    device="cuda_if_available",
    verbose=True,
    time_offsets=10,
)

def quick_run_cebra(X, title="CEBRA Embedding"):
    model = cebra_model.fit(X)
    embedding = model.transform(X)

    times = np.arange(X.shape[0]) / 500.0
    fig = plot_embedding_interactive(embedding, embedding_labels=times, title=title, markersize=3, cmap="rainbow")
    return fig
# Choose dataset to run on:
X1 = generate_cluster_dataset()  # or generate_loop_dataset()
plot_data(X1, title="Cluster Dataset Example")
X2 = generate_loop_dataset()+
plot_data(X2, title="Loop Dataset Example")


fig = quick_run_cebra(X1, title="Quick CEBRA run")
fig.write_html("artifical_cluster.html", auto_open=True)

fig = quick_run_cebra(X2, title="Quick CEBRA run")
fig.write_html("artifical_trajectory.html", auto_open=True)