import sys
from pathlib import Path
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from eeg.colors import valence_arousal_emotion_color
from data import eeg_dataloader
from cebra.integrations.plotly import plot_embedding_interactive  

archive_root = Path(__file__).resolve().parent.parent  # parent of helper
sys.path.insert(0, str(archive_root))

output_root = Path(r"C:\Users\bayer\MPI\embeddings\archive\results")
eeg_data_root = Path(r"E:\Cris_Work\preproc")

n_max = 50000

for npy_file in output_root.glob("sub-*/*.npy"):
    subject_key = npy_file.parent.name
    subject_folder = eeg_data_root / subject_key

    try:
        beh_df = eeg_dataloader.load_behavioral_labels(subject_folder)
    except Exception as e:
        print(f"Skipping {npy_file} – cannot load labels: {e}")
        continue

    t_behavior = beh_df["timestamp"].values
    valence = beh_df["valence"].values
    arousal = beh_df["arousal"].values

    embedding = np.load(npy_file)
    embedding_len = embedding.shape[0]

    # Interpolation
    embedding_timestamps = np.linspace(t_behavior.min(), t_behavior.max(), embedding_len)
    valence_interp = interp1d(t_behavior, valence, bounds_error=False, fill_value="extrapolate")(embedding_timestamps)
    arousal_interp = interp1d(t_behavior, arousal, bounds_error=False, fill_value="extrapolate")(embedding_timestamps)

    # Downsample
    step = max(1, embedding_len // n_max)
    idx = np.arange(0, embedding_len, step)

    # Colors
    colors_valence = valence_arousal_emotion_color(valence_interp[idx], arousal_interp[idx], mode="valence")
    colors_arousal = valence_arousal_emotion_color(valence_interp[idx], arousal_interp[idx], mode="arousal")
    colors_joint   = valence_arousal_emotion_color(valence_interp[idx], arousal_interp[idx], mode="combined")

    cmap = mcolors.LinearSegmentedColormap.from_list("rainbow", plt.cm.rainbow(np.linspace(0, 1, 256)))
    norm = plt.Normalize(vmin=embedding_timestamps.min(), vmax=embedding_timestamps.max())
    colors_time = [mcolors.to_hex(cmap(norm(t))) for t in embedding_timestamps[idx]]

    # Convert to hex
    colors_valence_hex = np.array([mcolors.to_hex(c) for c in colors_valence])
    colors_arousal_hex = np.array([mcolors.to_hex(c) for c in colors_arousal])
    colors_joint_hex   = np.array([mcolors.to_hex(c) for c in colors_joint])
    colors_time        = np.array([mcolors.to_hex(cmap(norm(t))) for t in embedding_timestamps[idx]])

    # --- Create plots ---
    fig_valence = plot_embedding_interactive(
        embedding[idx],
        embedding_labels=colors_valence_hex,
        title=f"{subject_key} - {npy_file.stem} - Valence"
    )
    

    fig_arousal = plot_embedding_interactive(
        embedding[idx],
        embedding_labels=colors_arousal_hex,
        title=f"{subject_key} - {npy_file.stem} - Arousal"
    )

    fig_time = plot_embedding_interactive(
        embedding[idx],
        embedding_labels=colors_time,
        title=f"{subject_key} - {npy_file.stem} - Time"
    )

    fig_behaviour = plot_embedding_interactive(
        embedding[idx],
        embedding_labels=colors_joint_hex,
        title=f"{subject_key} - {npy_file.stem} - Behaviour"
    )

    # --- Save as HTML ---
    html_valence_path   = npy_file.with_name(npy_file.stem + "_valence.html")
    html_arousal_path   = npy_file.with_name(npy_file.stem + "_arousal.html")
    html_time_path      = npy_file.with_name(npy_file.stem + "_time.html")
    html_behaviour_path = npy_file.with_name(npy_file.stem + "_behaviour.html")

    fig_valence.write_html(html_valence_path)
    fig_arousal.write_html(html_arousal_path)
    fig_time.write_html(html_time_path)
    fig_behaviour.write_html(html_behaviour_path)

    print(f"Saved {html_valence_path}, {html_arousal_path}, {html_time_path}, {html_behaviour_path}")
