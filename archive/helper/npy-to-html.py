from pathlib import Path
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from data import eeg_dataloader
from eeg.colors import valence_arousal_emotion_color
from cebra.integrations.plotly import plot_embedding_interactive

output_root = Path(r"C:\Users\bayer\MPI\embeddings\archive\results")
eeg_data_root = Path(r"E:\Cris_Work\preproc")
n_max = 50000

subject_embeddings = {}
for npy_file in output_root.glob("sub-*/*.npy"):
    subject_key = npy_file.parent.name
    subject_embeddings.setdefault(subject_key, []).append(npy_file)

for subject_key, npy_files in subject_embeddings.items():
    subject_folder = eeg_data_root / subject_key

    # Load behavioral data once per subject
    try:
        beh_df = eeg_dataloader.load_behavioral_labels(subject_folder)
    except Exception as e:
        print(f"Skipping {subject_key} – cannot load labels: {e}")
        continue

    t_behavior = beh_df["timestamp"].values
    valence = beh_df["valence"].values
    arousal = beh_df["arousal"].values

    # Cache for interpolated colors per embedding length
    color_cache = {}

    for npy_file in npy_files:
        embedding = np.load(npy_file)
        embedding_len = embedding.shape[0]

        # Check if all HTMLs already exist
        html_paths = [npy_file.with_name(npy_file.stem + suffix) for suffix in ("_valence.html", "_arousal.html", "_time.html", "_behaviour.html")]
        if all(p.exists() for p in html_paths):
            print(f"[INFO] Skipping {npy_file.stem} – all HTMLs exist")
            continue

        # Only compute interpolation/colors if not already cached for this length
        if embedding_len not in color_cache:
            embedding_timestamps = np.linspace(t_behavior.min(), t_behavior.max(), embedding_len)
            valence_interp = interp1d(t_behavior, valence, bounds_error=False, fill_value="extrapolate")(embedding_timestamps)
            arousal_interp = interp1d(t_behavior, arousal, bounds_error=False, fill_value="extrapolate")(embedding_timestamps)

            step = max(1, embedding_len // n_max)
            idx = np.arange(0, embedding_len, step)

            colors_valence_hex = np.array([mcolors.to_hex(c) for c in valence_arousal_emotion_color(valence_interp[idx], arousal_interp[idx], mode="valence")])
            colors_arousal_hex = np.array([mcolors.to_hex(c) for c in valence_arousal_emotion_color(valence_interp[idx], arousal_interp[idx], mode="arousal")])
            colors_joint_hex   = np.array([mcolors.to_hex(c) for c in valence_arousal_emotion_color(valence_interp[idx], arousal_interp[idx], mode="combined")])
            cmap = mcolors.LinearSegmentedColormap.from_list("rainbow", plt.cm.rainbow(np.linspace(0, 1, 256)))
            norm = plt.Normalize(vmin=embedding_timestamps.min(), vmax=embedding_timestamps.max())
            colors_time = np.array([mcolors.to_hex(cmap(norm(t))) for t in embedding_timestamps[idx]])

            # Store in cache
            color_cache[embedding_len] = (idx, colors_valence_hex, colors_arousal_hex, colors_joint_hex, colors_time)
        else:
            idx, colors_valence_hex, colors_arousal_hex, colors_joint_hex, colors_time = color_cache[embedding_len]

        # Plot only missing HTMLs
        plots = {
            "valence": (colors_valence_hex, "_valence.html"),
            "arousal": (colors_arousal_hex, "_arousal.html"),
            "time": (colors_time, "_time.html"),
            "behaviour": (colors_joint_hex, "_behaviour.html")
        }

        for mode, (colors, suffix) in plots.items():
            html_path = npy_file.with_name(npy_file.stem + suffix)
            if not html_path.exists():
                fig = plot_embedding_interactive(embedding[idx], embedding_labels=colors, title=f"{subject_key} - {npy_file.stem} - {mode.capitalize()}")
                fig.write_html(html_path)
                plt.close('all')

        print(f"[INFO] Processed embedding: {npy_file.stem} (skipped existing HTMLs)")
