"""
Why might the unsupervised CEBRA embedding fail to decode valence/arousal?

model_comparison.py showed every downstream model (linear, tree, kNN, MLP, LSTM)
performs at the "guess the population mean" level. This script tests one specific,
falsifiable explanation: CEBRA was fit with a time-contrastive window (time_offsets)
and an encoder receptive field that are both far shorter than the timescale at which
valence/arousal actually move, so the embedding structure it learned reflects fast
EEG dynamics, not slow affect.

Three diagnostics:
  1. Label timescale: circular autocorrelation of the angle vs. lag.
  2. Embedding timescale: cosine autocorrelation of the raw (500Hz) CEBRA embedding
     vs. lag, compared directly against (1) on the same axis, with CEBRA's own
     time_offsets and encoder receptive field marked.
  3. What the embedding learned instead: how well simple fast physiological
     features (broadband/alpha EEG power, elapsed time) explain the embedding,
     compared to how well the label explains it.

Also reports how much of the "explainable" structure is just each subject's own
personal baseline (within_subject_baseline_mae) vs. shared across subjects - i.e.
whether individual identity carries information the pooled model currently discards.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
mne.set_log_level("ERROR")

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from degree_dependency import (
    BEH_DIR, EMB_DIR, get_subject_data, compute_subject_mean_baseline,
)
from data import eeg_dataloader
from eeg.perprocessing import compute_angle_vector_length
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import cebra.models as cebra_models

OFFICIALLY_EXCLUDED = {"sub-011", "sub-017", "sub-044", "sub-047"}
OUT_DIR = Path(__file__).resolve().parent / "timescale_diagnostics_results"
OUT_DIR.mkdir(exist_ok=True)

CEBRA_SFREQ = 500.0
CEBRA_TIME_OFFSET_S = 500 / CEBRA_SFREQ  # 1.0s, the contrastive positive-sampling window
CEBRA_RECEPTIVE_FIELD_S = 10 / CEBRA_SFREQ  # 0.02s, offset10-model's actual context window


def usable_subjects():
    all_dirs = sorted(p.name for p in EMB_DIR.glob("sub-*"))
    return [s for s in all_dirs if s not in OFFICIALLY_EXCLUDED]


# ---------------------------
# 1. Label timescale: circular autocorrelation
# ---------------------------

def circular_autocorrelation(angle_deg: np.ndarray, max_lag: int) -> np.ndarray:
    rad = np.radians(angle_deg)
    z = np.exp(1j * rad)
    acf = np.empty(max_lag + 1)
    acf[0] = 1.0
    for lag in range(1, max_lag + 1):
        acf[lag] = np.mean(np.cos(np.angle(z[lag:] * np.conj(z[:-lag]))))
    return acf


def label_autocorrelation_curve(subject_ids, max_lag_s: int = 300):
    curves = []
    for s in subject_ids:
        beh = eeg_dataloader.load_behavioral_labels(BEH_DIR / s)
        if beh is None:
            continue
        angle, _ = compute_angle_vector_length(beh["valence"].values, beh["arousal"].values)
        lag_cap = min(max_lag_s, len(angle) - 10)
        curves.append(circular_autocorrelation(angle, lag_cap))
    min_len = min(len(c) for c in curves)
    mat = np.stack([c[:min_len] for c in curves])
    return np.arange(min_len), mat.mean(axis=0), mat.std(axis=0)


# ---------------------------
# 2. Embedding timescale: cosine autocorrelation at raw (500Hz) resolution
# ---------------------------

def cosine_autocorrelation(emb: np.ndarray, lags_samples: np.ndarray) -> np.ndarray:
    norm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    out = np.empty(len(lags_samples))
    for i, lag in enumerate(lags_samples):
        if lag == 0:
            out[i] = 1.0
        else:
            out[i] = np.mean(np.sum(norm[lag:] * norm[:-lag], axis=1))
    return out


def embedding_autocorrelation_curve(subject_ids, lags_seconds: np.ndarray, n_subjects: int = 10):
    lags_samples = np.round(lags_seconds * CEBRA_SFREQ).astype(int)
    curves = []
    for s in subject_ids[:n_subjects]:
        path = EMB_DIR / s / "embedding_TNone-None_BNone_CHall.npy"
        if not path.exists():
            continue
        emb = np.load(path)
        valid_lags = lags_samples[lags_samples < len(emb) - 1]
        acf = cosine_autocorrelation(emb, valid_lags)
        full = np.full(len(lags_samples), np.nan)
        full[:len(acf)] = acf
        curves.append(full)
    mat = np.stack(curves)
    return np.nanmean(mat, axis=0), np.nanstd(mat, axis=0)


def half_life_seconds(lags_s: np.ndarray, acf: np.ndarray) -> float:
    below = np.where(acf <= 0.5)[0]
    return float(lags_s[below[0]]) if len(below) else float("nan")


def plot_timescale_comparison(label_lags_s, label_acf, label_acf_std,
                               emb_lags_s, emb_acf, emb_acf_std):
    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.plot(emb_lags_s, emb_acf, color="indianred", marker="o", markersize=3, label="CEBRA embedding (cosine similarity)")
    ax.fill_between(emb_lags_s, emb_acf - emb_acf_std, emb_acf + emb_acf_std, color="indianred", alpha=0.15)

    ax.plot(label_lags_s, label_acf, color="steelblue", marker="o", markersize=3, label="Valence-arousal angle (circular)")
    ax.fill_between(label_lags_s, label_acf - label_acf_std, label_acf + label_acf_std, color="steelblue", alpha=0.15)

    ax.axvline(CEBRA_TIME_OFFSET_S, color="gray", linestyle="--", linewidth=1,
               label=f"CEBRA time_offsets ({CEBRA_TIME_OFFSET_S:.2g}s)")
    ax.axvline(CEBRA_RECEPTIVE_FIELD_S, color="black", linestyle=":", linewidth=1,
               label=f"offset10-model receptive field ({CEBRA_RECEPTIVE_FIELD_S:.2g}s)")
    ax.axhline(0.5, color="lightgray", linewidth=0.8)

    ax.set_xscale("log")
    ax.set_xlabel("Lag (seconds, log scale)")
    ax.set_ylabel("Autocorrelation")
    ax.set_title("Timescale mismatch: how fast the embedding decorrelates vs. how fast the label moves")
    ax.legend(fontsize=8, loc="upper right")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "timescale_comparison.png", dpi=150)
    plt.close()


# ---------------------------
# 3. What did the embedding learn instead?
# ---------------------------

def windowed_bandpower(x: np.ndarray, window_samples: int, sfreq: float,
                        band: "tuple | None") -> np.ndarray:
    # FFT-based, per-window power in a frequency band - avoids relying on
    # mne's raw.filter(), which turned out to be a near no-op on this data.
    n_windows = x.shape[-1] // window_samples
    trimmed = x[..., :n_windows * window_samples]
    reshaped = trimmed.reshape(*x.shape[:-1], n_windows, window_samples)
    freqs = np.fft.rfftfreq(window_samples, d=1.0 / sfreq)
    power = np.abs(np.fft.rfft(reshaped, axis=-1)) ** 2
    mask = (freqs > 0.5) if band is None else ((freqs >= band[0]) & (freqs <= band[1]))
    return power[..., mask].sum(axis=-1)


def load_physio_features(subject_id: str, n_seconds: int):
    try:
        raw = eeg_dataloader.load_subject(BEH_DIR / subject_id, data_type="preproc")
    except FileNotFoundError:
        return None
    sfreq = raw.info["sfreq"]
    window = int(round(sfreq))
    data = raw.get_data(picks="eeg")

    broadband = windowed_bandpower(data, window, sfreq, band=None)
    alpha = windowed_bandpower(data, window, sfreq, band=(8, 12))

    broadband_mean = np.log1p(broadband.mean(axis=0))[:n_seconds]
    alpha_mean = np.log1p(alpha.mean(axis=0))[:n_seconds]
    elapsed_time = np.arange(n_seconds, dtype=float)

    return pd.DataFrame({
        "broadband_power": broadband_mean,
        "alpha_power": alpha_mean,
        "elapsed_time": elapsed_time,
    })


def explain_embedding_variance(subject_ids, n_subjects: int = 6):
    rows = []
    for s in subject_ids[:n_subjects]:
        emb, angle = get_subject_data(s, BEH_DIR, EMB_DIR)
        if emb is None:
            continue
        n = len(angle)
        physio = load_physio_features(s, n)
        if physio is None or len(physio) < n:
            continue
        physio = physio.iloc[:n].reset_index(drop=True)
        emb = emb[:n]

        label_xy = np.stack([np.cos(np.radians(angle)), np.sin(np.radians(angle))], axis=1)

        split = int(n * 0.7)
        feature_sets = {
            "label (valence/arousal angle)": label_xy,
            "broadband EEG power": physio[["broadband_power"]].values,
            "alpha (8-12Hz) power": physio[["alpha_power"]].values,
            "elapsed time in session": physio[["elapsed_time"]].values,
            "all physiological features": physio.values,
        }

        for name, feats in feature_sets.items():
            scaler_x = StandardScaler().fit(feats[:split])
            Xtr, Xte = scaler_x.transform(feats[:split]), scaler_x.transform(feats[split:])
            ytr, yte = emb[:split], emb[split:]
            model = Ridge(alpha=1.0).fit(Xtr, ytr)
            pred = model.predict(Xte)
            r2 = r2_score(yte, pred, multioutput="variance_weighted")
            rows.append({"subject": s, "feature_set": name, "r2": r2})

    return pd.DataFrame(rows)


def plot_explained_variance(df: pd.DataFrame):
    import seaborn as sns
    order = df.groupby("feature_set")["r2"].mean().sort_values(ascending=False).index
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x="r2", y="feature_set", order=order, hue="feature_set",
                legend=False, palette="Set2", errorbar="sd")
    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel("R² explaining the CEBRA embedding (held-out, within-subject)")
    plt.ylabel("")
    plt.title("What explains the embedding better: the label, or fast EEG power / time?")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "embedding_explained_variance.png", dpi=150)
    plt.close()


# ---------------------------
# Main
# ---------------------------

def main():
    subject_ids = usable_subjects()
    print(f"Usable subjects: {len(subject_ids)}")

    print(f"\noffset10-model receptive field: {CEBRA_RECEPTIVE_FIELD_S*1000:.0f} ms")
    print(f"CEBRA time_offsets (contrastive window): {CEBRA_TIME_OFFSET_S:.2f} s")

    print("\nComputing label (angle) autocorrelation vs. lag...")
    label_lags, label_acf, label_acf_std = label_autocorrelation_curve(subject_ids)
    label_hl = half_life_seconds(label_lags, label_acf)
    print(f"  Label autocorrelation drops below 0.5 at lag = {label_hl:.0f}s")

    print("Computing CEBRA embedding autocorrelation vs. lag (raw 500Hz embeddings)...")
    lags_seconds = np.array([0.002, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 30, 60, 120, 240])
    emb_acf, emb_acf_std = embedding_autocorrelation_curve(subject_ids, lags_seconds)
    emb_hl = half_life_seconds(lags_seconds, emb_acf)
    print(f"  Embedding autocorrelation drops below 0.5 at lag = {emb_hl:.3f}s")
    print(f"  => label moves {label_hl / emb_hl:.0f}x slower than the embedding decorrelates" if emb_hl > 0 else "")

    plot_timescale_comparison(label_lags, label_acf, label_acf_std, lags_seconds, emb_acf, emb_acf_std)
    pd.DataFrame({"lag_s": label_lags, "label_acf": label_acf, "label_acf_std": label_acf_std}).to_csv(
        OUT_DIR / "label_autocorrelation.csv", index=False)
    pd.DataFrame({"lag_s": lags_seconds, "embedding_acf": emb_acf, "embedding_acf_std": emb_acf_std}).to_csv(
        OUT_DIR / "embedding_autocorrelation.csv", index=False)

    print("\nFitting embedding ~ [label | EEG power | elapsed time] (6 subjects, held-out split)...")
    explained = explain_embedding_variance(subject_ids)
    explained.to_csv(OUT_DIR / "embedding_explained_variance.csv", index=False)
    plot_explained_variance(explained)
    print(explained.groupby("feature_set")["r2"].agg(["mean", "std"]).sort_values("mean", ascending=False).round(3))

    print("\nHow much of the 'explainable' angle variance is just per-subject baseline?")
    baseline = compute_subject_mean_baseline(subject_ids, BEH_DIR, EMB_DIR)
    print(f"  Within-subject circular-mean MAE (each subject predicting their own mean): "
          f"{baseline.get('within_subject_baseline_mae', float('nan')):.1f} deg")
    print("  (compare to ~69-74 deg for every pooled cross-subject model in model_comparison.py -"
          " none of them currently receive subject identity at all)")


if __name__ == "__main__":
    main()
