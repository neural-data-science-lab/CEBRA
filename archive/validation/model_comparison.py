"""
Model comparison on the existing (unsupervised) CEBRA embeddings.

Question: given the embeddings we already have (embedding_TNone-None_BNone_CHall.npy,
per-subject, unsupervised time-contrastive CEBRA, 1s-aggregated - same features the
LSTM in degree_dependency.py uses), how much does model choice matter for decoding
the circular valence-arousal angle under Leave-One-Subject-Out CV?

Reuses get_subject_data / mean_angular_error / angular_rmse / angular_accuracy from
degree_dependency.py so results are directly comparable to the existing LSTM run.
"""
from pathlib import Path
from typing import Dict, List
import sys
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from degree_dependency import (
    BEH_DIR, EMB_DIR, get_subject_data,
    mean_angular_error, angular_rmse, angular_accuracy,
)

OFFICIALLY_EXCLUDED = {"sub-011", "sub-017", "sub-044", "sub-047"}
OUT_DIR = Path(__file__).resolve().parent / "model_comparison_results"
OUT_DIR.mkdir(exist_ok=True)


def angle_to_xy(angle_deg: np.ndarray) -> np.ndarray:
    rad = np.radians(angle_deg)
    return np.stack([np.cos(rad), np.sin(rad)], axis=1)


def xy_to_angle(xy: np.ndarray) -> np.ndarray:
    return (np.degrees(np.arctan2(xy[:, 1], xy[:, 0])) + 360) % 360


def circular_mean_deg(angles_deg: np.ndarray) -> float:
    rad = np.radians(angles_deg)
    return (np.degrees(np.arctan2(np.sin(rad).mean(), np.cos(rad).mean())) + 360) % 360


def load_all_subjects(subject_ids: List[str]):
    data = {}
    for s in tqdm(subject_ids, desc="Loading cached embeddings + behavior"):
        X, y = get_subject_data(s, BEH_DIR, EMB_DIR)
        if X is not None and y is not None and len(y) > 5:
            data[s] = (X, y)
    return data


def evaluate(y_pred: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
    return {
        "mae": mean_angular_error(y_pred, y_true),
        "rmse": angular_rmse(y_pred, y_true),
        "accuracy_10deg": angular_accuracy(y_pred, y_true, 10.0),
        "accuracy_20deg": angular_accuracy(y_pred, y_true, 20.0),
        "accuracy_30deg": angular_accuracy(y_pred, y_true, 30.0),
    }


def predict_persistence(y_train: np.ndarray, y_test: np.ndarray) -> np.ndarray:
    # Predict the previous timestep's true angle; the first test sample has
    # no predecessor of its own, so it falls back to the training circular mean.
    fallback = circular_mean_deg(y_train)
    return np.concatenate([[fallback], y_test[:-1]])


def predict_circular_mean(y_train: np.ndarray, n_test: int) -> np.ndarray:
    return np.full(n_test, circular_mean_deg(y_train))


MODEL_BUILDERS = {
    "Ridge": lambda: Ridge(alpha=1.0),
    "kNN": lambda: KNeighborsRegressor(n_neighbors=25, weights="distance"),
    "RandomForest": lambda: RandomForestRegressor(
        n_estimators=300, max_depth=14, n_jobs=-1, random_state=0),
    "MLP": lambda: MLPRegressor(
        hidden_layer_sizes=(64, 32), early_stopping=True,
        max_iter=300, random_state=0),
}


def run_loso(data: Dict[str, tuple], subject_ids: List[str]) -> pd.DataFrame:
    rows = []
    for test_subj in tqdm(subject_ids, desc="LOSO CV"):
        X_test, y_test = data[test_subj]
        train_subjs = [s for s in subject_ids if s != test_subj]
        X_train = np.concatenate([data[s][0] for s in train_subjs])
        y_train = np.concatenate([data[s][1] for s in train_subjs])

        # --- baselines (no fitting needed beyond a circular mean) ---
        rows.append({"subject": test_subj, "model": "Persistence",
                      **evaluate(predict_persistence(y_train, y_test), y_test)})
        rows.append({"subject": test_subj, "model": "CircularMean",
                      **evaluate(predict_circular_mean(y_train, len(y_test)), y_test)})

        # --- learned models, all predicting [cos, sin] then recovering angle ---
        scaler = StandardScaler().fit(X_train)
        Xtr = scaler.transform(X_train)
        Xte = scaler.transform(X_test)
        ytr_xy = angle_to_xy(y_train)

        for name, builder in MODEL_BUILDERS.items():
            model = builder()
            model.fit(Xtr, ytr_xy)
            pred_xy = model.predict(Xte)
            pred_angle = xy_to_angle(pred_xy)
            rows.append({"subject": test_subj, "model": name,
                          **evaluate(pred_angle, y_test)})

    return pd.DataFrame(rows)


def add_existing_lstm_results(results: pd.DataFrame, clean_subjects: set) -> pd.DataFrame:
    lstm_pkl = Path(__file__).resolve().parents[1] / "loso_degree_regression_results.pkl"
    if not lstm_pkl.exists():
        return results
    with open(lstm_pkl, "rb") as f:
        d = pickle.load(f)
    rows = []
    for subj, r in d["subject_results"].items():
        if subj not in clean_subjects:
            continue
        m = r["metrics"]
        rows.append({"subject": subj, "model": "LSTM (existing)", **m})
    if rows:
        results = pd.concat([results, pd.DataFrame(rows)], ignore_index=True)
    return results


def summarize_and_plot(results: pd.DataFrame):
    results.to_csv(OUT_DIR / "loso_results_by_subject.csv", index=False)

    metric_cols = ["mae", "rmse", "accuracy_10deg", "accuracy_20deg", "accuracy_30deg"]
    summary = results.groupby("model")[metric_cols].agg(["mean", "std"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary = summary.sort_values("mae_mean")
    summary.to_csv(OUT_DIR / "model_comparison_summary.csv")

    order = summary.index.tolist()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    sns.boxplot(data=results, x="model", y="mae", order=order, ax=axes[0], hue="model",
                legend=False, palette="Set2")
    axes[0].axhline(90, color="gray", linestyle="--", linewidth=1,
                     label="Expected MAE of a uniform-random guess")
    axes[0].set_title("Mean Angular Error by Model (LOSO, per-subject)")
    axes[0].set_ylabel("MAE (degrees)")
    axes[0].set_xlabel("")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].legend(fontsize=8)

    sns.barplot(data=results, x="model", y="accuracy_30deg", order=order, ax=axes[1],
                hue="model", legend=False, palette="Set2", errorbar="sd")
    axes[1].set_title("Accuracy within 30° by Model (LOSO)")
    axes[1].set_ylabel("Fraction of predictions within 30°")
    axes[1].set_xlabel("")
    axes[1].tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "model_comparison.png", dpi=150)
    plt.close()

    print("\n" + "=" * 70)
    print("MODEL COMPARISON — LOSO CV on existing unsupervised CEBRA embeddings")
    print("=" * 70)
    print(summary[["mae_mean", "mae_std", "accuracy_30deg_mean", "accuracy_30deg_std"]]
          .round(2).to_string())
    print("=" * 70)
    return summary


def main():
    all_subject_dirs = sorted(p.name for p in EMB_DIR.glob("sub-*"))
    clean_subjects = [s for s in all_subject_dirs if s not in OFFICIALLY_EXCLUDED]

    data = load_all_subjects(clean_subjects)
    subject_ids = sorted(data.keys())
    print(f"Usable subjects (have embedding + behavior, not officially excluded): {len(subject_ids)}")
    print(f"Excluded per excluded_participants.tsv: {sorted(OFFICIALLY_EXCLUDED)}")

    results = run_loso(data, subject_ids)
    results = add_existing_lstm_results(results, set(subject_ids))
    summarize_and_plot(results)


if __name__ == "__main__":
    main()
