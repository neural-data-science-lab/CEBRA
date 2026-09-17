# exploration_analysis.py
"""
Descriptive exploration of the valence-arousal labels.

The labeling tool constrains ratings to a square canvas (valence, arousal in
[-1, 1]) meant to approximate Russell's circumplex model, which is
theoretically a circle. Every plot here keeps that square-vs-circle framing
front and center, because in practice ratings pile up on the square's border
rather than filling the circle - and *how* they pile up (imbalance across
octants, consistency across subjects/time) is what tells us whether the
experiment behaved as expected or whether specific subjects should be
excluded before any downstream (e.g. CEBRA) analysis.
"""
from pathlib import Path
from typing import Dict, Optional, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import seaborn as sns
from scipy.stats import pearsonr, skew, kurtosis, circmean, circstd, circvar
import logging

sns.set_theme(style="whitegrid")
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ---------------------------
# Circumplex / octant constants
# ---------------------------

# Octants are centered on the 8 cardinal circumplex directions (0=pure
# positive valence, 90=pure high arousal, ...) rather than starting at 0,
# so a bin boundary never splits a "pure" emotional state in half.
OCTANT_LABELS = [
    "Pleasant\n(V+)",
    "Excited\n(V+ A+)",
    "Activated\n(A+)",
    "Tense\n(V- A+)",
    "Unpleasant\n(V-)",
    "Depressed\n(V- A-)",
    "Deactivated\n(A-)",
    "Relaxed\n(V+ A-)",
]
OCTANT_COLORS = sns.color_palette("husl", len(OCTANT_LABELS))
OCTANT_CMAP = ListedColormap(OCTANT_COLORS)

# QC thresholds for flagging subjects as candidates for exclusion/review.
MAX_MISSING_PCT = 5.0
DURATION_TOL_SEC = 30.0
MAX_DOMINANT_OCTANT_PROP = 0.7
MIN_VECTOR_LENGTH_MEAN = 0.25
MIN_GROUP_AGREEMENT = 0.05


def _ensure_angle_vector(df: pd.DataFrame) -> pd.DataFrame:
    if 'angle' in df.columns and 'vector_length' in df.columns:
        return df
    df = df.copy()
    v = df['valence'].to_numpy(dtype=float)
    a = df['arousal'].to_numpy(dtype=float)
    df['angle'] = (np.degrees(np.arctan2(a, v)) + 360) % 360
    df['vector_length'] = np.sqrt(v ** 2 + a ** 2) / np.sqrt(2)
    return df


def angles_to_octants(angles_deg: np.ndarray) -> np.ndarray:
    shifted = (np.asarray(angles_deg, dtype=float) + 22.5) % 360
    idx = np.floor(shifted / 45).astype(int) % len(OCTANT_LABELS)
    return np.array(OCTANT_LABELS)[idx]


def compute_octant_proportions(df: pd.DataFrame) -> pd.Series:
    a = df['angle'].dropna()
    if a.empty:
        return pd.Series(0.0, index=OCTANT_LABELS)
    counts = pd.Series(angles_to_octants(a.values)).value_counts()
    props = counts.reindex(OCTANT_LABELS, fill_value=0) / len(a)
    return props


def compute_subject_octant_table(all_labels: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = {s: compute_octant_proportions(df) for s, df in all_labels.items()}
    return pd.DataFrame(rows).T


# ---------------------------
# Circumplex reference overlay
# ---------------------------

def _draw_circumplex_reference(ax):
    """Square = labeling tool bounds; circle (radius 1) = the theoretical circumplex."""
    ax.add_patch(mpatches.Rectangle((-1, -1), 2, 2, fill=False, edgecolor='gray',
                                     linestyle='--', linewidth=1, label='Tool bounds (square)'))
    ax.add_patch(mpatches.Circle((0, 0), 1, fill=False, edgecolor='black',
                                  linestyle='-', linewidth=1.2, label='Circumplex (circle)'))
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_aspect('equal')


def plot_circumplex_panel(df: pd.DataFrame, save_path: Path, title: str):
    """Scatter (colored by time) + density, both against the square/circle reference."""
    sub = df[['valence', 'arousal']].dropna()
    if sub.empty:
        logger.warning(f"No valence/arousal data for circumplex panel: {title}")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))

    ax = axes[0]
    c = df.loc[sub.index, 'timestamp'] if 'timestamp' in df.columns else np.arange(len(sub))
    sc = ax.scatter(sub['valence'], sub['arousal'], c=c, cmap='viridis', s=6, alpha=0.5)
    _draw_circumplex_reference(ax)
    ax.set_title("Ratings over time")
    ax.set_xlabel("Valence")
    ax.set_ylabel("Arousal")
    ax.legend(loc='upper right', fontsize=7, framealpha=0.8)
    plt.colorbar(sc, ax=ax, label="Time (s)", shrink=0.8)

    ax = axes[1]
    sns.kdeplot(x=sub['valence'], y=sub['arousal'], fill=True, cmap='viridis',
                levels=50, thresh=0.02, ax=ax)
    _draw_circumplex_reference(ax)
    ax.set_title("Density")
    ax.set_xlabel("Valence")
    ax.set_ylabel("Arousal")

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved circumplex panel to {save_path}")


def plot_octant_occupancy(df: pd.DataFrame, save_path: Path, title: str):
    """Bar chart (class imbalance across octants) + polar histogram (raw angle resolution)."""
    a = df['angle'].dropna()
    if a.empty:
        logger.warning(f"No angle data for octant occupancy: {title}")
        return

    fig = plt.figure(figsize=(11, 5))

    ax = fig.add_subplot(1, 2, 1)
    props = compute_octant_proportions(df)
    bars = ax.bar(range(len(props)), props.values * 100, color=OCTANT_COLORS, edgecolor='black')
    ax.set_xticks(range(len(props)))
    ax.set_xticklabels(props.index, fontsize=8)
    ax.set_ylabel("% of samples")
    ax.axhline(100 / len(OCTANT_LABELS), color='red', linestyle='--', linewidth=1,
               label='Uniform expectation')
    ax.set_title("Octant occupancy (class imbalance)")
    ax.legend(fontsize=8)
    for bar, val in zip(bars, props.values * 100):
        ax.annotate(f"{val:.0f}%", (bar.get_x() + bar.get_width() / 2, val),
                    ha='center', va='bottom', fontsize=7)

    ax = fig.add_subplot(1, 2, 2, projection='polar')
    rad = np.radians(a.values)
    ax.hist(rad, bins=36, color='lightcoral', edgecolor='black', alpha=0.7)
    c_mean = circmean(rad, high=2 * np.pi, low=0)
    r_max = ax.get_ylim()[1]
    ax.annotate('', xy=(c_mean, r_max), xytext=(c_mean, 0),
                arrowprops=dict(facecolor='black', edgecolor='black', width=2, headwidth=8))
    ax.set_title(f"Angle histogram\n(circ. mean={np.degrees(c_mean) % 360:.0f}°)")

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved octant occupancy to {save_path}")


# ---------------------------
# Violin plots
# ---------------------------

def plot_subject_violin_halves(df: pd.DataFrame, save_dir: Path, subject_id: str):
    """Violin of valence/arousal/vector_length, split first vs. second half of the session -
    a quick check for whether a subject's rating behavior drifted over time."""
    if 'timestamp' not in df.columns:
        logger.warning(f"[{subject_id}] No timestamp; skipping half-session violin.")
        return
    cols = [c for c in ['valence', 'arousal', 'vector_length'] if c in df.columns]
    if not cols:
        return

    t = df['timestamp']
    mid = (t.min() + t.max()) / 2
    sub = df[cols + ['timestamp']].dropna(subset=cols, how='all').copy()
    sub['half'] = np.where(sub['timestamp'] <= mid, 'First half', 'Second half')
    long = sub.melt(id_vars='half', value_vars=cols, var_name='variable', value_name='value').dropna()
    if long.empty:
        return

    plt.figure(figsize=(7, 5))
    sns.violinplot(data=long, x='variable', y='value', hue='half', split=True,
                    inner='quartile', palette=['steelblue', 'indianred'])
    plt.title(f"{subject_id} — First vs. Second Half of Session")
    plt.xlabel("")
    plt.ylabel("Value")
    plt.tight_layout()
    save_path = save_dir / f"{subject_id}_violin_session_halves.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"[{subject_id}] Saved session-half violin to {save_path}")


def plot_group_violin_by_subject(df: pd.DataFrame, save_dir: Path):
    """Per-subject violins for valence/arousal/vector_length - shows bimodality (border
    clustering) that a boxplot hides, and lets outlier subjects stand out at a glance."""
    cols = [c for c in ['valence', 'arousal', 'vector_length'] if c in df.columns]
    if not cols or 'subject_id' not in df.columns:
        logger.warning("Cannot create group violin plots (missing columns).")
        return

    n_subjects = df['subject_id'].nunique()
    n = len(cols)
    fig, axes = plt.subplots(n, 1, figsize=(max(10, 0.35 * n_subjects), 3.5 * n))
    if n == 1:
        axes = [axes]

    order = sorted(df['subject_id'].unique())
    for ax, col in zip(axes, cols):
        sns.violinplot(data=df, x='subject_id', y=col, ax=ax, order=order,
                        hue='subject_id', legend=False, palette='Set3',
                        cut=0, density_norm='width', inner=None)
        ax.set_title(f"{col.capitalize()} by Subject")
        ax.tick_params(axis='x', rotation=90, labelsize=7)

    plt.tight_layout()
    save_path = save_dir / "group_violin_by_subject.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Saved group violin-by-subject to {save_path}")


# ---------------------------
# Timelines
# ---------------------------

def plot_subject_timeline(df: pd.DataFrame, save_dir: Path, subject_id: str):
    """Valence/arousal trace plus a color ribbon of the occupied octant over time."""
    if 'timestamp' not in df.columns:
        logger.warning(f"[{subject_id}] No timestamp column found; skipping timeline.")
        return

    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True,
                              gridspec_kw={'height_ratios': [4, 1]})

    ax = axes[0]
    if 'valence' in df.columns:
        ax.plot(df['timestamp'], df['valence'], color='steelblue', alpha=0.8, label='Valence')
    if 'arousal' in df.columns:
        ax.plot(df['timestamp'], df['arousal'], color='indianred', alpha=0.8, label='Arousal')
    ax.set_ylabel("Rating")
    ax.set_ylim(-1.05, 1.05)
    ax.legend(fontsize=8)
    ax.set_title(f"{subject_id} — Rating Timeline")

    ax = axes[1]
    a = df['angle']
    valid = a.notna()
    if valid.any():
        idx = np.array([OCTANT_LABELS.index(lbl) for lbl in angles_to_octants(a[valid].values)])
        strip = np.full(len(df), np.nan)
        strip[valid.values] = idx
        ax.imshow(strip[np.newaxis, :], aspect='auto', cmap=OCTANT_CMAP, vmin=0,
                  vmax=len(OCTANT_LABELS) - 1,
                  extent=[df['timestamp'].min(), df['timestamp'].max(), 0, 1])
    ax.set_yticks([])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Octant", fontsize=8)

    handles = [mpatches.Patch(color=OCTANT_COLORS[i], label=lbl.replace('\n', ' '))
               for i, lbl in enumerate(OCTANT_LABELS)]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, -0.08),
               ncol=4, fontsize=7, frameon=False)

    plt.tight_layout()
    save_path = save_dir / f"{subject_id}_timeline.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"[{subject_id}] Saved timeline to {save_path}")


def plot_group_timeline(all_labels: Dict[str, pd.DataFrame], save_dir: Path):
    """Group mean +/- SD over time, with faint per-subject traces to show consistency."""
    cols = ['valence', 'arousal']
    fig, axes = plt.subplots(len(cols), 1, figsize=(10, 3 * len(cols) + 1), sharex=True)

    for ax, col in zip(axes, cols):
        any_plotted = False
        for s, df in all_labels.items():
            if 'timestamp' not in df.columns or col not in df.columns:
                continue
            ax.plot(df['timestamp'], df[col], color='gray', alpha=0.12, linewidth=0.8)
            any_plotted = True
        if not any_plotted:
            continue
        combined = pd.concat(
            [df[['timestamp', col]].assign(subject_id=s) for s, df in all_labels.items()
             if 'timestamp' in df.columns and col in df.columns],
            ignore_index=True
        )
        agg = combined.groupby('timestamp')[col].agg(['mean', 'std'])
        ax.plot(agg.index, agg['mean'], color='black', linewidth=1.5, label=f'Mean {col}')
        ax.fill_between(agg.index, agg['mean'] - agg['std'], agg['mean'] + agg['std'],
                         color='black', alpha=0.15, label='±1 SD')
        ax.set_ylabel(col.capitalize())
        ax.set_ylim(-1.05, 1.05)
        ax.legend(fontsize=8)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Group Rating Timeline (individual subjects in gray)")
    plt.tight_layout()
    save_path = save_dir / "group_timeline.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Saved group timeline to {save_path}")


# ---------------------------
# Descriptive statistics utilities
# ---------------------------

def detect_outliers_iqr(series: pd.Series) -> int:
    s = series.dropna()
    if s.size == 0:
        return 0
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return 0
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return int(((s < lower) | (s > upper)).sum())


def compute_subject_descriptive_stats(df: pd.DataFrame, subject_id: str) -> Dict:
    stats: Dict = {'subject_id': subject_id, 'n_samples': len(df)}
    n_total = len(df)

    for col in ['valence', 'arousal', 'vector_length']:
        if col in df.columns:
            valid = df[col].dropna()
            stats[f'{col}_n'] = int(valid.size)
            stats[f'{col}_pct_missing'] = 100.0 * (1 - valid.size / n_total) if n_total > 0 else np.nan
            if valid.size > 0:
                q25, q75 = valid.quantile(0.25), valid.quantile(0.75)
                stats[f'{col}_mean'] = valid.mean()
                stats[f'{col}_median'] = valid.median()
                stats[f'{col}_std'] = valid.std()
                stats[f'{col}_min'] = valid.min()
                stats[f'{col}_max'] = valid.max()
                stats[f'{col}_q25'] = q25
                stats[f'{col}_q75'] = q75
                stats[f'{col}_iqr'] = q75 - q25
                stats[f'{col}_skew'] = float(skew(valid)) if valid.size > 2 else np.nan
                stats[f'{col}_kurtosis'] = float(kurtosis(valid)) if valid.size > 2 else np.nan
                stats[f'{col}_n_outliers_iqr'] = detect_outliers_iqr(valid)
            else:
                for suffix in ['mean', 'median', 'std', 'min', 'max', 'q25', 'q75', 'iqr', 'skew', 'kurtosis']:
                    stats[f'{col}_{suffix}'] = np.nan
                stats[f'{col}_n_outliers_iqr'] = 0
        else:
            stats[f'{col}_n'] = 0
            stats[f'{col}_pct_missing'] = 100.0
            for suffix in ['mean', 'median', 'std', 'min', 'max', 'q25', 'q75', 'iqr', 'skew', 'kurtosis']:
                stats[f'{col}_{suffix}'] = np.nan
            stats[f'{col}_n_outliers_iqr'] = 0

    if 'angle' in df.columns:
        a = df['angle'].dropna()
        stats['angle_n'] = int(a.size)
        stats['angle_pct_missing'] = 100.0 * (1 - a.size / n_total) if n_total > 0 else np.nan
        if a.size > 0:
            rad = np.radians(a.values)
            c_mean = circmean(rad, high=2 * np.pi, low=0)
            c_std = circstd(rad, high=2 * np.pi, low=0)
            c_var = circvar(rad, high=2 * np.pi, low=0)
            stats['angle_circ_mean_deg'] = float(np.degrees(c_mean)) % 360
            stats['angle_circ_std_deg'] = float(np.degrees(c_std))
            stats['angle_resultant_length_R'] = float(1 - c_var)  # 1 = concentrated, 0 = uniform
            dominant = compute_octant_proportions(df)
            stats['dominant_octant'] = dominant.idxmax().replace('\n', ' ')
            stats['dominant_octant_prop'] = float(dominant.max())
        else:
            stats['angle_circ_mean_deg'] = np.nan
            stats['angle_circ_std_deg'] = np.nan
            stats['angle_resultant_length_R'] = np.nan
            stats['dominant_octant'] = np.nan
            stats['dominant_octant_prop'] = np.nan
    else:
        stats['angle_n'] = 0
        stats['angle_pct_missing'] = 100.0
        stats['angle_circ_mean_deg'] = np.nan
        stats['angle_circ_std_deg'] = np.nan
        stats['angle_resultant_length_R'] = np.nan
        stats['dominant_octant'] = np.nan
        stats['dominant_octant_prop'] = np.nan

    return stats


def compute_full_stats_table(all_labels: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = [compute_subject_descriptive_stats(df, s) for s, df in all_labels.items()]
    return pd.DataFrame(rows).set_index('subject_id')


def compute_recording_summary(all_labels: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for s, df in all_labels.items():
        row = {'subject_id': s}
        if 'timestamp' in df.columns:
            ts = df['timestamp'].dropna().sort_values()
            if ts.size > 0:
                row['t_start'] = ts.iloc[0]
                row['t_end'] = ts.iloc[-1]
                row['duration'] = ts.iloc[-1] - ts.iloc[0]
                row['n_samples'] = int(ts.size)
                dt = ts.diff().dropna()
                median_dt = dt.median() if dt.size > 0 else np.nan
                row['median_dt'] = median_dt
                row['est_sampling_rate_hz'] = (1.0 / median_dt) if (pd.notna(median_dt) and median_dt > 0) else np.nan
            else:
                row.update({'t_start': np.nan, 't_end': np.nan, 'duration': np.nan,
                            'n_samples': 0, 'median_dt': np.nan, 'est_sampling_rate_hz': np.nan})
        else:
            row.update({'t_start': np.nan, 't_end': np.nan, 'duration': np.nan,
                        'n_samples': len(df), 'median_dt': np.nan, 'est_sampling_rate_hz': np.nan})
        rows.append(row)
    return pd.DataFrame(rows).set_index('subject_id')


def compute_missingness_table(all_labels: Dict[str, pd.DataFrame], cols: Optional[List[str]] = None) -> pd.DataFrame:
    if cols is None:
        cols = ['valence', 'arousal', 'angle', 'vector_length']
    rows = []
    for s, df in all_labels.items():
        n = len(df)
        row = {'subject_id': s}
        for c in cols:
            row[c] = 100.0 * df[c].isna().sum() / n if (c in df.columns and n > 0) else 100.0
        rows.append(row)
    return pd.DataFrame(rows).set_index('subject_id')


def select_example_subjects(stats_table: pd.DataFrame, completeness_col: str = 'valence_pct_missing') -> Dict[str, str]:
    if stats_table.empty:
        return {}
    if completeness_col in stats_table.columns and stats_table[completeness_col].notna().any():
        sorted_ids = stats_table[completeness_col].sort_values(ascending=True).index.tolist()
    elif 'n_samples' in stats_table.columns:
        sorted_ids = stats_table['n_samples'].sort_values(ascending=False).index.tolist()
    else:
        sorted_ids = stats_table.index.tolist()

    if len(sorted_ids) == 0:
        return {}

    examples = {'most_complete': sorted_ids[0]}
    if len(sorted_ids) > 1:
        examples['median'] = sorted_ids[len(sorted_ids) // 2]
        examples['least_complete'] = sorted_ids[-1]
    return examples


def plot_example_subjects_overlay(all_labels: Dict[str, pd.DataFrame], example_subjects: Dict[str, str],
                                   save_dir: Path):
    if not example_subjects:
        logger.warning("No example subjects selected; skipping overlay plot.")
        return

    cols = ['valence', 'arousal', 'vector_length']
    colors = {'most_complete': 'seagreen', 'median': 'steelblue', 'least_complete': 'indianred'}

    fig, axes = plt.subplots(1, len(cols), figsize=(5 * len(cols), 4.5))
    any_plotted = False
    for ax, col in zip(axes, cols):
        for label, subj_id in example_subjects.items():
            subj_df = all_labels.get(subj_id)
            if subj_df is None or col not in subj_df.columns:
                continue
            data = subj_df[col].dropna()
            if data.size < 2:
                continue
            sns.kdeplot(data, ax=ax, label=f"{label} ({subj_id})", color=colors.get(label), fill=False)
            any_plotted = True
        ax.set_title(f"{col.capitalize()} — Example Subjects")
        ax.set_xlabel(col.capitalize())
        ax.legend(fontsize=8)

    if not any_plotted:
        plt.close(fig)
        logger.warning("No data available to plot for example subjects overlay.")
        return

    plt.tight_layout()
    save_path = save_dir / "example_subjects_distribution_overlay.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Saved example subjects overlay to {save_path}")


# ---------------------------
# Consistency (valence/arousal only - angle/vector_length are deterministic
# functions of these two, so correlating them adds geometric noise, not signal)
# ---------------------------

def calculate_inter_subject_consistency(all_labels: Dict[str, pd.DataFrame]) -> Dict:
    subjects = list(all_labels.keys())
    if len(subjects) == 0:
        raise ValueError("all_labels is empty")

    ts_sets = []
    for s in subjects:
        df = all_labels[s]
        if 'timestamp' not in df.columns:
            raise ValueError(f"Subject {s} missing 'timestamp' column")
        ts_sets.append(set(df['timestamp'].dropna().unique()))
    common_ts = sorted(set.intersection(*ts_sets)) if len(ts_sets) > 0 else []
    all_ts = common_ts if len(common_ts) > 0 else sorted(set().union(*ts_sets))
    if len(common_ts) == 0:
        logger.warning("No common timestamps across all subjects. Falling back to union-alignment with NaNs.")

    def build_matrix(col):
        mat = pd.DataFrame(index=all_ts, columns=subjects, dtype=float)
        for s in subjects:
            subj_df = all_labels[s].set_index('timestamp')
            mat[s] = subj_df.loc[mat.index, col] if col in subj_df.columns else np.nan
        return mat

    valence_mat = build_matrix('valence')
    arousal_mat = build_matrix('arousal')

    def pairwise_corr_df(mat: pd.DataFrame):
        cols = mat.columns
        corr = pd.DataFrame(index=cols, columns=cols, dtype=float)
        for i in cols:
            for j in cols:
                a, b = mat[i], mat[j]
                valid = a.notna() & b.notna()
                corr.loc[i, j] = np.nan if valid.sum() < 2 else pearsonr(a[valid], b[valid])[0]
        return corr

    valence_corr = pairwise_corr_df(valence_mat)
    arousal_corr = pairwise_corr_df(arousal_mat)

    def mean_upper_tri(mat_df):
        arr = mat_df.values
        n = arr.shape[0]
        if n <= 1:
            return np.nan
        mask = np.triu(np.ones_like(arr, dtype=bool), k=1)
        vals = arr[mask]
        vals = vals[~np.isnan(vals)]
        return float(np.nan) if vals.size == 0 else float(np.mean(vals))

    return {
        'valence_matrix': valence_mat,
        'arousal_matrix': arousal_mat,
        'valence_matrix_corr': valence_corr,
        'arousal_matrix_corr': arousal_corr,
        'valence_consistency': mean_upper_tri(valence_corr),
        'arousal_consistency': mean_upper_tri(arousal_corr),
    }


def compute_subject_group_agreement(consistency_data: Dict) -> pd.DataFrame:
    """Correlate each subject's trace with the simple group-mean trace (self included) -
    a quick, non-rigorous signal for 'this subject didn't track the shared stimulus'."""
    rows = []
    for s in consistency_data['valence_matrix'].columns:
        v_corr = consistency_data['valence_matrix_corr'].loc[s].drop(s, errors='ignore')
        a_corr = consistency_data['arousal_matrix_corr'].loc[s].drop(s, errors='ignore')
        rows.append({
            'subject_id': s,
            'valence_group_agreement': v_corr.mean(),
            'arousal_group_agreement': a_corr.mean(),
        })
    df = pd.DataFrame(rows).set_index('subject_id')
    df['mean_group_agreement'] = df[['valence_group_agreement', 'arousal_group_agreement']].mean(axis=1)
    return df


def plot_consistency_matrices(consistency_data: Dict, save_path: Path):
    mats = {'Valence': consistency_data.get('valence_matrix_corr'),
            'Arousal': consistency_data.get('arousal_matrix_corr')}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, (title, mat) in zip(axes, mats.items()):
        sns.heatmap(mat, ax=ax, cmap='vlag', center=0, annot=False)
        ax.set_title(f"{title} — pairwise correlation across subjects")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Saved consistency matrices to {save_path}")


def plot_subject_octant_heatmap(octant_table: pd.DataFrame, save_dir: Path):
    """Subjects x octants occupancy - similar rows = consistent behavior across the group,
    a row that stands out is a strong visual cue for exclusion review."""
    if octant_table.empty:
        return
    plt.figure(figsize=(8, max(4, 0.25 * len(octant_table))))
    sns.heatmap(octant_table * 100, cmap='viridis', cbar_kws={'label': '% of samples'})
    plt.title("Octant Occupancy by Subject (class imbalance & consistency)")
    plt.xlabel("Octant")
    plt.ylabel("Subject")
    plt.tight_layout()
    save_path = save_dir / "subject_octant_heatmap.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Saved subject octant heatmap to {save_path}")


def plot_missingness_heatmap(missingness_df: pd.DataFrame, save_dir: Path):
    if missingness_df.empty:
        logger.warning("No missingness data to plot.")
        return
    plt.figure(figsize=(6, max(4, 0.3 * len(missingness_df))))
    sns.heatmap(missingness_df, annot=True, fmt='.1f', cmap='Reds', vmin=0, vmax=100,
                cbar_kws={'label': '% missing'})
    plt.title("Missing Data (%) by Subject and Variable")
    plt.xlabel("Variable")
    plt.ylabel("Subject")
    plt.tight_layout()
    save_path = save_dir / "missingness_heatmap.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Saved missingness heatmap to {save_path}")


def plot_recording_duration_comparison(recording_summary: pd.DataFrame, save_path: Path):
    if recording_summary.empty or 'duration' not in recording_summary.columns:
        return
    med = recording_summary['duration'].median()
    deviates = (recording_summary['duration'] - med).abs() > DURATION_TOL_SEC
    order = recording_summary.sort_values('duration')

    plt.figure(figsize=(max(10, 0.3 * len(order)), 4.5))
    colors = ['indianred' if deviates.loc[i] else 'steelblue' for i in order.index]
    plt.bar(order.index, order['duration'], color=colors, edgecolor='black')
    plt.axhline(med, color='black', linestyle='--', linewidth=1, label=f'Median = {med:.0f}s')
    plt.xticks(rotation=90, fontsize=7)
    plt.ylabel("Duration (s)")
    plt.title("Recording Duration by Subject (red = deviates from median by >{:.0f}s)".format(DURATION_TOL_SEC))
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Saved recording duration comparison to {save_path}")


# ---------------------------
# QC flags: the "should we exclude this subject?" summary
# ---------------------------

def compute_qc_flags(stats_table: pd.DataFrame, recording_summary: pd.DataFrame,
                      missingness_df: pd.DataFrame, octant_table: pd.DataFrame,
                      agreement_df: pd.DataFrame) -> pd.DataFrame:
    qc = pd.DataFrame(index=stats_table.index)

    mean_missing = missingness_df[['valence', 'arousal']].mean(axis=1) if not missingness_df.empty else pd.Series(dtype=float)
    qc['pct_missing'] = mean_missing
    qc['flag_missing_data'] = mean_missing > MAX_MISSING_PCT

    if not recording_summary.empty and 'duration' in recording_summary.columns:
        med = recording_summary['duration'].median()
        qc['duration'] = recording_summary['duration']
        qc['flag_duration_deviation'] = (recording_summary['duration'] - med).abs() > DURATION_TOL_SEC
    else:
        qc['duration'] = np.nan
        qc['flag_duration_deviation'] = False

    if not octant_table.empty:
        qc['dominant_octant_prop'] = octant_table.max(axis=1)
        qc['flag_dominant_octant'] = qc['dominant_octant_prop'] > MAX_DOMINANT_OCTANT_PROP
    else:
        qc['dominant_octant_prop'] = np.nan
        qc['flag_dominant_octant'] = False

    if 'vector_length_mean' in stats_table.columns:
        qc['vector_length_mean'] = stats_table['vector_length_mean']
        qc['flag_low_range'] = stats_table['vector_length_mean'] < MIN_VECTOR_LENGTH_MEAN
    else:
        qc['vector_length_mean'] = np.nan
        qc['flag_low_range'] = False

    if not agreement_df.empty:
        qc['mean_group_agreement'] = agreement_df['mean_group_agreement']
        qc['flag_low_group_agreement'] = agreement_df['mean_group_agreement'] < MIN_GROUP_AGREEMENT
    else:
        qc['mean_group_agreement'] = np.nan
        qc['flag_low_group_agreement'] = False

    flag_cols = [c for c in qc.columns if c.startswith('flag_')]
    qc['total_flags'] = qc[flag_cols].sum(axis=1)
    return qc.sort_values('total_flags', ascending=False)


def plot_qc_flags_summary(qc: pd.DataFrame, save_path: Path):
    flag_cols = [c for c in qc.columns if c.startswith('flag_')]
    flagged = qc[qc['total_flags'] > 0]
    if flagged.empty:
        logger.info("No subjects triggered any QC flag; skipping QC summary plot.")
        return

    plt.figure(figsize=(7, max(3, 0.5 * len(flagged) + 1.5)))
    ax = sns.heatmap(flagged[flag_cols].astype(int), cmap='Reds', cbar=False, linewidths=0.5,
                      linecolor='white', annot=True, fmt='d')
    ax.set_yticklabels(flagged.index, rotation=0, fontsize=9)
    ax.set_xticklabels([c.replace('flag_', '').replace('_', ' ') for c in flag_cols],
                        rotation=30, ha='right', fontsize=9)
    plt.title("Subjects Flagged for Exclusion Review")
    plt.xlabel("")
    plt.ylabel("Subject")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Saved QC flags summary to {save_path}")


# ---------------------------
# High-level per-subject / group analysis
# ---------------------------

def subject_analyses(df: pd.DataFrame, subject_key: str, output_root: Path) -> None:
    save_dir = Path(output_root) / "exploration" / subject_key
    save_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"[{subject_key}] Saving subject-level outputs to {save_dir}")

    df.describe().to_csv(save_dir / "describe.csv")
    detailed_stats = compute_subject_descriptive_stats(df, subject_key)
    pd.DataFrame([detailed_stats]).to_csv(save_dir / "detailed_stats.csv", index=False)

    plot_circumplex_panel(df, save_dir / f"{subject_key}_circumplex.png", f"{subject_key} — Circumplex")
    plot_octant_occupancy(df, save_dir / f"{subject_key}_octant_occupancy.png", f"{subject_key} — Octant Occupancy")
    plot_subject_violin_halves(df, save_dir, subject_key)
    plot_subject_timeline(df, save_dir, subject_key)


def group_analyses(df: pd.DataFrame, all_labels: Dict[str, pd.DataFrame], output_root: Path) -> None:
    save_dir = Path(output_root) / "exploration" / "group_analysis"
    save_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"[GROUP] Saving group-level outputs to {save_dir}")

    plot_circumplex_panel(df, save_dir / "group_circumplex.png", "Group — Circumplex")
    plot_octant_occupancy(df, save_dir / "group_octant_occupancy.png", "Group — Octant Occupancy")
    plot_group_violin_by_subject(df, save_dir)
    plot_group_timeline(all_labels, save_dir)


# ---------------------------
# Main orchestrator
# ---------------------------

def run_explorative_analyses(
    output_root: Path,
    all_labels: Dict[str, pd.DataFrame],
):
    all_labels = {s: _ensure_angle_vector(df) for s, df in all_labels.items()}

    exploration_root = Path(output_root) / "exploration"
    overview_dir = exploration_root / "overview"
    overview_dir.mkdir(parents=True, exist_ok=True)

    combined_df = pd.concat(
        [df.assign(subject_id=s) for s, df in all_labels.items()],
        ignore_index=True
    )

    try:
        combined_df.describe().to_csv(overview_dir / "group_describe.csv")
    except Exception as e:
        logger.warning(f"Failed to save group describe: {e}")

    logger.info("Starting group-level analysis...")
    group_analyses(combined_df, all_labels, output_root)
    logger.info("Group-level analysis complete.")

    logger.info("Computing full per-subject statistics table...")
    stats_table = compute_full_stats_table(all_labels)
    stats_table.to_csv(overview_dir / "full_stats_table.csv")

    logger.info("Computing recording summary (duration, sampling rate)...")
    recording_summary = compute_recording_summary(all_labels)
    recording_summary.to_csv(overview_dir / "recording_summary.csv")
    plot_recording_duration_comparison(recording_summary, overview_dir / "recording_duration_comparison.png")

    logger.info("Computing missingness summary...")
    missingness_df = compute_missingness_table(all_labels)
    missingness_df.to_csv(overview_dir / "missingness_summary.csv")
    plot_missingness_heatmap(missingness_df, overview_dir)

    logger.info("Computing octant occupancy per subject...")
    octant_table = compute_subject_octant_table(all_labels)
    octant_table.to_csv(overview_dir / "subject_octant_table.csv")
    plot_subject_octant_heatmap(octant_table, overview_dir)

    logger.info("Calculating inter-subject consistency...")
    consistency_data = calculate_inter_subject_consistency(all_labels)
    agreement_df = compute_subject_group_agreement(consistency_data)
    agreement_df.to_csv(overview_dir / "subject_group_agreement.csv")

    consistency_summary = {
        'valence_mean_correlation': consistency_data['valence_consistency'],
        'arousal_mean_correlation': consistency_data['arousal_consistency'],
        'n_subjects': len(all_labels)
    }
    pd.DataFrame([consistency_summary]).to_csv(overview_dir / "consistency_summary.csv", index=False)
    plot_consistency_matrices(consistency_data, overview_dir / "consistency_matrices.png")

    logger.info("Computing QC / exclusion-candidate flags...")
    qc = compute_qc_flags(stats_table, recording_summary, missingness_df, octant_table, agreement_df)
    qc.to_csv(overview_dir / "qc_flags.csv")
    plot_qc_flags_summary(qc, overview_dir / "qc_flags_summary.png")

    logger.info("Selecting representative example subjects...")
    example_subjects = select_example_subjects(stats_table)
    if example_subjects:
        pd.DataFrame([{'label': k, 'subject_id': v} for k, v in example_subjects.items()]).to_csv(
            overview_dir / "example_subjects.csv", index=False
        )
        plot_example_subjects_overlay(all_labels, example_subjects, overview_dir)

    logger.info("Starting per-subject analyses...")
    for subject_id, subj_df in all_labels.items():
        logger.info(f"Analyzing subject {subject_id}...")
        subject_analyses(subj_df, subject_id, output_root)
    logger.info("Per-subject analyses complete.")

    final_summary = {
        'total_subjects': len(all_labels),
        'valence_consistency': consistency_data['valence_consistency'],
        'arousal_consistency': consistency_data['arousal_consistency'],
        'mean_pct_missing_valence': missingness_df['valence'].mean() if 'valence' in missingness_df else np.nan,
        'mean_pct_missing_arousal': missingness_df['arousal'].mean() if 'arousal' in missingness_df else np.nan,
        'n_subjects_flagged': int((qc['total_flags'] > 0).sum()),
    }
    pd.DataFrame([final_summary]).to_csv(overview_dir / "final_analysis_summary.csv", index=False)

    print("\n" + "=" * 60)
    print("EMOTION LABELING VALIDITY ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Total subjects analyzed: {len(all_labels)}")
    print(f"Inter-subject consistency (valence mean pairwise corr): {consistency_data['valence_consistency']:.3f}")
    print(f"Inter-subject consistency (arousal mean pairwise corr): {consistency_data['arousal_consistency']:.3f}")
    print("-" * 60)
    print("Mean % missing data across subjects:")
    for col in ['valence', 'arousal', 'angle', 'vector_length']:
        if col in missingness_df:
            print(f"  {col:>14}: {missingness_df[col].mean():.1f}%")
    print("-" * 60)
    print("Octant occupancy (group-level class imbalance):")
    group_props = compute_octant_proportions(combined_df)
    for label, val in group_props.items():
        print(f"  {label.replace(chr(10), ' '):>20}: {val * 100:5.1f}%")
    flagged = qc[qc['total_flags'] > 0]
    if not flagged.empty:
        print("-" * 60)
        print(f"{len(flagged)} subject(s) flagged for exclusion review (see qc_flags.csv):")
        for subj_id, row in flagged.iterrows():
            print(f"  {subj_id}: {int(row['total_flags'])} flag(s)")
    if example_subjects:
        print("-" * 60)
        print("Representative example subjects (by data completeness):")
        for label, subj_id in example_subjects.items():
            print(f"  {label:>16}: {subj_id}")
    print("=" * 60)

    logger.info(f"[INFO] Exploration complete. Outputs saved to {exploration_root}")
