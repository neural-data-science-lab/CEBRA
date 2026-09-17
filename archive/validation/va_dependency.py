"""
Research Goal: Determine whether CEBRA-derived EEG embeddings can predict emotional states (as continuous circular degree angles on the valence-arousal wheel) in a generalizable way across new subjects.
- subjects labeled on the edges of a square frame, splitting into separate valence/arousal would create artificial bimodal distributions at -1 and 1, losing the circular structure. 

"""
# --------------------------------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------------------------------
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from data import eeg_dataloader


# --------------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------------
BEH_DIR = Path(r"E:\Cris_Work\preproc")
EMB_DIR = Path(r"C:\Users\bayer\MPI\embeddings\archive\results")


# --------------------------------------------------------------------------------------------
# Custom Loss Functions for Circular Regression
# --------------------------------------------------------------------------------------------
class VectorLossWeighted(nn.Module):
    """Vector loss with proper per-sample weighting"""
    def __init__(self, reduction='none'):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred_degrees, target_degrees, weights=None):
        pred_rad = pred_degrees * torch.pi / 180.0
        target_rad = target_degrees * torch.pi / 180.0

        pred_vec = torch.stack((torch.cos(pred_rad), torch.sin(pred_rad)), dim=-1)
        target_vec = torch.stack((torch.cos(target_rad), torch.sin(target_rad)), dim=-1)

        # Per-sample loss
        loss = torch.sum((pred_vec - target_vec) ** 2, dim=-1)
        
        if weights is not None:
            loss = loss * weights.squeeze()
        
        if self.reduction == 'mean':
            return torch.mean(loss)
        return loss
    
class FocalAngularLoss(nn.Module):
    """
    Focal loss for circular regression - focuses on hard examples
    Reduces loss contribution from easy (well-predicted) samples
    """
    def __init__(self, gamma=2.0, reduction='none'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, pred_degrees, target_degrees, weights=None):
        pred_rad = pred_degrees * torch.pi / 180.0
        target_rad = target_degrees * torch.pi / 180.0
        
        # Cosine similarity (1 = perfect, -1 = opposite)
        cos_diff = torch.cos(pred_rad - target_rad)
        
        # Base loss: 1 - cos(diff), range [0, 2]
        base_loss = 1.0 - cos_diff
        
        # Focal term: (1 - accuracy)^gamma
        # When prediction is good (base_loss → 0), focal term → 0
        # When prediction is bad (base_loss → 2), focal term → 2^gamma
        focal_term = base_loss ** self.gamma
        loss = focal_term * base_loss
        
        if weights is not None:
            loss = loss * weights.squeeze()
        
        if self.reduction == 'mean':
            return torch.mean(loss)
        return loss

class CosineLossWeighted(nn.Module):
    """Cosine loss with proper per-sample weighting"""
    def __init__(self, reduction='none'):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred_degrees: torch.Tensor, target_degrees: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        pred_rad = pred_degrees * torch.pi / 180.0
        target_rad = target_degrees * torch.pi / 180.0
        cos_diff = torch.cos(pred_rad - target_rad)

        # Per-sample loss
        loss = 1.0 - cos_diff
        
        if weights is not None:
            loss = loss * weights.squeeze()
        
        if self.reduction == 'mean':
            return torch.mean(loss)
        return loss
    
# --------------------------------------------------------------------------------------------
# Model Definition
# --------------------------------------------------------------------------------------------
class LSTM_Degree(nn.Module):
    """LSTM network for degree regression (0-360°) on temporal sequences."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,  # LSTM Hidden state size
        num_layers: int = 2,    # Number of stacked LSTM layers
        dropout: float = 0.3,
        bidirectional: bool = True,
        output_activation: str = 'sigmoid',
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # 1. LSTM Layer: Processes the input sequence (batch, sequence_len, input_dim)
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,  # Input/Output Tensors are (batch, seq, feature)
            dropout=dropout,
            bidirectional=bidirectional
        )

        # Output dimensions change if bidirectional is True
        output_multiplier = 2 if bidirectional else 1
        
        # 2. Fully Connected (FC) Head: Regresses from the final hidden state to the angle
        # Note: We take the output of the final time step
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * output_multiplier, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)  # Final output layer
        )
        
        # Output scaling, copied from DegreeMLP
        if output_activation == 'sigmoid':
            self.output_activation = nn.Sigmoid()
            self.output_scale = 360.0
        elif output_activation == 'tanh':
            self.output_activation = nn.Tanh()
            self.output_scale = 180.0
            self.output_offset = 180.0
        else:
            self.output_activation = None
            self.output_scale = 1.0
            self.output_offset = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_len, input_dim).
        
        Returns:
            torch.Tensor: Predicted angles in degrees, shape (batch_size, 1).
        """
        # x shape: (batch_size, sequence_len, input_dim)
        
        # LSTM output: (output, (h_n, c_n))
        # output shape: (batch_size, sequence_len, num_directions * hidden_dim)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use the output from the last time step for prediction.
        # This assumes the label y corresponds to the last time point in the sequence.
        # last_output shape: (batch_size, num_directions * hidden_dim)
        last_output = lstm_out[:, -1, :] 
        
        # Pass through the FC head
        output = self.fc(last_output)

        # Apply output activation and scaling
        if self.output_activation:
            output = self.output_activation(output)
            if self.output_activation is nn.Sigmoid:
                return output * self.output_scale
            elif self.output_activation is nn.Tanh:
                return (output + 1) * self.output_scale
        
        # For 'None' activation, use modulo
        return torch.fmod(torch.abs(output), 360.0)

# --------------------------------------------------------------------------------------------
# Metrics for Circular Data
# --------------------------------------------------------------------------------------------
def mean_angular_error(preds: np.ndarray, target: np.ndarray) -> float:
    """
    Computes the Mean Angular Error (MAE) for circular data.

    Calculates the minimum angular difference between predicted and target angles,
    accounting for the circular wrap-around.

    Args:
        preds (np.ndarray): Predicted angles in degrees.
        target (np.ndarray): Target angles in degrees.

    Returns:
        float: The mean absolute angular difference.
    """
    diff = (preds - target + 180) % 360 - 180
    return np.mean(np.abs(diff))

def angular_rmse(preds: np.ndarray, target: np.ndarray) -> float:
    """
    Computes the Root Mean Squared Angular Error (RMSE) for circular data.

    Args:
        preds (np.ndarray): Predicted angles in degrees.
        target (np.ndarray): Target angles in degrees.

    Returns:
        float: The root mean squared angular difference.
    """
    diff = (preds - target + 180) % 360 - 180
    return np.sqrt(np.mean(diff**2))


def angular_accuracy(preds: np.ndarray, target: np.ndarray, tolerance: float = 10.0) -> float:
    """
    Computes the percentage of predictions within a given angular tolerance.

    Args:
        preds (np.ndarray): Predicted angles in degrees.
        target (np.ndarray): Target angles in degrees.
        tolerance (float): The acceptable error margin in degrees.

    Returns:
        float: The proportion of accurate predictions.
    """
    diff = (preds - target + 180) % 360 - 180
    return np.mean(np.abs(diff) <= tolerance)

# --------------------------------------------------------------------------------------------
# Data Augmentation for Circular Data
# --------------------------------------------------------------------------------------------
def augment_circular_data(X, y, target_samples_per_bin=500, n_bins=36):
    """
    Oversample underrepresented angular bins using interpolation
    """
    bins = np.linspace(0, 360, n_bins + 1)
    bin_indices = np.digitize(y, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    X_aug_list = [X]
    y_aug_list = [y]
    
    for bin_idx in range(n_bins):
        mask = bin_indices == bin_idx
        n_samples = mask.sum()
        
        if n_samples < target_samples_per_bin and n_samples > 0:
            # How many synthetic samples needed
            n_synthetic = target_samples_per_bin - n_samples
            
            # Get samples in this bin
            X_bin = X[mask]
            y_bin = y[mask]
            
            # Random interpolation between pairs
            for _ in range(n_synthetic):
                idx1, idx2 = np.random.choice(len(X_bin), 2, replace=True)
                alpha = np.random.uniform(0.3, 0.7)
                
                # Interpolate features
                X_synthetic = alpha * X_bin[idx1] + (1 - alpha) * X_bin[idx2]
                
                # Interpolate angles (circular)
                angle1_rad = np.radians(y_bin[idx1])
                angle2_rad = np.radians(y_bin[idx2])
                
                # Vector interpolation in 2D then back to angle
                x_interp = alpha * np.cos(angle1_rad) + (1 - alpha) * np.cos(angle2_rad)
                y_interp = alpha * np.sin(angle1_rad) + (1 - alpha) * np.sin(angle2_rad)
                angle_synthetic = (np.degrees(np.arctan2(y_interp, x_interp)) + 360) % 360
                
                X_aug_list.append(X_synthetic.reshape(1, -1))
                y_aug_list.append(np.array([angle_synthetic]))
    
    X_augmented = np.vstack(X_aug_list)
    y_augmented = np.concatenate(y_aug_list)
    
    return X_augmented, y_augmented

# --------------------------------------------------------------------------------------------
# Training Function
# --------------------------------------------------------------------------------------------
def train_circular_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    device: str = "cpu",
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    patience: int = 10,
    loss_function: str = "focal",  # Changed default to 'focal'
    use_weighted_loss: bool = True,
    use_augmentation: bool = True,  # NEW parameter
    track_progress: bool = True,
    model_params: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, float], Dict[str, List[float]], np.ndarray]:
    """
    IMPROVED: Trains with proper weighted loss, focal loss, and data augmentation.
    """
    if model_params is None:
        model_params = {}
    
    # --- NEW: Apply data augmentation ---
    if use_augmentation:
        print(f"[Augmentation] Original training size: {len(X_train)}")
        X_train, y_train = augment_circular_data(X_train, y_train, target_samples_per_bin=300)
        print(f"[Augmentation] Augmented training size: {len(X_train)}")
    
    # --- CHANGED: Initialize new loss functions ---
    if loss_function == "focal":
        criterion = FocalAngularLoss(gamma=2.0, reduction='none')
    elif loss_function == "vector":
        criterion = VectorLossWeighted(reduction='none')
    else:  # cosine
        criterion = CosineLossWeighted(reduction='none')
    
    model = DegreeMLP(input_dim=X_train.shape[1], **model_params).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)  # Changed to AdamW
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, verbose=False)

    X_tr = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_tr = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)

    # --- CHANGED: Compute weights AFTER augmentation ---
    weights_t = None
    if use_weighted_loss:
        sample_weights = compute_angular_weights(y_train)
        weights_t = torch.tensor(sample_weights, dtype=torch.float32).unsqueeze(1).to(device)

    history = {"train_loss": [], "val_loss": [], "val_mae": [], "val_rmse": [], "val_acc_10": []}
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(X_tr.size(0))
        epoch_loss = 0.0
        n_batches = 0

        iterator = range(0, X_tr.size(0), batch_size)
        if track_progress:
            iterator = tqdm(iterator, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)

        for i in iterator:
            idx = perm[i:i+batch_size]
            xb, yb = X_tr[idx], y_tr[idx]

            optimizer.zero_grad()
            outputs = model(xb)
            
            # --- CRITICAL FIX: Proper per-sample weighted loss ---
            if use_weighted_loss:
                wb = weights_t[idx]
                losses = criterion(outputs, yb, weights=wb)  # Per-sample losses
            else:
                losses = criterion(outputs, yb, weights=None)
            
            loss = torch.mean(losses)  # Reduce after weighting
            # --------------------------------------------------------
            
            loss.backward()
            
            # --- NEW: Gradient clipping ---
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

            if track_progress:
                iterator.set_postfix({"batch_loss": f"{loss.item():.4f}"})

        avg_train_loss = epoch_loss / n_batches

        # Validation (unweighted)
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t)
            val_losses = criterion(val_outputs, y_val_t, weights=None)
            val_loss = torch.mean(val_losses).item()
            
            val_preds = val_outputs.cpu().numpy().ravel()
            val_mae = mean_angular_error(val_preds, y_val)
            val_rmse = angular_rmse(val_preds, y_val)
            val_acc_10 = angular_accuracy(val_preds, y_val, tolerance=10)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["val_mae"].append(val_mae)
        history["val_rmse"].append(val_rmse)
        history["val_acc_10"].append(val_acc_10)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if track_progress:
                    print(f"Early stopping at epoch {epoch + 1}.")
                break

    if best_model_state:
        model.load_state_dict(best_model_state)

    # Final Test evaluation
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_t)
        test_preds = test_outputs.cpu().numpy().ravel()

    metrics = {
        "mae": mean_angular_error(test_preds, y_test),
        "rmse": angular_rmse(test_preds, y_test),
        "accuracy_10deg": angular_accuracy(test_preds, y_test, tolerance=10),
        "accuracy_20deg": angular_accuracy(test_preds, y_test, tolerance=20),  # NEW
        "accuracy_30deg": angular_accuracy(test_preds, y_test, tolerance=30),  # NEW
    }

    return metrics, history, test_preds

def compute_angular_weights(y_train: np.ndarray, n_bins: int = 36, smoothing: float = 1.0) -> np.ndarray:
    """
    Compute sample weights inversely proportional to angular bin frequency.
    
    Args:
        y_train: Training angles in degrees
        n_bins: Number of bins for histogram
        smoothing: Smoothing factor to prevent extreme weights
    
    Returns:
        Array of weights for each sample
    """
    bins = np.linspace(0, 360, n_bins + 1)
    hist, _ = np.histogram(y_train, bins=bins)
    
    # Inverse frequency with smoothing
    bin_weights = 1.0 / (hist + smoothing)
    
    # Normalize: average weight should be 1.0
    bin_weights = bin_weights / bin_weights.sum() * n_bins
    
    # Cap extreme weights
    bin_weights = np.clip(bin_weights, 0.1, 10.0)
    
    # Assign weight to each sample
    sample_weights = np.zeros(len(y_train))
    for i, angle in enumerate(y_train):
        bin_idx = int(angle // (360 / n_bins))
        bin_idx = min(bin_idx, n_bins - 1)
        sample_weights[i] = bin_weights[bin_idx]
    
    return sample_weights

# ------------------------------
# Load data
# ------------------------------

def get_subject_data(subj: str, beh_dir: Path, emb_dir: Path) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load and preprocess a single subject's EEG embeddings and degree data.
    
    The function aggregates the embeddings to a 1-second resolution to match
    the behavioral data and computes the degree angle from valence and arousal.

    Args:
        subj (str): The subject ID (e.g., 'sub-01').
        beh_dir (Path): Base directory for behavioral data.
        emb_dir (Path): Base directory for embedding data.

    Returns:
        Tuple[Optional[np.ndarray], Optional[np.ndarray]]: 
            - Aggregated EEG embeddings, or None if files are not found.
            - Aligned degree angles, or None if files are not found.
    """
    # Load embeddings
    emb_path = emb_dir / subj / "embedding_TNone-None_BNone_CHall.npy"
    embeddings = np.load(emb_path)
    embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=1e6, neginf=-1e6)
    # aggregate to 1-second windows (500 samples = 1 second)
    n_seconds = embeddings.shape[0] // 500
    embeddings_trimmed = embeddings[:n_seconds * 500]  # Trim to exact seconds
    embeddings_reshaped = embeddings_trimmed.reshape(n_seconds, 500, 3)
    embeddings_1sec = embeddings_reshaped.mean(axis=1)
    
    # Load behavioral labels
    beh_df = eeg_dataloader.load_behavioral_labels(beh_dir / subj)
    # Compute degrees from valence/arousal
    valence = beh_df["valence"].values
    arousal = beh_df["arousal"].values
    degrees, _ = colors.compute_angle_vector_length(valence, arousal)
    # Ensure degrees are in [0, 360] range
    degrees_aligned = degrees % 360
    # trim behavioral data to match the length of the aggregated embeddings
    degrees_aligned = degrees_aligned[:n_seconds]
    
    return embeddings_1sec, degrees_aligned

def create_sequences(X: np.ndarray, y: np.ndarray, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converts time-series data into overlapping sequences for LSTM input.
    
    Args:
        X: Feature array of shape (n_samples, input_dim).
        y: Label array of shape (n_samples,).
        seq_len: Length of the input sequence (time steps).
        
    Returns:
        X_seq: Sequences of shape (n_sequences, seq_len, input_dim).
        y_seq: Labels corresponding to the last time step of each sequence, shape (n_sequences,).
    """
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_len + 1):
        # The input sequence is X[i:i+seq_len]
        X_seq.append(X[i : i + seq_len])
        # The target is the label corresponding to the last time step in the sequence
        y_seq.append(y[i + seq_len - 1])
        
    return np.array(X_seq), np.array(y_seq)

def load_subjects_data(
    subject_list: List[str], beh_dir: Path, emb_dir: Path
) -> Tuple[np.ndarray, np.ndarray]:
    """Load and concatenate data from multiple subjects."""
    X_list, y_list = [], []
    for subj in subject_list:
        X, y = get_subject_data(subj, beh_dir, emb_dir)
        if X is not None:
            X_list.append(X)
            y_list.append(y)
    if not X_list:
        return np.array([]), np.array([])
    return np.concatenate(X_list), np.concatenate(y_list)

def stratified_train_val_split(X_train: np.ndarray, y_train: np.ndarray, n_bins: int = 36, val_size: float = 0.1, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Stratified split that preserves angular distribution in both train and val sets.
    
    Args:
        X_train: Feature matrix
        y_train: Angle labels
        n_bins: Number of angular bins for stratification
        val_size: Fraction for validation
        random_state: Random seed
    
    Returns:
        X_tr, X_val, y_tr, y_val
    """
    from sklearn.model_selection import StratifiedShuffleSplit
    
    # Bin angles for stratification
    bins = np.digitize(y_train, np.linspace(0, 360, n_bins + 1)) - 1
    bins = np.clip(bins, 0, n_bins - 1)
    
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)
    
    for train_idx, val_idx in sss.split(X_train, bins):
        return X_train[train_idx], X_train[val_idx], y_train[train_idx], y_train[val_idx]
    
# Leave-One-Subject-Out Cross-Validation to tests generalization to new subjects
def leave_one_subject_out_validation(
    subject_ids: List[str],
    beh_dir: Path,
    emb_dir: Path,
    **train_params
) -> Dict[str, Any]:
    """
    Performs Leave-One-Subject-Out (LOSO) cross-validation.

    The model is trained on data from all subjects except one, and then tested
    on the held-out subject. This process is repeated for every subject to
    evaluate the model's generalizability to new individuals.

    Args:
        subject_ids (List[str]): List of all subject IDs.
        beh_dir (Path): Base directory for behavioral data.
        emb_dir (Path): Base directory for embedding data.
        **train_params: Keyword arguments for the `train_circular_mlp` function.

    Returns:
        Dict[str, Any]: A dictionary containing aggregated metrics, per-subject
                        results, and all predictions/true labels.
    """
    all_metrics = []
    all_predictions = []
    all_true_labels = []
    all_subject_results = {}
    
    for test_subj in tqdm(subject_ids, desc="LOSO CV"):
        # Split subjects
        train_subjects = [s for s in subject_ids if s != test_subj]
        
        # Load training data (concatenate all training subjects)
        X_train_list, y_train_list = [], []
        for subj in train_subjects:
            X, y = get_subject_data(subj, beh_dir, emb_dir)
            if X is not None:
                X_train_list.append(X)
                y_train_list.append(y)
        
        X_train = np.concatenate(X_train_list)
        y_train = np.concatenate(y_train_list)
        
        # Load test data (single subject)
        X_test, y_test = get_subject_data(test_subj, beh_dir, emb_dir)
        
        if X_test is None:
            continue
        
        # Split training into train/val (90/10)
        X_tr, X_val, y_tr, y_val = stratified_train_val_split(
            X_train, y_train, n_bins=36, val_size=0.1, random_state=42
        )
        
        # Standardize (fit on training data)
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
        
        # Train model
        metrics, history, preds = train_circular_mlp(
            X_tr, y_tr, X_val, y_val, X_test, y_test, **train_params
        )
        
        all_metrics.append(metrics)
        all_predictions.append(preds)
        all_true_labels.append(y_test)
        all_subject_results[test_subj] = {
            "metrics": metrics,
            "predictions": preds,
            "y_true": y_test
        }
    
    # Aggregate metrics across subjects
    avg_metrics = {
        key: np.mean([m[key] for m in all_metrics])
        for key in all_metrics[0].keys()
    }
    std_metrics = {
        key: np.std([m[key] for m in all_metrics])
        for key in all_metrics[0].keys()
    }
    
    return {
        "avg_metrics": avg_metrics,
        "std_metrics": std_metrics,
        "subject_results": all_subject_results,
        "all_predictions": np.concatenate(all_predictions),
        "all_true": np.concatenate(all_true_labels)
    }

# --------------------------------------------------------------------------------------------
# Baselines and Data Analysis
# --------------------------------------------------------------------------------------------
# naive baseline (predict mean):minimum performance threshold
def compute_baseline_metrics(subject_ids, beh_dir, emb_dir):
    """Compute baseline: predict each subject's mean angle."""
    
    all_mae = []
    all_rmse = []
    
    for test_subj in subject_ids:
        # Get training subjects' mean angle
        train_subjects = [s for s in subject_ids if s != test_subj]
        train_angles = []
        
        for subj in train_subjects:
            _, y = get_subject_data(subj, beh_dir, emb_dir)
            if y is not None:
                train_angles.append(y)
        
        train_mean = np.concatenate(train_angles).mean()
        
        # Test subject
        _, y_test = get_subject_data(test_subj, beh_dir, emb_dir)
        if y_test is None:
            continue
            
        # Predict constant mean for all test points
        y_pred = np.full_like(y_test, train_mean)
        
        mae = mean_angular_error(y_pred, y_test)
        rmse = angular_rmse(y_pred, y_test)
        
        all_mae.append(mae)
        all_rmse.append(rmse)
    
    return {
        "baseline_mae": np.mean(all_mae),
        "baseline_rmse": np.mean(all_rmse)
    }

# per-subject mean (upper bound check)
def compute_subject_mean_baseline(subject_ids, beh_dir, emb_dir):
    """What if we predicted each subject's own mean? (Not valid for new subjects, but informative)"""
    
    all_mae = []
    
    for subj in subject_ids:
        _, y = get_subject_data(subj, beh_dir, emb_dir)
        if y is None:
            continue
        
        # Compute subject's circular mean
        y_rad = np.radians(y)
        mean_x = np.mean(np.cos(y_rad))
        mean_y = np.mean(np.sin(y_rad))
        subj_mean = (np.degrees(np.arctan2(mean_y, mean_x)) + 360) % 360
        
        # Predict this for all their timepoints
        y_pred = np.full_like(y, subj_mean)
        all_mae.append(mean_angular_error(y_pred, y))
    
    return {"within_subject_baseline_mae": np.mean(all_mae)}


# ------------------------------
# Visualizations
# ------------------------------
def plot_training_history(history: dict) -> None:
    """Plot training/validation metrics over epochs."""
    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(15, 4))
    
    # Loss plot
    plt.subplot(1, 4, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over epochs')
    plt.legend()

    # MAE plot
    plt.subplot(1, 4, 2)
    plt.plot(epochs, history['val_mae'], label='Val MAE', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('MAE (degrees)')
    plt.title('Validation MAE')
    plt.legend()

    # RMSE plot
    plt.subplot(1, 4, 3)
    plt.plot(epochs, history['val_rmse'], label='Val RMSE', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE (degrees)')
    plt.title('Validation RMSE')
    plt.legend()

    # Accuracy plot
    plt.subplot(1, 4, 4)
    plt.plot(epochs, history['val_acc_10'], label='Acc@10°', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy (±10°)')
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_circular_scatter(y_true, y_pred, title="Circular Regression"):
    """
    Plot predictions vs true angles on a unit circle.
    """
    y_true_rad = np.radians(y_true)
    y_pred_rad = np.radians(y_pred)

    plt.figure(figsize=(6,6))
    # Plot true angles
    plt.scatter(np.cos(y_true_rad), np.sin(y_true_rad), label="True", alpha=0.7)
    # Plot predicted angles
    plt.scatter(np.cos(y_pred_rad), np.sin(y_pred_rad), label="Predicted", alpha=0.7)
    
    plt.title(title)
    plt.xlabel("cos(angle)")
    plt.ylabel("sin(angle)")
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.show()


def plot_angular_error_histogram(y_true, y_pred, bins=36):
    """
    Plot histogram of angular errors (absolute difference on circle).
    """
    # Compute angular error
    diff = (y_pred - y_true + 180) % 360 - 180
    error = np.abs(diff)

    plt.figure(figsize=(6,4))
    plt.hist(error, bins=bins, color='skyblue', edgecolor='black')
    plt.title("Angular Error Distribution")
    plt.xlabel("Absolute Error (degrees)")
    plt.ylabel("Count")
    plt.grid(True)
    plt.show()

def plot_true_vs_pred(y_true, y_pred):
    """
    Plot true vs predicted angles with circular wrap-around.
    """
    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, alpha=0.7)
    plt.plot([0,360], [0,360], 'r--', label="y=x")
    plt.xlim(0,360)
    plt.ylim(0,360)
    plt.xlabel("True Angle (°)")
    plt.ylabel("Predicted Angle (°)")
    plt.title("True vs Predicted Angles")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_polar_comparison(y_true, y_pred):
    """
    Plot true vs predicted angles in polar coordinates.
    """
    y_true_rad = np.radians(y_true)
    y_pred_rad = np.radians(y_pred)

    plt.figure(figsize=(6,6))
    ax = plt.subplot(111, polar=True)
    ax.scatter(y_true_rad, np.ones_like(y_true_rad), label="True", alpha=0.7)
    ax.scatter(y_pred_rad, np.ones_like(y_pred_rad)*1.1, label="Predicted", alpha=0.7)
    ax.set_yticklabels([])
    ax.set_title("Circular Regression Polar Plot")
    ax.legend()
    plt.show()

def plot_loss(history):
    plt.figure(figsize=(6,4))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training History")
    plt.legend()
    plt.grid(True)
    plt.show()


# ---------------------------
# Main 
# ---------------------------

if __name__ == "__main__":

    # Find all subjects with behavioral data
    subject_ids = [p.name for p in BEH_DIR.glob("sub-*") if (BEH_DIR / p.name / "beh").exists()]
    print(f"[INFO] Found {len(subject_ids)} subjects")

    # Filter subjects that have embeddings
    available_subjects = [subj for subj in subject_ids if (EMB_DIR / subj / "embedding_TNone-None_BNone_CHall.npy").exists()]
    print(f"[INFO] {len(available_subjects)} subjects have embeddings available")

    # Set training parameters
    train_params = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "epochs": 150,  # Increased
        "batch_size": 128,  # Increased
        "lr": 1e-3,  # Increased
        "patience": 15,  # Increased
        "loss_function": "focal",  # Changed to focal
        "use_weighted_loss": True,
        "use_augmentation": True,  # NEW - Enable augmentation
        "track_progress": True,
        "model_params": {
            "hidden_dims": [512, 256, 128, 64],  # Deeper network
            "dropout": 0.4,  # Higher dropout
            "use_batch_norm": True,
            "output_activation": "sigmoid"
        }
    }

    print(f"[INFO] Using device: {train_params['device']}")
    print(f"[INFO] Using loss function: {train_params['loss_function']}")

    # Run the LOSO cross-validation
    loso_results = leave_one_subject_out_validation(available_subjects, BEH_DIR, EMB_DIR, **train_params)
    
    # Print overall results
    print("\n" + "="*50)
    print("Final LOSO Cross-Validation Results")
    print("="*50)
    
    print("\nMean Test Metrics (across all subjects):")
    for k, v in loso_results["avg_metrics"].items():
        std_v = loso_results["std_metrics"][k]
        print(f"  {k}: {v:.3f} ± {std_v:.3f}")

    # Compute and print baseline metrics for comparison
    print("\n" + "="*50)
    print("Baseline Performance")
    print("="*50)
    naive_baseline = compute_baseline_metrics(available_subjects, BEH_DIR, EMB_DIR)
    print("Naive (predicting mean of all others):")
    for k, v in naive_baseline.items():
        print(f"  {k}: {v:.3f}")
        
    subject_baseline = compute_subject_mean_baseline(available_subjects, BEH_DIR, EMB_DIR)
    print("\nWithin-Subject (predicting own mean):")
    for k, v in subject_baseline.items():
        print(f"  {k}: {v:.3f}")
        
    # Analyze label distribution
    print("\n" + "="*50)
    print("Data Label Distribution Analysis")
    print("="*50)
    _ = analyze_label_distribution(available_subjects, BEH_DIR, EMB_DIR)

    # Save results
    results_file = "loso_degree_regression_results.pkl"
    with open(results_file, "wb") as f:
        pickle.dump(loso_results, f)
    print(f"\n[INFO] Full results saved to {results_file}")

    # Visualization of aggregated predictions
    all_true = loso_results["all_true"]
    all_preds = loso_results["all_predictions"]
    
    print("\nGenerating final visualizations...")
    plot_true_vs_pred(all_true, all_preds)
    plot_angular_error_histogram(all_true, all_preds)
    plot_polar_comparison(all_true, all_preds)
    