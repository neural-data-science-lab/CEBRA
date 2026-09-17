"""
Research Goal: Determine whether CEBRA-derived EEG embeddings can predict emotional states 
(as continuous circular degree angles on the valence-arousal wheel) in a generalizable way 
across new subjects using an LSTM for sequence modeling.
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
# NOTE: Assuming `data.eeg_dataloader` and `colors.compute_angle_vector_length` 
# are available in the local environment, as they were referenced in the original script.
# For a runnable script, these dependencies would need to be defined or imported.
# For this rewrite, I will mock the necessary missing functions/classes.
try:
    from data import eeg_dataloader
    from eeg.perprocessing import compute_angle_vector_length
except ImportError:
    # MOCKING MISSING DEPENDENCIES FOR CODE COMPLETENESS
    print("[WARNING] Mocking missing dependencies: eeg_dataloader and colors.")
    class MockDataloader:
        @staticmethod
        def load_behavioral_labels(path):
            # Mock behavioral data for demonstration
            n_samples = 1000
            valence = np.cos(np.linspace(0, 4*np.pi, n_samples)) * 0.8
            arousal = np.sin(np.linspace(0, 4*np.pi, n_samples)) * 0.8
            return {"valence": valence, "arousal": arousal}
    eeg_dataloader = MockDataloader()
    
    class MockColors:
        @staticmethod
        def compute_angle_vector_length(valence, arousal):
            # Compute angle in degrees (0-360) and vector length
            degrees = np.degrees(np.arctan2(arousal, valence))
            degrees = (degrees + 360) % 360 # Ensure 0-360 range
            vector_length = np.sqrt(valence**2 + arousal**2)
            return degrees, vector_length
    colors = MockColors()

# --------------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------------
# NOTE: Update these paths to valid locations in your environment
BEH_DIR = Path(r"E:\Cris_Work\preproc")
EMB_DIR = Path(r"C:\Users\bayer\MPI\embeddings\archive\results")

# --- NEW/UPDATED CONSTANTS FOR LSTM ---
SEQUENCE_LENGTH = 10 # 10 seconds of history to predict the current angle
EMBEDDING_DIM = 3    # Based on the embedding file processing in get_subject_data

# --------------------------------------------------------------------------------------------
# Custom Loss Functions for Circular Regression (Kept as is)
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
# Model Definition (LSTM_Degree is now the primary model)
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
        
        # Output scaling
        if output_activation == 'sigmoid':
            self.output_activation = nn.Sigmoid()
            self.output_scale = 360.0
            self.output_offset = 0.0 # Adjusted for sigmoid
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
        # last_output shape: (batch_size, num_directions * hidden_dim)
        last_output = lstm_out[:, -1, :] 
        
        # Pass through the FC head
        output = self.fc(last_output)

        # Apply output activation and scaling
        if self.output_activation:
            output = self.output_activation(output)
            if isinstance(self.output_activation, nn.Sigmoid):
                return output * self.output_scale
            elif isinstance(self.output_activation, nn.Tanh):
                return (output + 1) * self.output_scale + self.output_offset
        
        # For 'None' activation, use modulo
        return torch.fmod(torch.abs(output), 360.0)

# --------------------------------------------------------------------------------------------
# Metrics for Circular Data (Kept as is)
# --------------------------------------------------------------------------------------------
def mean_angular_error(preds: np.ndarray, target: np.ndarray) -> float:
    """Computes the Mean Angular Error (MAE) for circular data."""
    diff = (preds - target + 180) % 360 - 180
    return np.mean(np.abs(diff))

def angular_rmse(preds: np.ndarray, target: np.ndarray) -> float:
    """Computes the Root Mean Squared Angular Error (RMSE) for circular data."""
    diff = (preds - target + 180) % 360 - 180
    return np.sqrt(np.mean(diff**2))

def angular_accuracy(preds: np.ndarray, target: np.ndarray, tolerance: float = 10.0) -> float:
    """Computes the percentage of predictions within a given angular tolerance."""
    diff = (preds - target + 180) % 360 - 180
    return np.mean(np.abs(diff) <= tolerance)

# --------------------------------------------------------------------------------------------
# Data Augmentation for Circular Data (Kept as is)
# --------------------------------------------------------------------------------------------
def augment_circular_data(X, y, target_samples_per_bin=500, n_bins=36):
    """
    Oversample underrepresented angular bins using interpolation
    NOTE: This augmentation is designed for non-sequential data (like MLP).
    For LSTM, interpolation needs to be applied to the sequence itself (X).
    Since the original was designed for an MLP, we will keep the original augmentation
    logic and apply it to the flattened sequence data before converting back to sequences.
    This is an imperfect but pragmatic approach given the original's intent.
    """
    
    # Flatten sequences for binning/interpolation if X is (N, T, D)
    if X.ndim == 3:
        # For sequence data, we'll only augment the *labels* for binning
        X_flat = X[:, -1, :] # Use last time step's feature vector for interpolation
    else:
        X_flat = X

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
            X_bin = X[mask] # Keep as sequence if it was one
            X_flat_bin = X_flat[mask]
            y_bin = y[mask]
            
            # Random interpolation between pairs
            for _ in range(n_synthetic):
                idx1, idx2 = np.random.choice(len(X_bin), 2, replace=True)
                alpha = np.random.uniform(0.3, 0.7)
                
                # Interpolate features (on the flattened vector for consistency with MLP augmentation)
                X_synthetic_flat = alpha * X_flat_bin[idx1] + (1 - alpha) * X_flat_bin[idx2]
                
                # Interpolate angles (circular)
                angle1_rad = np.radians(y_bin[idx1])
                angle2_rad = np.radians(y_bin[idx2])
                
                # Vector interpolation in 2D then back to angle
                x_interp = alpha * np.cos(angle1_rad) + (1 - alpha) * np.cos(angle2_rad)
                y_interp = alpha * np.sin(angle1_rad) + (1 - alpha) * np.sin(angle2_rad)
                angle_synthetic = (np.degrees(np.arctan2(y_interp, x_interp)) + 360) % 360
                
                # IMPORTANT: For LSTM, we must create a synthetic *sequence*. 
                # Since we don't know the temporal dependency, we'll repeat the interpolated 
                # final feature vector to create a placeholder sequence.
                # A proper time-series augmentation would use sequence-level techniques.
                if X.ndim == 3:
                    T = X.shape[1]
                    X_synthetic = np.repeat(X_synthetic_flat[np.newaxis, :], T, axis=0)
                    X_aug_list.append(X_synthetic[np.newaxis, ...])
                else:
                    X_aug_list.append(X_synthetic_flat.reshape(1, -1))

                y_aug_list.append(np.array([angle_synthetic]))
    
    X_augmented = np.concatenate(X_aug_list, axis=0)
    y_augmented = np.concatenate(y_aug_list)
    
    # If the original input was 2D (which it won't be in the final LSTM pipeline), ensure output matches
    if X.ndim == 2 and X_augmented.ndim == 3:
        # This case should not happen if SEQUENCE_LENGTH is > 1
        X_augmented = X_augmented[:, -1, :]
    
    return X_augmented, y_augmented


# --------------------------------------------------------------------------------------------
# Training Function (Modified for LSTM)
# --------------------------------------------------------------------------------------------
def train_circular_lstm( # RENAMED from train_circular_mlp
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
    loss_function: str = "focal",
    use_weighted_loss: bool = True,
    use_augmentation: bool = True,
    track_progress: bool = True,
    model_params: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, float], Dict[str, List[float]], np.ndarray]:
    """
    Trains an LSTM with proper weighted loss, focal loss, and data augmentation.
    """
    if model_params is None:
        model_params = {}
    
    # --- Check for correct input shape (N, T, D) ---
    assert X_train.ndim == 3, f"X_train must be 3D (N, T, D), got {X_train.ndim}D"
    
    # --- Apply data augmentation ---
    # NOTE: Augmentation applied to sequences.
    if use_augmentation:
        print(f"[Augmentation] Original training size: {len(X_train)}")
        X_train, y_train = augment_circular_data(X_train, y_train, target_samples_per_bin=300)
        print(f"[Augmentation] Augmented training size: {len(X_train)}")
    
    # --- Initialize loss functions ---
    if loss_function == "focal":
        criterion = FocalAngularLoss(gamma=2.0, reduction='none')
    elif loss_function == "vector":
        criterion = VectorLossWeighted(reduction='none')
    else:  # cosine
        criterion = CosineLossWeighted(reduction='none')
    
    # --- Initialize LSTM Model ---
    # Input dim is the last dimension of the sequence array (N, T, D) -> D
    input_dim = X_train.shape[-1]
    model = LSTM_Degree(input_dim=input_dim, **model_params).to(device) # USING LSTM_Degree
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, verbose=False)

    # Convert to Tensors
    X_tr = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_tr = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)

    # --- Compute weights AFTER augmentation ---
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
            
            # Proper per-sample weighted loss
            if use_weighted_loss:
                wb = weights_t[idx]
                losses = criterion(outputs, yb, weights=wb)  # Per-sample losses
            else:
                losses = criterion(outputs, yb, weights=None)
            
            loss = torch.mean(losses)  # Reduce after weighting
            
            loss.backward()
            
            # Gradient clipping
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
        "accuracy_20deg": angular_accuracy(test_preds, y_test, tolerance=20),
        "accuracy_30deg": angular_accuracy(test_preds, y_test, tolerance=30),
    }

    return metrics, history, test_preds

def compute_angular_weights(y_train: np.ndarray, n_bins: int = 36, smoothing: float = 1.0) -> np.ndarray:
    """
    Compute sample weights inversely proportional to angular bin frequency.
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
        # Determine the bin index
        bin_idx = np.digitize(angle, bins) - 1
        bin_idx = min(bin_idx, n_bins - 1)
        sample_weights[i] = bin_weights[bin_idx]
    
    return sample_weights

# ------------------------------
# Load data (Modified for sequence generation)
# ------------------------------

def get_subject_data(subj: str, beh_dir: Path, emb_dir: Path) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load and preprocess a single subject's EEG embeddings and degree data.
    
    Aggregates embeddings to 1-second resolution and computes the degree angle.
    
    NOTE: This is the raw 1-second data. The sequences will be created later.
    """
    try:
        # Load embeddings
        emb_path = emb_dir / subj / "embedding_TNone-None_BNone_CHall.npy"
        # Use existing data if mock is active, otherwise attempt load
        if 'MockColors' in globals() and not emb_path.exists():
            print(f"[MOCK] Creating mock data for {subj}")
            embeddings = np.random.rand(5000, EMBEDDING_DIM) # 10 seconds of data (5000 samples)
        else:
            embeddings = np.load(emb_path)
        
        embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # aggregate to 1-second windows (500 samples = 1 second)
        SAMPLES_PER_SECOND = 500
        n_seconds = embeddings.shape[0] // SAMPLES_PER_SECOND
        embeddings_trimmed = embeddings[:n_seconds * SAMPLES_PER_SECOND]
        embeddings_reshaped = embeddings_trimmed.reshape(n_seconds, SAMPLES_PER_SECOND, embeddings.shape[1])
        embeddings_1sec = embeddings_reshaped.mean(axis=1)
        
        # Load behavioral labels
        beh_df = eeg_dataloader.load_behavioral_labels(beh_dir / subj)
        # Compute degrees from valence/arousal
        valence = beh_df["valence"].values
        arousal = beh_df["arousal"].values
        # Ensure V/A arrays are long enough to cover n_seconds
        if len(valence) < n_seconds:
            print(f"[WARNING] V/A data for {subj} too short. Skipping.")
            return None, None
            
        degrees, _ = compute_angle_vector_length(valence[:n_seconds], arousal[:n_seconds])
        
        # Ensure degrees are in [0, 360] range
        degrees_aligned = degrees % 360
        # trim behavioral data to match the length of the aggregated embeddings
        degrees_aligned = degrees_aligned[:n_seconds]
        
        return embeddings_1sec, degrees_aligned
    
    except FileNotFoundError:
        print(f"[ERROR] Could not find necessary files for subject {subj}. Skipping.")
        return None, None
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred for subject {subj}: {e}. Skipping.")
        return None, None

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
    subject_list: List[str], beh_dir: Path, emb_dir: Path, seq_len: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Load, concatenate, and sequence data from multiple subjects."""
    X_seq_list, y_seq_list = [], []
    for subj in subject_list:
        X, y = get_subject_data(subj, beh_dir, emb_dir)
        if X is not None:
            # Create sequences for this subject
            X_seq, y_seq = create_sequences(X, y, seq_len)
            if X_seq.size > 0:
                X_seq_list.append(X_seq)
                y_seq_list.append(y_seq)
            else:
                print(f"[WARNING] Data for {subj} too short to create sequences of length {seq_len}. Skipping.")
    
    if not X_seq_list:
        return np.array([]), np.array([])
    return np.concatenate(X_seq_list), np.concatenate(y_seq_list)

def stratified_train_val_split(X_train: np.ndarray, y_train: np.ndarray, n_bins: int = 36, val_size: float = 0.1, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Stratified split that preserves angular distribution in both train and val sets.
    """
    # Bin angles for stratification
    bins = np.digitize(y_train, np.linspace(0, 360, n_bins + 1)) - 1
    bins = np.clip(bins, 0, n_bins - 1)
    
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)
    
    for train_idx, val_idx in sss.split(X_train, bins):
        return X_train[train_idx], X_train[val_idx], y_train[train_idx], y_train[val_idx]
    
# Leave-One-Subject-Out Cross-Validation (Modified for sequence generation)
def leave_one_subject_out_validation(
    subject_ids: List[str],
    beh_dir: Path,
    emb_dir: Path,
    seq_len: int, # NEW parameter
    **train_params
) -> Dict[str, Any]:
    """
    Performs Leave-One-Subject-Out (LOSO) cross-validation for LSTM.
    """
    all_metrics = []
    all_predictions = []
    all_true_labels = []
    all_subject_results = {}
    
    for test_subj in tqdm(subject_ids, desc="LOSO CV"):
        # Split subjects
        train_subjects = [s for s in subject_ids if s != test_subj]
        
        # Load and sequence training data (concatenate all training subjects)
        X_train, y_train = load_subjects_data(train_subjects, beh_dir, emb_dir, seq_len)
        
        # Load and sequence test data (single subject)
        X_test_raw, y_test_raw = get_subject_data(test_subj, beh_dir, emb_dir)
        
        if X_train.size == 0 or X_test_raw is None:
            print(f"[WARNING] Skipping {test_subj} due to insufficient data.")
            continue
            
        X_test, y_test = create_sequences(X_test_raw, y_test_raw, seq_len)
        
        if X_test.size == 0:
            print(f"[WARNING] Test subject {test_subj} data too short for sequences. Skipping.")
            continue
        
        # Split training into train/val (90/10)
        X_tr, X_val, y_tr, y_val = stratified_train_val_split(
            X_train, y_train, n_bins=36, val_size=0.1, random_state=42
        )
        
        # Standardize (fit on training data)
        # Scaler is 2D, so reshape 3D to 2D for fit/transform
        scaler = StandardScaler()
        
        # Flatten and fit/transform training
        N_tr, T, D = X_tr.shape
        X_tr_flat = X_tr.reshape(-1, D)
        X_tr_flat = scaler.fit_transform(X_tr_flat)
        X_tr = X_tr_flat.reshape(N_tr, T, D)
        
        # Flatten and transform validation
        N_val, _, _ = X_val.shape
        X_val_flat = X_val.reshape(-1, D)
        X_val_flat = scaler.transform(X_val_flat)
        X_val = X_val_flat.reshape(N_val, T, D)
        
        # Flatten and transform test
        N_test, _, _ = X_test.shape
        X_test_flat = X_test.reshape(-1, D)
        X_test_flat = scaler.transform(X_test_flat)
        X_test = X_test_flat.reshape(N_test, T, D)
        
        # Train model (using the renamed function)
        metrics, history, preds = train_circular_lstm(
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
    if not all_metrics:
        print("[ERROR] No subjects were processed successfully.")
        return {}
        
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
# Baselines and Data Analysis (Modified to handle sequence data implicitly)
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
            _, y = get_subject_data(subj, beh_dir, emb_dir) # Use raw data
            if y is not None:
                train_angles.append(y)
        
        if not train_angles:
            continue
            
        # Compute circular mean of training data
        all_train_y = np.concatenate(train_angles)
        y_rad = np.radians(all_train_y)
        mean_x = np.mean(np.cos(y_rad))
        mean_y = np.mean(np.sin(y_rad))
        train_mean = (np.degrees(np.arctan2(mean_y, mean_x)) + 360) % 360
        
        # Test subject
        _, y_test = get_subject_data(test_subj, beh_dir, emb_dir) # Use raw data
        if y_test is None:
            continue
            
        # Predict constant mean for all test points
        y_pred = np.full_like(y_test, train_mean)
        
        mae = mean_angular_error(y_pred, y_test)
        rmse = angular_rmse(y_pred, y_test)
        
        all_mae.append(mae)
        all_rmse.append(rmse)
    
    if not all_mae: return {}
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
        
    if not all_mae: return {}
    return {"within_subject_baseline_mae": np.mean(all_mae)}

def analyze_label_distribution(subject_ids, beh_dir, emb_dir):
    """Analyze label distribution across all subjects."""
    all_angles = []
    for subj in subject_ids:
        _, y = get_subject_data(subj, beh_dir, emb_dir)
        if y is not None:
            all_angles.append(y)
            
    if not all_angles:
        print("[ANALYSIS] No data found for distribution analysis.")
        return None
        
    all_angles = np.concatenate(all_angles)
    
    plt.figure(figsize=(6,4))
    plt.hist(all_angles, bins=36, range=(0, 360), color='teal', edgecolor='black', alpha=0.7)
    plt.title("Overall Angle Label Distribution (0-360°)")
    plt.xlabel("Angle (Degrees)")
    plt.ylabel("Frequency")
    plt.xticks(np.linspace(0, 360, 9))
    plt.grid(axis='y', alpha=0.5)
    plt.show()

    y_rad = np.radians(all_angles)
    mean_x = np.mean(np.cos(y_rad))
    mean_y = np.mean(np.sin(y_rad))
    circ_mean = (np.degrees(np.arctan2(mean_y, mean_x)) + 360) % 360
    
    print(f"Total samples: {len(all_angles)}")
    print(f"Circular Mean Angle: {circ_mean:.2f}°")
    
    return all_angles


# ------------------------------
# Visualizations (Kept as is)
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
    """Plot predictions vs true angles on a unit circle."""
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
    """Plot histogram of angular errors (absolute difference on circle)."""
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
    """Plot true vs predicted angles with circular wrap-around."""
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
    """Plot true vs predicted angles in polar coordinates."""
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
# Main (Modified for LSTM)
# ---------------------------

if __name__ == "__main__":

    # Find all subjects with behavioral data (using mock check if necessary)
    subject_ids = [p.name for p in BEH_DIR.glob("sub-*") if (BEH_DIR / p.name / "beh").exists()]
    
    # MOCK SUBJECTS if no real data is found (for testing the code structure)
    if not subject_ids:
        subject_ids = [f"sub-{i:02d}" for i in range(1, 11)] # 10 mock subjects
        print(f"[WARNING] No real subjects found, using {len(subject_ids)} mock subjects.")
    
    print(f"[INFO] Found {len(subject_ids)} potential subjects")

    # Filter subjects that have embeddings (mock check included)
    available_subjects = [subj for subj in subject_ids if (EMB_DIR / subj / "embedding_TNone-None_BNone_CHall.npy").exists() or 'MockColors' in globals()]
    print(f"[INFO] {len(available_subjects)} subjects have embeddings available/mocked")
    
    if not available_subjects:
        print("[FATAL] No available subjects to run experiment. Exiting.")
    else:

        # Set training parameters
        train_params = {
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "epochs": 150,
            "batch_size": 128,
            "lr": 1e-3,
            "patience": 15,
            "loss_function": "focal",
            "use_weighted_loss": True,
            "use_augmentation": True,
            "track_progress": True,
            "model_params": {
                # NEW LSTM-specific parameters
                "hidden_dim": 128,
                "num_layers": 2,
                "dropout": 0.4,
                "bidirectional": True,
                "output_activation": "sigmoid"
            }
        }
    
        # NEW Sequence Length Parameter
        SEQ_LEN = SEQUENCE_LENGTH # Use the constant defined above (10)

        print(f"[INFO] Using device: {train_params['device']}")
        print(f"[INFO] Using loss function: {train_params['loss_function']}")
        print(f"[INFO] Using Sequence Length (T): {SEQ_LEN} seconds")

        # Run the LOSO cross-validation
        # PASS SEQUENCE_LENGTH to LOSO function
        loso_results = leave_one_subject_out_validation(
            available_subjects, BEH_DIR, EMB_DIR, SEQ_LEN, **train_params
        )
        
        if loso_results:
            # Print overall results
            print("\n" + "="*50)
            print(f"Final LOSO Cross-Validation Results (LSTM T={SEQ_LEN})")
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
            results_file = f"loso_lstm_degree_regression_T{SEQ_LEN}_results.pkl"
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