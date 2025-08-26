'''
quick test for different label predictions

'''
from scipy.interpolate import interp1d
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from tqdm import tqdm

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# Load embedding
embedding = r"C:\Users\bayer\MPI\embeddings\archive\results\sub-001\embedding_TNone-None_BNone_CHall.npy"
X = np.load(embedding)  # shape (695935, 3)
print(X.shape)

# Load behavioral labels
beh_path = r"E:\Cris_Work\preproc\sub-001\beh\sub-001_task-AVR_beh_preprocessed.tsv"
beh = pd.read_csv(beh_path, sep="\t", header=0)
print(beh.head())
print(beh.shape)

# Create time arrays
# Behavioral data: 1 Hz 
t_beh = beh['timestamp'].values  
# Embedding data: 500 Hz 
t_embed = np.linspace(t_beh[0], t_beh[-1], X.shape[0])

# Interpolate behavioral columns onto embedding timeline
beh_interp = pd.DataFrame()
for col in ['valence', 'arousal', 'flubber_frequency', 'flubber_amplitude']:
    f = interp1d(t_beh, beh[col].values, kind='linear', bounds_error=False, fill_value='extrapolate')
    beh_interp[col] = f(t_embed)

print(beh_interp.shape) 
y = beh_interp['flubber_amplitude'].to_numpy()


# Scale features
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

# Cross-validation
cv = KFold(n_splits=5, shuffle=True, random_state=42)

def cross_val_with_progress(model, X, y, cv, scale_y=False):
    scores = []
    for fold, (train_idx, test_idx) in enumerate(
        tqdm(cv.split(X), total=cv.get_n_splits(), desc=f"{model.__class__.__name__} CV")
    ):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        if scale_y:
            # Scale target
            y_mean, y_std = y_train.mean(), y_train.std()
            y_train_scaled = (y_train - y_mean) / y_std
            model.fit(X_train, y_train_scaled)
            y_pred_scaled = model.predict(X_test)
            y_pred = y_pred_scaled * y_std + y_mean  # inverse scale
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        
        score = r2_score(y_test, y_pred)
        scores.append(score)
        tqdm.write(f"Fold {fold+1} R^2: {score:.4f}")
    return np.array(scores)

# Convert target to categorical based on median
y_cat = (beh_interp['flubber_amplitude'] > np.median(beh_interp['flubber_amplitude'])).astype(int)

# Scale features if not already done
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

# Quick test with different classifiers
cv = KFold(n_splits=5, shuffle=True, random_state=42)

classifiers = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel='rbf', gamma='scale')
}

for name, clf in classifiers.items():
    scores = cross_val_score(clf, X_scaled, y_cat, cv=cv, scoring='accuracy')
    print(f"{name} Accuracy per fold: {scores}")
    print(f"{name} Mean Accuracy: {scores.mean():.4f}\n")

# Linear Regression
lr = LinearRegression()
lr_scores = cross_val_with_progress(lr, X_scaled, y, cv, scale_y=False)
print(f"\nLinear Regression Mean R^2: {lr_scores.mean():.4f}")
print(f"R^2 per fold: {lr_scores}\n")


# MLP Regressor
mlp = MLPRegressor(hidden_layer_sizes=(128, 64, 32), max_iter=5000, random_state=42, early_stopping=True)
mlp_scores = cross_val_with_progress(mlp, X_scaled, y, cv, scale_y=True)
print(f"\nMLP Regressor Mean R^2: {mlp_scores.mean():.4f}")
print(f"R^2 per fold: {mlp_scores}\n")

# SVR
subset_idx = np.random.choice(X_scaled.shape[0], size=10000, replace=False)
X_sub, y_sub = X_scaled[subset_idx], y[subset_idx]
svr = SVR(kernel='rbf', C=1.0, gamma='scale')
svr_scores = cross_val_with_progress(svr, X_sub, y_sub, cv=KFold(n_splits=5, shuffle=True))

print(f"\nSVR Mean R^2: {svr_scores.mean():.4f}")
print(f"R^2 per fold: {svr_scores}\n")
