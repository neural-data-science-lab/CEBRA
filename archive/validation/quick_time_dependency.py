'''
Brief examination if prediction of time indexing is possible based on the embedding
Q: Does the emebedding / the trajectory encode temporal information? 


- models predict time index from embedding

'''

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from tqdm import tqdm
import matplotlib.pyplot as plt


# Load embedding
file = r"C:\Users\bayer\MPI\embeddings\archive\results\sub-001\embedding_TNone-None_BNone_CHall.npy"
X = np.load(file)   # shape: (N_samples = 69000, N_dims = 3)
print(X[:5,:5])
# Time index target
y = np.arange(X.shape[0])



# #Plots
# fig, axes = plt.subplots(3, 1, figsize=(16, 12), layout='constrained')

# # Plot each component
# axes[0].scatter(y, X[:, 0])
# axes[0].set_title("component_1")

# axes[1].scatter(y, X[:, 1])
# axes[1].set_title("component_2")

# axes[2].scatter(y, X[:, 2])
# axes[2].set_title("component_3")
# axes[2].set_xlabel("time index")

# plt.show()



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
