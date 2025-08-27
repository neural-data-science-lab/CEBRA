import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import normalize
import hdbscan
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.learning.kmeans import RiemannianKMeans

# ------------------------------
# 1. Load your 3D embeddings
# ------------------------------
data = np.load(r"C:\Users\bayer\MPI\embeddings\archive\results\sub-001\embedding_TNone-None_BNone_CHall.npy")
embedding_len = data.shape[0]

# ------------------------------
# 2. Normalize embeddings to lie on the unit sphere
# ------------------------------
data_normalized = normalize(data, axis=1)

# ------------------------------
# 3. Riemannian KMeans clustering
# ------------------------------
sphere = Hypersphere(dim=2)  # 3D embeddings lie on S^3
n_clusters = 2  # adjust based on your data
rkmeans = RiemannianKMeans(n_clusters=n_clusters, space=sphere)
labels_rkmeans = rkmeans.fit_predict(data_normalized)

# ------------------------------
# 4. HDBSCAN clustering with cosine distance
# ------------------------------
from sklearn.metrics.pairwise import cosine_distances
import hdbscan

clusterer = hdbscan.HDBSCAN(
    min_cluster_size=20,
    min_samples=5,
    metric='euclidean',  # fast
    algorithm='best'
)
labels_hdbscan = clusterer.fit_predict(data_normalized)


# ------------------------------
# 5. Downsampling for plotting
# ------------------------------
n_max = 50000
step = max(1, embedding_len // n_max)
idx = np.arange(0, embedding_len, step)

data_plot = data[idx]  # original coordinates for plotting
labels_rkmeans_plot = labels_rkmeans[idx]
labels_hdbscan_plot = labels_hdbscan[idx]

# ------------------------------
# 6. Convert to DataFrame for Plotly
# ------------------------------
df = pd.DataFrame(data_plot, columns=['X', 'Y', 'Z'])
df['RiemannianKMeans'] = labels_rkmeans_plot.astype(str)
df['HDBSCAN'] = labels_hdbscan_plot.astype(str)

# ------------------------------
# 7. Plot Riemannian KMeans
# ------------------------------
fig_rkmeans = px.scatter_3d(
    df, x='X', y='Y', z='Z', color='RiemannianKMeans',
    color_discrete_sequence=px.colors.qualitative.Dark24,
    opacity=1, size_max=1
)
fig_rkmeans.update_layout(title=f'Riemannian KMeans Clustering ({n_clusters} clusters)', legend_title='Cluster')
fig_rkmeans.write_html("riemannian_kmeans_3d_plot.html")
print("Riemannian KMeans plot saved as riemannian_kmeans_3d_plot.html")

# ------------------------------
# 8. Plot HDBSCAN
# ------------------------------
fig_hdb = px.scatter_3d(
    df, x='X', y='Y', z='Z', color='HDBSCAN',
    color_discrete_sequence=px.colors.qualitative.Dark24,
    opacity=1, size_max=1
)
fig_hdb.update_layout(title='HDBSCAN Clustering (Cosine, 3D)', legend_title='Cluster')
fig_hdb.write_html("hdbscan_3d_plot.html")
print("HDBSCAN plot saved as hdbscan_3d_plot.html")
