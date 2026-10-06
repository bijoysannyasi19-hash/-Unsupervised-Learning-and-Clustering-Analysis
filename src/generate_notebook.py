import nbformat as nbf

def create_notebook(output_path):
    nb = nbf.v4.new_notebook()
    
    cells = []
    
    # Markdown cell
    cells.append(nbf.v4.new_markdown_cell("""# Week 3 Task: Unsupervised Learning and Clustering Analysis
This notebook covers the end-to-end process of clustering wholesale customers based on their annual spending. 
It follows the exact same logic as the Python scripts in `/src`."""))

    # Imports
    cells.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import warnings
warnings.filterwarnings('ignore')

plt.style.use('default')
sns.set_theme(style="whitegrid")"""))

    # Load Data
    cells.append(nbf.v4.new_markdown_cell("## 1. Load Data\nLoading the UCI Wholesale Customers dataset."))
    cells.append(nbf.v4.new_code_cell("""df = pd.read_csv('../data/wholesale_customers.csv')
print(df.head())
print(df.info())
print(df.describe())"""))

    # Preprocessing
    cells.append(nbf.v4.new_markdown_cell("## 2. Preprocessing\n- Remove duplicates\n- Exclude `Channel` and `Region`\n- Apply `log1p` to fix heavy right-skew\n- Scale using `StandardScaler`"))
    cells.append(nbf.v4.new_code_cell("""df_clean = df.drop_duplicates().copy()
labels = df_clean[['Channel', 'Region']].copy()
features = df_clean.drop(['Channel', 'Region'], axis=1)

original_features = features.copy()
features_log = np.log1p(features)
scaler = StandardScaler()
features_scaled = pd.DataFrame(scaler.fit_transform(features_log), columns=features.columns, index=features.index)

print(features_scaled.head())"""))

    # Determine k
    cells.append(nbf.v4.new_markdown_cell("## 3. Determine Optimal k (K-Means)\nUsing Elbow method, Silhouette Score, Calinski-Harabasz, and Davies-Bouldin."))
    cells.append(nbf.v4.new_code_cell("""k_range = range(2, 11)
results = []
for k in k_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    labels_k = kmeans.fit_predict(features_scaled)
    results.append({
        'k': k,
        'inertia': kmeans.inertia_,
        'silhouette': silhouette_score(features_scaled, labels_k),
        'calinski_harabasz': calinski_harabasz_score(features_scaled, labels_k),
        'davies_bouldin': davies_bouldin_score(features_scaled, labels_k)
    })
k_df = pd.DataFrame(results)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes[0, 0].plot(k_df['k'], k_df['inertia'], marker='o')
axes[0, 0].set_title('Elbow Method (Inertia)')
axes[0, 1].plot(k_df['k'], k_df['silhouette'], marker='o', color='orange')
axes[0, 1].set_title('Silhouette Score')
axes[1, 0].plot(k_df['k'], k_df['calinski_harabasz'], marker='o', color='green')
axes[1, 0].set_title('Calinski-Harabasz Score')
axes[1, 1].plot(k_df['k'], k_df['davies_bouldin'], marker='o', color='red')
axes[1, 1].set_title('Davies-Bouldin Score')
plt.tight_layout()
plt.show()"""))

    # Final Clustering
    cells.append(nbf.v4.new_markdown_cell("## 4. Run Final Clustering Models\nI chose `k=4` based on metrics and interpretability."))
    cells.append(nbf.v4.new_code_cell("""k_opt = 4
kmeans = KMeans(n_clusters=k_opt, init='k-means++', n_init=10, random_state=42)
kmeans_labels = kmeans.fit_predict(features_scaled)

hc = AgglomerativeClustering(n_clusters=k_opt, linkage='ward')
hc_labels = hc.fit_predict(features_scaled)

gmm = GaussianMixture(n_components=k_opt, covariance_type='full', random_state=42)
gmm_labels = gmm.fit_predict(features_scaled)"""))

    # Cluster Profiles
    cells.append(nbf.v4.new_markdown_cell("## 5. Cluster Profiles\nAnalyzing the average original spending per cluster."))
    cells.append(nbf.v4.new_code_cell("""original_features['Cluster'] = kmeans_labels
profiles = original_features.groupby('Cluster').mean()
print(profiles)

plt.figure(figsize=(10, 6))
sns.heatmap((profiles - profiles.mean()) / profiles.std(), annot=True, cmap='coolwarm', center=0)
plt.title('Standardized Cluster Profiles')
plt.show()"""))
    
    nb['cells'] = cells
    
    with open(output_path, 'w') as f:
        nbf.write(nb, f)
    print(f"Notebook generated at {output_path}")

if __name__ == "__main__":
    create_notebook('notebooks/clustering_analysis.ipynb')
