import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np
import pandas as pd
from math import pi
from sklearn.metrics import silhouette_samples

def set_style():
    plt.style.use('default')
    sns.set_theme(style="whitegrid")
    
def plot_metrics(k_df, save_path=None):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('K-Means Evaluation Metrics vs Number of Clusters (k)', fontsize=16)
    
    axes[0, 0].plot(k_df['k'], k_df['inertia'], marker='o', linestyle='-')
    axes[0, 0].set_title('Elbow Method (Inertia)')
    axes[0, 0].set_xlabel('Number of Clusters (k)')
    axes[0, 0].set_ylabel('Inertia')
    
    axes[0, 1].plot(k_df['k'], k_df['silhouette'], marker='o', linestyle='-', color='orange')
    axes[0, 1].set_title('Silhouette Score (Higher is better)')
    axes[0, 1].set_xlabel('Number of Clusters (k)')
    axes[0, 1].set_ylabel('Silhouette Score')
    
    axes[1, 0].plot(k_df['k'], k_df['calinski_harabasz'], marker='o', linestyle='-', color='green')
    axes[1, 0].set_title('Calinski-Harabasz Score (Higher is better)')
    axes[1, 0].set_xlabel('Number of Clusters (k)')
    axes[1, 0].set_ylabel('Score')
    
    axes[1, 1].plot(k_df['k'], k_df['davies_bouldin'], marker='o', linestyle='-', color='red')
    axes[1, 1].set_title('Davies-Bouldin Score (Lower is better)')
    axes[1, 1].set_xlabel('Number of Clusters (k)')
    axes[1, 1].set_ylabel('Score')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.close()

def plot_dendrogram(X, save_path=None):
    plt.figure(figsize=(10, 6))
    Z = linkage(X, method='ward')
    dendrogram(Z, truncate_mode='level', p=5)
    plt.title('Hierarchical Clustering Dendrogram (Ward Linkage, Truncated)')
    plt.xlabel('Cluster Size / Index')
    plt.ylabel('Distance')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.close()

def plot_pca_2d(X_pca, labels, pca_model, save_path=None):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette='tab10', legend='full', s=50)
    plt.title('PCA 2D Scatter Plot Colored by Cluster')
    plt.xlabel(f'PC1 ({pca_model.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca_model.explained_variance_ratio_[1]:.2%} variance)')
    plt.legend(title='Cluster')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.close()

def plot_pca_3d(X_pca, labels, pca_model, save_path=None):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=labels, cmap='tab10', s=40)
    ax.set_title('PCA 3D Scatter Plot Colored by Cluster')
    ax.set_xlabel(f'PC1 ({pca_model.explained_variance_ratio_[0]:.2%})')
    ax.set_ylabel(f'PC2 ({pca_model.explained_variance_ratio_[1]:.2%})')
    ax.set_zlabel(f'PC3 ({pca_model.explained_variance_ratio_[2]:.2%})')
    legend1 = ax.legend(*scatter.legend_elements(), title="Cluster")
    ax.add_artist(legend1)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.close()

def plot_cluster_sizes(labels, save_path=None):
    plt.figure(figsize=(8, 5))
    sizes = pd.Series(labels).value_counts().sort_index()
    sns.barplot(x=sizes.index, y=sizes.values, palette='tab10')
    plt.title('Cluster Size Distribution')
    plt.xlabel('Cluster')
    plt.ylabel('Number of Instances')
    for i, v in enumerate(sizes.values):
        plt.text(i, v + 2, str(v), ha='center')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.close()

def plot_cluster_profiles_heatmap(profiles, save_path=None):
    plt.figure(figsize=(10, 6))
    # Standardize profiles across features to show relative Highs and Lows in the heatmap
    normalized_profiles = (profiles - profiles.mean()) / profiles.std()
    sns.heatmap(normalized_profiles, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Cluster Profiles Heatmap (Standardized Feature Means)')
    plt.ylabel('Cluster')
    plt.xlabel('Feature')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.close()

def plot_boxplots(df, labels, features, save_path=None):
    df_plot = df.copy()
    df_plot['Cluster'] = labels
    
    n_features = len(features)
    cols = 2
    rows = (n_features + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, rows * 4))
    axes = axes.flatten()
    
    for i, feature in enumerate(features):
        sns.boxplot(data=df_plot, x='Cluster', y=feature, ax=axes[i], palette='tab10')
        axes[i].set_title(f'Distribution of {feature} per Cluster')
        axes[i].set_ylabel('Original Value')
    
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
        
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.close()

def plot_radar_chart(profiles, save_path=None):
    # Prepare data
    categories = list(profiles.columns)
    N = len(categories)
    
    # Scale profiles for radar to [0, 1] range based on max value in feature for plotting
    scaled_profiles = profiles / profiles.max()
    
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    plt.xticks(angles[:-1], categories, color='grey', size=10)
    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75], ["25%", "50%", "75%"], color="grey", size=8)
    plt.ylim(0, 1)
    
    colors = plt.cm.tab10.colors
    for i, (idx, row) in enumerate(scaled_profiles.iterrows()):
        values = row.tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=f'Cluster {idx}', color=colors[i % len(colors)])
        ax.fill(angles, values, alpha=0.1, color=colors[i % len(colors)])
        
    plt.title('Radar Chart of Cluster Profiles (Relative Scale)', size=14, y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.close()

def plot_pairplot(df, labels, save_path=None):
    df_plot = df.copy()
    # Log transform for pairplot to make it readable, as original values are skewed
    df_plot = np.log1p(df_plot) 
    df_plot['Cluster'] = labels
    
    # Sample data if too large to render fast, but 440 rows is fine
    g = sns.pairplot(df_plot, hue='Cluster', palette='tab10', corner=True, diag_kind='kde')
    g.fig.suptitle('Pairplot of Log-Transformed Features Colored by Cluster', y=1.02)
    if save_path:
        g.savefig(save_path, dpi=200)
    plt.close()

def plot_gmm_bic(X, k_range, save_path=None):
    bics = []
    aics = []
    from sklearn.mixture import GaussianMixture
    for k in k_range:
        gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=42).fit(X)
        bics.append(gmm.bic(X))
        aics.append(gmm.aic(X))
        
    plt.figure(figsize=(8, 5))
    plt.plot(k_range, bics, marker='o', label='BIC')
    plt.plot(k_range, aics, marker='s', label='AIC')
    plt.title('GMM Model Selection: BIC and AIC')
    plt.xlabel('Number of Components (k)')
    plt.ylabel('Information Criterion')
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.close()

def plot_silhouette_per_cluster(X, labels, save_path=None):
    from sklearn.metrics import silhouette_samples, silhouette_score
    import matplotlib.cm as cm
    
    n_clusters = len(np.unique(labels))
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))
    
    ax1.set_xlim([-0.2, 1])
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])
    
    silhouette_avg = silhouette_score(X, labels)
    sample_silhouette_values = silhouette_samples(X, labels)
    
    y_lower = 10
    for i in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[labels == i]
        ith_cluster_silhouette_values.sort()
        
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        
        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10
        
    ax1.set_title("Silhouette Plot for the Various Clusters")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")
    
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax1.set_yticks([])
    ax1.set_xticks(np.arange(-0.2, 1.1, 0.2))
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.close()
