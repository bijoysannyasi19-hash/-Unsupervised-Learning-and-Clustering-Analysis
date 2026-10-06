from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import pandas as pd
from clustering import run_kmeans

def evaluate_kmeans_k(X, k_range):
    """Evaluates K-Means for different values of k."""
    results = []
    for k in k_range:
        kmeans, labels = run_kmeans(X, k)
        results.append({
            'k': k,
            'inertia': kmeans.inertia_,
            'silhouette': silhouette_score(X, labels),
            'calinski_harabasz': calinski_harabasz_score(X, labels),
            'davies_bouldin': davies_bouldin_score(X, labels)
        })
    return pd.DataFrame(results)

def evaluate_model(X, labels, model_name):
    """Evaluates a single clustering model."""
    return {
        'Model': model_name,
        'Silhouette': silhouette_score(X, labels),
        'Calinski-Harabasz': calinski_harabasz_score(X, labels),
        'Davies-Bouldin': davies_bouldin_score(X, labels)
    }
