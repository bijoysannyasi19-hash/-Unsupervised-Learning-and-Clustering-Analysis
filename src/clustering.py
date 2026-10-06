import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

def run_kmeans(X, k, random_state=42):
    """Runs K-Means clustering."""
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=random_state)
    labels = kmeans.fit_predict(X)
    return kmeans, labels

def run_hierarchical(X, k, linkage='ward'):
    """Runs Hierarchical (Agglomerative) clustering."""
    hc = AgglomerativeClustering(n_clusters=k, linkage=linkage)
    labels = hc.fit_predict(X)
    return hc, labels

def run_gmm(X, k, random_state=42):
    """Runs Gaussian Mixture Model clustering."""
    gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=random_state)
    labels = gmm.fit_predict(X)
    return gmm, labels

def run_pca(X, n_components=None):
    """Runs PCA for dimensionality reduction."""
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X)
    return pca, X_pca
