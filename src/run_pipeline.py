import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Import our modules
from data_loader import download_data, load_data
from preprocessing import preprocess_data, audit_data
from clustering import run_kmeans, run_hierarchical, run_gmm, run_pca
from evaluation import evaluate_kmeans_k, evaluate_model
from visualization import (set_style, plot_metrics, plot_dendrogram, plot_pca_2d, 
                           plot_pca_3d, plot_cluster_sizes, plot_cluster_profiles_heatmap, 
                           plot_boxplots, plot_radar_chart, plot_pairplot, plot_gmm_bic,
                           plot_silhouette_per_cluster)
from generate_report import generate_report

def draw_architecture(save_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    
    boxes = [
        ('Raw Data\n(data/)', 0.1, 0.7),
        ('Preprocessing\n(src/preprocessing.py)', 0.35, 0.7),
        ('Clustering\n(K-Means, HC, GMM)', 0.6, 0.7),
        ('Evaluation\n(Metrics)', 0.85, 0.7),
        ('Cluster Profiling\n(results/)', 0.6, 0.4),
        ('Visualization\n(figures/)', 0.6, 0.1),
        ('Report Gen\n(report/)', 0.85, 0.1)
    ]
    
    for text, x, y in boxes:
        rect = patches.Rectangle((x, y), 0.2, 0.15, linewidth=1.5, edgecolor='black', facecolor='lightblue', zorder=2)
        ax.add_patch(rect)
        ax.text(x + 0.1, y + 0.075, text, ha='center', va='center', fontsize=10, zorder=3)
        
    # Arrows
    style = "Simple, tail_width=0.5, head_width=4, head_length=8"
    kw = dict(arrowstyle=style, color="gray")
    
    arrows = [
        ((0.3, 0.775), (0.35, 0.775)),
        ((0.55, 0.775), (0.6, 0.775)),
        ((0.8, 0.775), (0.85, 0.775)),
        ((0.7, 0.7), (0.7, 0.55)),
        ((0.7, 0.4), (0.7, 0.25)),
        ((0.8, 0.175), (0.85, 0.175))
    ]
    
    for (x1, y1), (x2, y2) in arrows:
        a = patches.FancyArrowPatch((x1, y1), (x2, y2), **kw, zorder=1)
        ax.add_patch(a)
        
    plt.title("Pipeline Architecture")
    plt.tight_layout()
    plt.savefig(save_path, dpi=72)
    plt.close()

def main():
    print("Starting clustering pipeline...")
    set_style()
    
    # 1. Load Data
    url = "https://raw.githubusercontent.com/TrainingByPackt/Data-Science-with-Python/master/Chapter01/Data/Wholesale%20customers%20data.csv"
    data_path = "data/wholesale_customers.csv"
    download_data(url, data_path)
    df = load_data(data_path)
    
    # 2. Preprocess
    print("Preprocessing data...")
    orig_features, scaled_features, labels_cat = preprocess_data(df)
    
    # 3. K-Means Evaluation
    print("Evaluating K-Means...")
    k_range = range(2, 11)
    k_metrics_df = evaluate_kmeans_k(scaled_features, k_range)
    
    # 4. Final Models (k=4 chosen based on metrics and interpretability)
    k_opt = 4
    print(f"Running final models with k={k_opt}...")
    kmeans_model, kmeans_labels = run_kmeans(scaled_features, k_opt)
    hc_model, hc_labels = run_hierarchical(scaled_features, k_opt)
    
    # 5. Model Evaluation
    print("Calculating final evaluation metrics...")
    eval_kmeans = evaluate_model(scaled_features, kmeans_labels, 'K-Means')
    eval_hc = evaluate_model(scaled_features, hc_labels, 'Hierarchical')
    final_metrics_df = pd.DataFrame([eval_kmeans, eval_hc])
    final_metrics_df.to_csv('results/model_metrics.csv', index=False)
    
    # 6. Cluster Profiling
    print("Generating cluster profiles...")
    orig_features['Cluster'] = kmeans_labels
    cluster_profiles = orig_features.groupby('Cluster').mean()
    cluster_profiles.to_csv('results/cluster_profiles.csv')
    pd.DataFrame({'Instance_Index': orig_features.index, 'Cluster': kmeans_labels}).to_csv('results/cluster_assignments.csv', index=False)
    
    # 7. PCA
    print("Running PCA...")
    pca_model, X_pca = run_pca(scaled_features, n_components=3)
    
    # 8. Visualizations
    print("Generating visualizations...")
    plot_metrics(k_metrics_df, 'figures/elbow_metrics.png')
    plot_silhouette_per_cluster(scaled_features.values, kmeans_labels, 'figures/silhouette_k4.png')
    plot_dendrogram(scaled_features, 'figures/dendrogram.png')
    plot_pca_2d(X_pca, kmeans_labels, pca_model, 'figures/pca_2d.png')
    plot_pca_3d(X_pca, kmeans_labels, pca_model, 'figures/pca_3d.png')
    plot_cluster_sizes(kmeans_labels, 'figures/cluster_sizes.png')
    plot_cluster_profiles_heatmap(cluster_profiles, 'figures/cluster_profiles_heatmap.png')
    plot_boxplots(orig_features.drop('Cluster', axis=1), kmeans_labels, orig_features.columns[:-1], 'figures/boxplots.png')
    plot_radar_chart(cluster_profiles, 'figures/radar_chart.png')
    plot_pairplot(orig_features.drop('Cluster', axis=1), kmeans_labels, 'figures/pairplot.png')
    plot_gmm_bic(scaled_features, k_range, 'figures/gmm_bic_aic.png')
    
    # 9. Architecture Diagram
    print("Generating architecture diagram...")
    draw_architecture('docs/architecture.png')
    
    # 10. Generate Report
    print("Generating DOCX report...")
    generate_report('results', 'figures', 'report/Clustering_Report.docx')
    
    print("Pipeline completed successfully.")

if __name__ == "__main__":
    main()
