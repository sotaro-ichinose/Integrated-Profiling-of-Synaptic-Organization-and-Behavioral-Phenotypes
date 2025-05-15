#!/usr/bin/env python3
"""
PCA projection of WT vs VPA with centroid vector, confidence ellipses, and loading vectors.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import euclidean
from matplotlib.patches import Ellipse
from tkinter import Tk, filedialog
import os

# === Plot style ===
plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 16,
    'axes.labelsize': 16,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16
})

def compute_centroid(df, group):
    """Compute centroid of a group in PC1-PC2 space."""
    sub = df[df['Group'] == group]
    return sub[['PC1', 'PC2']].mean().values

def plot_confidence_ellipse(x, y, ax, n_std=1.96, **kwargs):
    """Plot 95% confidence ellipse for a group."""
    cov = np.cov(x, y)
    mean_x, mean_y = np.mean(x), np.mean(y)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    vx, vy = eigvecs[:, 0]
    angle = np.degrees(np.arctan2(vy, vx))
    width, height = 2 * n_std * np.sqrt(eigvals)
    ellipse = Ellipse((mean_x, mean_y), width, height, angle=angle,
                      facecolor='none', **kwargs)
    ax.add_patch(ellipse)

def main():
    # === GUI-based CSV file selection ===
    Tk().withdraw()
    file_path = filedialog.askopenfilename(
        title="Select CSV file for PCA computation",
        filetypes=[("CSV Files", "*.csv")]
    )
    if not file_path:
        print("❌ No file selected. Exiting.")
        return

    # === Load data ===
    df = pd.read_csv(file_path)
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_dir = os.path.dirname(file_path)

    # === Group filtering ===
    df['Group'] = df['Group'].str.strip()
    groups_of_interest = ['WT', 'VPA']
    df = df[df['Group'].isin(groups_of_interest)].copy()

    # === PCA preparation ===
    features = [col for col in df.columns if col not in ['Group', 'MouseID']]
    X = StandardScaler().fit_transform(df[features])
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
    df_pca['Group'] = df['Group'].values
    df_pca['MouseID'] = df['MouseID'].values

    # === Variable loadings ===
    loadings = pca.components_.T
    loading_df = pd.DataFrame(loadings, index=features, columns=["PC1_loading", "PC2_loading"])
    print("\nVariable loadings:")
    print(loading_df)

    # === Save outputs ===
    loading_csv = os.path.join(output_dir, f"{base_name}_pca_all_loadings.csv")
    loading_df.to_csv(loading_csv)
    print(f"✅ Saved PCA loadings to {loading_csv}")

    # === Save PC scores per MouseID ===
    #pc_scores_csv = os.path.join(output_dir, f"{base_name}_PCA_scores_by_MouseID.csv")
    #df_pca[['MouseID', 'Group', 'PC1', 'PC2']].to_csv(pc_scores_csv, index=False)
    #print(f"✅ Saved PC1 and PC2 scores by MouseID to {pc_scores_csv}")

    # === Save original data + PC1, PC2 ===
    df_with_pcs = df.copy()
    df_with_pcs["PC1"] = df_pca["PC1"].values
    df_with_pcs["PC2"] = df_pca["PC2"].values
    augmented_csv = os.path.join(output_dir, f"{base_name}_with_pc1_pc2.csv")
    df_with_pcs.to_csv(augmented_csv, index=False)
    print(f"✅ Saved original data with PC1/PC2 to {augmented_csv}")

    # === Compute centroids and displacement vector ===
    c_wt = compute_centroid(df_pca, 'WT')
    c_vpa = compute_centroid(df_pca, 'VPA')
    displacement = c_vpa - c_wt
    dist_obs = euclidean(c_wt, c_vpa)

    # === Plotting ===
    fig, ax = plt.subplots(figsize=(10, 6))
    palette = dict(zip(groups_of_interest, sns.color_palette("Set2", 2)))

    # Scatterplot with centroids
    sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='Group',
                    palette=palette, s=100, ax=ax)
    ax.scatter(*c_wt, marker='X', color=palette['WT'], s=150, edgecolor='black')
    ax.scatter(*c_vpa, marker='X', color=palette['VPA'], s=150, edgecolor='black')

    # Confidence ellipses
    for g in groups_of_interest:
        sub = df_pca[df_pca['Group'] == g]
        plot_confidence_ellipse(sub['PC1'], sub['PC2'], ax,
                                edgecolor=palette[g], linestyle='--')

    # Loading vectors
    loadings = pca.components_.T
    for i, var in enumerate(features):
        ax.arrow(0, 0, loadings[i, 0]*7, loadings[i, 1]*7,
                 color='red', alpha=0.7, head_width=0.05, head_length=0.05)
        ax.text(loadings[i, 0]*9, loadings[i, 1]*9, var,
                color='red', ha='center', va='center', fontsize=16)

    # Styling
    ax.axhline(0, color='gray', linestyle='--')
    ax.axvline(0, color='gray', linestyle='--')
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    ax.set_title("PCA of Behavioral and Synaptic Features: WT vs VPA")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
