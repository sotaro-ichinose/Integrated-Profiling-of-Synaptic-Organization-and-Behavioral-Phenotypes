#!/usr/bin/env python3
"""
Recompute PCA on all samples with GUI file selection, display explained variance,
export loadings and PC scores, and plot PCA projection with loading vectors.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tkinter import Tk, filedialog
import os

# === Global plot style ===
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 12,
    'axes.labelsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12
})

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

    # === Feature selection ===
    feature_cols = [c for c in df.columns if c not in ["Group", "MouseID"]]
    scaler = StandardScaler()
    X_all = scaler.fit_transform(df[feature_cols])

    # === Perform PCA ===
    pca = PCA(n_components=2)
    X_all = scaler.fit_transform(df[feature_cols])
    X_all_pca = pca.fit_transform(X_all)

    # === Initial loadings
    loadings = pca.components_.T
    loading_df = pd.DataFrame(loadings, index=feature_cols, columns=["PC1_loading", "PC2_loading"])

    # === Fix PC signs for interpretability
    if loading_df.loc["Density of PSD-95 (10% Depth)", "PC1_loading"] < 0:
        X_all_pca[:, 0] *= -1
        pca.components_[0, :] *= -1
        loadings[:, 0] *= -1
        loading_df["PC1_loading"] *= -1

    if loading_df.loc["OF Center Region", "PC2_loading"] > 0:
        X_all_pca[:, 1] *= -1
        pca.components_[1, :] *= -1
        loadings[:, 1] *= -1
        loading_df["PC2_loading"] *= -1

    df["PC1"] = X_all_pca[:, 0]
    df["PC2"] = X_all_pca[:, 1]


    # === Explained variance ===
    exp_var = pca.explained_variance_ratio_
    print(f"Explained variance: PC1 = {exp_var[0]*100:.1f}%, PC2 = {exp_var[1]*100:.1f}%")

    # === Variable loadings ===
    loadings = pca.components_.T
    loading_df = pd.DataFrame(loadings, index=feature_cols, columns=["PC1_loading", "PC2_loading"])
    print("\nVariable loadings:")
    print(loading_df)

    # === Save outputs ===
    score_csv = os.path.join(output_dir, f"{base_name}_pca_all_scores.csv")
    loading_csv = os.path.join(output_dir, f"{base_name}_pca_all_loadings.csv")
    df.to_csv(score_csv, index=False)
    loading_df.to_csv(loading_csv)
    print(f"✅ Saved PCA scores to {score_csv}")
    print(f"✅ Saved PCA loadings to {loading_csv}")

    # === Plot PCA projection ===
    plt.figure(figsize=(6, 4))
    sns.scatterplot(data=df, x="PC1", y="PC2", hue="Group", s=100, palette="Set2")

    # Plot variable vectors (loadings)
    for i, var in enumerate(feature_cols):
        plt.arrow(0, 0, loadings[i, 0] * 4, loadings[i, 1] * 4,
                  color='red', alpha=0.7, head_width=0.1, head_length=0.1)
        plt.text(loadings[i, 0] * 6, loadings[i, 1] * 6, var,
                 color='red', ha='center', va='center')

    plt.title("PCA Projection of All Samples")
    plt.xlabel(f"PC1 ({exp_var[0]*100:.1f}% variance)")
    plt.ylabel(f"PC2 ({exp_var[1]*100:.1f}% variance)")
    plt.axhline(0, color='grey', linestyle='--')
    plt.axvline(0, color='grey', linestyle='--')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    plot_path = os.path.join(output_dir, f"{base_name}_pca_all_projection.png")
    plt.savefig(plot_path, dpi=300)
    print(f"📊 Saved PCA projection plot to {plot_path}")
    plt.show()

if __name__ == "__main__":
    main()
