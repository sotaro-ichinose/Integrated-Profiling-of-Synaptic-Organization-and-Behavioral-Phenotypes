import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import tkinter as tk
from tkinter import filedialog
import os

def main():
    # ==== Plot settings ====
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 12,
        'axes.labelsize': 12,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12
    })

    # ==== GUI file selection ====
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select a CSV file for PCA bootstrap",
        filetypes=[("CSV files", "*.csv")]
    )
    if not file_path:
        print("❌ No file selected. Exiting.")
        return

    df = pd.read_csv(file_path)
    output_dir = os.path.dirname(file_path)

    # ==== Select feature columns ====
    feature_cols = [col for col in df.columns if col not in ['Group', 'MouseID']]
    X = df[feature_cols].values
    X_scaled = StandardScaler().fit_transform(X)

    # ==== Bootstrap PCA ====
    n_bootstrap = 1000
    loadings_pc1 = {var: [] for var in feature_cols}
    loadings_pc2 = {var: [] for var in feature_cols}

    # ==== Anchor variables for sign alignment ====
    anchor_PC1 = "Density of PSD-95 (10% Depth)"
    anchor_PC2 = "OF Center Region"
    anchor_idx1 = feature_cols.index(anchor_PC1)
    anchor_idx2 = feature_cols.index(anchor_PC2)

    for _ in range(n_bootstrap):
        X_resampled = resample(X_scaled, replace=True)
        pca = PCA(n_components=2).fit(X_resampled)

        # Align sign for consistency
        sign1 = np.sign(pca.components_[0, anchor_idx1])
        sign2 = -np.sign(pca.components_[1, anchor_idx2])
        pca.components_[0, :] *= sign1
        pca.components_[1, :] *= sign2

        for i, var in enumerate(feature_cols):
            loadings_pc1[var].append(pca.components_[0, i])
            loadings_pc2[var].append(pca.components_[1, i])

    # ==== Plot scatter for each variable ====
    palette = sns.color_palette('Set2', n_colors=len(feature_cols))

    for i, var in enumerate(feature_cols):
        fig, ax = plt.subplots(figsize=(4.8, 4.8))
        ax.scatter(loadings_pc1[var], loadings_pc2[var], alpha=0.4, s=10, color=palette[i])
        ax.axhline(0, color='gray', linestyle='--')
        ax.axvline(0, color='gray', linestyle='--')
        ax.set_xlabel("PC1 loading")
        ax.set_ylabel("PC2 loading")
        ax.set_title(f"PC1 vs PC2 Loadings: {var}")
        plt.tight_layout()
        png_path = os.path.join(output_dir, f"pca_PC1_PC2_{var}.png")
        eps_path = os.path.join(output_dir, f"pca_PC1_PC2_{var}.eps")
        plt.savefig(png_path, dpi=300)
        plt.savefig(eps_path, format='eps')
        plt.close()

    print(f"✅ All loading scatter plots saved to: {output_dir}")

if __name__ == '__main__':
    main()
