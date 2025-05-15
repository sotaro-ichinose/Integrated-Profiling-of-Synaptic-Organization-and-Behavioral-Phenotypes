#!/usr/bin/env python3
"""
Generate scatterplots of PCA scores (PC1, PC2) vs original variables, with Pearson r and p-values.
CSV and output directory are selected via GUI.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from tkinter import Tk, filedialog
import os

def main():
    # === Matplotlib settings ===
    plt.rcParams.update({
        'font.size': 8,
        'axes.titlesize': 8,
        'axes.labelsize': 8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8
    })

    # === File selection ===
    root = Tk()
    root.withdraw()
    csv_path = filedialog.askopenfilename(title="Select CSV file for PCA", filetypes=[("CSV files", "*.csv")])
    if not csv_path:
        print("❌ No file selected. Exiting.")
        return

    output_dir = filedialog.askdirectory(title="Select Output Directory for Plots")
    if not output_dir:
        print("❌ No output directory selected. Exiting.")
        return

    # === Configuration ===
    group_column = "Group"  # Adjust if needed
    group_palette = ['#66C2A5', '#FC8D62']

    # === Load data ===
    df = pd.read_csv(csv_path)
    if group_column not in df.columns:
        print(f"❌ Column '{group_column}' not found in CSV. Exiting.")
        return

    groups = df[group_column].values
    exclude_columns = [group_column, "MouseID"]
    X = df.drop(columns=[col for col in exclude_columns if col in df.columns]).select_dtypes(include='number')
    features = X.columns.tolist()

    # === Perform PCA ===
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2)
    scores = pca.fit_transform(X_scaled)

    # === Map group labels to colors ===
    unique_groups = np.unique(groups)
    group_color_map = {group: group_palette[i % len(group_palette)] for i, group in enumerate(unique_groups)}

    # === Generate scatter plots: PC1/PC2 vs each variable ===
    for pc_index in [0, 1]:
        pc_scores = scores[:, pc_index]
        pc_label = f"PC{pc_index + 1}"

        for variable in features:
            y = df[variable].values
            r, p = pearsonr(pc_scores, y)

            plt.figure(figsize=(2.4, 2))
            for group in unique_groups:
                idx = groups == group
                plt.scatter(pc_scores[idx], y[idx],
                            label=group,
                            s=20, alpha=0.8,
                            color=group_color_map[group])

            plt.xlabel(f"{pc_label} Score")
            plt.ylabel(variable)
            plt.title(f"{pc_label} vs {variable}\nr = {r:.2f}, p = {p:.4f}")
            plt.grid(True)
            plt.tight_layout()

            # Save as EPS
            safe_var = variable.replace("/", "_per_").replace(" ", "_")
            eps_path = os.path.join(output_dir, f"{pc_label}_vs_{safe_var}.eps")
            plt.savefig(eps_path, format='eps', bbox_inches='tight')
            plt.close()

    print(f"✅ All plots saved to: '{output_dir}'")

if __name__ == "__main__":
    main()
