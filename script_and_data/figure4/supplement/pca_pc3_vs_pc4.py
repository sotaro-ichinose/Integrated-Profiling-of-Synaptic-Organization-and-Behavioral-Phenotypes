import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import tkinter as tk
from tkinter import filedialog

def main():
    # === File selection via GUI ===
    root = tk.Tk()
    root.withdraw()
    csv_path = filedialog.askopenfilename(
        title="Select CSV file for PCA",
        filetypes=[("CSV files", "*.csv")]
    )
    if not csv_path:
        print("❌ No file selected. Exiting.")
        return

    group_column = "Group"  # Column name for group labels (e.g., WT, VPA)

    # === Load data ===
    df = pd.read_csv(csv_path)
    if group_column not in df.columns:
        print(f"❌ '{group_column}' column not found in the CSV.")
        return

    groups = df[group_column].values
    X = df.drop(columns=["MouseID", group_column], errors="ignore").select_dtypes(include='number')
    features = X.columns.tolist()

    # === PCA (up to 4 components) ===
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=4)
    scores = pca.fit_transform(X_scaled)

    # === Group color mapping (supports up to 2 groups) ===
    unique_groups = np.unique(groups)
    color_palette = ['#66C2A5', '#FC8D62']
    group_colors = {group: color_palette[i % len(color_palette)] for i, group in enumerate(unique_groups)}

    # === Plot PC3 vs PC4 ===
    plt.figure(figsize=(4, 3))
    for group in unique_groups:
        idx = groups == group
        plt.scatter(scores[idx, 2], scores[idx, 3],
                    label=group, s=80, alpha=0.8,
                    color=group_colors[group], edgecolor='k')

    plt.axhline(0, color='gray', linestyle='--')
    plt.axvline(0, color='gray', linestyle='--')
    plt.xlabel(f"PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)")
    plt.ylabel(f"PC4 ({pca.explained_variance_ratio_[3]*100:.1f}%)")
    plt.title("PCA Scatter Plot: PC3 vs PC4")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
