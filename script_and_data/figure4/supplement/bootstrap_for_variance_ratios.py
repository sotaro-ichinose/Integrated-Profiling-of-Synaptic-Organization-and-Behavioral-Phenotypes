import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import tkinter as tk
from tkinter import filedialog
import seaborn as sns

def main():
    # === GUI for CSV file selection ===
    root = tk.Tk()
    root.withdraw()
    csv_path = filedialog.askopenfilename(
        title="Select a CSV file for PCA bootstrap analysis",
        filetypes=[("CSV Files", "*.csv")]
    )
    if not csv_path:
        print("❌ No file selected. Exiting.")
        return

    group_column = "Group"

    # === Load data ===
    df = pd.read_csv(csv_path)
    features = df.drop(columns=["MouseID", group_column], errors="ignore")
    X = features.select_dtypes(include='number')

    # === Standardization ===
    X_scaled = StandardScaler().fit_transform(X)

    # === Bootstrap PCA explained variance ratios ===
    rng = np.random.default_rng(seed=42)
    n_iter = 1000
    n_samples = X_scaled.shape[0]
    pc1_ratios = []
    pc2_ratios = []

    for _ in range(n_iter):
        indices = rng.integers(0, n_samples, n_samples)
        X_boot = X_scaled[indices]
        pca_boot = PCA(n_components=2)
        pca_boot.fit(X_boot)
        pc1_ratios.append(pca_boot.explained_variance_ratio_[0])
        pc2_ratios.append(pca_boot.explained_variance_ratio_[1])

    # === Histogram Plot ===
    palette = sns.color_palette("Set2", n_colors=2)

    plt.figure(figsize=(7, 3))
    plt.hist(pc1_ratios, bins=30, alpha=0.6, color=palette[0], label="PC1")
    plt.hist(pc2_ratios, bins=30, alpha=0.6, color=palette[1], label="PC2")
    plt.xlabel("Explained Variance Ratio")
    plt.ylabel("Frequency")
    plt.legend()
    plt.title("Bootstrap Distribution of Explained Variance Ratios")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
