#!/usr/bin/env python3
"""
ARI between Z-based resilience and behavior-based clustering, with PCA plot.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog
import os

# === Global figure style settings ===
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.labelsize': 16,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16
})

def main():
    # === Select CSV file via GUI ===
    Tk().withdraw()
    path = filedialog.askopenfilename(title="Select CSV with behavior data and Z_Resilient_Label")
    if not path:
        raise SystemExit("❌ No file selected.")

    df = pd.read_csv(path)
    df["Group"] = df["Group"].astype(str).str.strip()  # Trim spaces for safety
    base = os.path.splitext(os.path.basename(path))[0]

    # === Filter for VPA group only ===
    vpa = df[df["Group"] == "VPA"].copy()
    if "Z_Resilient_Label" not in vpa.columns:
        raise ValueError("❗ 'Z_Resilient_Label' column is missing.")

    # === Select behavior features only (exclude synaptic variables) ===
    exclude_cols = ["MouseID", "Density of PSD-95 (10% Depth)", "Density of gephyrin (10% Depth)"]
    behavior_cols = [
        "SI Time with stranger",
        "YM Re-entry Ratio"
    ]

    X = StandardScaler().fit_transform(vpa[behavior_cols])

    # === K-means clustering (k=2) ===
    kmeans = KMeans(n_clusters=2, random_state=0, n_init='auto')
    clusters = kmeans.fit_predict(X)
    vpa["Cluster"] = clusters

    # === Convert Z-based label to binary (True → 1, False → 0) ===
    vpa["Z_Label_Binary"] = vpa["Z_Resilient_Label"].map({
        "VPA resilient": 1, "VPA non resilient": 0
    })

    valid = vpa[["Z_Label_Binary", "Cluster"]].dropna()
    true_labels = valid["Z_Label_Binary"].astype(int).values
    pred_labels = valid["Cluster"].astype(int).values

    ari = adjusted_rand_score(true_labels, pred_labels)
    print(f"✅ Adjusted Rand Index (ARI): {ari:.3f}")

    # === Dimensionality reduction for visualization ===
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    vpa["PC1"] = X_pca[:, 0]
    vpa["PC2"] = X_pca[:, 1]

    # === Color and marker settings ===
    label_colors = {0: "#66c2a5" , 1: "#fc8d62"}
    label_markers = {"VPA resilient": "o", "VPA non resilient": "s"}

    fig, ax = plt.subplots(figsize=(8, 4))
    for label in [0, 1]:
        for res in ["VPA resilient", "VPA non resilient"]:
            subset = vpa[(vpa["Cluster"] == label) & (vpa["Z_Resilient_Label"] == res)]
            ax.scatter(
                subset["PC1"], subset["PC2"],
                color=label_colors[label],
                marker=label_markers[res],
                label=f"Cluster {label} / {res}",
                edgecolor='black', alpha=0.8, s=80
            )

    ax.set_title(f"PCA of behavior (ARI = {ari:.2f})")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.axhline(0, color='gray', linestyle='--')
    ax.axvline(0, color='gray', linestyle='--')

    # === Display legend outside the plot (right side) ===
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0.)

    # === Adjust layout to leave space for legend ===
    fig.tight_layout()
    fig.subplots_adjust(right=0.5)

    # === Save ===
    base = os.path.splitext(os.path.basename(path))[0]
    input_dir = os.path.dirname(path)
    output_png = os.path.join(input_dir, f"{base}_cluste.png")
    output_eps = os.path.join(input_dir, f"{base}_cluster.eps")
    output_csv = os.path.join(input_dir, f"{base}_cluster.csv")

    fig.savefig(output_png, dpi=300)
    fig.savefig(output_eps, format="eps")
    vpa.to_csv(output_csv, index=False)
    plt.show()

    print(f"✅ Saved figure: {output_png}, {output_eps}")
    print(f"✅ Saved results: {output_csv}")

# Entry point
if __name__ == "__main__":
    main()
