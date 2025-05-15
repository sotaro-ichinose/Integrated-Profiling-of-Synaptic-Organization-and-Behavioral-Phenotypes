#!/usr/bin/env python3
"""
Raincloud plot (half-violin + jittered points) of 1-D Z-distance from WT centroid,
comparing WT and VPA only. ±1.96 SD = resilience threshold.

INPUT  : CSV with 'PC1', 'PC2', 'Group'
OUTPUT : PNG + EPS raincloud figure, CSV with Z_Distance & Resilient columns
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog
import os

def main():
    # Plot settings
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 16,
        'axes.labelsize': 16,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 16
    })

    # === Load CSV via GUI ===
    root = Tk()
    root.withdraw()
    csv_path = filedialog.askopenfilename(
        title="Select PCA-result CSV",
        filetypes=[("CSV files", "*.csv")]
    )
    if not csv_path:
        raise SystemExit("❌ No file selected – exiting.")

    df_raw = pd.read_csv(csv_path)
    required_cols = ("PC1", "PC2", "Group")
    if not all(col in df_raw.columns for col in required_cols):
        raise ValueError(f"Missing required columns: {required_cols}")

    # === Compute projection axis and Z-distance ===
    df = df_raw.copy()
    wt_mask = df["Group"] == "WT"
    vpa_mask = df["Group"] == "VPA"

    wt_cent = df.loc[wt_mask, ["PC1", "PC2"]].mean().values
    vpa_cent = df.loc[vpa_mask, ["PC1", "PC2"]].mean().values
    axis = vpa_cent - wt_cent
    axis /= np.linalg.norm(axis)

    proj = (df[["PC1", "PC2"]].values - wt_cent).dot(axis)
    mu, sigma = proj[wt_mask].mean(), proj[wt_mask].std(ddof=0)
    df["Z_Distance"] = (proj - mu) / sigma

    # === Label resilience based on threshold ===
    threshold = 1.96
    df["Resilient"] = df["Z_Distance"].abs() <= threshold

    # === Plot WT vs VPA only ===
    df_plot = df[df["Group"].isin(["WT", "VPA"])]
    groups = ["WT", "VPA"]
    colors = {
        "WT": "#fc8d62",
        "VPA": "#66c2a5"
    }
    data = [df_plot[df_plot["Group"] == g]["Z_Distance"].values for g in groups]

    # === Create raincloud plot ===
    fig, ax = plt.subplots(figsize=(7, 4))

    # Half-violin
    parts = ax.violinplot(data, positions=np.arange(len(groups)), vert=False,
                          widths=0.6, showmeans=False, showextrema=False)
    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor(colors[groups[i]])
        body.set_alpha(0.6)
        verts = body.get_paths()[0].vertices
        verts[verts[:, 1] < i, 1] = i  # retain upper half only

    # Jittered scatter
    for i, g in enumerate(groups):
        vals = df_plot[df_plot["Group"] == g]["Z_Distance"].values
        jitter = np.random.normal(loc=i - 0.1, scale=0.05, size=len(vals))
        ax.scatter(vals, jitter, color="black", s=12, alpha=0.7, zorder=3)

    # Resilience threshold band
    ax.axvspan(-threshold, threshold, color="lightgrey", alpha=0.3, label=f"|Z| ≤ {threshold}")

    # Axis settings
    ax.set_yticks(np.arange(len(groups)))
    ax.set_yticklabels(groups)
    ax.set_xlabel("Z-distance from WT centroid")
    ax.set_title("Z-distance distribution (±1.96 SD)")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    fig.tight_layout()

    # === Rename Group for VPA depending on resilience ===
    df.loc[(df["Group"] == "VPA") & (df["Resilient"]), "Z_Resilient_Label"] = "VPA resilient"
    df.loc[(df["Group"] == "VPA") & (~df["Resilient"]), "Z_Resilient_Label"] = "VPA non resilient"

    # === Save ===
    base = os.path.splitext(os.path.basename(csv_path))[0]
    input_dir = os.path.dirname(csv_path)
    output_png = os.path.join(input_dir, f"{base}_raincloud.png")
    output_eps = os.path.join(input_dir, f"{base}_raincloud..eps")
    output_csv = os.path.join(input_dir, f"{base}_with_resilience.csv")

    fig.savefig(output_png, dpi=300)
    fig.savefig(output_eps, format="eps")
    df.to_csv(output_csv, index=False)
    plt.show()

    print(f"✅ Saved figure: {output_png}, {output_eps}")
    print(f"✅ Saved results: {output_csv}")

    plt.show()

if __name__ == '__main__':
    main()
