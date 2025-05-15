import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel, ttest_ind
import tkinter as tk
from tkinter import filedialog

def main():
    # === File selection ===
    root = tk.Tk()
    root.withdraw()
    csv_path = filedialog.askopenfilename(
        title="Select a CSV file for dorsal vs ventral analysis",
        filetypes=[("CSV Files", "*.csv")]
    )
    
    if not csv_path:
        print("No file selected. Exiting.")
        return

    output_dir = os.path.dirname(csv_path)
    base_name = os.path.splitext(os.path.basename(csv_path))[0]

    # === Plot settings ===
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 12,
        'axes.labelsize': 12,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12
    })

    # === Load data ===
    df = pd.read_csv(csv_path)
    df["X_percent"] = df["X"] / 600 * 100  # Convert X to % depth

    # === Group stats ===
    means = df.groupby("region")["X_percent"].mean()
    sds = df.groupby("region")["X_percent"].std()
    regions = ["dorsal", "ventral"]

    # === Paired t-test ===
    pivot = df.pivot(index="ID", columns="region", values="X_percent").dropna()
    t_stat, p_value = ttest_rel(pivot["dorsal"], pivot["ventral"])

    # === Plot ===
    fig, ax = plt.subplots(figsize=(3.5, 3))
    ax.bar(regions, means[regions], yerr=sds[regions],
           color=['#66C2A5', '#FC8D62'], capsize=5)

    for _, row in pivot.iterrows():
        ax.plot(regions, [row["dorsal"], row["ventral"]],
                color="gray", marker="o", markersize=6, linewidth=1, alpha=0.7)

    # Significance annotation
    max_y = 100
    if p_value < 0.001:
        stars = '***'
    elif p_value < 0.01:
        stars = '**'
    elif p_value < 0.05:
        stars = '*'
    else:
        stars = 'n.s.'
    ax.text(0.5, max_y * 0.6, stars, ha='center', va='bottom', fontsize=14)

    ax.set_ylabel("Depth (%)")
    ax.set_ylim(0, max_y * 1.05)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(regions)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.tight_layout()

    # === Save plots ===
    png_path = os.path.join(output_dir, f"{base_name}_dorsal_vs_ventral_SD.png")
    eps_path = os.path.join(output_dir, f"{base_name}_dorsal_vs_ventral_SD.eps")
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(eps_path, format='eps', bbox_inches='tight')
    plt.close()

    # === Save stats ===
    txt_path = os.path.join(output_dir, f"{base_name}_summary.txt")
    with open(txt_path, 'w') as f:
        for region in regions:
            f.write(f"{region}: {means[region]:.3f} ± {sds[region]:.3f}\n")
        f.write(f"p-value (paired t-test): {p_value:.4f}\n")

    print("✅ Processing completed:")

if __name__ == "__main__":
    main()
