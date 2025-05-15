import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import os
import tkinter as tk
from tkinter import filedialog

def main():
    # === Plot settings ===
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 12,
        'axes.labelsize': 12,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12
    })

    # === File selection ===
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select CSV file",
        filetypes=[("CSV Files", "*.csv")]
    )
    if not file_path:
        print("❌ No file selected. Exiting.")
        return

    df = pd.read_csv(file_path)

    # === Define depth bins and labels ===
    depth_bins = [(0, 10), (10, 20), (20, 40), (40, 70), (70, 100)]
    labels = [f"{d[0]}–{d[1]}%" for d in depth_bins]
    metrics = {
        "Exc_density": {"color": "#66c2a5", "label": "Excitatory"},
        "Inh_density": {"color": "#fc8d62", "label": "Inhibitory"}
    }

    # === Store results as rows for CSV ===
    summary_rows = []

    # === Create plot ===
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    for ax, (metric, props) in zip(axes, metrics.items()):
        r_vals = []
        p_vals = []

        for i, (d_min, d_max) in enumerate(depth_bins):
            df_sub = df[(df['Depth'] >= d_min) & (df['Depth'] < d_max)]
            df_mean = df_sub.groupby('MouseID')[[metric, 'Behavior']].mean().reset_index()

            if len(df_mean) > 2:
                r, p = pearsonr(df_mean[metric], df_mean['Behavior'])
            else:
                r, p = np.nan, np.nan

            r_vals.append(r)
            p_vals.append(p)

            summary_rows.append({
                "Metric": metric,
                "Depth_bin": f"{d_min}-{d_max}%",
                "Pearson_r": r,
                "p_value": p
            })

        # Bar plot of r values
        bars = ax.bar(labels, r_vals, color=props['color'], edgecolor='black')
        ax.set_title(props['label'], fontsize=14)
        ax.set_xlabel("Depth (%)")
        ax.axhline(0, color='gray', linestyle='--')
        ax.set_ylim(-1.0, 1.0)

        # Asterisk for significant p-values
        for i, (r, p) in enumerate(zip(r_vals, p_vals)):
            if p < 0.05:
                ax.text(i, r + 0.05 * np.sign(r), "*", ha='center', va='bottom', fontsize=14)

        # Overlay p-values
        ax2 = ax.twinx()
        ax2.scatter(labels, p_vals, color='red', label='p-value')
        ax2.axhline(0.05, color='red', linestyle='--', label='p = 0.05')
        ax2.set_ylim(0, 1.05)
        if ax == axes[-1]:
            ax2.legend(loc='upper right', fontsize=10)
        else:
            ax2.set_yticklabels([])

    axes[0].set_ylabel("Pearson r")
    fig.suptitle("Depth-wise Pearson Correlation with Behavior", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)

    # === Save results ===
    base = os.path.splitext(os.path.basename(file_path))[0]
    out_dir = os.path.dirname(file_path)
    fig_path_png = os.path.join(out_dir, f"{base}_Layerwise_Correlation_Exc_Inh.png")
    fig_path_eps = os.path.join(out_dir, f"{base}_Layerwise_Correlation_Exc_Inh.eps")
    csv_path = os.path.join(out_dir, f"{base}_r_and_p_values.csv")

    plt.savefig(fig_path_png, dpi=300)
    plt.savefig(fig_path_eps, format='eps')
    pd.DataFrame(summary_rows).to_csv(csv_path, index=False)

    print(f"✅ Results saved:\n- Plot: {fig_path_png} / .eps\n- Summary CSV: {csv_path}")
    plt.show()

if __name__ == "__main__":
    main()
