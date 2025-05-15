import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog
import os

def main():
    # === Plot style settings ===
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 16,
        'axes.labelsize': 16,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 16
    })

    # === GUI: Select CSV file ===
    Tk().withdraw()
    file_path = filedialog.askopenfilename(
        title="Select a CSV file with cluster labels and behavioral values",
        filetypes=[("CSV Files", "*.csv")]
    )
    if not file_path:
        raise SystemExit("❌ No file selected.")

    df = pd.read_csv(file_path)
    base = os.path.splitext(os.path.basename(file_path))[0]

    # === Define behavior variables (2) ===
    features = ["SI Time with stranger", "YM Re-entry Ratio"]
    if "Cluster" not in df.columns:
        raise ValueError("❗ Column 'Cluster' not found. Please use a CSV file with clustering results.")

    # === Compute mean and SEM per cluster ===
    summary = df.groupby("Cluster")[features].agg(['mean', 'sem']).reset_index()

    # === Reshape for plotting ===
    plot_data = []
    for i, row in summary.iterrows():
        for feat in features:
            plot_data.append({
                "Cluster": f"Cluster {int(row['Cluster'])}",
                "Feature": feat,
                "Mean": row[(feat, 'mean')],
                "SEM": row[(feat, 'sem')]
            })
    df_plot = pd.DataFrame(plot_data)

    # === Plotting ===
    fig, ax = plt.subplots(figsize=(6, 4))

    # Custom cluster colors
    colors = {"Cluster 0": "#66c2a5", "Cluster 1": "#fc8d62"}

    # Bar plot with error bars
    x = np.arange(len(features))
    width = 0.35
    for i, cluster in enumerate(sorted(df_plot["Cluster"].unique())):
        vals = df_plot[df_plot["Cluster"] == cluster]
        ax.bar(x + i * width, vals["Mean"], width,
               yerr=vals["SEM"], capsize=5,
               label=cluster, color=colors.get(cluster, f"C{i}"))

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(features, rotation=15, ha='right')
    ax.set_ylabel("Mean ± SEM")
    ax.set_title("Behavioral Profile by Cluster")
    ax.legend(loc='upper right', frameon=True)

    fig.tight_layout()

    # === Save ===
    base = os.path.splitext(os.path.basename(file_path))[0]
    input_dir = os.path.dirname(file_path)
    output_png = os.path.join(input_dir, f"{base}_cluster_profile.png")
    output_eps = os.path.join(input_dir, f"{base}_cluster_profile.eps")

    fig.savefig(output_png, dpi=300)
    fig.savefig(output_eps, format="eps")

    plt.show()

if __name__ == '__main__':
    main()
