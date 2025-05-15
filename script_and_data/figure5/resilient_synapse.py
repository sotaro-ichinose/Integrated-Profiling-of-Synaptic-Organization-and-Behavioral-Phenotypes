import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tkinter import Tk, filedialog
import os

# === Plot style ===
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.labelsize': 16,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16
})

def main():
    # === File selection dialog ===
    Tk().withdraw()
    file_path = filedialog.askopenfilename(title="Select CSV with Z_Resilient_Label and synapse densities")
    if not file_path:
        raise SystemExit("❌ No file selected.")

    df = pd.read_csv(file_path)
    base = os.path.splitext(os.path.basename(file_path))[0]

    # === Variables and group labels ===
    group_col = "Z_Resilient_Label"
    features = ["Density of PSD-95 (90% Depth)", "Density of gephyrin (90% Depth)"]
    groups = ["VPA resilient", "VPA non resilient"]
    colors = {"VPA resilient": "#66c2a5", "VPA non resilient": "#fc8d62"}

    # === Plotting ===
    fig, axes = plt.subplots(1, 2, figsize=(8, 4.2))
    for i, feat in enumerate(features):
        ax = axes[i]
        for g in groups:
            vals = df[df[group_col] == g][feat].dropna()
            x = 0 if g == groups[0] else 1
            ax.bar(x, vals.mean(), yerr=vals.std(), color=colors[g], width=0.6, alpha=0.7, label=g, capsize=5)
            ax.scatter([x] * len(vals), vals, color='black', zorder=10)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Resilient", "Non-resilient"])
        ax.set_ylabel(feat)
        ax.set_title(feat.split("(")[0].strip())

    fig.suptitle("Deep-layer synapse density by resilience status")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    # === Save ===
    base = os.path.splitext(os.path.basename(file_path))[0]
    input_dir = os.path.dirname(file_path)
    output_png = os.path.join(input_dir, f"{base}_bar.png")
    output_eps = os.path.join(input_dir, f"{base}_bar.eps")

    fig.savefig(output_png, dpi=300)
    fig.savefig(output_eps, format="eps")
    plt.show()

    print(f"✅ Saved: {output_png}, {output_eps}")

if __name__ == "__main__":
    main()
