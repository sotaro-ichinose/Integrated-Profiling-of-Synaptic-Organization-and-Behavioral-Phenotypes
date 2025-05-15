#!/usr/bin/env python3

import os
import tkinter as tk
from tkinter import filedialog
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway, levene

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
        title="Select CSV file for Trial data",
        filetypes=[("CSV Files", "*.csv")]
    )
    if not file_path:
        print("❌ No file selected. Exiting.")
        return

    # === Output paths ===
    save_dir = os.path.dirname(file_path)
    base = os.path.splitext(os.path.basename(file_path))[0]

    # === Load data ===
    df = pd.read_csv(file_path)

    # === Analyze spacing consistency ===
    interval_data = df.dropna(subset=['Interval'])
    interval_groups = [g['Interval'].values for _, g in interval_data.groupby('Trial')]
    f_interval = f_oneway(*interval_groups)
    levene_interval = levene(*interval_groups)

    # === Analyze direction consistency ===
    slope_data = df.dropna(subset=['Slope'])
    slope_groups = [g['Slope'].values for _, g in slope_data.groupby('Trial')]
    f_slope = f_oneway(*slope_groups)
    levene_slope = levene(*slope_groups)

    # === Display statistical results ===
    print("\n=== Statistical Results ===")
    spacing_anova_p = f_interval.pvalue
    spacing_levene_p = levene_interval.pvalue
    direction_anova_p = f_slope.pvalue
    direction_levene_p = levene_slope.pvalue

    print(f"[Spacing Consistency]  ANOVA p = {spacing_anova_p:.4f}, Levene p = {spacing_levene_p:.4f}")
    print(f"[Direction Consistency] ANOVA p = {direction_anova_p:.4f}, Levene p = {direction_levene_p:.4f}")

    # === Save statistical results to CSV ===
    stats_df = pd.DataFrame([
        {
            "Measure": "Spacing Consistency",
            "ANOVA_p": spacing_anova_p,
            "Levene_p": spacing_levene_p
        },
        {
            "Measure": "Direction Consistency",
            "ANOVA_p": direction_anova_p,
            "Levene_p": direction_levene_p
        }
    ])
    stats_path = os.path.join(save_dir, f"{base}_consistency_stats.csv")
    stats_df.to_csv(stats_path, index=False)
    print(f"📄 Statistical results saved as: {stats_path}")

    # === Visualization ===
    sns.set_palette("Set2")
    fig, axes = plt.subplots(1, 2, figsize=(7, 3), sharey=False)

    sns.boxplot(data=interval_data, x='Trial', y='Interval', ax=axes[0])
    axes[0].set_title("Spacing Consistency")
    axes[0].set_ylabel("Interval (pixels)")

    sns.boxplot(data=slope_data, x='Trial', y='Slope', ax=axes[1])
    axes[1].set_title("Direction Consistency")
    axes[1].set_ylabel("Slope")

    plt.tight_layout()

    # === Save plot ===
    fig_path = os.path.join(save_dir, f"{base}_consistency_plot.png")
    plt.savefig(fig_path, dpi=300)
    print(f"\n✅ Plot saved as: {fig_path}")
    plt.show()

if __name__ == "__main__":
    main()
