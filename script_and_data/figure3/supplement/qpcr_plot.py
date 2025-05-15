#!/usr/bin/env python3
"""
Visualize qPCR fold change (2^-ΔΔCt) per gene with WT vs VPA comparison.

- Loads a CSV file via GUI.
- Computes group means, SEM, and Welch's t-test on ΔΔCt.
- Displays bar plots with individual points and p-value annotations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import filedialog
from scipy.stats import ttest_ind

# Global plotting style
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 12,
    'axes.labelsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12
})

def main():
    # === File selection via GUI ===
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select qPCR result CSV",
        filetypes=[("CSV files", "*.csv")]
    )
    if not file_path:
        print("❌ No file selected. Exiting.")
        return

    # === Load CSV with Japanese encoding support ===
    df = pd.read_csv(file_path, encoding="shift_jis")
    df.columns = df.columns.str.strip()

    # === Validate required columns ===
    required_cols = {"Sample Name", "Target Name", "ΔΔCт"}
    if not required_cols.issubset(df.columns):
        print("❌ Required columns not found: Sample Name, Target Name, ΔΔCт")
        return

    # === Infer group from Sample Name ===
    df['Group'] = df['Sample Name'].apply(lambda x: 'WT' if str(x).startswith('WT') else 'VPA')
    df['FoldChange'] = 2 ** (-df['ΔΔCт'])

    # === Loop through each gene ===
    genes = df['Target Name'].unique()
    palette = sns.color_palette("Set2", 2)

    for gene in genes:
        sub = df[df['Target Name'] == gene]

        # Compute mean and SEM per group
        summary = (
            sub.groupby('Group')
            .agg(
                FoldChange_Mean=('FoldChange', 'mean'),
                FoldChange_SE=('FoldChange', lambda x: x.std(ddof=1) / np.sqrt(len(x)))
            )
            .reindex(['WT', 'VPA'])  # Ensure consistent group order
            .reset_index()
        )

        # === Bar plot with individual dots ===
        fig, ax = plt.subplots(figsize=(3.6, 3))
        ax.bar(
            summary['Group'],
            summary['FoldChange_Mean'],
            yerr=summary['FoldChange_SE'],
            capsize=5,
            color=[palette[0], palette[1]],
            edgecolor='none',
            width=0.6
        )

        for i, group in enumerate(['WT', 'VPA']):
            points = sub[sub['Group'] == group]['FoldChange']
            ax.scatter([i] * len(points), points, color='black', zorder=2, alpha=0.8)

        ax.set_ylabel("Fold Change (2$^{-ΔΔC_T}$)")
        ax.set_title(f"{gene} expression")
        ax.set_ylim(bottom=0)
        ax.spines[['right', 'top']].set_visible(False)

        # === Statistical test on ΔΔCt ===
        wt_vals = sub[sub['Group'] == 'WT']['ΔΔCт']
        vpa_vals = sub[sub['Group'] == 'VPA']['ΔΔCт']
        _, p_val = ttest_ind(wt_vals, vpa_vals, equal_var=False)

        # Annotate p-value above bars
        max_y = summary['FoldChange_Mean'].max() + summary['FoldChange_SE'].max()
        if np.isnan(p_val):
            asterisk = ''
            p_text = "n/a"
        else:
            asterisk = (
                '***' if p_val < 0.001 else
                '**' if p_val < 0.01 else
                '*' if p_val < 0.05 else 'n.s.'
            )
            p_text = f"{asterisk}\n(p = {p_val:.4f})" if p_val >= 0.0001 else f"{asterisk}\n(p < 0.0001)"
        ax.text(0.5, max_y * 0.96, p_text, ha='center', va='bottom')

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
