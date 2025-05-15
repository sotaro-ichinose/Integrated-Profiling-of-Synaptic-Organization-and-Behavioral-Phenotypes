#!/usr/bin/env python3
"""
Statistical and visualization pipeline for depth-wise synapse density comparisons.
Includes Games-Howell test with FDR correction and layered line plot generation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
import tkinter as tk
from tkinter import filedialog, simpledialog
from statsmodels.stats.multitest import multipletests
import matplotlib.colors as mcolors
import pingouin as pg

# === Plot style ===
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 12,
    'axes.labelsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12
})

def lighten_color(color, amount=0.5):
    """Brighten or lighten a color."""
    try:
        c = mcolors.cnames[color]
    except:
        c = color
    c = mcolors.to_rgb(c)
    return [(1.0 - (1.0 - x) * (1 - amount)) for x in c]

def run_gameshowell_and_add_pvalues(df):
    """Run Games-Howell test and compute FDR-corrected q-values for VPA+E vs VPA."""
    depth_values = df["Depth (%)"]
    p_values = []

    for i, depth in enumerate(depth_values):
        row_data = df.iloc[i]
        values, group1, group2 = [], [], []

        for col in df.columns:
            match = re.match(r"(\d{2}_ACC)\s(WT|WT\+E|VPA|VPA\+E)(\d+)$", col)
            if match:
                label = match.group(2)
                val = row_data[col]
                if pd.isna(val): continue
                values.append(val)
                group1.append("WT" if "WT" in label else "VPA")
                group2.append("+E" if "+E" in label else "None")

        if len(values) < 2:
            p_values.append(np.nan)
            continue

        anova_df = pd.DataFrame({
            "value": values,
            "group1": group1,
            "group2": group2
        })
        anova_df["group_combined"] = anova_df["group1"] + "_" + anova_df["group2"]

        gh_results = pg.pairwise_gameshowell(
            data=anova_df,
            dv="value",
            between="group_combined"
        )

        p_val = gh_results.query(
            '(A == "VPA_None" and B == "VPA_+E") or (A == "VPA_+E" and B == "VPA_None")'
        )["pval"].values

        p_values.append(p_val[0] if len(p_val) > 0 else np.nan)

    df["P_value_VPA_vs_VPA+E"] = p_values

    # Apply FDR correction
    pvals = df["P_value_VPA_vs_VPA+E"].fillna(1.0).values
    _, qvals, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')
    df["q_fdr"] = qvals

    return df

def plot_results(df, csv_path):
    """Generate layered line plot with mean ± SD and significance stars."""
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    output_dir = os.path.dirname(csv_path)

    root = tk.Tk()
    root.withdraw()
    xlabel = simpledialog.askstring("X-axis label", "Enter label for X-axis:", initialvalue="Density of gephyrin (/$\mu$m$^2$)")
    ylabel = simpledialog.askstring("Y-axis label", "Enter label for Y-axis:", initialvalue="Depth (%)")

    depth = df["Depth (%)"]
    group_cols = {
        "WT": [col for col in df.columns if "WT" in col and "+E" not in col and "_Mean" not in col],
        "WT+E": [col for col in df.columns if "WT+E" in col and "_Mean" not in col],
        "VPA": [col for col in df.columns if "VPA" in col and "+E" not in col and "_Mean" not in col],
        "VPA+E": [col for col in df.columns if "VPA+E" in col and "_Mean" not in col],
    }

    colors = {"WT": '#9999FF', "WT+E": '#0000FF', "VPA": '#FF9999', "VPA+E": '#FF0000'}
    linestyles = {"WT": "dashed", "WT+E": "solid", "VPA": "dashed", "VPA+E": "solid"}

    fig, ax = plt.subplots(figsize=(3.5, 3))

    for group in ["WT", "VPA", "WT+E", "VPA+E"]:
        mean_col, sd_col = f"{group}_Mean", f"{group}_SD"
        if mean_col in df.columns and sd_col in df.columns:
            ax.plot(df[mean_col], depth, color=colors[group], linewidth=2, label=group, linestyle=linestyles[group])
            fill_color = lighten_color(colors[group], amount=0.5)
            ax.fill_betweenx(depth, df[mean_col] - df[sd_col], df[mean_col] + df[sd_col], color=fill_color)

    # Annotate FDR-corrected significance
    if 'q_fdr' in df.columns:
        for i, (q, yval) in enumerate(zip(df['q_fdr'], df['Depth (%)'])):
            if pd.notnull(q):
                stars = '***' if q < 0.001 else '**' if q < 0.01 else '*' if q < 0.05 else ''
                if stars:
                    max_val = max(
                        df.loc[i, 'VPA+E_Mean'] + df.loc[i, 'VPA+E_SD'] if 'VPA+E_Mean' in df.columns else 0,
                        df.loc[i, 'VPA_Mean'] + df.loc[i, 'VPA_SD'] if 'VPA_Mean' in df.columns else 0
                    )
                    ax.text(max_val + 0.05, yval, stars, fontsize=12, verticalalignment='center', color='black')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.invert_yaxis()
    ax.set_xlim(left=0)
    ax.set_ylim(top=0)
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.legend(loc='upper left')
    ax.grid(False)

    png_path = os.path.join(output_dir, f"{base_name}_E_comparison.png")
    eps_path = os.path.join(output_dir, f"{base_name}_E_comparison.eps")
    df_out_path = os.path.join(output_dir, f"{base_name}_p_q.csv")

    plt.savefig(png_path, bbox_inches='tight')
    plt.savefig(eps_path, format='eps', bbox_inches='tight')
    plt.close()
    df.to_csv(df_out_path, index=False)

    print(f"✅ Done:\n - Plots: {png_path}, {eps_path}\n - Statistics CSV: {df_out_path}")

def main():
    """Main entry point: select CSV, process statistics, and plot results."""
    root = tk.Tk()
    root.withdraw()
    csv_path = filedialog.askopenfilename(
        title="📄 Select processed CSV",
        filetypes=[("CSV Files", "*.csv")]
    )

    if not csv_path:
        print("⛔ No file selected. Exiting.")
        return

    df = pd.read_csv(csv_path)
    df = run_gameshowell_and_add_pvalues(df)
    plot_results(df, csv_path)

if __name__ == "__main__":
    main()
