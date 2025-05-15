import os
import tkinter as tk
from tkinter import filedialog, simpledialog
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests

# --- Matplotlib settings ---
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 12,
    'axes.labelsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12
})

def plot_results(csv_path):
    # --- Load data ---
    df = pd.read_csv(csv_path)
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    output_dir = os.path.dirname(csv_path)

    # --- Ask axis labels ---
    root = tk.Tk()
    root.withdraw()
    xlabel = simpledialog.askstring(
        "X-axis label", "Enter label for X-axis:", initialvalue="Density of gephyrin (/$\mu$m$^2$)"
    )
    ylabel = simpledialog.askstring(
        "Y-axis label", "Enter label for Y-axis:", initialvalue="Depth (%)"
    )

    # --- FDR correction on P-values ---
    if 'P_value' in df.columns:
        pvals = df['P_value'].fillna(1.0).values
        _, qvals, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')
        df['q_fdr'] = qvals
        stats_csv = os.path.join(output_dir, f"{base_name}_p_q_values.csv")
        df[['Depth (%)', 'P_value', 'q_fdr']].to_csv(stats_csv, index=False)
        print(f"Saved statistical results with FDR correction: {stats_csv}")

    # --- Identify columns ---
    depth = df["Depth (%)"]
    wt_cols = [col for col in df.columns if "WT" in col and all(x not in col for x in ["_Mean", "_SD", "_N"])]
    vpa_cols = [col for col in df.columns if "VPA" in col and all(x not in col for x in ["_Mean", "_SD", "_N"])]

    # --- Create plot ---
    fig, ax = plt.subplots(figsize=(3.5, 3))

    # Individual traces
    for col in wt_cols:
        ax.plot(df[col], depth, color='#9999FF', linewidth=1)
    for col in vpa_cols:
        ax.plot(df[col], depth, color='#FF9999', linewidth=1)

    # Mean ± SD
    if 'WT_Mean' in df.columns and 'WT_SD' in df.columns:
        ax.plot(df['WT_Mean'], depth, color='blue', label='WT', linewidth=2)
        ax.fill_betweenx(depth,
                         df['WT_Mean'] - df['WT_SD'],
                         df['WT_Mean'] + df['WT_SD'],
                         color='#dcdcff')
    if 'VPA_Mean' in df.columns and 'VPA_SD' in df.columns:
        ax.plot(df['VPA_Mean'], depth, color='red', label='VPA', linewidth=2)
        ax.fill_betweenx(depth,
                         df['VPA_Mean'] - df['VPA_SD'],
                         df['VPA_Mean'] + df['VPA_SD'],
                         color='#ffdcdc')

    # Annotate significance stars based on q_fdr
    if 'q_fdr' in df.columns:
        for i, yval in enumerate(df['Depth (%)']):
            q = df.loc[i, 'q_fdr']
            if q < 0.001:
                stars = '***'
            elif q < 0.01:
                stars = '**'
            elif q < 0.05:
                stars = '*'
            else:
                stars = ''
            if stars:
                max_val = 0
                if 'WT_Mean' in df.columns:
                    max_val = max(max_val, df.loc[i, 'WT_Mean'] + df.loc[i, 'WT_SD'])
                if 'VPA_Mean' in df.columns:
                    max_val = max(max_val, df.loc[i, 'VPA_Mean'] + df.loc[i, 'VPA_SD'])
                ax.text(max_val + 0.05 * max_val, yval, stars,
                        fontsize=12, va='center', color='black')

    # Format axes
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.invert_yaxis()
    ax.grid(False)
    ax.legend()
    ax.set_xlim(left=0)
    ax.set_ylim(top=0)
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # Save figures
    png_path = os.path.join(output_dir, f"{base_name}_laminar.png")
    eps_path = os.path.join(output_dir, f"{base_name}_laminar.eps")
    plt.savefig(png_path, bbox_inches='tight', dpi=300)
    plt.savefig(eps_path, format='eps', bbox_inches='tight')
    plt.close()
    print(f"Plots saved:\n - {png_path}\n - {eps_path}")

def main():
    root = tk.Tk()
    root.withdraw()
    csv_path = filedialog.askopenfilename(
        title="Select CSV file for laminar plotting",
        filetypes=[("CSV Files", "*.csv")]
    )
    if csv_path:
        plot_results(csv_path)
    else:
        print("No file selected.")

if __name__ == "__main__":
    main()
