import os
import tkinter as tk
from tkinter import filedialog
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import numpy as np

plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 12,
    'axes.labelsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12
})

def run_all_barplots():
    # ==== Select CSV file via GUI ====
    root = tk.Tk()
    root.withdraw()
    csv_path = filedialog.askopenfilename(
        title="Select a CSV file",
        filetypes=[("CSV files", "*.csv")]
    )
    if not csv_path:
        print("❌ No file selected.")
        return

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Failed to read CSV: {e}")
        return

    if 'Group' not in df.columns:
        print("❌ 'Group' column not found.")
        return

    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    if not numeric_columns:
        print("❌ No numeric columns found.")
        return

    save_dir = os.path.dirname(csv_path)
    base_name = os.path.splitext(os.path.basename(csv_path))[0]

    stats_output = []  # Store statistical results here

    for col in numeric_columns:
        fig, ax = plt.subplots(figsize=(3.5, 3))

        summary = df.groupby("Group")[col].agg(['mean', 'std']).reindex(['WT', 'VPA'])
        summary.plot(kind='bar', y='mean', yerr='std', ax=ax,
                     legend=False, color=['#66C2A5', '#FC8D62'], capsize=5, zorder=2)

        for i, group in enumerate(['WT', 'VPA']):
            y_vals = df[df['Group'] == group][col].dropna().values
            ax.scatter([i] * len(y_vals), y_vals, color='black', zorder=3)

        wt_vals = df[df['Group'] == 'WT'][col].dropna()
        vpa_vals = df[df['Group'] == 'VPA'][col].dropna()
        if len(wt_vals) > 1 and len(vpa_vals) > 1:
            _, p_val = ttest_ind(wt_vals, vpa_vals, equal_var=False)
        else:
            p_val = np.nan

        max_y = max(df[col].max(), (summary['mean'] + summary['std']).max())
        ax.plot([0, 1], [max_y * 1.1] * 2, color='black')

        if np.isnan(p_val):
            asterisk = ''
            p_text = "n/a"
        else:
            asterisk = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'n.s.'
            p_text = f"{asterisk}\n(p = {p_val:.4f})" if p_val >= 0.0001 else f"{asterisk}\n(p < 0.0001)"

        ax.text(0.5, max_y * 1.12, p_text, ha='center', va='bottom')

        ax.set_ylabel(col)
        ax.set_xlabel("")
        ax.set_title("")
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='x', labelrotation=0)

        col_safe = col.replace(" ", "_").replace("(", "").replace(")", "").replace("²", "2").replace("μ", "u")
        png_path = os.path.join(save_dir, f"{base_name}_{col_safe}_plot.png")
        eps_path = os.path.join(save_dir, f"{base_name}_{col_safe}_plot.eps")
        fig.tight_layout()
        fig.savefig(png_path, dpi=300)
        fig.savefig(eps_path, format='eps')
        plt.close()

        print(f"✅ Saved plot for {col}: {png_path}")

        stats_output.append(
            f"{col}:\n"
            f"  WT  = {wt_vals.mean():.3f} ± {wt_vals.std():.3f} (n={len(wt_vals)})\n"
            f"  VPA = {vpa_vals.mean():.3f} ± {vpa_vals.std():.3f} (n={len(vpa_vals)})\n"
            f"  p = {p_val:.4f}  ({asterisk})\n"
        )

    # ==== Save statistical summary as CSV ====
    import csv

    stats_csv_path = os.path.join(save_dir, f"{base_name}_summary_stats.csv")

    with open(stats_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Variable', 'WT Mean', 'WT SD', 'WT N',
                        'VPA Mean', 'VPA SD', 'VPA N', 'p-value', 'Significance'])

        for col in numeric_columns:
            wt_vals = df[df['Group'] == 'WT'][col].dropna()
            vpa_vals = df[df['Group'] == 'VPA'][col].dropna()
            if len(wt_vals) > 1 and len(vpa_vals) > 1:
                _, p_val = ttest_ind(wt_vals, vpa_vals, equal_var=False)
            else:
                p_val = np.nan

            asterisk = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'n.s.' if not np.isnan(p_val) else ''

            writer.writerow([
                col,
                f"{wt_vals.mean():.3f}", f"{wt_vals.std():.3f}", len(wt_vals),
                f"{vpa_vals.mean():.3f}", f"{vpa_vals.std():.3f}", len(vpa_vals),
                f"{p_val:.4f}" if not np.isnan(p_val) else 'n/a',
                asterisk
            ])

    print(f"📄 Saved summary statistics as CSV: {stats_csv_path}")


if __name__ == '__main__':
    run_all_barplots()
