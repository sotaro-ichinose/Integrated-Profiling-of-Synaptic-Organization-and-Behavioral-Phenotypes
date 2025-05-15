# Required libraries (install if needed):
# pip install pandas scikit-learn matplotlib seaborn pingouin scikit-posthocs

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import tkinter as tk
from tkinter import filedialog
import pingouin as pg
import itertools

# === Plot configuration ===
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 12,
    'axes.labelsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12
})

def main():
    # === Select CSV via GUI ===
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if not file_path:
        print("❌ No file selected.")
        return

    # === Load data ===
    df = pd.read_csv(file_path)
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    if not numeric_cols:
        raise ValueError("No numeric columns found.")

    # === Let user choose Y-axis column ===
    def get_y_axis_column():
        sel = tk.Tk()
        sel.title("Select column for Y-axis")
        variable = tk.StringVar(sel)
        variable.set(numeric_cols[0])
        tk.Label(sel, text="Select a column for Y-axis:").pack()
        option_menu = tk.OptionMenu(sel, variable, *numeric_cols)
        option_menu.pack()
        tk.Button(sel, text="OK", command=sel.quit).pack()
        sel.mainloop()
        y_col = variable.get()
        sel.destroy()
        return y_col

    y_col = get_y_axis_column()

    # === Column check ===
    if not {'Group', 'Excercise'}.issubset(df.columns):
        raise ValueError("Both 'Group' and 'Excercise' columns are required.")

    # === Create combined group labels ===
    df['Group_Excercise'] = df['Group'].astype(str) + "_" + df['Excercise'].astype(str)

    # === Statistical analysis ===
    # Games-Howell test
    print("\n--- Games-Howell Test ---")
    gh_results = pg.pairwise_gameshowell(data=df, dv=y_col, between='Group_Excercise')
    print(gh_results)

    # Hedges' g
    def compute_hedges_g(x1, x2):
        n1, n2 = len(x1), len(x2)
        s1, s2 = np.var(x1, ddof=1), np.var(x2, ddof=1)
        spooled = np.sqrt(((n1-1)*s1 + (n2-1)*s2) / (n1+n2-2))
        d = (np.mean(x1) - np.mean(x2)) / spooled
        correction = 1 - (3 / (4*(n1+n2) - 9))
        return d * correction

    groups = df.groupby('Group_Excercise')
    pair_results = []
    for (grp1, data1), (grp2, data2) in itertools.combinations(groups, 2):
        g = compute_hedges_g(data1[y_col].values, data2[y_col].values)
        pair_results.append({'Group1': grp1, 'Group2': grp2, "Hedges' g": g})

    hedges_df = pd.DataFrame(pair_results)

    # Merge results
    merged_results = pd.merge(
        gh_results[['A', 'B', 'pval']],
        hedges_df,
        left_on=['A', 'B'],
        right_on=['Group1', 'Group2'],
        how='left'
    )

    # === Visualization ===
    plt.figure(figsize=(3.5, 3))
    sns.set(style="white")

    sns.barplot(
        data=df, x='Group', y=y_col, hue='Excercise',
        estimator='mean', errorbar='sd',
        capsize=0.1, palette='Set2', alpha=0.8,
        err_kws={'linewidth': 1.5},
        order=['WT', 'VPA']
    )

    sns.stripplot(
        data=df, x='Group', y=y_col, hue='Excercise',
        palette='dark:k', dodge=True, alpha=0.5,
        jitter=False, linewidth=0.5, legend=False
    )

    # Annotate VPA(+) vs VPA(-)
    vpapos_vs_vpaneg = merged_results[
        ((merged_results['A'] == 'VPA_(+)') & (merged_results['B'] == 'VPA_(-)')) |
        ((merged_results['A'] == 'VPA_(-)') & (merged_results['B'] == 'VPA_(+)'))
    ]

    if not vpapos_vs_vpaneg.empty:
        pval = vpapos_vs_vpaneg['pval'].values[0]
        g_val = vpapos_vs_vpaneg["Hedges' g"].values[0]
        xpos1, xpos2 = 0.8, 1.2
        y_max = df[y_col].max()
        h = y_max * 1.1
        plt.plot([xpos1, xpos1, xpos2, xpos2], [h, h+0.02*h, h+0.02*h, h], lw=1.5, c='black')
        plt.text((xpos1+xpos2)/2, h+0.12*h,
                 f"p = {pval:.3f}\ng = {g_val:.2f}",
                 ha='center', va='bottom', fontsize=10)

    plt.xlabel('')
    plt.ylabel(y_col)
    plt.legend(title='Exercise')
    sns.despine()

    # === Save plot ===
    output_dir = os.path.dirname(file_path)
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    png_path = os.path.join(output_dir, f"{base_name}_{y_col}.png")
    eps_path = os.path.join(output_dir, f"{base_name}_{y_col}.eps")
    plt.tight_layout()
    plt.savefig(png_path, bbox_inches='tight')
    plt.savefig(eps_path, format='eps', bbox_inches='tight')
    plt.close()

    # === Save statistics ===
    stat_path = os.path.join(output_dir, f"{base_name}_{y_col}_p_g.csv")
    merged_results.to_csv(stat_path, index=False)
    print(f"\n✅ Statistical results saved to: {stat_path}")

# Entry point
if __name__ == "__main__":
    main()
