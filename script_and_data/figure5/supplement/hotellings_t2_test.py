#!/usr/bin/env python3
"""
Performs Hotelling’s T² test in PCA space between two user-selected groups via dropdown menus.
"""

import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import f
from numpy.linalg import inv
import tkinter as tk
from tkinter import filedialog

def select_file():
    root = tk.Tk()
    root.withdraw()
    return filedialog.askopenfilename(
        title="Select CSV for PCA and Hotelling's T² Test",
        filetypes=[("CSV files", "*.csv")]
    )

def select_two_groups(options):
    win = tk.Tk()
    win.title("Select Two Groups")
    win.geometry("300x150")

    var1 = tk.StringVar(win); var2 = tk.StringVar(win)
    var1.set(options[0]); var2.set(options[1])

    tk.Label(win, text="Group 1:").pack()
    tk.OptionMenu(win, var1, *options).pack()
    tk.Label(win, text="Group 2:").pack()
    tk.OptionMenu(win, var2, *options).pack()

    def on_ok():
        win.quit()
    tk.Button(win, text="OK", command=on_ok).pack(pady=10)

    win.mainloop()
    g1, g2 = var1.get(), var2.get()
    win.destroy()
    return g1, g2

def hotelling_t2(df, group1, group2):
    # extract PC1/PC2
    data1 = df.loc[df['Group']==group1, ['PC1','PC2']].values
    data2 = df.loc[df['Group']==group2, ['PC1','PC2']].values

    n1, n2 = len(data1), len(data2)
    m1, m2 = data1.mean(axis=0), data2.mean(axis=0)
    S1 = np.cov(data1, rowvar=False)
    S2 = np.cov(data2, rowvar=False)
    Sp = ((n1-1)*S1 + (n2-1)*S2) / (n1+n2-2)
    Sp_inv = inv(Sp)
    diff = m1 - m2

    # T²
    T2 = (n1*n2)/(n1+n2) * diff.T @ Sp_inv @ diff

    # convert to F
    p = 2
    F_stat = (n1+n2-p-1)*T2 / ((n1+n2-2)*p)
    df1_f, df2_f = p, (n1+n2-p-1)
    p_val = 1 - f.cdf(F_stat, df1_f, df2_f)

    return T2, F_stat, df1_f, df2_f, p_val

def main():
    file_path = select_file()
    if not file_path:
        print("❌ No file selected. Exiting.")
        return

    df = pd.read_csv(file_path)

    # PCA on all features except Group/MouseID
    features = [c for c in df.columns if c not in ('Group','MouseID')]
    X = StandardScaler().fit_transform(df[features])
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(X)
    df['PC1'], df['PC2'] = pcs[:,0], pcs[:,1]

    # choose groups
    groups = sorted(df['Group'].dropna().unique())
    g1, g2 = select_two_groups(groups)
    if g1==g2 or g1 not in groups or g2 not in groups:
        print("❌ Please select two different valid groups. Exiting.")
        return

    # compute Hotelling’s T²
    T2, F_stat, df1_f, df2_f, p_val = hotelling_t2(df, g1, g2)

    # print
    print("\n=== Hotelling’s T² Test ===")
    print(f"Groups: {g1} vs {g2}")
    print(f"T²: {T2:.4f}")
    print(f"F-statistic: {F_stat:.4f}")
    print(f"d.f.: ({df1_f}, {df2_f})")
    print(f"p-value: {p_val:.4f}\n")

    # save
    out = pd.DataFrame([{
        'Group1': g1, 'Group2': g2,
        'T2_statistic': T2,
        'F_statistic': F_stat,
        'df1': df1_f, 'df2': df2_f,
        'p_value': p_val
    }])
    out_path = os.path.join(
        os.path.dirname(file_path),
        f"Hotelling_{g1}_vs_{g2}.csv"
    )
    out.to_csv(out_path, index=False)
    print(f"✅ Results saved to: {out_path}")

if __name__ == "__main__":
    main()
