import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests
import tkinter as tk
from tkinter import filedialog
import os

def main():
    # === GUI for file selection ===
    root = tk.Tk()
    root.withdraw()
    csv_path = filedialog.askopenfilename(
        title="Select CSV file containing behavioral and synaptic data",
        filetypes=[("CSV Files", "*.csv")]
    )
    if not csv_path:
        print("❌ No file selected. Exiting.")
        return

    # === Settings ===
    alpha_fdr = 0.05
    output_path = os.path.splitext(csv_path)[0] + "_selected_bin_correlations_5bin.csv"

    # Define depth bins
    depth_bins = [(0, 10), (10, 20), (20, 40), (40, 70), (70, 100)]

    # Synaptic marker prefixes
    synaptic_markers = ["PSD_num", "gephyrin_num"]

    # Behavioral metrics
    behavior_cols = [
        "YM Alternation Rate",
        "YM Re-entry Ratio",
        "SI Time with stranger",
        "OF Total Distance",
        "OF Center Region",
        "EPM Open Arm Time"
    ]

    # === Load data ===
    df = pd.read_csv(csv_path)

    # === Correlation computation ===
    results = []

    for marker in synaptic_markers:
        for lower, upper in depth_bins:
            # Find all columns matching this bin
            bin_cols = [
                col for col in df.columns
                if col.startswith(marker) and
                   lower <= int(col.split("_")[-1]) < upper
            ]
            if not bin_cols:
                continue

            bin_mean = df[bin_cols].mean(axis=1)
            bin_label = f"{marker}_{lower}-{upper}"

            for behavior in behavior_cols:
                x = df[behavior]
                y = bin_mean
                mask = x.notna() & y.notna()

                if mask.sum() < 3:
                    r, p = np.nan, np.nan
                else:
                    r, p = pearsonr(x[mask], y[mask])
                results.append((behavior, bin_label, r, p))

    res_df = pd.DataFrame(results, columns=["Behavior", "Synapse_Bin", "r", "p_raw"])

    # === FDR correction ===
    reject, q_vals, _, _ = multipletests(res_df["p_raw"], alpha=alpha_fdr, method="fdr_bh")
    res_df["q_fdr"] = q_vals
    res_df["significant"] = reject
    res_df["sig_mark"] = res_df["significant"].map({True: "*", False: ""})

    # === Save result table ===
    res_df.to_csv(output_path, index=False)
    print(f"✅ Correlation table saved to: {output_path}")

if __name__ == "__main__":
    main()
