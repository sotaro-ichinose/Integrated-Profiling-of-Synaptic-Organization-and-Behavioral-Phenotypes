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
    output_path = os.path.splitext(csv_path)[0] + "_selected_bin_correlations_1bin.csv"

    # Depth-bin settings: (marker_name, lower_bound, upper_bound)
    selected_bins = [
        ("PSD_num", 10, 20),
        ("gephyrin_num", 10, 20)
    ]

    # Behavioral metrics
    behavior_cols = [
        "YM Alternation Rate",
        "YM Re-entry Ratio",
        "SI Time with stranger",
        "OF Total Distance",
        "OF Center Region",
        "EPM Open Arm Time"
    ]

    # === Load CSV ===
    df = pd.read_csv(csv_path)

    # === Extract mean values for selected bins ===
    synapse_data = {}
    for marker, lower, upper in selected_bins:
        cols_in_bin = [
            col for col in df.columns
            if col.startswith(marker) and
               lower <= int(col.split("_")[-1]) < upper
        ]
        if cols_in_bin:
            bin_label = f"{marker}_{lower}-{upper}"
            synapse_data[bin_label] = df[cols_in_bin].mean(axis=1)

    # === Pearson correlation ===
    results = []
    for beh in behavior_cols:
        for syn_label, syn_series in synapse_data.items():
            x = df[beh]
            y = syn_series
            mask = x.notna() & y.notna()
            if mask.sum() < 3:
                r, p = np.nan, np.nan
            else:
                r, p = pearsonr(x[mask], y[mask])
            results.append((beh, syn_label, r, p))

    res_df = pd.DataFrame(results, columns=["Behavior", "Synapse_Bin", "r", "p_raw"])

    # === FDR correction ===
    reject, q_vals, _, _ = multipletests(res_df["p_raw"], alpha=alpha_fdr, method="fdr_bh")
    res_df["q_fdr"] = q_vals
    res_df["significant"] = reject
    res_df["sig_mark"] = res_df["significant"].map({True: "*", False: ""})

    # === Save output ===
    res_df.to_csv(output_path, index=False)
    print(f"✅ Correlation table saved to: {output_path}")

if __name__ == "__main__":
    main()
