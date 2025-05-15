import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog
import os

plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.labelsize': 16,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16
})

# --- Cliff's delta with bootstrap ---
def cliffs_delta(x, y, n_boot=1000):
    nx, ny = len(x), len(y)
    all_pairs = [(xi, yi) for xi in x for yi in y]
    more = sum(1 for xi, yi in all_pairs if xi > yi)
    less = sum(1 for xi, yi in all_pairs if xi < yi)
    delta = (more - less) / (nx * ny)

    boot_deltas = []
    for _ in range(n_boot):
        xb = np.random.choice(x, size=nx, replace=True)
        yb = np.random.choice(y, size=ny, replace=True)
        m = sum(xi > yi for xi in xb for yi in yb)
        l = sum(xi < yi for xi in xb for yi in yb)
        boot_deltas.append((m - l) / (nx * ny))
    lower = np.percentile(boot_deltas, 2.5)
    upper = np.percentile(boot_deltas, 97.5)
    return delta, lower, upper

# --- Main routine ---
def main():
    # === Select CSV ===
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select behavioral CSV file",
        filetypes=[("CSV files", "*.csv")]
    )
    if not file_path:
        raise SystemExit("❌ No file selected.")

    df = pd.read_csv(file_path)
    group_col = 'Z_Resilient_Label'

    # Filter only VPA resilient and non resilient
    df = df[df[group_col].isin(["VPA resilient", "VPA non resilient"])].copy()
    group1 = "VPA non resilient"
    group2 = "VPA resilient"

    # Limit to 8 specific features
    selected_features = [
        "Density of PSD-95 (10% Depth)",
        "Density of gephyrin (10% Depth)",
        "YM Alternation Rate",
        "YM Re-entry Ratio",
        "SI Time with stranger",
        "OF Total Distance",
        "OF Center Region",
        "EPM Open Arm Time"
    ]

    # === Compute Cliff’s Δ for each feature ===
    results = []
    for feat in selected_features:
        if feat not in df.columns:
            continue
        x = df[df[group_col] == group1][feat].dropna().values
        y = df[df[group_col] == group2][feat].dropna().values
        if len(x) < 2 or len(y) < 2:
            continue
        d, lo, hi = cliffs_delta(x, y)
        results.append((feat, d, lo, hi))

    df_res = pd.DataFrame(results, columns=["Feature", "Delta", "Lower", "Upper"])
    df_res = df_res.sort_values("Delta")

    # === Plot ===
    fig, ax = plt.subplots(figsize=(10, len(df_res) * 0.4 + 1))
    y = np.arange(len(df_res))

    ax.errorbar(df_res["Delta"], y,
                xerr=[df_res["Delta"] - df_res["Lower"], df_res["Upper"] - df_res["Delta"]],
                fmt='o', capsize=4)

    ax.axvline(0, linestyle='--', color = "gray")
    ax.set_yticks(y)
    ax.set_yticklabels(df_res["Feature"])
    ax.set_xlabel("Cliff’s Delta")
    ax.set_title("Behavioral differences (Cliff’s Δ)\nVPA resilient vs non resilient")

    fig.tight_layout()

    # === Save ===
    base = os.path.splitext(os.path.basename(file_path))[0]
    input_dir = os.path.dirname(file_path)
    output_png = os.path.join(input_dir, f"{base}_cliffs_delta_vpaonly.png")
    output_eps = os.path.join(input_dir, f"{base}_cliffs_delta_vpaonly.eps")
    output_csv = os.path.join(input_dir, f"{base}_cliffs_delta_vpaonly.csv")

    fig.savefig(output_png, dpi=300)
    fig.savefig(output_eps, format="eps")
    df_res.to_csv(output_csv, index=False)
    plt.show()

    print(f"✅ Saved figure: {output_png}, {output_eps}")
    print(f"✅ Saved results: {output_csv}")

main()
