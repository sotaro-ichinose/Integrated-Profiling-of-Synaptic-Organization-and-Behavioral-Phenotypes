import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import tkinter as tk
from tkinter import filedialog
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, LeaveOneOut
from scipy.stats import ttest_ind

def main():
    # === GUI to select CSV ===
    root = tk.Tk()
    root.withdraw()
    csv_path = filedialog.askopenfilename(
        title="Select a CSV file for LDA",
        filetypes=[("CSV files", "*.csv")]
    )
    if not csv_path:
        print("❌ No file selected. Exiting.")
        return

    # === Load data ===
    df = pd.read_csv(csv_path)
    group_column = "Group"
    output_dir = os.path.dirname(csv_path)

    # Exclude group and MouseID columns
    exclude_columns = [group_column, "MouseID"]
    features = df.drop(columns=exclude_columns, errors="ignore").select_dtypes(include='number')
    X = features.values
    y = df[group_column].values

    # === LDA with LOOCV ===
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("lda", LinearDiscriminantAnalysis())
    ])
    loo = LeaveOneOut()
    scores = cross_val_score(pipeline, X, y, cv=loo)
    print(f"LOOCV Accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

    # === Fit LDA to all data ===
    lda = LinearDiscriminantAnalysis()
    lda.fit(StandardScaler().fit_transform(X), y)
    coef = lda.coef_[0]
    feature_names = features.columns

    # === Save coefficients ===
    coef_path = os.path.join(output_dir, "LDA_coefficients.csv")
    coef_df = pd.DataFrame({
        "Feature": feature_names,
        "LDA_Coefficient": coef
    })
    coef_df.to_csv(coef_path, index=False)
    print(f"✅ LDA coefficients saved to: {coef_path}")

    # === Coefficient plot ===
    plt.figure(figsize=(8, 3))
    plt.barh(feature_names, coef)
    plt.axvline(0, color='gray', linestyle='--')
    plt.title("LDA Coefficients (Feature Contribution to Group Separation)")
    plt.xlabel("Coefficient")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()

    # === LDA projection (1D) ===
    X_scaled = StandardScaler().fit_transform(X)
    X_lda = lda.transform(X_scaled)

    plt.figure(figsize=(4, 3))
    colors = {'WT': '#FC8D62', 'VPA': '#66C2A5'}
    for group in np.unique(y):
        idx = y == group
        plt.scatter(X_lda[idx], [0] * np.sum(idx), label=group, color=colors[group], s=100)

    plt.xlabel("LDA Score")
    plt.yticks([])
    plt.axvline(0, color='gray', linestyle='--')
    plt.title("LDA Score Distribution by Group")
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("LDA class order:", np.unique(y))

    # === t-test comparison ===
    group1 = df[df[group_column] == "WT"]
    group2 = df[df[group_column] == "VPA"]
    results = []

    for col in features.columns:
        wt_vals = group1[col].values
        vpa_vals = group2[col].values
        wt_mean, vpa_mean = np.mean(wt_vals), np.mean(vpa_vals)
        wt_std, vpa_std = np.std(wt_vals), np.std(vpa_vals)
        t_stat, p_val = ttest_ind(wt_vals, vpa_vals, equal_var=False)
        lda_coef = coef[list(features.columns).index(col)]

        results.append({
            "Feature": col,
            "WT_mean": wt_mean,
            "VPA_mean": vpa_mean,
            "WT > VPA": wt_mean > vpa_mean,
            "t-statistic": t_stat,
            "p-value": p_val,
            "LDA Coefficient": lda_coef,
            "LDA suggests WT > VPA": lda_coef > 0
        })

    comparison_df = pd.DataFrame(results)
    comparison_df = comparison_df.sort_values(by="p-value")
    comparison_path = os.path.join(output_dir, "LDA_vs_ttest_comparison.csv")
    comparison_df.to_csv(comparison_path, index=False)
    print(f"✅ Comparison table saved to: {comparison_path}")

if __name__ == "__main__":
    main()
