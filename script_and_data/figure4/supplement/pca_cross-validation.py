import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import tkinter as tk
from tkinter import filedialog

def main():
    # === Plot settings ===
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 12,
        'axes.labelsize': 12,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12
    })

    # === GUI to select CSV file ===
    root = tk.Tk()
    root.withdraw()
    csv_path = filedialog.askopenfilename(
        title="Select CSV file for PCA cross-validation",
        filetypes=[("CSV Files", "*.csv")]
    )
    if not csv_path:
        print("❌ No file selected. Exiting.")
        return

    # === Load and standardize data ===
    df = pd.read_csv(csv_path)

    # Exclude non-feature columns (e.g., Group and MouseID)
    exclude_columns = ["Group", "MouseID"]
    feature_cols = [col for col in df.columns
                    if df[col].dtype in ['float64', 'int64'] and col not in exclude_columns]
    X = df[feature_cols].values
    X_scaled = StandardScaler().fit_transform(X)

    # === Parameters ===
    n_features = X_scaled.shape[1]
    max_components = min(n_features, 20)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    mean_errors = []
    std_errors = []

    # === Cross-validation: evaluate reconstruction error for 1 to max_components ===
    for n_comp in range(1, max_components + 1):
        mse_list = []
        for train_idx, test_idx in kf.split(X_scaled):
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]

            pca = PCA(n_components=n_comp)
            pca.fit(X_train)

            X_test_proj = pca.transform(X_test)
            X_test_reconstructed = pca.inverse_transform(X_test_proj)

            mse = mean_squared_error(X_test, X_test_reconstructed)
            mse_list.append(mse)

        mean_errors.append(np.mean(mse_list))
        std_errors.append(np.std(mse_list))

    # === Plot ===
    plt.figure(figsize=(4, 3))
    plt.errorbar(
        range(1, max_components + 1),
        mean_errors,
        yerr=std_errors,
        fmt='-o',
        capsize=4,
        label="Reconstruction Error ± SD"
    )
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Reconstruction Error (MSE)")
    plt.title("PCA Cross-Validation")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
