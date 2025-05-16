#!/usr/bin/env python3
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, cross_val_score, permutation_test_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import os

def main():
    # === GUI for file selection ===
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select CSV file",
        filetypes=[("CSV files", "*.csv")]
    )
    if not file_path:
        print("❌ No file selected. Exiting.")
        return

    # === Settings ===
    group_column = "Group"
    n_components = 8
    n_splits = 5
    n_permutations = 1000
    output_csv = os.path.splitext(file_path)[0] + "_classification_with_pvalues.csv"

    # === Load and preprocess data ===
    df = pd.read_csv(file_path)
    exclude_columns = ["Group", "MouseID"]
    feature_cols = [c for c in df.columns
                    if df[c].dtype in ['float64', 'int64'] and c not in exclude_columns]
    X = df[feature_cols].values
    y = df[group_column].values

    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    # === Classifiers ===
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "SVM (RBF Kernel)": SVC(kernel='rbf', probability=True, random_state=42)
    }

    # === Evaluation ===
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = []

    for name, model in models.items():
        acc_scores = cross_val_score(model, X_pca, y, cv=cv, scoring='accuracy')
        auc_scores = cross_val_score(model, X_pca, y, cv=cv, scoring='roc_auc')

        # Permutation test
        score, perm_scores, p_value = permutation_test_score(
            model, X_pca, y,
            scoring='accuracy',
            cv=cv,
            n_permutations=n_permutations,
            random_state=42,
            n_jobs=-1
        )

        results.append({
            "Model": name,
            "Accuracy_Mean": acc_scores.mean(),
            "Accuracy_SD": acc_scores.std(),
            "ROC_AUC_Mean": auc_scores.mean(),
            "ROC_AUC_SD": auc_scores.std(),
            "Permutation_Accuracy": score,
            "Permutation_p_value": p_value
        })

        print(f"=== {name} ===")
        print(f"Accuracy: {acc_scores.mean():.3f} ± {acc_scores.std():.3f}")
        print(f"ROC AUC : {auc_scores.mean():.3f} ± {auc_scores.std():.3f}")
        print(f"Permutation Accuracy: {score:.3f}, p = {p_value:.4f}\n")

    # === Save results ===
    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f"✅ Results saved to: {output_csv}")

if __name__ == "__main__":
    main()
