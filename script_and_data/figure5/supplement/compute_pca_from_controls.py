#!/usr/bin/env python3
"""
Project intervention groups onto PCA space computed from control groups.
Visualize PCA scores, centroids, and 95% confidence ellipses.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tkinter import filedialog, Tk
from matplotlib.patches import Ellipse
import os

plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 12,
    'axes.labelsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12
})

def plot_confidence_ellipse(x, y, ax, n_std=1.96, **kwargs):

    cov = np.cov(x, y)
    mean_x, mean_y = np.mean(x), np.mean(y)
    eigenvals, eigenvecs = np.linalg.eigh(cov)
    order = eigenvals.argsort()[::-1]
    eigenvals, eigenvecs = eigenvals[order], eigenvecs[:, order]
    vx, vy = eigenvecs[:, 0]
    angle = np.degrees(np.arctan2(vy, vx))
    width, height = 2 * n_std * np.sqrt(eigenvals)
    ellipse = Ellipse((mean_x, mean_y), width, height, angle=angle, facecolor='none', **kwargs)
    ax.add_patch(ellipse)

def main():
    # === File selection ===
    Tk().withdraw()
    file_path = filedialog.askopenfilename(title="Select PCA CSV",
                                           filetypes=[("CSV files", "*.csv")])
    if not file_path:
        raise SystemExit("No file selected.")
    df = pd.read_csv(file_path)
    base = os.path.splitext(os.path.basename(file_path))[0]

    # === Feature extraction ===
    feature_cols = [col for col in df.columns if col not in ['Group', 'MouseID']]
    control_groups = ['WT Control', 'VPA Control']
    exercise_groups = ['WT Exercise', 'VPA Exercise']

    df_control = df[df['Group'].isin(control_groups)].reset_index(drop=True)
    df_exercise = df[df['Group'].isin(exercise_groups)].reset_index(drop=True)

    # === PCA computation from control only ===
    scaler = StandardScaler()
    X_control = scaler.fit_transform(df_control[feature_cols])
    pca = PCA(n_components=2)
    X_control_pca = pca.fit_transform(X_control)

    # === Projection of exercise groups ===
    X_exercise = scaler.transform(df_exercise[feature_cols])
    X_exercise_pca = pca.transform(X_exercise)

    # === Combine results ===
    df_control_pca = pd.DataFrame(X_control_pca, columns=['PC1', 'PC2'])
    df_control_pca['Group'] = df_control['Group'].values
    df_control_pca['MouseID'] = df_control['MouseID'].values

    df_exercise_pca = pd.DataFrame(X_exercise_pca, columns=['PC1', 'PC2'])
    df_exercise_pca['Group'] = df_exercise['Group'].values
    df_exercise_pca['MouseID'] = df_exercise['MouseID'].values

    df_pca_all = pd.concat([df_control_pca, df_exercise_pca], ignore_index=True)

    # === Compute group centroids ===
    centroids = df_pca_all.groupby('Group')[['PC1', 'PC2']].mean()

    # === Color palette ===
    palette = sns.color_palette('Set2', n_colors=df_pca_all['Group'].nunique())
    group_to_color = dict(zip(df_pca_all['Group'].unique(), palette))

    # === Plot ===
    plt.figure(figsize=(6, 4))
    sns.scatterplot(data=df_pca_all, x='PC1', y='PC2', hue='Group', s=100,
                    marker='o', palette=group_to_color)

    ax = plt.gca()
    for group_name, group_df in df_pca_all.groupby('Group'):
        plot_confidence_ellipse(
            group_df['PC1'].values,
            group_df['PC2'].values,
            ax,
            n_std=1.96,
            edgecolor=group_to_color[group_name],
            linestyle='--',
        )

    # Arrows from control to exercise
    for src, tgt in [('WT Control', 'WT Exercise'), ('VPA Control', 'VPA Exercise')]:
        if src in centroids.index and tgt in centroids.index:
            src_point = centroids.loc[src]
            tgt_point = centroids.loc[tgt]
            plt.arrow(src_point['PC1'], src_point['PC2'],
                      (tgt_point['PC1'] - src_point['PC1']) * 0.9,
                      (tgt_point['PC2'] - src_point['PC2']) * 0.9,
                      head_width=0.1, head_length=0.2, fc='black', ec='black')

    for group, row in centroids.iterrows():
        plt.scatter(row['PC1'], row['PC2'],
                    color=group_to_color[group], marker='X', s=100, edgecolor='black')

    # Axis and layout
    plt.title('Projection of Exercise Groups onto Control PCA Space')
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    plt.axhline(0, color='gray', linestyle='--')
    plt.axvline(0, color='gray', linestyle='--')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    # === Save results ===
    output_path = os.path.join(os.path.dirname(file_path), f"{base}_pca_projected.csv")
    df_pca_all.to_csv(output_path, index=False)
    print(f"✅ Saved projected PCA results to: {output_path}")

if __name__ == "__main__":
    main()
