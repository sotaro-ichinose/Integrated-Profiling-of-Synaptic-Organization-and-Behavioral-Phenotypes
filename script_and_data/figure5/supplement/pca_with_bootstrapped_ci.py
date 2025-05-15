import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import euclidean
from tqdm import tqdm
from matplotlib.patches import Ellipse
from tkinter import Tk, filedialog
import os

# === Plot style ===
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 12,
    'axes.labelsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12
})

def compute_centroid(df, group):
    sub = df[df['Group'] == group]
    return sub[['PC1', 'PC2']].mean().values

def plot_confidence_ellipse(x, y, ax, n_std=1.96, **kwargs):

    cov = np.cov(x, y)
    mean_x, mean_y = np.mean(x), np.mean(y)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    vx, vy = eigvecs[:, 0]
    angle = np.degrees(np.arctan2(vy, vx))
    width, height = 2 * n_std * np.sqrt(eigvals)
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

    # === Parameters ===
    groups_pairs = [('WT Control', 'WT Exercise'), ('VPA Control', 'VPA Exercise')]
    all_groups = ['VPA Control', 'WT Control', 'VPA Exercise', 'WT Exercise']
    n_boot = 1000

    # === PCA ===
    scaler = StandardScaler()
    features = [c for c in df.columns if c not in ['Group', 'MouseID']]
    X = scaler.fit_transform(df[features])
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # === Flip PC signs if needed for interpretability ===
    loadings = pca.components_.T
    loading_df = pd.DataFrame(loadings, index=features, columns=["PC1_loading", "PC2_loading"])

    if "Density of PSD-95 (10% Depth)" in loading_df.index:
        if loading_df.loc["Density of PSD-95 (10% Depth)", "PC1_loading"] < 0:
            X_pca[:, 0] *= -1
            pca.components_[0, :] *= -1
            loadings[:, 0] *= -1
            loading_df["PC1_loading"] *= -1

    if "OF Center Region" in loading_df.index:
        if loading_df.loc["OF Center Region", "PC2_loading"] > 0:
            X_pca[:, 1] *= -1
            pca.components_[1, :] *= -1
            loadings[:, 1] *= -1
            loading_df["PC2_loading"] *= -1

    df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
    df_pca['Group'] = df['Group'].str.strip()
    df_pca['MouseID'] = df['MouseID']

    # === Compute observed distances ===
    obs = {}
    for g1, g2 in groups_pairs:
        obs[(g1, g2)] = euclidean(compute_centroid(df_pca, g1),
                                  compute_centroid(df_pca, g2))

    # === Bootstrap distances ===
    results = {}
    for g1, g2 in groups_pairs:
        data1 = df_pca[df_pca['Group'] == g1][['PC1','PC2']].values
        data2 = df_pca[df_pca['Group'] == g2][['PC1','PC2']].values
        boot = []
        for _ in tqdm(range(n_boot), desc=f"Bootstrap {g1} vs {g2}"):
            s1 = data1[np.random.choice(len(data1), len(data1), replace=True)]
            s2 = data2[np.random.choice(len(data2), len(data2), replace=True)]
            boot.append(euclidean(s1.mean(axis=0), s2.mean(axis=0)))
        boot = np.array(boot)
        results[(g1, g2)] = {
            'dist': obs[(g1, g2)],
            'ci_low': np.percentile(boot, 2.5),
            'ci_high': np.percentile(boot, 97.5),
            'p': np.mean(boot >= obs[(g1, g2)])
        }

    # === Save bootstrap results ===
    df_res = pd.DataFrame.from_dict(results, orient='index')
    df_res.to_csv(os.path.join(os.path.dirname(file_path),
                               f"{base}_bootstrap_centroid.csv"))

    # === Plot ===
    fig, ax = plt.subplots(figsize=(6, 4))
    palette = dict(zip(all_groups, sns.color_palette("Set2", len(all_groups))))

    sns.scatterplot(data=df_pca, x='PC1', y='PC2',
                    hue='Group', palette=palette, s=100, ax=ax)

    # centroids, arrows, labels
    for (g1, g2), v in results.items():
        c1 = compute_centroid(df_pca, g1)
        c2 = compute_centroid(df_pca, g2)
        ax.scatter(*c1, marker='X', color=palette[g1], s=150, edgecolor='black')
        ax.scatter(*c2, marker='X', color=palette[g2], s=150, edgecolor='black')
        ax.arrow(c1[0], c1[1], (c2[0]-c1[0])*0.9, (c2[1]-c1[1])*0.9,
                 head_width=0.1, head_length=0.1, fc='black', ec='black')
        mx, my = (c1 + c2) *2.5 
        ax.text(mx, my, f"d={v['dist']:.2f}\nCI=[{v['ci_low']:.2f},{v['ci_high']:.2f}]\np={v['p']:.3f}",
                ha='center', va='center', bbox=dict(boxstyle='round', fc='white', ec='gray'))

    # confidence ellipses
    for g in all_groups:
        sub = df_pca[df_pca['Group'] == g]
        plot_confidence_ellipse(sub['PC1'], sub['PC2'], ax,
                                edgecolor=palette[g], linestyle='--')

    ax.axhline(0, color='gray', linestyle='--')
    ax.axvline(0, color='gray', linestyle='--')
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    ax.set_title("PCA Projection with Bootstrapped CI & 95% Ellipses")
    ax.legend(bbox_to_anchor=(1.05,1), loc='upper left')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
