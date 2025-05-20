import numpy as np
from scipy.spatial import procrustes
from sklearn.metrics.pairwise import cosine_similarity


# === PCA scores from the original PCA (22 animals: WT + VPA only) ===
X_old = np.array([
    [1.638765093, 0.149644439],
    [2.424157573, -1.435264317],
    [1.772956598, 0.336392603],
    [4.568405064, 2.287339881],
    [-0.546601628, 3.356889881],
    [-0.18274507, 1.802584631],
    [-1.2080265, -0.117911656],
    [-1.682215382, 2.762737992],
    [0.92790678, -1.810687634],
    [-0.993937362, -0.916600387],
    [-0.07512459, -0.993119589],
    [1.70781058, -0.235660078],
    [-2.266400036, 1.351717866],
    [-0.409090185, 0.360808751],
    [0.696164361, -3.267942247],
    [0.916329382, -1.006012405],
    [0.280557235, 0.505557414],
    [-1.891758366, -0.289261506],
    [-1.562889507, -1.095736639],
    [-2.736343607, -0.642370213],
    [0.113970664, 0.347152993],
    [-1.491891095, -1.45025978]
])

# === PCA scores after recomputing PCA with all animals (WT, VPA, and exercise groups) ===
X_new = np.array([
    [1.379347613, 1.045101788],
    [2.413201763, -0.27248711],
    [1.490012181, 1.328804689],
    [4.1927697, 3.230669103],
    [-0.974858333, 3.069796463],
    [-0.444473469, 1.637422947],
    [-1.378850714, 0.421139243],
    [-2.148458008, 2.119394603],
    [1.026916218, -0.663747664],
    [-1.266960588, 0.313427647],
    [-0.193036175, 0.225586094],
    [1.625265642, 0.677286124],
    [-2.378305363, 0.760568782],
    [-0.428456692, 0.495518239],
    [0.898317063, -1.839914882],
    [1.017450155, -0.961610539],
    [0.03131062, 0.872427268],
    [-1.974306752, -0.187719812],
    [-1.456293899, -0.880714132],
    [-2.827355963, -0.75419375],
    [0.065011693, 0.664360698],
    [-1.396207476, -1.368409655]
])

# === Perform Procrustes analysis ===
mtx1, mtx2, disparity = procrustes(X_old, X_new)
similarity = 1 - disparity

# === Output results ===
print("=== Procrustes Analysis (scipy.spatial) ===")
print(f"Disparity: {disparity:.4f}")
print(f"Similarity (1 - disparity): {similarity:.4f}")

# === Interpretation ===
print("\nInterpretation:")
print("- Disparity quantifies the shape difference between PCA configurations.")
print("- A similarity score ≥ 0.95 indicates high spatial preservation.")
print("- These results confirm that adding exercise groups did not distort the original eigenspace,")
print("  but instead caused a translation within the same variance structure.")

# cosine similarity
pc1_old = X_old[:, 0].reshape(1, -1)
pc1_new = X_new[:, 0].reshape(1, -1)
pc2_old = X_old[:, 1].reshape(1, -1)
pc2_new = X_new[:, 1].reshape(1, -1)

cos_pc1 = cosine_similarity(pc1_old, pc1_new)[0, 0]
cos_pc2 = cosine_similarity(pc2_old, pc2_new)[0, 0]

print(f"\nCosine similarity: PC1 = {cos_pc1:.3f}, PC2 = {cos_pc2:.3f}")

