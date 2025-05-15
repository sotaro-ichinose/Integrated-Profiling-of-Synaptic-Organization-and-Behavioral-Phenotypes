import os
import numpy as np
import matplotlib.pyplot as plt
from tifffile import imread
from scipy.spatial.distance import euclidean
from tkinter import Tk, filedialog
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

def compute_histogram_and_stats(file_path):
    try:
        img = imread(file_path)
        if img.ndim != 3:
            return None, None, None, None

        # 2-channel image: (ch, H, W) or (H, W, ch)
        if img.shape[0] == 2:
            ch0, ch1 = img[0], img[1]
        elif img.shape[2] == 2:
            ch0, ch1 = img[..., 0], img[..., 1]
        else:
            return None, None, None, None

        # Compute histograms (12-bit, normalized)
        hist0, _ = np.histogram(ch0, bins=4096, range=(0, 2**12 - 1), density=True)
        hist1, _ = np.histogram(ch1, bins=4096, range=(0, 2**12 - 1), density=True)
        avg_hist = (hist0 + hist1) / 2

        # Mean intensity and saturation ratio
        mean_val = np.mean((ch0 + ch1) / 2)
        sat_ratio = np.mean((ch0 == 4095) | (ch1 == 4095)) / 2

        return avg_hist, file_path, mean_val, sat_ratio
    except:
        return None, None, None, None

def analyze_folder(folder):
    tif_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith('.tif')]
    results = []

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(compute_histogram_and_stats, f) for f in tif_files]
        for future in tqdm(as_completed(futures), total=len(futures), desc="📊 Analyzing histograms"):
            hist, path, mean_val, sat_ratio = future.result()
            if hist is not None:
                results.append((hist, path, mean_val, sat_ratio))

    return results

def run_histogram_analysis_with_eps(folder_path):
    # Step 1: Run analysis
    results = analyze_folder(folder_path)
    if not results:
        print("❌ No valid histograms to process.")
        return

    histograms = np.array([r[0] for r in results])
    valid_files = [r[1] for r in results]
    median_hist = np.median(histograms, axis=0)
    distances = [euclidean(h, median_hist) for h in histograms]
    sorted_indices = np.argsort(distances)
    recommended_paths = [valid_files[i] for i in sorted_indices[:5]]

    print("\n📈 Saving histogram plots for top 5 candidates...")

    for i in sorted_indices[:5]:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(histograms[i], label=os.path.basename(valid_files[i]), color='black')
        ax.plot(median_hist, '--', label='Median Histogram', color='red')
        ax.set_xlabel("Pixel Intensity (0–4095)")
        ax.set_ylabel("Normalized Frequency")
        ax.set_xlim(0, 4200)
        ax.set_ylim(0, 0.005)
        ax.legend()
        fig.tight_layout()

        base = os.path.splitext(os.path.basename(valid_files[i]))[0]
        eps_path = os.path.join(folder_path, f"{base}_histogram.eps")
        png_path = os.path.join(folder_path, f"{base}_histogram.png")

        fig.savefig(eps_path, format='eps')
        fig.savefig(png_path, format='png')
        plt.close()

    print("✅ Histogram plots saved.")

def main():
    root = Tk()
    root.withdraw()
    folder = filedialog.askdirectory(title="📂 Select folder containing 2-channel TIF images")
    if not folder:
        print("❌ No folder selected. Exiting.")
        return

    run_histogram_analysis_with_eps(folder)

    results = analyze_folder(folder)
    if not results:
        print("❌ No valid 2-channel images found.")
        return

    # Calculate median histogram and distances
    hists = np.array([r[0] for r in results])
    median_hist = np.median(hists, axis=0)
    dists = [euclidean(h, median_hist) for h in hists]
    mean_diffs = [r[2] - np.mean([r[2] for r in results]) for r in results]
    sat_ratios = [r[3] for r in results]

    sorted_indices = np.argsort(dists)

    print("\n🎯 Top 5 recommended reference candidates (closest to median):")
    for i in sorted_indices[:5]:
        print(f"{os.path.basename(results[i][1])} | L2 distance: {dists[i]:.4f} | Mean intensity deviation: {mean_diffs[i]:+.1f} | Saturation: {sat_ratios[i]*100:.2f}%")

    print("\n👁️ Please select the final reference image(s) manually based on visual inspection and absence of artifacts.")

if __name__ == "__main__":
    main()
