import os
import numpy as np
from tkinter import filedialog, Tk
import tifffile
from skimage.exposure import match_histograms
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

def process_file(file_path, reference, output_folder):
    try:
        img = tifffile.imread(file_path)

        # If Z-stack exists, use only the first Z-slice
        if img.ndim == 4 and img.shape[1] == 2:
            img = img[0]
        elif img.ndim == 3 and img.shape[2] == 2:
            img = np.transpose(img, (2, 0, 1))  # (H, W, 2) → (2, H, W)

        if img.ndim != 3 or img.shape[0] != 2:
            return f"Skipped: {os.path.basename(file_path)} is not in (2ch, H, W) format."

        matched = np.zeros_like(img)

        # Histogram matching for each channel
        for ch in range(2):
            matched[ch] = match_histograms(img[ch], reference[ch], channel_axis=None)

        output_path = os.path.join(output_folder, os.path.basename(file_path).replace(".tif", "_matched.tif"))
        tifffile.imwrite(output_path, matched.astype(img.dtype))

        return f"Saved: {output_path}"

    except Exception as e:
        return f"⚠️ Error ({os.path.basename(file_path)}): {e}"

def main():
    root = Tk()
    root.withdraw()

    target_folder = filedialog.askdirectory(title="Select folder containing target TIF images")
    if not target_folder:
        print("❌ No folder selected. Exiting.")
        return

    reference_path = filedialog.askopenfilename(
        title="Select reference image (2-channel TIF)",
        filetypes=[("TIFF files", "*.tif")]
    )
    if not reference_path:
        print("❌ No reference image selected. Exiting.")
        return

    reference = tifffile.imread(reference_path)

    if reference.ndim == 4 and reference.shape[1] == 2:
        reference = reference[0]
    elif reference.ndim == 3 and reference.shape[2] == 2:
        reference = np.transpose(reference, (2, 0, 1))

    if reference.ndim != 3 or reference.shape[0] != 2:
        print("❌ Reference image must be a TIF file in (2ch, H, W) format.")
        return

    tif_files = [os.path.join(target_folder, f) for f in os.listdir(target_folder) if f.lower().endswith(".tif")]
    if not tif_files:
        print("❌ No .tif files found in the selected folder.")
        return

    output_folder = os.path.join(target_folder, "histogram_matched")
    os.makedirs(output_folder, exist_ok=True)

    print(f"🔵 Found {len(tif_files)} files. Starting multiprocessing...")

    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(process_file, file_path, reference, output_folder): file_path
            for file_path in tif_files
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            print(future.result())

    print("🎉 Histogram matching completed for all images!")

if __name__ == "__main__":
    main()
