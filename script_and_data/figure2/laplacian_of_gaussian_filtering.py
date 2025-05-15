import os
import numpy as np
import tifffile
from tkinter import Tk, filedialog
from skimage.filters import laplace, gaussian
from multiprocessing import Pool, cpu_count

# GUI folder selection
def select_folder():
    root = Tk()
    root.withdraw()
    folder = filedialog.askdirectory(title="Select folder containing TIF images")
    return folder

# Apply LoG filter with different sigma per channel
def apply_log_per_channel(img):
    result = np.zeros_like(img, dtype=np.float32)
    # Channel 1 (index 0): sigma = 2.4
    result[0] = laplace(gaussian(img[0], sigma=2.4))
    # Channel 2 (index 1): sigma = 1.5
    result[1] = laplace(gaussian(img[1], sigma=1.5))
    return result

# Process a single image
def process_image(path_out_tuple):
    path, out_dir = path_out_tuple
    img = tifffile.imread(path).astype(np.float32)
    if img.shape[0] != 2:
        print(f"Skipped: {os.path.basename(path)} is not a 2-channel image.")
        return
    log_img = apply_log_per_channel(img)
    filename = os.path.splitext(os.path.basename(path))[0] + "_LoG.tif"
    save_path = os.path.join(out_dir, filename)
    tifffile.imwrite(save_path, log_img.astype(np.float32))
    print(f"Saved: {filename}")

# Main routine
def main():
    folder = select_folder()
    if not folder:
        print("No folder selected. Exiting.")
        return

    out_dir = os.path.join(folder, "LoG_output")
    os.makedirs(out_dir, exist_ok=True)

    tif_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".tif")]

    # Run in parallel using multiprocessing
    with Pool(processes=8) as pool:
        pool.map(process_image, [(f, out_dir) for f in tif_files])

    print("LoG filtering completed for all images.")

if __name__ == "__main__":
    main()
