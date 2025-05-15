import os
import numpy as np
import pandas as pd
import tifffile
from skimage import measure
from skimage.filters import threshold_otsu
from multiprocessing import Pool, cpu_count
from functools import partial
from tkinter import Tk, filedialog
from skimage.segmentation import watershed
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.morphology import local_maxima

# Parameters
pixel_size_um = 0.2084
pixel_area_um2 = pixel_size_um ** 2

min_area_pixels = 2
max_area_pixels = 46
min_circularity = 0
max_circularity = 1.0

block_size = 256

def analyze_channel(channel_data, block_id, threshold_value, ch):
    binary = channel_data > threshold_value

    # Distance transform + Watershed
    distance = ndimage.distance_transform_edt(binary)
    local_maxi = local_maxima(distance) & binary
    markers = measure.label(local_maxi)
    label_image = watershed(-distance, markers, mask=binary)
    props = measure.regionprops(label_image)

    filtered_props = []
    for prop in props:
        area = prop.area
        perimeter = prop.perimeter
        if perimeter == 0:
            continue
        circularity = (4 * np.pi * area) / (perimeter ** 2)
        if min_area_pixels <= area <= max_area_pixels and min_circularity <= circularity <= max_circularity:
            filtered_props.append((area, circularity))

    num_particles = len(filtered_props)
    total_area_pixels = sum([a for a, _ in filtered_props])
    total_area_um2 = total_area_pixels * pixel_area_um2
    mean_area_um2 = total_area_um2 / num_particles if num_particles > 0 else 0

    return {
        "block_id": block_id,
        "channel": ch,
        "num_particles": num_particles,
        "total_area_pixels": total_area_pixels,
        "total_area_um2": total_area_um2,
        "mean_area_um2_per_particle": mean_area_um2
    }

def process_block(block_info, threshold_ch1, threshold_ch2):
    block, block_id = block_info
    res_ch1 = analyze_channel(block[0], block_id, threshold_ch1, ch=1)
    res_ch2 = analyze_channel(block[1], block_id, threshold_ch2, ch=2)
    return res_ch1, res_ch2

def main():
    # Folder selection
    root = Tk()
    root.withdraw()
    input_folder = filedialog.askdirectory(title="Select Folder")

    if not input_folder:
        print("No folder selected.")
        return

    for root_dir, dirs, files in os.walk(input_folder):
        relative_depth = len(os.path.relpath(root_dir, input_folder).split(os.sep))
        if relative_depth <= 2:
            for file in files:
                if file.lower().endswith(('.tif', '.tiff')):
                    input_path = os.path.join(root_dir, file)
                    folder_name = os.path.splitext(os.path.basename(file))[0]
                    output_folder = os.path.join(root_dir, folder_name)
                    os.makedirs(output_folder, exist_ok=True)

                    img = tifffile.imread(input_path)  # Shape: (ch, H, W)
                    if img.ndim != 3 or img.shape[0] != 2:
                        print(f"Skipped: {file} is not in (2ch, H, W) format.")
                        continue

                    _, height, width = img.shape

                    # Thresholds (can be replaced with Otsu if needed)
                    threshold_ch1 = 25
                    threshold_ch2 = 40
                    print(f"{file} - Thresholds: ch1={threshold_ch1:.1f}, ch2={threshold_ch2:.1f}")

                    # Block preparation
                    blocks = []
                    for y in range(0, height, block_size):
                        for x in range(0, width, block_size):
                            block = img[:, y:y+block_size, x:x+block_size]
                            if block.shape[1:] == (block_size, block_size):
                                blocks.append((block, len(blocks)))

                    print(f"Running multiprocessing with {cpu_count()} cores...")
                    with Pool(processes=cpu_count()) as pool:
                        results = pool.map(partial(process_block,
                                                   threshold_ch1=threshold_ch1,
                                                   threshold_ch2=threshold_ch2),
                                           blocks)

                    results_ch1 = [res[0] for res in results]
                    results_ch2 = [res[1] for res in results]

                    df_ch1 = pd.DataFrame(results_ch1)
                    df_ch2 = pd.DataFrame(results_ch2)

                    df_ch1.to_csv(os.path.join(output_folder, f"{folder_name}_analyze_particles_ch1.csv"), index=False)
                    df_ch2.to_csv(os.path.join(output_folder, f"{folder_name}_analyze_particles_ch2.csv"), index=False)

                    print(f"Finished: {file}")

if __name__ == "__main__":
    main()
