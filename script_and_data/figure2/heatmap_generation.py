#!/usr/bin/env python3
"""
Compute E–I log ratios and generate block-wise heatmaps for multiple metrics.

This script:
 1. Prompts user to select a parent folder.
 2. For each subfolder, finds `*_ch1.csv` and `*_ch2.csv`, computes E–I log10 ratio, and saves `<subfolder>_E-I.csv`.
 3. Reads `image_dimensions.csv` in the parent folder for spatial layout.
 4. Loads E–I CSV and precomputed channel CSVs to build heatmaps for:
    - E–I balance
    - GEP number & area (ch1)
    - PSD number & area (ch2)
 5. Saves each heatmap as a 16-bit TIFF in the respective subfolder.

Requirements:
  pip install pandas numpy imageio tkinter
"""
import os
import numpy as np
import pandas as pd
import imageio
from tkinter import Tk, filedialog

# GUI folder selector
def select_parent_folder():
    root = Tk()
    root.withdraw()
    return filedialog.askdirectory(title="Select Parent Folder")

# Step 1: compute E–I log ratio CSV for each subfolder
def compute_ei_logs(parent_folder):
    for sub in os.listdir(parent_folder):
        subdir = os.path.join(parent_folder, sub)
        if not os.path.isdir(subdir):
            continue
        # locate channel CSVs
        ch1 = ch2 = None
        for f in os.listdir(subdir):
            if f.endswith('_ch1.csv'):
                ch1 = os.path.join(subdir, f)
            elif f.endswith('_ch2.csv'):
                ch2 = os.path.join(subdir, f)
        if not (ch1 and ch2):
            print(f"Skipping {sub}: channel CSVs missing.")
            continue
        df1 = pd.read_csv(ch1)
        df2 = pd.read_csv(ch2)
        if 'num_particles' not in df1 or 'num_particles' not in df2:
            print(f"Skipping {sub}: 'num_particles' column missing.")
            continue
        ratio = df2['num_particles'] / df1['num_particles']
        log_ratio = np.log10(ratio.replace(0, np.nan))
        log_ratio = log_ratio.fillna(0).to_numpy()
        out = pd.DataFrame({
            'Index': df1.iloc[:, 0],
            'E-I_log10': log_ratio
        })
        out_path = os.path.join(parent_folder, f"{sub}_E-I.csv")
        out.to_csv(out_path, index=False)
        print(f"Saved E–I log CSV: {out_path}")

# Step 2: heatmap creation
def create_heatmap(data, w_blocks, h_blocks, save_path):
    heat = np.zeros((h_blocks*10, w_blocks*10), np.float32)
    for i, val in enumerate(data[:w_blocks*h_blocks]):
        x = (i % w_blocks)*10
        y = (i // w_blocks)*10
        heat[y:y+10, x:x+10] = val
    heat16 = (heat * 32).astype(np.uint16)
    imageio.imwrite(save_path, heat16)
    print(f"Heatmap saved: {save_path}")

# Main pipeline
def main():
    parent = select_parent_folder()
    if not parent:
        print("No folder selected. Exiting.")
        return

    # compute E–I logs
    compute_ei_logs(parent)

    # load dimensions
    dim_path = os.path.join(parent, 'image_dimensions.csv')
    try:
        dims = pd.read_csv(dim_path)
    except FileNotFoundError:
        print(f"{dim_path} not found. Exiting.")
        return

    # per-subfolder heatmap loop
    for sub in os.listdir(parent):
        subdir = os.path.join(parent, sub)
        if not os.path.isdir(subdir):
            continue
        print(f"\nProcessing heatmaps for {sub}...")
        row = dims[dims['File Name'].str.replace('.tif','')==sub]
        if row.empty:
            print(f"No dimensions for {sub}. Skipping.")
            continue
        w = int(row['Width (pixels)'].values[0] // 256)
        h = int(row['Height (pixels)'].values[0] // 256)
        # load E–I
        ei_path = os.path.join(parent, f"{sub}_E-I.csv")
        if not os.path.exists(ei_path):
            print(f"Missing E–I CSV for {sub}. Skipping.")
            continue
        ei_data = pd.read_csv(ei_path)['E-I_log10'].to_list()
        # load channel results
        ch1 = os.path.join(subdir, f"{sub}_analyze_particles_ch1.csv")
        ch2 = os.path.join(subdir, f"{sub}_analyze_particles_ch2.csv")
        if not (os.path.exists(ch1) and os.path.exists(ch2)):
            print(f"Missing channel CSV in {sub}. Skipping.")
            continue
        df1 = pd.read_csv(ch1); df2 = pd.read_csv(ch2)
        nums1 = df1['num_particles'].to_list(); areas1 = df1['mean_area_um2_per_particle'].to_list()
        nums2 = df2['num_particles'].to_list(); areas2 = df2['mean_area_um2_per_particle'].to_list()
        # define outputs
        out_map = {
            'EIbalance': (ei_data, f"{sub}_EIbalance.tif"),
            'gep_number': (nums1, f"{sub}_gep_number.tif"),
            'gep_area'  : (areas1, f"{sub}_gep_area.tif"),
            'PSD_number': (nums2, f"{sub}_PSD_number.tif"),
            'PSD_area'  : (areas2, f"{sub}_PSD_area.tif"),
        }
        # generate all
        for key,(data, fname) in out_map.items():
            create_heatmap(data, w, h, os.path.join(subdir, fname))

    print("\nAll done.")

if __name__ == '__main__':
    main()
