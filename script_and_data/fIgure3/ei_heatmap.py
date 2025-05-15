import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tkinter import Tk, filedialog

def select_folder():
    root = Tk()
    root.withdraw()
    return filedialog.askdirectory(title="Select Folder")

def load_dimensions_csv(folder_path):
    path = os.path.join(folder_path, "image_dimensions.csv")
    if not os.path.exists(path):
        print(f"'{path}' not found.")
        return None
    df = pd.read_csv(path)
    required_columns = ['File Name', 'Width (pixels)', 'Height (pixels)']
    if not all(col in df.columns for col in required_columns):
        print(f"'image_dimensions.csv' must contain {required_columns}.")
        return None
    return df

def generate_heatmap(csv_path, dimensions_df, folder_path):
    df = pd.read_csv(csv_path)
    if 'E-I_log10' not in df.columns:
        print(f"'E-I_log10' column not found in {os.path.basename(csv_path)}. Skipping.")
        return

    # Get original image name (e.g., "xxx_E-I.csv" → "xxx.tif")
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    image_name = base_name.replace('_E-I', '') + '.tif'

    row = dimensions_df[dimensions_df['File Name'] == image_name]
    if row.empty:
        print(f"No dimensions found for image '{image_name}'. Skipping.")
        return

    width = row['Width (pixels)'].values[0]
    height = row['Height (pixels)'].values[0]
    blocks_x = width // 256
    blocks_y = height // 256

    print(f"Processing {image_name}: {blocks_x}×{blocks_y} blocks")

    ratios = df['E-I_log10'].tolist()
    heatmap = np.zeros((blocks_y * 2, blocks_x * 2))

    for i in range(min(len(ratios), blocks_x * blocks_y)):
        x = (i % blocks_x) * 2
        y = (i // blocks_x) * 2
        heatmap[y:y+2, x:x+2] = ratios[i]

    # Plot heatmap
    plt.figure(figsize=(12, 10))
    cmap = plt.get_cmap('coolwarm')
    norm = mcolors.Normalize(vmin=-1.2, vmax=1.2)
    plt.imshow(heatmap, cmap=cmap, norm=norm)
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()
    plt.title(f'E/I Balance (log10): {image_name}')

    # Save heatmap
    save_basename = os.path.splitext(image_name)[0]
    out_png = os.path.join(folder_path, f"{save_basename}_heatmap.png")
    out_eps = os.path.join(folder_path, f"{save_basename}_heatmap.eps")
    plt.savefig(out_png, bbox_inches='tight')
    plt.savefig(out_eps, bbox_inches='tight')
    plt.close()

    print(f"Saved heatmap: {out_png} and .eps")

def main():
    folder = select_folder()
    if not folder:
        print("No folder selected.")
        return

    dimensions_df = load_dimensions_csv(folder)
    if dimensions_df is None:
        return

    for file in os.listdir(folder):
        if file.endswith('_E-I.csv'):
            csv_path = os.path.join(folder, file)
            generate_heatmap(csv_path, dimensions_df, folder)

if __name__ == '__main__':
    main()
