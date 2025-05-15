import os
import csv
from tkinter import Tk, filedialog
import tifffile

# GUI to select folder
def select_folder():
    root = Tk()
    root.withdraw()
    folder_selected = filedialog.askdirectory(title="Select Folder")
    return folder_selected

# Function to collect image dimensions
def collect_image_dimensions(folder_path):
    dimensions = []
    for subdir, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.tif', '.tiff')):
                file_path = os.path.join(subdir, file)
                try:
                    img = tifffile.imread(file_path)
                    if img.ndim == 3:
                        # Format: (channels, height, width)
                        _, height, width = img.shape
                    elif img.ndim == 2:
                        # Format: (height, width)
                        height, width = img.shape
                    else:
                        print(f"⚠️ Skipped {file_path}: unexpected image shape {img.shape}")
                        continue

                    folder_name = os.path.basename(subdir)
                    dimensions.append((folder_name, file, width, height))
                    print(f"✅ Processed: {file_path}")

                except Exception as e:
                    print(f"❌ Error reading {file_path}: {e}")
    return dimensions

# Function to save dimensions to CSV
def save_to_csv(dimensions, output_file):
    with open(output_file, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Folder Name', 'File Name', 'Width (pixels)', 'Height (pixels)'])
        writer.writerows(dimensions)

# Main function
def main():
    folder_path = select_folder()
    if not folder_path:
        print("No folder selected.")
        return

    dimensions = collect_image_dimensions(folder_path)

    if dimensions:
        output_file = os.path.join(folder_path, 'image_dimensions.csv')
        save_to_csv(dimensions, output_file)
        print(f"\n🎉 Image dimensions saved to: {output_file}")
    else:
        print("No .tif or .tiff files found in the selected folder.")

if __name__ == '__main__':
    main()
