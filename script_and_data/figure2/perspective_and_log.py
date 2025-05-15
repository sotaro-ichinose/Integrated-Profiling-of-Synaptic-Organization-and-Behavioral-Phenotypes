import os
import numpy as np
import cv2
from tkinter import Tk, filedialog, simpledialog
from PIL import Image
import imageio
import pandas as pd

# Logarithmic Y-axis correction
def nonlinear_y_correction_reverse(image, a):
    H, W = image.shape[:2]
    corrected = np.zeros_like(image)
    for y_new in range(H):
        y_orig = H * (1 - np.log1p(a * (H - y_new)) / np.log1p(a * H))
        y0 = int(np.floor(y_orig))
        y1 = min(y0 + 1, H - 1)
        weight = y_orig - y0
        if image.ndim == 2:
            corrected[y_new, :] = (1 - weight) * image[y0, :] + weight * image[y1, :]
        else:
            corrected[y_new, :, :] = (1 - weight) * image[y0, :, :] + weight * image[y1, :, :]
    return corrected.astype(image.dtype)

# Perspective transform
def perspective_transform(image, points):
    pts1 = np.float32(points)
    width, height = 200, 300
    pts2 = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(image, matrix, (width, height))

# Mouse callback for point selection
points = []
def select_point(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append((x, y))
        print(f"Point selected: {x}, {y}")
        cv2.circle(param, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Select Points", param)

# Mean brightness calculation (X-axis division into 10 bins)
def calculate_mean_brightness(image):
    height, width = image.shape[:2]
    results = []
    for x in range(0, width, width // 10):
        column = image[:, x]
        mean_val = np.mean(column) / 32
        results.append((int(100 * x / width), mean_val))
    return results

# Export to CSV
def output_to_csv(data, save_dir, folder_name, file_suffix):
    df = pd.DataFrame(data, columns=['Depth (%)', 'Mean'])
    csv_path = os.path.join(save_dir, f'{folder_name}_{file_suffix}.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved mean profile to: {csv_path}")

# Main function
def main():
    root = Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(title="Select target image folder")
    if not folder_path:
        print("No folder selected. Exiting.")
        return

    folder_name = os.path.basename(folder_path)
    psd_path = os.path.join(folder_path, f"{folder_name}_PSD_number.tif")
    if not os.path.exists(psd_path):
        print(f"{psd_path} not found. Exiting.")
        return

    psd_img = np.array(Image.open(psd_path))
    if psd_img.ndim != 2:
        raise ValueError("The PSD_number image must be 2D grayscale.")

    # Contrast stretch for display
    psd_min, psd_max = np.min(psd_img), np.max(psd_img)
    if psd_max > psd_min:
        psd_img_disp = ((psd_img - psd_min) / (psd_max - psd_min) * 255).astype(np.uint8)
    else:
        psd_img_disp = np.zeros_like(psd_img, dtype=np.uint8)

    print("Please select 4 points in this order: surface → deep → deep → surface.")
    cv2.imshow("Select Points", psd_img_disp)
    cv2.setMouseCallback("Select Points", select_point, psd_img_disp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(points) != 4:
        print("Four points were not selected. Exiting.")
        return

    a = simpledialog.askfloat("Log Correction Factor", "Enter value for 'a' (e.g., 0.015):", initialvalue=0.015)
    if a is None:
        print("Log factor input cancelled. Exiting.")
        return

    suffixes = ["EIbalance", "gep_number", "gep_area", "PSD_number", "PSD_area"]

    for suffix in suffixes:
        target_path = os.path.join(folder_path, f"{folder_name}_{suffix}.tif")
        if not os.path.exists(target_path):
            print(f"{target_path} not found. Skipping.")
            continue

        img = np.array(Image.open(target_path))
        if img.ndim != 2:
            print(f"{target_path} is not 2D grayscale. Skipping.")
            continue

        transformed = perspective_transform(img, points)
        corrected = nonlinear_y_correction_reverse(transformed, a)

        output_path = os.path.join(folder_path, f"{folder_name}_{suffix}_processing.tif")
        imageio.imwrite(output_path, corrected.astype(img.dtype))
        print(f"Saved corrected image: {output_path}")

        brightness_data = calculate_mean_brightness(transformed)
        output_to_csv(brightness_data, folder_path, folder_name, suffix)

    print("\nAll files processed successfully.")

if __name__ == '__main__':
    main()
