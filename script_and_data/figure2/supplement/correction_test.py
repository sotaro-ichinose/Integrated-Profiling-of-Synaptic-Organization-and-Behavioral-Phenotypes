import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tkinter import filedialog, simpledialog, Tk
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

# --- Reverse nonlinear Y-axis correction ---
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
    return corrected.astype(np.uint8)

# --- Apply log correction to Y coordinates ---
def apply_log_correction_to_coords(coords, a, H):
    y = coords[:, 1]
    log_base = np.log1p(a * H)
    y_new = H - (1 / a) * (np.exp(log_base * (1 - y / H)) - 1)
    corrected_coords = coords.copy()
    corrected_coords[:, 1] = y_new
    return corrected_coords

# --- Define original lines (functions) ---
def get_original_lines():
    return [
        lambda x: 0.978 * x,
        lambda x: 0.704 * x + 180,
        lambda x: 0.447 * x + 342,
        lambda x: 0.222 * x + 487,
        lambda x: 0.004 * x + 618
    ]

# --- Fit line to points (slope and intercept) ---
def fit_line(x, y):
    coef = np.polyfit(x, y, 1)
    return coef[0], coef[1]

# --- Main procedure ---
def main():
    root = Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(
        title="Select an image",
        filetypes=[("Image files", "*.tif;*.tiff;*.png;*.jpg;*.jpeg")]
    )
    if not file_path:
        print("❌ No image selected. Exiting.")
        return

    try:
        image_pil = Image.open(file_path)
        image_np = np.array(image_pil)
        image = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR) if image_np.ndim == 3 else image_np
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return

    a = simpledialog.askfloat("Log Correction Coefficient", "Enter value for a (e.g., 0.015):", initialvalue=0.015)
    if a is None:
        print("Log coefficient entry cancelled. Exiting.")
        return

    lines = get_original_lines()
    results = []
    save_dir = os.path.dirname(file_path)
    height, width = image.shape[:2]

    for trial in range(1, 11):
        print(f"\n[Trial {trial}/10] Please click 4 points in clockwise order.")

        selected_points = []
        def select_point(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(selected_points) < 4:
                selected_points.append((x, y))
                cv2.circle(image_display, (x, y), 3, (0, 255, 0), -1)
                cv2.imshow("Select 4 Points", image_display)

        image_display = image.copy()
        cv2.imshow("Select 4 Points", image_display)
        cv2.setMouseCallback("Select 4 Points", select_point)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        if len(selected_points) != 4:
            print("⚠️ 4 points not selected. Skipping trial.")
            continue

        matrix = cv2.getPerspectiveTransform(
            np.float32(selected_points),
            np.float32([[0, 0], [width, 0], [width, height], [0, height]])
        )

        # --- Plot original lines on raw image ---
        plt.figure(figsize=(6, 9))
        plt.title(f"Original Lines (Trial {trial})")
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 else image, cmap='gray')
        for i, func in enumerate(lines):
            x_vals = np.linspace(0, width, 200)
            y_vals = func(x_vals)
            mask = (y_vals >= 0) & (y_vals < height)
            plt.plot(x_vals[mask], y_vals[mask], label=f"Line {i+1}", linewidth=1)
        plt.gca().invert_yaxis()
        plt.legend()
        plt.savefig(os.path.join(save_dir, f"original_lines_trial{trial}.png"), dpi=300, bbox_inches="tight")
        plt.close()

        # --- Apply perspective + log correction to full image ---
        transformed_full = cv2.warpPerspective(image, matrix, (width, height))
        corrected_full = nonlinear_y_correction_reverse(transformed_full, a)
        corrected_rgb = cv2.cvtColor(corrected_full, cv2.COLOR_BGR2RGB) if corrected_full.ndim == 3 else corrected_full
        Image.fromarray(corrected_rgb).save(os.path.join(save_dir, f"transformed_image_trial{trial}.png"))

        # --- Process and plot each line ---
        plt.figure(figsize=(6, 9))
        plt.title(f"Transformed Lines (Trial {trial})")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.gca().invert_yaxis()

        for i, func in enumerate(lines):
            x_vals = np.linspace(0, width, 200)
            y_vals = func(x_vals)
            mask = (y_vals >= 0) & (y_vals < height)
            coords = np.stack([x_vals[mask], y_vals[mask]], axis=1)

            transformed = cv2.perspectiveTransform(np.array([coords], dtype=np.float32), matrix)[0]
            corrected_coords = apply_log_correction_to_coords(transformed, a=a, H=height)

            slope, intercept = fit_line(corrected_coords[:, 0], corrected_coords[:, 1])
            results.append({
                "Trial": trial,
                "Line": f"Line {i+1}",
                "Slope": slope,
                "Intercept": intercept
            })

            plt.scatter(corrected_coords[:, 0], corrected_coords[:, 1], s=2, label=f"Line {i+1}")
            x_fit = np.linspace(corrected_coords[:, 0].min(), corrected_coords[:, 0].max(), 100)
            y_fit = slope * x_fit + intercept
            plt.plot(x_fit, y_fit, linewidth=1)

        plt.legend()
        plt.savefig(os.path.join(save_dir, f"transformed_lines_trial{trial}.png"), dpi=300, bbox_inches="tight")
        plt.close()

    # --- Save all results to CSV ---
    df = pd.DataFrame(results)
    csv_path = os.path.join(save_dir, "transformed_lines.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n✅ All results saved to: {csv_path}")

if __name__ == "__main__":
    main()
