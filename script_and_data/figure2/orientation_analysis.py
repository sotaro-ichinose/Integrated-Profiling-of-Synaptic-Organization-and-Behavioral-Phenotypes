import numpy as np
import tifffile
import matplotlib.pyplot as plt
from tkinter import filedialog, Tk
from scipy.ndimage import gaussian_filter

def main():
    # === GUI: Select TIF image ===
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title="Select an 8-bit TIF image", filetypes=[("TIF files", "*.tif")])
    if not file_path:
        print("No file selected. Exiting.")
        return

    # === Load image ===
    img = tifffile.imread(file_path).astype(float)
    if img.ndim > 2:
        img = img[0]  # Use the first Z-slice
    if img.dtype != np.uint8 and img.dtype != np.float64:
        raise ValueError("An 8-bit or float image is required.")

    # === Structure tensor calculation ===
    sigma = 3.0  # Spatial scale
    Ix = gaussian_filter(img, sigma=sigma, order=[0, 1])
    Iy = gaussian_filter(img, sigma=sigma, order=[1, 0])

    Ixx = gaussian_filter(Ix * Ix, sigma=sigma)
    Iyy = gaussian_filter(Iy * Iy, sigma=sigma)
    Ixy = gaussian_filter(Ix * Iy, sigma=sigma)

    # === Compute orientation (0–180°) ===
    theta = 0.5 * np.arctan2(2 * Ixy, Ixx - Iyy)
    theta_deg = np.rad2deg(theta) % 180

    # === Weighted orientation histogram ===
    orientation_strength = np.sqrt((Ixx - Iyy)**2 + 4 * Ixy**2)  # Anisotropy measure
    num_bins = 36
    bins = np.linspace(0, 180, num_bins + 1)
    hist = np.zeros(num_bins)

    for t, w in zip(theta_deg.flatten(), orientation_strength.flatten()):
        bin_idx = int(t // (180 / num_bins)) % num_bins
        hist[bin_idx] += w

    # === Polar plot ===
    angles_rad = np.deg2rad((bins[:-1] + bins[1:]) / 2)

    fig = plt.figure(figsize=(3.6, 3.6))
    ax = fig.add_subplot(111, polar=True)
    ax.bar(angles_rad, hist, width=np.deg2rad(180 / num_bins), alpha=0.7)

    # Configure orientation
    ax.set_theta_zero_location("S")  # 0° at bottom
    ax.set_theta_direction(-1)       # Clockwise

    # Limit to semicircle
    ax.set_thetamin(0)
    ax.set_thetamax(180)

    # === Mean orientation angle ± SD (circular) ===
    angles_deg = (bins[:-1] + bins[1:]) / 2
    angles_rad_vec = np.deg2rad(angles_deg)
    weights = hist

    x = np.sum(weights * np.cos(2 * angles_rad_vec))
    y = np.sum(weights * np.sin(2 * angles_rad_vec))
    mean_angle_rad = 0.5 * np.arctan2(y, x)
    mean_angle_deg = np.rad2deg(mean_angle_rad) % 180

    R = np.sqrt(x**2 + y**2) / np.sum(weights)
    circular_std_rad = np.sqrt(-2 * np.log(R)) / 2  # Corrected for 180° periodicity
    circular_std_deg = np.rad2deg(circular_std_rad)

    # Display mean ± SD lines
    mean_angle_display = np.deg2rad(mean_angle_deg)
    std_angle_plus = np.deg2rad((mean_angle_deg + circular_std_deg) % 360)
    std_angle_minus = np.deg2rad((mean_angle_deg - circular_std_deg) % 360)

    ax.plot([mean_angle_display, mean_angle_display], [0, max(hist)], color='red', linewidth=2, label='Mean angle')
    ax.plot([std_angle_plus, std_angle_plus], [0, max(hist)], color='red', linestyle='--', linewidth=1)
    ax.plot([std_angle_minus, std_angle_minus], [0, max(hist)], color='red', linestyle='--', linewidth=1)

    # Hide radial tick labels
    ax.set_yticklabels([])

    # Display mean ± SD in text
    mean_text = f"{mean_angle_deg:.1f}° ± {circular_std_deg:.1f}°"
    ax.text(np.deg2rad(135), max(hist) * 0.9, mean_text,
            ha='center', va='top', fontsize=12, color='red',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", lw=1))

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
