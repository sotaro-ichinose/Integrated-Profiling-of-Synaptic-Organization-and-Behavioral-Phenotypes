import pandas as pd
import os
import tkinter as tk
from tkinter import filedialog

def run_all_barplots():
    # Let the user select a folder
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(title="Select a folder containing CSV files")

    if not folder_path:
        print("No folder selected. Exiting...")
        return

    # Get all CSV files in the folder
    csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]

    if not csv_files:
        print("No CSV files found in the selected folder. Exiting...")
        return

    # Collect results
    results = []

    for csv_file in csv_files:
        file_path = os.path.join(folder_path, csv_file)
        file_name = os.path.splitext(csv_file)[0]

        try:
            df = pd.read_csv(file_path, delimiter=None, engine='python')

            # Count X-axis boundary crossings
            x = df['X'].values
            x_cross_140 = ((x[:-1] >= 140) & (x[1:] < 140)).sum()
            x_cross_160 = ((x[:-1] <= 160) & (x[1:] > 160)).sum()
            count_x = x_cross_140 + x_cross_160

            # Count Y-axis boundary crossings
            y = df['Y'].values
            y_cross_140 = ((y[:-1] >= 140) & (y[1:] < 140)).sum()
            y_cross_160 = ((y[:-1] <= 160) & (y[1:] > 160)).sum()
            count_y = y_cross_140 + y_cross_160

            results.append([file_name, count_x, count_y])

        except Exception as e:
            print(f"⚠️ Error processing {csv_file}: {e}")

    # Save as DataFrame
    results_df = pd.DataFrame(results, columns=['Filename', 'Count_X_crossings', 'Count_Y_crossings'])
    output_path = os.path.join(folder_path, "results_of_EPM_crossings.csv")
    results_df.to_csv(output_path, index=False)

    print(f"✅ Processed results saved as: {output_path}")

if __name__ == '__main__':
    run_all_barplots()
