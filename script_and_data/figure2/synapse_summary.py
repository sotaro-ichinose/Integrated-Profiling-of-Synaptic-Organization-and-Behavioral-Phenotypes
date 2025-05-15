#!/usr/bin/env python3
"""
Summarize PSD/gephyrin synapse counts and E/I ratio (Exc/Inh) by depth across animals.
Also computes Welch’s t-test (WT vs VPA) and exports per-depth statistics.

INPUT : Parent folder containing subfolders with *_PSD_number.csv, *_gep_number.csv, etc.
OUTPUT: CSV summaries (e.g., _PSD_number.csv, _gep_area.csv, _EIbalance_mean_ratio.csv, _EIbalance_log.csv)
"""

import os
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog
import re
from collections import defaultdict
from scipy.stats import ttest_ind

def select_folder():
    root = tk.Tk()
    root.withdraw()
    return filedialog.askdirectory(title="Select parent folder")

def find_csv_files(folder, keyword):
    matched_files = []
    for dirpath, _, filenames in os.walk(folder):
        for file in filenames:
            if file.endswith(f"{keyword}.csv"):
                matched_files.append(os.path.join(dirpath, file))
    return matched_files

def extract_animal_id(folder_name):
    match = re.match(r'^(\d{6}_ACC (?:WT|VPA)\d{1})', folder_name)
    return match.group(1) if match else folder_name

def create_summary_csv(folder, csv_files, output_suffix, apply_log=False):
    folder_name = os.path.basename(folder)
    output_path = os.path.join(folder, f"{folder_name}_{output_suffix}.csv")
    df = pd.DataFrame({"Depth (%)": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]})
    grouped_data = defaultdict(list)

    for csv_file in csv_files:
        animal_id = extract_animal_id(os.path.basename(os.path.dirname(csv_file)))
        try:
            data = pd.read_csv(csv_file, usecols=[1]).iloc[:10, 0].replace(0, np.nan)
            if apply_log:
                data = np.log10(data)
            grouped_data[animal_id].append(data.values)
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")

    wt_cols, vpa_cols = [], []

    for animal_id, value_list in grouped_data.items():
        stacked = np.vstack(value_list)
        mean_values = np.nanmean(stacked, axis=0)
        df[animal_id] = mean_values
        if "WT" in animal_id:
            wt_cols.append(animal_id)
        elif "VPA" in animal_id:
            vpa_cols.append(animal_id)

    if wt_cols:
        df['WT_Mean'] = df[wt_cols].mean(axis=1)
        df['WT_SD'] = df[wt_cols].std(axis=1)
        df['WT_N'] = df[wt_cols].count(axis=1)

    if vpa_cols:
        df['VPA_Mean'] = df[vpa_cols].mean(axis=1)
        df['VPA_SD'] = df[vpa_cols].std(axis=1)
        df['VPA_N'] = df[vpa_cols].count(axis=1)

    p_vals = []
    for i in range(len(df)):
        wt_vals = df.loc[i, wt_cols].dropna().values
        vpa_vals = df.loc[i, vpa_cols].dropna().values
        if len(wt_vals) > 1 and len(vpa_vals) > 1:
            _, p = ttest_ind(wt_vals, vpa_vals, equal_var=False)
        else:
            p = np.nan
        p_vals.append(p)

    df["P_value"] = p_vals
    df.to_csv(output_path, index=False)
    print(f"✅ Created summary CSV: {output_path}")

def compute_ei_ratio(folder, psd_files, gep_files):
    _compute_ei_base(folder, psd_files, gep_files, apply_log=False)

def compute_ei_ratio_log(folder, psd_files, gep_files):
    _compute_ei_base(folder, psd_files, gep_files, apply_log=True)

def _compute_ei_base(folder, psd_files, gep_files, apply_log):
    folder_name = os.path.basename(folder)
    suffix = "EIbalance_log" if apply_log else "EIbalance_mean_ratio"
    output_path = os.path.join(folder, f"{folder_name}_{suffix}.csv")
    df = pd.DataFrame({"Depth (%)": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]})
    exc_data, inh_data = defaultdict(list), defaultdict(list)

    for path in psd_files:
        aid = extract_animal_id(os.path.basename(os.path.dirname(path)))
        try:
            data = pd.read_csv(path, usecols=[1]).iloc[:10, 0].replace(0, np.nan)
            exc_data[aid].append(data.values)
        except Exception as e:
            print(f"Error reading PSD: {e}")

    for path in gep_files:
        aid = extract_animal_id(os.path.basename(os.path.dirname(path)))
        try:
            data = pd.read_csv(path, usecols=[1]).iloc[:10, 0].replace(0, np.nan)
            inh_data[aid].append(data.values)
        except Exception as e:
            print(f"Error reading GEP: {e}")

    ei_data = {}
    wt_cols, vpa_cols = [], []

    for aid in exc_data:
        if aid in inh_data:
            exc_avg = np.nanmean(np.vstack(exc_data[aid]), axis=0)
            inh_avg = np.nanmean(np.vstack(inh_data[aid]), axis=0)
            with np.errstate(divide='ignore', invalid='ignore'):
                ei_ratio = exc_avg / inh_avg
                if apply_log:
                    ei_ratio = np.log10(ei_ratio)
            ei_data[aid] = ei_ratio

    for aid, values in ei_data.items():
        df[aid] = values
        if "WT" in aid:
            wt_cols.append(aid)
        elif "VPA" in aid:
            vpa_cols.append(aid)

    if wt_cols:
        df['WT_Mean'] = df[wt_cols].mean(axis=1)
        df['WT_SD'] = df[wt_cols].std(axis=1)
        df['WT_N'] = df[wt_cols].count(axis=1)
    if vpa_cols:
        df['VPA_Mean'] = df[vpa_cols].mean(axis=1)
        df['VPA_SD'] = df[vpa_cols].std(axis=1)
        df['VPA_N'] = df[vpa_cols].count(axis=1)

    p_vals = []
    for i in range(len(df)):
        wt_vals = df.loc[i, wt_cols].dropna().values
        vpa_vals = df.loc[i, vpa_cols].dropna().values
        if len(wt_vals) > 1 and len(vpa_vals) > 1:
            _, p = ttest_ind(wt_vals, vpa_vals, equal_var=False)
        else:
            p = np.nan
        p_vals.append(p)

    df["P_value"] = p_vals
    df.to_csv(output_path, index=False)
    print(f"✅ Created {'log ' if apply_log else ''}E/I summary CSV: {output_path}")

def main():
    selected_folder = select_folder()
    if not selected_folder:
        print("❌ No folder selected.")
        return

    targets = [
        ("PSD_number", False),
        ("gep_number", False),
        ("PSD_area", False),
        ("gep_area", False),
    ]

    for keyword, apply_log in targets:
        files = find_csv_files(selected_folder, keyword)
        if files:
            create_summary_csv(selected_folder, files, keyword, apply_log=apply_log)
        else:
            print(f"⚠️ No '{keyword}.csv' files found.")

    psd_files = find_csv_files(selected_folder, "PSD_number")
    gep_files = find_csv_files(selected_folder, "gep_number")

    if psd_files and gep_files:
        compute_ei_ratio(selected_folder, psd_files, gep_files)
        compute_ei_ratio_log(selected_folder, psd_files, gep_files)

if __name__ == "__main__":
    main()
