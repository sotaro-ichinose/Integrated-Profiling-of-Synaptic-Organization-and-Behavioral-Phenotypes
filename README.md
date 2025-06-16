# Analysis Scripts and Data

This repository contains scripts and supporting data for the analysis presented in the manuscript:

**"Integrated Profiling of Synaptic E/I Balance Reveals Altered Synaptic Organization and Phenotypic Variability in a Prenatal ASD Model"**  
*(Currently under peer review)*

---

## Contents

- `scripts_and_data/`:  
  Main folder containing all figure-specific analysis pipelines.  
  Each subfolder corresponds to a main figure from the manuscript.

  - `figure1/`, `figure2/`, ...:  
    Each folder includes both `.py` (analysis script) and `.csv` (processed data) files  
    necessary to reproduce the corresponding figure.

    - `supplement/` (within each figure folder, if applicable):  
      Contains scripts and data for figure supplements (e.g., Figure 1—figure supplement 1D–F).  
      These are organized similarly with `.py` and `.csv` files.

- `script_figure_mapping.csv`:  
  Index linking each figure and supplement to the specific input files, scripts, and outputs.

- `LICENSE.md`:  
  MIT license for code, and CC BY 4.0 license for data.

---

## How to Reproduce Figures

To reproduce a figure:

1. Navigate to `scripts_and_data/figureX/` (e.g., `figure3/`)
2. Run the corresponding `.py` script using Python 3.10
3. Input `.csv` files are included in the same directory
4. For supplemental figures, use scripts under `figureX/supplement/`

Each figure's pipeline is self-contained. See `script_figure_mapping.csv` for a global index.

---

## License

### 🔹 Code (.py and .ijm)

All Python scripts (`.py`) and ImageJ macros (`.ijm`) in this repository are licensed under the [MIT License](LICENSE.md).  
You are free to use, modify, and redistribute the code with attribution.

### 🔹 Data (.csv)

The data files in this repository are associated with a manuscript currently under peer review.  
These are shared solely for review or demonstration purposes.

**Note:** Redistribution or reuse of the data is not permitted until the manuscript is officially published.  
Please contact the corresponding author for any data-related inquiries.

---

## Contact

For questions, please contact:  
**Sotaro Ichinose** – [ichinose@gunma-u.ac.jp]

---

## Citation

(Include this section after acceptance)

