#!/usr/bin/env python3
"""
Power analysis plot for a two-sample t-test with Cohen's d = 0.94
"""

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.power import TTestIndPower

plt.rcParams.update({
    'font.size': 12,            # Base font size
    'axes.titlesize': 12,       # Title size
    'axes.labelsize': 12,       # Axis label size
    'xtick.labelsize': 12,      # X-axis tick size
    'ytick.labelsize': 12,      # Y-axis tick size
    'legend.fontsize': 12       # Legend font size
})

def main():
    # === Parameters ===
    effect_size = 0.94  # Cohen's d estimated from VPA comparison
    alpha = 0.05        # Significance level
    analysis = TTestIndPower()

    # === Sample size range (per group) ===
    n_range = np.arange(3, 51, 1)  # from 3 to 50 samples per group

    # === Compute statistical power ===
    power_values = analysis.power(effect_size=effect_size,
                                  nobs1=n_range,
                                  alpha=alpha,
                                  ratio=1.0)  # equal group sizes

    # === Plot the power curve ===
    plt.figure(figsize=(3.6, 3))
    plt.plot(n_range, power_values, label=f"d = {effect_size}")
    plt.axhline(0.8, linestyle='--', color='gray', label='Power = 0.80')
    plt.xlabel('Sample size per group (n)')
    plt.ylabel('Statistical Power')
    plt.title("Power Curve for Two-Sample t-Test")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
