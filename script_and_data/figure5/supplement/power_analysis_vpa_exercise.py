# power_analysis_vpa_exercise.py

"""
Post-hoc power analysis for PC1 scores between VPA_Control and VPA_Exercise groups.
"""

import numpy as np
from statsmodels.stats.power import TTestIndPower

# ==== Input data: PC1 scores ====

control_scores = np.array([
    1.379347613,
    2.413201763,
    1.490012181,
    4.192769700,
    1.026916218,
   -1.266960588,
   -0.193036175,
    1.625265642,
    0.898317063,
    1.017450155,
    0.031310620
])

exercise_scores = np.array([
   -0.717366373,
   -0.417641798,
    0.787729964
])

# ==== Sample sizes ====
n_control = len(control_scores)
n_exercise = len(exercise_scores)

# ==== Means and standard deviations ====
mean_ctrl = control_scores.mean()
mean_ex = exercise_scores.mean()
sd_ctrl = control_scores.std(ddof=1)
sd_ex = exercise_scores.std(ddof=1)

# ==== Pooled standard deviation ====
pooled_sd = np.sqrt(((n_control - 1) * sd_ctrl**2 +
                     (n_exercise - 1) * sd_ex**2) /
                    (n_control + n_exercise - 2))

# ==== Cohen's d ====
d = (mean_ex - mean_ctrl) / pooled_sd

# ==== Post-hoc power calculation ====
analysis = TTestIndPower()
power = analysis.power(effect_size=abs(d),
                       nobs1=n_control,
                       alpha=0.05,
                       ratio=n_exercise/n_control)

# ==== Required sample size per group for 80% power ====
required_n = analysis.solve_power(effect_size=abs(d),
                                  power=0.8,
                                  alpha=0.05,
                                  ratio=1.0)

# ==== Output results ====
print("=== Power Analysis Results ===")
print(f"Group sizes: Control = {n_control}, Exercise = {n_exercise}")
print(f"Cohen's d: {d:.3f}")
print(f"Post-hoc power (alpha=0.05): {power:.3f}")
print(f"Required n per group for 80% power: {required_n:.1f}")
