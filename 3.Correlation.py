# -*- coding: utf-8 -*-
"""
@author: Liang
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

save_dir = Path(r"your file path")
save_dir.mkdir(parents=True, exist_ok=True)
save_png_path = save_dir / "Group_Posterior_Bivariate_Hierarchical.png"

# Group-level Hyper-posteriors
# group_beta_hbo & group_beta_hbr
beta_hbo_samples = trace_group_mv.posterior["group_beta_hbo"].values.flatten()
beta_hbr_samples = trace_group_mv.posterior["group_beta_hbr"].values.flatten()

corr_samples = trace_group_mv.posterior["hbo_hbr_correlation"].values.flatten()

# 95% HDI
hbo_lower, hbo_upper = np.percentile(beta_hbo_samples, [2.5, 97.5])
hbr_lower, hbr_upper = np.percentile(beta_hbr_samples, [2.5, 97.5])
corr_lower, corr_upper = np.percentile(corr_samples, [2.5, 97.5])


fig, axes = plt.subplots(1, 3, figsize=(16, 5))

#  HbO's Beta (≈ 1.0) 
axes[0].hist(beta_hbo_samples, bins=50, color="#E67E22", alpha=0.75, density=True)
axes[0].axvline(0, color="black", linestyle="--", linewidth=1.5, label="Ref (0)")
axes[0].axvline(
    1.0, color="#27AE60", linestyle="--", linewidth=2, label="True HbO (1.0)"
)
axes[0].axvline(hbo_lower, color="#C0392B", linestyle=":", linewidth=2)
axes[0].axvline(hbo_upper, color="#C0392B", linestyle=":", linewidth=2)
axes[0].set_title(
    f"Group Posterior of Beta (HbO)\n95% HDI: [{hbo_lower:.2f},"
    f" {hbo_upper:.2f}]",
    fontsize=12,
    fontweight="bold",
)
axes[0].set_xlabel(r"Effect Size $\beta_{HbO}$")
axes[0].legend(loc="upper left")
axes[0].grid(True, linestyle=":", alpha=0.6)

# HbR's Beta (≈ -0.4)
axes[1].hist(beta_hbr_samples, bins=50, color="#2980B9", alpha=0.75, density=True)
axes[1].axvline(0, color="black", linestyle="--", linewidth=1.5, label="Ref (0)")
axes[1].axvline(
    -0.4, color="#8E44AD", linestyle="--", linewidth=2, label="True HbR (-0.4)"
)
axes[1].axvline(hbr_lower, color="#1A5276", linestyle=":", linewidth=2)
axes[1].axvline(hbr_upper, color="#1A5276", linestyle=":", linewidth=2)
axes[1].set_title(
    f"Group Posterior of Beta (HbR)\n95% HDI: [{hbr_lower:.2f},"
    f" {hbr_upper:.2f}]",
    fontsize=12,
    fontweight="bold",
)
axes[1].set_xlabel(r"Effect Size $\beta_{HbR}$")
axes[1].legend(loc="upper right")
axes[1].grid(True, linestyle=":", alpha=0.6)

# Correlation
axes[2].hist(corr_samples, bins=50, color="#8E44AD", alpha=0.75, density=True)
axes[2].axvline(
    0, color="black", linestyle="--", linewidth=1.5, label="No Correlation (0)"
)
axes[2].axvline(corr_lower, color="#4A235A", linestyle=":", linewidth=2)
axes[2].axvline(corr_upper, color="#4A235A", linestyle=":", linewidth=2)
axes[2].set_title(
    f"Group Residual HbO-HbR Correlation\n95% HDI: [{corr_lower:.2f},"
    f" {corr_upper:.2f}]",
    fontsize=12,
    fontweight="bold",
)
axes[2].set_xlabel(r"Correlation Coefficient $\rho$")
axes[2].legend(loc="upper right")
axes[2].grid(True, linestyle=":", alpha=0.6)

plt.tight_layout()


plt.savefig(save_png_path, dpi=300, bbox_inches="tight")
print(f"Successfully: {save_png_path}")

plt.show()