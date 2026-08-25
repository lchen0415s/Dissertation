# -*- coding: utf-8 -*-
"""
@author: Liang
"""

#2D (for multi simples)
import pymc as pm
import numpy as np


n_subs, n_times = all_y_hbo_std.shape

with pm.Model() as group_glm:

    # Priors
    alpha_hbo = pm.Normal('alpha_hbo', mu=0, sigma=10, shape=n_subs)
    alpha_hbr = pm.Normal('alpha_hbr', mu=0, sigma=10, shape=n_subs)

    beta_hbo = pm.Normal('beta_hbo', mu=0, sigma=5, shape=n_subs)
    beta_hbr = pm.Normal('beta_hbr', mu=0, sigma=5, shape=n_subs)

    sigma_hbo = pm.HalfNormal('sigma_hbo', sigma=5, shape=n_subs)
    sigma_hbr = pm.HalfNormal('sigma_hbr', sigma=5, shape=n_subs)

    # Mu = Alpha + Beta * X
    mu_hbo = alpha_hbo[:, None] + beta_hbo[:, None] * X_task_std[None, :]
    mu_hbr = alpha_hbr[:, None] + beta_hbr[:, None] * X_task_std[None, :]

    # Likelihood
    Y_obs_hbo = pm.Normal('Y_obs_hbo', mu=mu_hbo, sigma=sigma_hbo[:, None], observed=all_y_hbo_std)
    Y_obs_hbr = pm.Normal('Y_obs_hbr', mu=mu_hbr, sigma=sigma_hbr[:, None], observed=all_y_hbr_std)

    # MCMC
    print(f"MCMC sampling...")
    # chains=2
    trace_group = pm.sample(draws=1000, tune=1000, chains=2, target_accept=0.9, random_seed=42)

print("\nDone.")