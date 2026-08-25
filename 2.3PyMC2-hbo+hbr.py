# -*- coding: utf-8 -*-
"""
@author: Liang
"""

import numpy as np
import pymc as pm

# 3D
Y_obs_group_mv = np.stack([all_y_hbo_std, all_y_hbr_std], axis=-1)
n_subs, n_times = all_y_hbo_std.shape

# Hierarchical Bivariate
with pm.Model() as group_mv_glm:

  # Group-level Hyper-priors
  # [0] = HbO, [1] = HbR
  group_mu_beta = pm.Normal('group_mu_beta', mu=0, sigma=1, shape=2)
  group_sd_beta = pm.HalfNormal('group_sd_beta', sigma=1, shape=2)

  group_mu_alpha = pm.Normal('group_mu_alpha', mu=0, sigma=1, shape=2)
  group_sd_alpha = pm.HalfNormal('group_sd_alpha', sigma=1, shape=2)


  alpha = pm.Normal(
      'alpha', mu=group_mu_alpha, sigma=group_sd_alpha, shape=(n_subs, 2)
  )
  beta = pm.Normal(
      'beta', mu=group_mu_beta, sigma=group_sd_beta, shape=(n_subs, 2)
  )


  pm.Deterministic('group_beta_hbo', group_mu_beta[0])
  pm.Deterministic('group_beta_hbr', group_mu_beta[1])

  # Mu
  mu = alpha[:, None, :] + beta[:, None, :] * X_task_std[None, :, None]

  # Covariance Matrix
  sd_dist = pm.HalfNormal.dist(sigma=1)
  chol, corr, stds = pm.LKJCholeskyCov(
      'chol', n=2, eta=2.0, sd_dist=sd_dist, compute_corr=True
  )

  pm.Deterministic('hbo_hbr_correlation', corr[0, 1])

  # Likelihood
  obs = pm.MvNormal('obs', mu=mu, chol=chol, observed=Y_obs_group_mv)

  # MCMC
  print(f'Hierarchical Bayesian GLM sampling...')
  trace_group_mv = pm.sample(
      draws=1000,
      tune=1000,
      chains=2,
      target_accept=0.9,
      random_seed=42,
      return_inferencedata=True,
  )

print('\nDone.')