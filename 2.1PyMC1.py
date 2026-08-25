# -*- coding: utf-8 -*-
"""
@author: Liang
"""

import matplotlib.pyplot as plt
import mne
import numpy as np

all_y_hbo_std = []
all_y_hbr_std = []


X_task = global_sim
X_task_std = X_task / np.max(X_task)


for data_proc, processed in zip(group_data_list, processed_list):

  # First HbO / HbR channel
  hbo_ch = mne.pick_types(processed.info, fnirs="hbo")[0]
  hbr_ch = mne.pick_types(processed.info, fnirs="hbr")[0]

  # channel data
  y_hbo = data_proc[hbo_ch]
  y_hbr = data_proc[hbr_ch]


  y_hbo_std = (y_hbo - np.mean(y_hbo)) / np.std(y_hbo)
  y_hbr_std = (y_hbr - np.mean(y_hbr)) / np.std(y_hbr)

  all_y_hbo_std.append(y_hbo_std)
  all_y_hbr_std.append(y_hbr_std)


min_length = min(min(len(y) for y in all_y_hbo_std), len(X_task_std))

all_y_hbo_std = np.array([y[:min_length] for y in all_y_hbo_std])
all_y_hbr_std = np.array([y[:min_length] for y in all_y_hbr_std])
X_task_std = X_task_std[:min_length]

print("Ready for PyMC.")
print(f"HbO: {all_y_hbo_std.shape}")
print(f"HbR: {all_y_hbr_std.shape}")
print(f"{X_task_std.shape}")