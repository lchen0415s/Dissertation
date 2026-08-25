# -*- coding: utf-8 -*-
"""
@author: Liang
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import mne
from scipy.stats import gamma

# HRF
def hrf_kernel(sfreq, peak=6, dispersion=1, length=30):
    t = np.arange(0, length, 1/sfreq)
    a = (peak/dispersion)**2
    scale = dispersion**2 / peak
    h = gamma.pdf(t, a, scale=scale)
    return h / np.max(h)   # HRF peak = 1

# Event
df_events = pd.DataFrame({
    "onset": [30, 90, 150],
    "duration": [20, 20, 20],
    "value": [1, 1, 1],
    "label": ["Simulated_Task"]*3
})

root = Path(r"your file path")
output_dirs = [d for d in root.iterdir() if d.is_dir() and d.name.endswith("_newoutput")]
print(f"Find {len(output_dirs)} newoutput files.\n")

group_data_list = []
processed_list = []
global_sim = None

# Trial amplitude
np.random.seed(42)

for out_dir in output_dirs:
    fif_files = list(out_dir.glob("*.fif"))
    if not fif_files:
        continue

    fif = fif_files[0]
    raw = mne.io.read_raw_fif(fif, preload=True, verbose=False)

    annotations = mne.Annotations(
        onset=df_events['onset'].values,
        duration=df_events['duration'].values,
        description=df_events['label'].values
    )
    raw.set_annotations(annotations)


    od = mne.preprocessing.nirs.optical_density(raw.copy())
    tddr = mne.preprocessing.nirs.temporal_derivative_distribution_repair(od)
    conc = mne.preprocessing.nirs.beer_lambert_law(tddr)
    processed = conc.copy().filter(l_freq=0.025, h_freq=0.5, verbose=False)


    sfreq = processed.info['sfreq']
    times = processed.times
    onsets = [30, 90, 150]
    duration = 20

    # Trial amplitude A_i ~ N(1.0, 0.15^2)
    trial_amplitudes = np.random.normal(loc=1.0, scale=0.15, size=len(onsets))

    stim_varying = np.zeros_like(times)
    for idx, onset in enumerate(onsets):
        mask = (times >= onset) & (times < onset + duration)
        stim_varying[mask] = trial_amplitudes[idx]

   # HRF convolution & Normalization
    h = hrf_kernel(sfreq)
    sim_raw = np.convolve(stim_varying, h, mode='full')[:len(times)]

    sim = sim_raw / np.max(sim_raw)

    # HbO=1.0x std, HbR=-0.4x std
    # HbO / HbR channel index
    hbo_picks = mne.pick_types(processed.info, fnirs="hbo")
    hbr_picks = mne.pick_types(processed.info, fnirs="hbr")

    hbo_ch = hbo_picks[0]
    hbr_ch = hbr_picks[0]

    # std
    std_hbo = np.std(processed._data[hbo_ch])
    std_hbr = np.std(processed._data[hbr_ch])


    processed._data[hbo_ch] += sim * std_hbo * 1.0   # HbO (+1.0 x std)
    processed._data[hbr_ch] += sim * std_hbr * -0.4  # HbR (-0.4 x std)

    processed_list.append(processed)   
    group_data_list.append(processed.get_data())
    global_sim = sim

    print(f"{out_dir.name[:8]}... prepocessing + signal are done.")