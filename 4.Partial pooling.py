# -*- coding: utf-8 -*-
"""
@author: Liang
"""

import matplotlib.pyplot as plt

save_dir = Path(r'your file path')
save_dir.mkdir(parents=True, exist_ok=True)
save_png_path = save_dir / 'MultiChannel_Shrinkage_Results.png'

# 4 channels' Beta
# group_channel_beta: (draws*chains, 4_channels, 2_types)
channel_beta_samples = trace_4ch.posterior[
    'group_channel_beta'
].values.reshape(-1, 4, 2)

fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharey='row')

colors_hbo = ['#E67E22', '#BDC3C7', '#BDC3C7', '#BDC3C7']
colors_hbr = ['#2980B9', '#BDC3C7', '#BDC3C7', '#BDC3C7']


for ch in range(4):
  b_hbo = channel_beta_samples[:, ch, 0]
  b_hbr = channel_beta_samples[:, ch, 1]

  hbo_low, hbo_high = np.percentile(b_hbo, [2.5, 97.5])
  hbr_low, hbr_high = np.percentile(b_hbr, [2.5, 97.5])

  # HbO
  axes[0, ch].hist(
      b_hbo, bins=40, color=colors_hbo[ch], alpha=0.8, density=True
  )
  axes[0, ch].axvline(
      0, color='black', linestyle='--', linewidth=1.5, label='Ref (0)'
  )
  axes[0, ch].axvline(hbo_low, color='#C0392B', linestyle=':')
  axes[0, ch].axvline(hbo_high, color='#C0392B', linestyle=':')

  title_prefix = 'Ch 0 (Active)' if ch == 0 else f'Ch {ch} (Noise Only)'
  axes[0, ch].set_title(
      f'{title_prefix} - HbO\nMean: {np.mean(b_hbo):.2f} [{hbo_low:.2f},'
      f' {hbo_high:.2f}]',
      fontweight='bold',
  )
  axes[0, ch].grid(True, linestyle=':', alpha=0.6)
  if ch == 0:
    axes[0, ch].set_ylabel('Density (HbO)')

  # HbR
  axes[1, ch].hist(
      b_hbr, bins=40, color=colors_hbr[ch], alpha=0.8, density=True
  )
  axes[1, ch].axvline(
      0, color='black', linestyle='--', linewidth=1.5, label='Ref (0)'
  )
  axes[1, ch].axvline(hbr_low, color='#1A5276', linestyle=':')
  axes[1, ch].axvline(hbr_high, color='#1A5276', linestyle=':')

  axes[1, ch].set_title(
      f'{title_prefix} - HbR\nMean: {np.mean(b_hbr):.2f} [{hbr_low:.2f},'
      f' {hbr_high:.2f}]',
      fontweight='bold',
  )
  axes[1, ch].grid(True, linestyle=':', alpha=0.6)
  if ch == 0:
    axes[1, ch].set_ylabel('Density (HbR)')

plt.suptitle(
    'Hierarchical Bayesian Spatial Shrinkage & Pooling across 4 Channels',
    fontsize=14,
    fontweight='bold',
    y=1.02,
)
plt.tight_layout()


plt.savefig(save_png_path, dpi=300, bbox_inches='tight')
print(f'Successfully: {save_png_path}')

plt.show()