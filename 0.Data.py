# -*- coding: utf-8 -*-
"""
@author: Liang
"""

from pathlib import Path
import mne

root = Path(r"your file path")

subject_dirs = [d for d in root.iterdir() if d.is_dir()]

print(f"Find {len(subject_dirs)} files.")


for sub_dir in subject_dirs:
    sub_name = sub_dir.name
    print(f"\nLoading:{sub_name}")

    snirf_files = list(sub_dir.glob("*.snirf"))
    if len(snirf_files) == 0:
        print(f"{sub_name} has no SNIRF.")
        continue

    output_dir = root / f"{sub_name}_newoutput"
    output_dir.mkdir(exist_ok=True)

    print(f"Output:{output_dir}")


    for snirf in snirf_files:
        print(f"Loading:{snirf.name}")
        raw = mne.io.read_raw_snirf(snirf, preload=True)

        out_path = output_dir / f"{snirf.stem}.fif"
        raw.save(out_path, overwrite=True)

        print(f"Output:{out_path.name}")

print("\nDone.")