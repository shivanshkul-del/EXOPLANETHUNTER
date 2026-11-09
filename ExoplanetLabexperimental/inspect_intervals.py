import numpy as np
import pandas as pd
from pathlib import Path

files = [
    Path("data/run_20261026_163841.csv"),
    Path("data/run_20261026_170039.csv"),
    Path("data/run_20261026_170150.csv"),
]

for path in files:
    df = pd.read_csv(path)
    t = df["time_ms"].to_numpy(dtype=float) / 1000
    y = df["brightness_pct"].to_numpy(dtype=float)

    valid = np.isfinite(t) & np.isfinite(y)
    t, y = t[valid], y[valid]

    smooth = pd.Series(y).rolling(
        3, center=True, min_periods=1
    ).median().to_numpy()

    baseline = np.percentile(smooth, 80)
    threshold = baseline - max(
        5.0, 0.30 * (baseline - np.percentile(smooth, 1))
    )

    inside = smooth < threshold
    starts = np.where(inside & ~np.r_[False, inside[:-1]])[0]
    ends = np.where(inside & ~np.r_[inside[1:], False])[0]

    dip_times = [
        float(np.mean(t[a:b+1]))
        for a, b in zip(starts, ends) if b > a
    ]
    intervals = np.diff(dip_times)

    print(f"\n{path.name}")
    print("Intervals (seconds):")
    print(np.round(intervals, 3))
    print(f"Mean: {np.mean(intervals):.3f} s")
    print(f"Median: {np.median(intervals):.3f} s")
    print(f"Standard deviation: {np.std(intervals):.3f} s")
