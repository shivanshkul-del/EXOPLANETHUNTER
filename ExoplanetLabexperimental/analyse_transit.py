import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if len(sys.argv) < 2:
    print("Usage: py analyse_transit.py data/your_run.csv")
    raise SystemExit(1)

path = Path(sys.argv[1])

if not path.exists():
    raise FileNotFoundError(f"CSV not found: {path}")

df = pd.read_csv(path)

required = {"time_ms", "light_mV", "brightness_pct"}
if not required.issubset(df.columns):
    raise ValueError(f"Unexpected CSV columns: {list(df.columns)}")

df = df.dropna(subset=list(required)).sort_values("time_ms")

t = df["time_ms"].to_numpy(dtype=float) / 1000.0
brightness = df["brightness_pct"].to_numpy(dtype=float)

if len(t) < 20:
    raise ValueError("Not enough readings to analyse.")

# Use the 80th percentile as an initial baseline estimate.
baseline = float(np.percentile(brightness, 80))
minimum = float(np.percentile(brightness, 1))

if baseline <= 0:
    raise ValueError("Invalid brightness baseline.")

depth = max(0.0, (baseline - minimum) / baseline)
radius_ratio = np.sqrt(depth)

print(f"File: {path}")
print(f"Readings: {len(df)}")
print(f"Recording duration: {t[-1] - t[0]:.2f} seconds")
print(f"Estimated baseline: {baseline:.2f}%")
print(f"Approximate dip depth: {baseline - minimum:.2f} percentage points")
print(f"Fractional dip depth: {depth:.5f}")
print(f"Apparent radius ratio: {radius_ratio:.4f}")

# Find candidate dips using a threshold based on the observed range.
threshold = baseline - 0.25 * (baseline - minimum)
below = brightness < threshold

edges = np.diff(np.r_[False, below, False].astype(int))
starts = np.where(edges == 1)[0]
ends = np.where(edges == -1)[0]

events = []
for start, end in zip(starts, ends):
    if end - start >= 2:
        segment = brightness[start:end]
        min_index = start + int(np.argmin(segment))
        events.append((t[start], t[end - 1], t[min_index]))

print(f"Candidate dips detected: {len(events)}")

if events:
    durations = [end - start for start, end, _ in events]
    print(f"Median candidate dip duration: {np.median(durations):.3f} seconds")

if len(events) >= 2:
    minima = np.array([event[2] for event in events])
    print(f"Median interval between candidate dips: {np.median(np.diff(minima)):.3f} seconds")
else:
    print("Not enough candidate dips to estimate a repeated-dip interval.")

plt.figure(figsize=(11, 5))
plt.plot(t - t[0], brightness, label="Measured brightness")
plt.axhline(baseline, linestyle="--", label="Estimated baseline")
plt.axhline(threshold, linestyle=":", label="Exploratory dip threshold")
plt.xlabel("Elapsed time (seconds)")
plt.ylabel("Relative brightness (%)")
plt.title("Exoplanet Transit Laboratory — Experimental Light Curve")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
