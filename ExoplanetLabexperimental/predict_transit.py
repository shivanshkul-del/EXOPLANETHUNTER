import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path("data")
FILES = [
    DATA_DIR / "run_20261026_163841.csv",
    DATA_DIR / "run_20261026_170039.csv",
    DATA_DIR / "run_20261026_170150.csv",
]

def load_dips(path):
    df = pd.read_csv(path)
    required = {"time_ms", "brightness_pct"}
    if not required.issubset(df.columns):
        raise ValueError(f"{path}: expected columns {required}")

    t = df["time_ms"].to_numpy(dtype=float) / 1000
    y = df["brightness_pct"].to_numpy(dtype=float)
    valid = np.isfinite(t) & np.isfinite(y)
    t, y = t[valid], y[valid]

    if len(t) < 20:
        raise ValueError(f"{path}: too few valid readings")

    # Smooth small sensor fluctuations without erasing the sharp dips.
    smooth = pd.Series(y).rolling(3, center=True, min_periods=1).median().to_numpy()
    baseline = np.percentile(smooth, 80)
    threshold = baseline - max(5.0, 0.30 * (baseline - np.percentile(smooth, 1)))
    inside = smooth < threshold

    # Find each continuous dip region.
    starts = np.where(inside & ~np.r_[False, inside[:-1]])[0]
    ends = np.where(inside & ~np.r_[inside[1:], False])[0]
    dips = []
    for a, b in zip(starts, ends):
        if b > a:
            dips.append((t[a:b+1].mean(), t[b] - t[a], smooth[a:b+1].min()))
    return t, y, np.array(dips), baseline

all_results = []
for path in FILES:
    if not path.exists():
        print(f"Missing file: {path}")
        sys.exit(1)
    t, y, dips, baseline = load_dips(path)
    if len(dips) < 4:
        print(f"Only {len(dips)} dips detected in {path.name}; check threshold/settings.")
        sys.exit(1)
    intervals = np.diff(dips[:, 0])
    period = float(np.median(intervals))
    all_results.append((path.name, t, y, dips, baseline, period))
    print(f"\n{path.name}")
    print(f"  Dips detected: {len(dips)}")
    print(f"  Median dip interval: {period:.3f} s")
    print(f"  Median dip duration: {np.median(dips[:, 1]):.3f} s")

# Test a simple prediction on the last recording:
# use its first 70% of detected dips to estimate the period,
# then predict the remaining dip times.
name, t, y, dips, baseline, _ = all_results[-1]
split = max(4, int(len(dips) * 0.70))
train_times = dips[:split, 0]
test_times = dips[split:, 0]

if len(test_times) == 0:
    print("\nNot enough dips for a held-out test.")
    sys.exit(1)

period = float(np.median(np.diff(train_times)))
predicted = train_times[-1] + period * np.arange(1, len(test_times) + 1)
errors = predicted - test_times

print("\nHELD-OUT TIMING TEST")
print(f"Recording: {name}")
print(f"Training dips: {len(train_times)}")
print(f"Test dips: {len(test_times)}")
print(f"Estimated period from training: {period:.3f} s")
print(f"Mean absolute timing error: {np.mean(np.abs(errors)):.3f} s")
print("Note: this is a periodic baseline, not yet a trained AI model.")

plt.figure(figsize=(11, 5))
plt.plot(t, y, label="Measured brightness", linewidth=1)
plt.axhline(baseline, linestyle="--", label="Estimated baseline")
for i, dip in enumerate(dips):
    plt.axvline(dip[0], alpha=0.20)
for actual, pred in zip(test_times, predicted):
    plt.axvline(actual, linestyle=":", alpha=0.8)
    plt.axvline(pred, linestyle="--", alpha=0.8)
plt.xlabel("Elapsed time (s)")
plt.ylabel("Relative brightness (%)")
plt.title("Exoplanet Transit Laboratory — Timing Prediction Test")
plt.legend()
plt.tight_layout()
plt.show()
