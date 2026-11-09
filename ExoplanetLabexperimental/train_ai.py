import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib

FILES = [
    Path("data/run_20261026_163841.csv"),
    Path("data/run_20261026_170039.csv"),
    Path("data/run_20261026_170150.csv"),
]

def get_intervals(path):
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

    dip_times = []
    for a, b in zip(starts, ends):
        if b > a:
            dip_times.append(float(np.mean(t[a:b+1])))

    return np.diff(dip_times)

# Three previous intervals are the input features.
X_train, y_train = [], []

for path in FILES[:2]:
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV: {path}")

    intervals = get_intervals(path)
    for i in range(3, len(intervals)):
        X_train.append(intervals[i-3:i])
        y_train.append(intervals[i])

test_path = FILES[2]
if not test_path.exists():
    raise FileNotFoundError(f"Missing CSV: {test_path}")

test_intervals = get_intervals(test_path)
X_test, y_test = [], []

for i in range(3, len(test_intervals)):
    X_test.append(test_intervals[i-3:i])
    y_test.append(test_intervals[i])

if len(X_train) < 8 or len(X_test) < 3:
    raise ValueError(
        f"Not enough examples: training={len(X_train)}, "
        f"testing={len(X_test)}"
    )

X_train = np.asarray(X_train)
y_train = np.asarray(y_train)
X_test = np.asarray(X_test)
y_test = np.asarray(y_test)

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=3,
    min_samples_leaf=2,
    random_state=42
)
model.fit(X_train, y_train)

ai_predictions = model.predict(X_test)

# Baseline uses only the training recordings.
baseline_period = float(np.median(y_train))
baseline_predictions = np.full(len(y_test), baseline_period)

ai_mae = mean_absolute_error(y_test, ai_predictions)
baseline_mae = mean_absolute_error(y_test, baseline_predictions)

print("\nEXOPLANET TRANSIT LAB — MODEL COMPARISON")
print("----------------------------------------")
print(f"Training examples: {len(X_train)}")
print(f"Unseen test examples: {len(X_test)}")
print(f"Baseline median interval: {baseline_period:.4f} s")
print(f"Baseline test MAE: {baseline_mae:.4f} s")
print(f"ML model test MAE: {ai_mae:.4f} s")

print("\nIndividual test predictions:")
for i, (actual, ai, base) in enumerate(
    zip(y_test, ai_predictions, baseline_predictions), start=1
):
    print(
        f"{i}. Actual={actual:.3f}s | "
        f"ML={ai:.3f}s | Baseline={base:.3f}s"
    )

Path("models").mkdir(exist_ok=True)
joblib.dump(model, "models/transit_interval_model.joblib")
print("\nSaved trained model to models/transit_interval_model.joblib")

if ai_mae < baseline_mae:
    print("Result: ML performed better on this test recording.")
else:
    print("Result: ML did not outperform the simple baseline on this test.")
    print("That is a valid scientific result; do not hide it.")
