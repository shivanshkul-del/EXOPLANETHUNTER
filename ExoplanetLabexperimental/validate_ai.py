import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

FILES = [
    Path("data/run_20261026_163841.csv"),
    Path("data/run_20261026_170039.csv"),
    Path("data/run_20261026_170150.csv"),
]

def intervals_from_csv(path):
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
    return np.diff(dip_times)

def make_examples(values):
    X, y = [], []
    for i in range(3, len(values)):
        X.append(values[i-3:i])
        y.append(values[i])
    return np.asarray(X), np.asarray(y)

series = [intervals_from_csv(p) for p in FILES]

print("LEAVE-ONE-RECORDING-OUT VALIDATION")
print("---------------------------------")

for test_index in range(len(FILES)):
    train_values = [
        series[i] for i in range(len(FILES)) if i != test_index
    ]
    X_parts, y_parts = [], []

    for values in train_values:
        X, y = make_examples(values)
        if len(y):
            X_parts.append(X)
            y_parts.append(y)

    X_train = np.vstack(X_parts)
    y_train = np.concatenate(y_parts)

    X_test, y_test = make_examples(series[test_index])

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=3,
        min_samples_leaf=2,
        random_state=42
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    baseline = np.full(len(y_test), np.median(y_train))

    print(f"\nTest recording: {FILES[test_index].name}")
    print(f"Training examples: {len(y_train)}")
    print(f"Test examples: {len(y_test)}")
    print(f"Baseline MAE: {mean_absolute_error(y_test, baseline):.4f} s")
    print(f"ML model MAE: {mean_absolute_error(y_test, pred):.4f} s")

print("\nCompare all three runs before drawing conclusions.")
