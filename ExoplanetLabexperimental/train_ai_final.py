import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib

TRAIN_FILE = Path("data/run_20261026_170039.csv")
TEST_FILE = Path("data/run_20261026_170150.csv")
MODEL_FILE = Path("models/transit_interval_model_final.joblib")


def extract_intervals(path):
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path.resolve()}")

    df = pd.read_csv(path)
    required = {"time_ms", "brightness_pct"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing columns in {path.name}: {required}")

    t = df["time_ms"].to_numpy(dtype=float) / 1000.0
    y = df["brightness_pct"].to_numpy(dtype=float)
    valid = np.isfinite(t) & np.isfinite(y)
    t, y = t[valid], y[valid]

    smooth = pd.Series(y).rolling(
        3, center=True, min_periods=1
    ).median().to_numpy()

    baseline = np.percentile(smooth, 80)
    minimum = np.percentile(smooth, 1)
    threshold = baseline - max(5.0, 0.30 * (baseline - minimum))
    inside = smooth < threshold

    starts = np.where(inside & ~np.r_[False, inside[:-1]])[0]
    ends = np.where(inside & ~np.r_[inside[1:], False])[0]

    dip_times = [
        float(np.mean(t[a:b + 1]))
        for a, b in zip(starts, ends) if b > a
    ]
    intervals = np.diff(dip_times)

    if len(intervals) < 6:
        raise ValueError(
            f"Only {len(intervals)} intervals detected in {path.name}"
        )

    return intervals


def make_examples(intervals):
    X, y = [], []
    for i in range(3, len(intervals)):
        X.append(intervals[i - 3:i])
        y.append(intervals[i])
    return np.asarray(X), np.asarray(y)


def main():
    print("EXOPLANET HUNTER - REVISED AI MODEL")
    print("Run 2 = training; Run 3 = testing")
    print("Run 1 is excluded.")
    print("-" * 40)

    train_intervals = extract_intervals(TRAIN_FILE)
    test_intervals = extract_intervals(TEST_FILE)

    X_train, y_train = make_examples(train_intervals)
    X_test, y_test = make_examples(test_intervals)

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=3,
        min_samples_leaf=2,
        random_state=42
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    baseline_value = float(np.median(y_train))
    baseline_predictions = np.full(len(y_test), baseline_value)

    ai_mae = mean_absolute_error(y_test, predictions)
    baseline_mae = mean_absolute_error(y_test, baseline_predictions)

    print(f"Training examples: {len(y_train)}")
    print(f"Test examples: {len(y_test)}")
    print(f"Training median interval: {baseline_value:.4f} s")
    print(f"Baseline test MAE: {baseline_mae:.4f} s")
    print(f"AI test MAE: {ai_mae:.4f} s")

    print("\nIndividual predictions:")
    for i, (actual, predicted, base) in enumerate(
        zip(y_test, predictions, baseline_predictions), 1
    ):
        print(
            f"{i}. Actual={actual:.3f}s | "
            f"AI={predicted:.3f}s | Baseline={base:.3f}s"
        )

    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_FILE)
    print(f"\nMODEL SAVED: {MODEL_FILE.resolve()}")

    if ai_mae < baseline_mae:
        print("AI performed better on this test recording.")
    elif ai_mae > baseline_mae:
        print("Baseline performed better on this test recording.")
    else:
        print("Both methods had equal error.")

    print("Note: one test recording does not establish general reliability.")


if __name__ == "__main__":
    main()
