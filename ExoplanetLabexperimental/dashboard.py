from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_PATH = ROOT / "models" / "transit_interval_model_final.joblib"

RUN_FILES = [
    "run_20261026_163841.csv",
    "run_20261026_170039.csv",
    "run_20261026_170150.csv",
]

st.set_page_config(
    page_title="Exoplanet Transit Laboratory",
    page_icon="🔭",
    layout="wide",
)

st.title("EXOPLANET TRANSIT LABORATORY")
st.caption("Experimental transit analysis and machine-learning interval prediction")

st.info(
    "This is a laboratory analogue of the exoplanet transit method. "
    "The AI analyses detected brightness dips and predicts the next dip interval. "
    "It does not confirm real exoplanets or directly measure a real planet's size."
)

@st.cache_data
def load_run(path_string):
    df = pd.read_csv(path_string)
    required = {"time_ms", "light_mV", "brightness_pct"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing CSV columns: {', '.join(sorted(missing))}")
    df = df[list(required)].copy()
    df = df[["time_ms", "light_mV", "brightness_pct"]]
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna().sort_values("time_ms").reset_index(drop=True)
    df["time_s"] = df["time_ms"] / 1000.0
    return df

def detect_dips(df):
    brightness = df["brightness_pct"].to_numpy(dtype=float)
    times = df["time_s"].to_numpy(dtype=float)

    if len(brightness) < 5:
        return np.array([]), np.array([]), np.array([])

    smooth = (
        pd.Series(brightness)
        .rolling(window=3, center=True, min_periods=1)
        .median()
        .to_numpy()
    )

    baseline = float(np.percentile(smooth, 80))
    low_level = float(np.percentile(smooth, 10))
    threshold = baseline - max(5.0, 0.30 * (baseline - low_level))
    is_low = smooth < threshold

    regions = []
    start = None
    for i, low in enumerate(is_low):
        if low and start is None:
            start = i
        elif not low and start is not None:
            if i - start >= 2:
                regions.append((start, i))
            start = None
    if start is not None and len(is_low) - start >= 2:
        regions.append((start, len(is_low)))

    dip_times = []
    dip_depths = []
    for a, b in regions:
        dip_times.append(float(np.mean(times[a:b])))
        dip_depths.append(float(baseline - np.min(smooth[a:b])))

    dip_times = np.asarray(dip_times, dtype=float)
    dip_depths = np.asarray(dip_depths, dtype=float)
    intervals = np.diff(dip_times) if len(dip_times) >= 2 else np.array([])
    return dip_times, dip_depths, intervals

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)

available_runs = [
    name for name in RUN_FILES if (DATA_DIR / name).exists()
]

if not available_runs:
    st.error(f"No expected CSV files were found in: {DATA_DIR}")
    st.stop()

selected_run = st.selectbox(
    "Select experimental recording",
    available_runs,
    index=len(available_runs) - 1,
)

try:
    df = load_run(str(DATA_DIR / selected_run))
    dip_times, dip_depths, intervals = detect_dips(df)
except Exception as exc:
    st.error(f"Could not analyse the selected CSV: {exc}")
    st.stop()

duration = float(df["time_s"].max() - df["time_s"].min())
mean_brightness = float(df["brightness_pct"].mean())
median_interval = float(np.median(intervals)) if len(intervals) else float("nan")
interval_std = float(np.std(intervals, ddof=1)) if len(intervals) > 1 else float("nan")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Samples", f"{len(df):,}")
c2.metric("Recording duration", f"{duration:.1f} s")
c3.metric("Detected dips", str(len(dip_times)))
c4.metric(
    "Median dip interval",
    f"{median_interval:.3f} s" if np.isfinite(median_interval) else "N/A",
)

tab1, tab2, tab3, tab4 = st.tabs([
    "Brightness signal",
    "Dip analysis",
    "AI prediction",
    "Raw data",
])

with tab1:
    st.subheader("Brightness versus time")
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(df["time_s"], df["brightness_pct"], linewidth=1, label="Measured brightness")
    if len(dip_times):
        for i, t in enumerate(dip_times):
            ax.axvline(
                t,
                linestyle="--",
                linewidth=0.8,
                alpha=0.6,
                label="Detected dip" if i == 0 else None,
            )
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Brightness (%)")
    ax.set_title("Measured light signal during the experiment")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        f"Mean measured brightness: {mean_brightness:.2f}%. "
        "Vertical dashed lines mark dips detected by the software."
    )

with tab2:
    st.subheader("Detected transit-like events")
    if len(dip_times):
        dips_df = pd.DataFrame({
            "Dip number": np.arange(1, len(dip_times) + 1),
            "Time (s)": dip_times,
            "Estimated dip depth (brightness percentage points)": dip_depths,
        })
        st.dataframe(
            dips_df.round(4),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.warning("No dips were detected in this recording.")

    if len(intervals):
        interval_df = pd.DataFrame({
            "Interval number": np.arange(1, len(intervals) + 1),
            "Interval (s)": intervals,
        })
        st.subheader("Intervals between detected dips")
        st.dataframe(interval_df.round(4), use_container_width=True, hide_index=True)

        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.plot(
            np.arange(1, len(intervals) + 1),
            intervals,
            marker="o",
            linewidth=1.5,
        )
        ax.axhline(
            np.median(intervals),
            linestyle="--",
            label=f"Median = {np.median(intervals):.3f} s",
        )
        ax.set_xlabel("Interval number")
        ax.set_ylabel("Interval (seconds)")
        ax.set_title("Variation in measured dip intervals")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.warning("At least two detected dips are needed to calculate intervals.")

with tab3:
    st.subheader("Machine-learning prediction")
    model = load_model()

    if model is None:
        st.error(f"Model file not found: {MODEL_PATH}")
    elif len(intervals) < 3:
        st.warning(
            "The selected recording has fewer than three measured intervals. "
            "The model needs at least three intervals to make a prediction."
        )
    else:
        try:
            # The trained model uses the three preceding intervals as features.
            features = np.asarray(intervals[-3:], dtype=float).reshape(1, -1)
            prediction = float(np.asarray(model.predict(features)).reshape(-1)[0])

            if not np.isfinite(prediction) or prediction <= 0:
                st.warning("The model returned an invalid interval prediction.")
            else:
                p1, p2 = st.columns(2)
                p1.metric("Predicted next dip interval", f"{prediction:.4f} s")
                p2.metric(
                    "Recent measured median",
                    f"{np.median(intervals[-3:]):.4f} s",
                )

                st.write("Intervals supplied to the model:")
                st.code(", ".join(f"{x:.4f} s" for x in intervals[-3:]))

                if len(dip_times):
                    next_time = float(dip_times[-1] + prediction)
                    st.metric("Estimated next dip time", f"{next_time:.3f} s")
                    st.caption(
                        "This estimated time is relative to the start of the selected "
                        "recording and assumes the observed interval pattern continues."
                    )

                st.markdown("**How the AI works**")
                st.write(
                    "A Random Forest regression model learns patterns from previous "
                    "dip intervals and estimates the next interval. The result is a "
                    "prediction, not a guaranteed event time."
                )
                st.warning(
                    "The model was trained on a very small number of experimental "
                    "examples. Its apparent accuracy on one recording does not prove "
                    "that it will generalise to new conditions or real astronomical data."
                )
        except Exception as exc:
            st.error(
                "Prediction failed. Check that the saved model was trained using "
                "exactly three preceding intervals as input features. "
                f"Details: {exc}"
            )

with tab4:
    st.subheader("Recorded sensor data")
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.download_button(
        "Download selected CSV",
        data=(DATA_DIR / selected_run).read_bytes(),
        file_name=selected_run,
        mime="text/csv",
    )

st.divider()
st.caption(
    "EXOPLANET TRANSIT LABORATORY | Offline CSV analysis | "
    "The dashboard does not display live sensor readings."
)
