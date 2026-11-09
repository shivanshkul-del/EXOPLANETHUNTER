import re
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import serial
import serial.tools.list_ports
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# --------------------------------------------
# EXOPLANET HUNTER - LIVE DASHBOARD
# --------------------------------------------
st.set_page_config(
    page_title="Exoplanet Hunter - Live",
    page_icon="🔭",
    layout="wide",
)

st.title("🔭 Exoplanet Hunter")
st.subheader("Live ESP32 Transit Monitor")
st.caption(
    "Live LDR readings, transit timing and next-dip prediction. "
    "Predictions are experimental and require validation."
)

MODEL_PATH = Path("models/transit_interval_model_final.joblib")
BAUD_RATE = 115200
MAX_POINTS = 1800

# Matches the current Arduino Serial Plotter output, for example:
# Baseline_100:100.0,Transit_Threshold:95.0,Star_Brightness:87.5,Floor_0:0.0
BRIGHTNESS_LABEL = re.compile(
    r"Star_Brightness\s*:\s*(-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)


# --------------------------------------------
# Serial connection and model loading
# --------------------------------------------
@st.cache_resource
def open_serial(port, baud):
    connection = serial.Serial(
        port=port,
        baudrate=baud,
        timeout=0,
        xonxoff=False,
    )
    time.sleep(1.5)
    connection.reset_input_buffer()
    return connection


def reset_serial_connection(port, baud):
    try:
        connection = open_serial(port, baud)
        if connection.is_open:
            connection.close()
    except Exception:
        pass
    try:
        open_serial.clear()
    except Exception:
        pass


@st.cache_resource
def load_interval_model(path_string, modified_time):
    # modified_time makes Streamlit reload if the model file changes.
    return joblib.load(path_string)


def parse_brightness(raw):
    """Parse labeled Serial Plotter output or numeric CSV output."""
    match = BRIGHTNESS_LABEL.search(raw)
    if match:
        try:
            value = float(match.group(1))
            return value if np.isfinite(value) else None
        except ValueError:
            return None

    fields = raw.strip().split(",")
    # CSV firmware format: time_ms,light_mV,brightness_pct
    if len(fields) >= 3:
        try:
            value = float(fields[-1].strip())
            return value if np.isfinite(value) else None
        except ValueError:
            return None

    return None


# --------------------------------------------
# Session state
# --------------------------------------------
defaults = {
    "live_values": [],
    "live_times": [],
    "dip_times": [],
    "intervals": [],
    "in_dip": False,
    "dip_start": None,
    "last_brightness": None,
    "serial_error": None,
    "serial_buffer": "",
    "low_candidate_count": 0,
    "low_candidate_start": None,
    "low_candidate_min_brightness": None,
    "low_candidate_min_time": None,
    "recovery_count": 0,
    "dip_min_brightness": None,
    "dip_min_time": None,
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value.copy() if isinstance(value, list) else value


def reset_detector():
    st.session_state.in_dip = False
    st.session_state.dip_start = None
    st.session_state.low_candidate_count = 0
    st.session_state.low_candidate_start = None
    st.session_state.low_candidate_min_brightness = None
    st.session_state.low_candidate_min_time = None
    st.session_state.recovery_count = 0
    st.session_state.dip_min_brightness = None
    st.session_state.dip_min_time = None


# --------------------------------------------
# Sidebar controls
# --------------------------------------------
st.sidebar.header("Live connection")
available_ports = [p.device for p in serial.tools.list_ports.comports()]
port_options = list(dict.fromkeys(available_ports + ["COM4"]))
preferred_index = port_options.index("COM4") if "COM4" in port_options else 0
selected_port = st.sidebar.selectbox(
    "ESP32 serial port", port_options, index=preferred_index
)
st.sidebar.write(f"Baud rate: {BAUD_RATE}")

threshold = st.sidebar.slider(
    "Dip detection threshold (%)",
    min_value=50.0,
    max_value=100.0,
    value=95.0,
    step=1.0,
)
st.sidebar.caption(
    "Normal brightness should be above the threshold and transit brightness "
    "below it. A dip starts after 3 consecutive low readings."
)

if st.sidebar.button("Clear live history"):
    for key in ("live_values", "live_times", "dip_times", "intervals"):
        st.session_state[key] = []
    st.session_state.serial_buffer = ""
    st.session_state.last_brightness = None
    reset_detector()
    st.rerun()

if st.sidebar.button("Disconnect / reset connection"):
    reset_serial_connection(selected_port, BAUD_RATE)
    st.session_state.serial_buffer = ""
    st.rerun()


# --------------------------------------------
# Read serial data and detect dips
# --------------------------------------------
try:
    ser = open_serial(selected_port, BAUD_RATE)
    st.session_state.serial_error = None

    if ser.in_waiting:
        chunk = ser.read(ser.in_waiting).decode("utf-8", errors="ignore")
        st.session_state.serial_buffer += chunk

    lines = st.session_state.serial_buffer.split("\n")
    st.session_state.serial_buffer = lines.pop()

    for raw in lines:
        raw = raw.strip()
        if not raw:
            continue

        brightness = parse_brightness(raw)
        if brightness is None:
            continue

        now = time.monotonic()
        st.session_state.live_times.append(now)
        st.session_state.live_values.append(brightness)
        st.session_state.last_brightness = brightness

        # Start a dip only after 3 consecutive readings below threshold.
        if not st.session_state.in_dip:
            if brightness < threshold:
                if st.session_state.low_candidate_count == 0:
                    st.session_state.low_candidate_start = now
                    st.session_state.low_candidate_min_brightness = brightness
                    st.session_state.low_candidate_min_time = now

                st.session_state.low_candidate_count += 1

                if (
                    st.session_state.low_candidate_min_brightness is None
                    or brightness < st.session_state.low_candidate_min_brightness
                ):
                    st.session_state.low_candidate_min_brightness = brightness
                    st.session_state.low_candidate_min_time = now

                if st.session_state.low_candidate_count >= 3:
                    st.session_state.in_dip = True
                    st.session_state.dip_start = st.session_state.low_candidate_start
                    st.session_state.dip_min_brightness = st.session_state.low_candidate_min_brightness
                    st.session_state.dip_min_time = st.session_state.low_candidate_min_time
                    st.session_state.recovery_count = 0
            else:
                st.session_state.low_candidate_count = 0
                st.session_state.low_candidate_start = None
                st.session_state.low_candidate_min_brightness = None
                st.session_state.low_candidate_min_time = None

        # While in a dip, track its minimum and wait for recovery.
        else:
            if (
                st.session_state.dip_min_brightness is None
                or brightness < st.session_state.dip_min_brightness
            ):
                st.session_state.dip_min_brightness = brightness
                st.session_state.dip_min_time = now

            recovery_threshold = min(threshold + 1.0, 100.0)
            if brightness >= recovery_threshold:
                st.session_state.recovery_count += 1
            else:
                st.session_state.recovery_count = 0

            # Complete a dip after 3 consecutive recovered readings.
            if st.session_state.recovery_count >= 3:
                dip_time = st.session_state.dip_min_time
                if dip_time is not None:
                    previous_dips = st.session_state.dip_times
                    if not previous_dips or dip_time > previous_dips[-1]:
                        previous_dips.append(dip_time)
                        if len(previous_dips) >= 2:
                            interval = previous_dips[-1] - previous_dips[-2]
                            if 1.0 < interval < 30.0:
                                st.session_state.intervals.append(float(interval))
                reset_detector()

except Exception as exc:
    st.session_state.serial_error = str(exc)

# Keep a bounded display history; preserve dip and interval history.
st.session_state.live_times = st.session_state.live_times[-MAX_POINTS:]
st.session_state.live_values = st.session_state.live_values[-MAX_POINTS:]


# --------------------------------------------
# Connection status and metrics
# --------------------------------------------
if st.session_state.serial_error:
    st.error(f"Cannot read {selected_port}: {st.session_state.serial_error}")
    st.info(
        "Check the USB cable and COM port. Close Arduino Serial Monitor and "
        "Serial Plotter before connecting to the dashboard."
    )
else:
    st.success(f"Connected to ESP32 on {selected_port}")

last_value = st.session_state.last_brightness
intervals = st.session_state.intervals
dip_times = st.session_state.dip_times
c1, c2, c3, c4 = st.columns(4)
c1.metric("Latest brightness", f"{last_value:.1f}%" if last_value is not None else "Waiting...")
c2.metric("Detected dips", len(dip_times))
c3.metric("Last dip interval", f"{intervals[-1]:.3f} s" if intervals else "Waiting...")
c4.metric(
    "Median interval",
    f"{np.median(intervals[-12:]):.3f} s" if intervals else "Waiting...",
)


# --------------------------------------------
# Live light curve - native Streamlit chart
# --------------------------------------------
st.subheader("Live light curve")
if st.session_state.live_values and st.session_state.live_times:
    relative_times = (
        np.asarray(st.session_state.live_times, dtype=float)
        - float(st.session_state.live_times[0])
    )
    chart_df = pd.DataFrame({
        "Elapsed time (s)": relative_times,
        "Brightness (%)": np.asarray(st.session_state.live_values, dtype=float),
    })
    st.line_chart(
        chart_df,
        x="Elapsed time (s)",
        y="Brightness (%)",
        height=350,
    )
    st.caption(f"Showing {len(chart_df)} recent sensor readings.")
else:
    st.info("Waiting for brightness readings from the ESP32...")


# --------------------------------------------
# Next-dip prediction
# --------------------------------------------
st.subheader("Next-dip prediction")
if len(intervals) < 3:
    st.info(
        "Need 3 valid completed intervals for the model. That usually means "
        "at least 4 completed dips. "
        f"Currently available: {len(intervals)} interval(s) from {len(dip_times)} completed dip(s)."
    )
else:
    recent = np.asarray(intervals[-3:], dtype=float).reshape(1, -1)
    st.write(
        "Previous three measured intervals (s):",
        ", ".join(f"{value:.3f}" for value in recent[0]),
    )

    if not MODEL_PATH.exists():
        st.error(f"Prediction model not found: {MODEL_PATH.resolve()}")
    else:
        try:
            model = load_interval_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
            expected_features = getattr(model, "n_features_in_", None)
            if expected_features is not None and expected_features != recent.shape[1]:
                st.error(
                    f"The saved model expects {expected_features} input feature(s), "
                    f"but the dashboard is supplying {recent.shape[1]}. "
                    "The model file was not changed. We need to match the input format "
                    "used when it was trained."
                )
            else:
                prediction = float(model.predict(recent)[0])
                if not np.isfinite(prediction) or not (1.0 <= prediction <= 30.0):
                    st.warning(
                        f"Model returned {prediction:.3f} s, outside the expected 1–30 s range. "
                        "Check the experimental intervals before relying on it."
                    )
                else:
                    st.metric("AI predicted next interval", f"{prediction:.3f} s")
                    if dip_times:
                        elapsed = time.monotonic() - dip_times[-1]
                        remaining = prediction - elapsed
                        st.metric(
                            "Estimated time until next dip",
                            f"{max(0.0, remaining):.2f} s",
                        )
                        if remaining < -0.25:
                            st.caption(
                                "The predicted time has passed. Compare the next detected dip "
                                "with this prediction."
                            )
        except Exception as exc:
            st.error(f"Could not load or use the saved prediction model: {exc}")

    st.caption("The model predicts an interval, not planet size or habitability.")


# --------------------------------------------
# Recent readings and intervals
# --------------------------------------------
with st.expander("Show recent sensor readings"):
    if st.session_state.live_values:
        recent_df = pd.DataFrame({
            "Elapsed time (s)": np.round(
                np.asarray(st.session_state.live_times)
                - st.session_state.live_times[0],
                3,
            ),
            "Brightness (%)": st.session_state.live_values,
        })
        st.dataframe(recent_df.tail(30), use_container_width=True)
    else:
        st.write("No readings received yet.")

with st.expander("Show detected dip intervals"):
    if intervals:
        interval_df = pd.DataFrame({
            "Interval number": range(1, len(intervals) + 1),
            "Interval (s)": np.round(intervals, 4),
        })
        st.dataframe(interval_df.tail(50), use_container_width=True)
    else:
        st.write("No complete dip intervals detected yet.")

# Refresh the app frequently so new serial readings are processed.
st_autorefresh(interval=500, key="live_refresh")
