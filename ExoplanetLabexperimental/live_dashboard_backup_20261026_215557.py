import time
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import serial
import serial.tools.list_ports
import matplotlib.pyplot as plt
from streamlit_autorefresh import st_autorefresh

# --------------------------------------------
# EXOPLANET HUNTER - LIVE DASHBOARD
# Original dashboard and model files untouched
# --------------------------------------------

st.set_page_config(
    page_title="Exoplanet Hunter - Live",
    page_icon="🔭",
    layout="wide"
)

st.title("🔭 Exoplanet Hunter")
st.subheader("Live ESP32 Transit Monitor")

st.caption(
    "Live LDR readings, transit timing and next-dip prediction. "
    "Predictions are experimental and require validation."
)

MODEL_PATH = Path("models/transit_interval_model_final.joblib")
BAUD_RATE = 115200

# Your ESP32 firmware prints Star_Brightness:<value>
BRIGHTNESS_PATTERN = re.compile(
   r"(?:Star_Brightness\s*:\s*|^\s*\d+\s*,\s*\d+(?:\.\d+)?\s*,\s*)(-?\d+(?:\.\d+)?)"
)

# --------------------------------------------
# Persistent serial connection
# --------------------------------------------

@st.cache_resource
def open_serial(port, baud):
    connection = serial.Serial(
        port=port,
        baudrate=baud,
        timeout=0,
        xonxoff=False
    )
    time.sleep(2)
    connection.reset_input_buffer()
    return connection


def close_serial():
    try:
        connection = open_serial.clear()
    except Exception:
        pass


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
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value.copy() if isinstance(value, list) else value

# --------------------------------------------
# Sidebar controls
# --------------------------------------------

st.sidebar.header("Live connection")

available_ports = [
    p.device for p in serial.tools.list_ports.comports()
]

default_index = available_ports.index("COM4") if "COM4" in available_ports else 0

port_options = list(dict.fromkeys(["COM4"] + available_ports))

selected_port = st.sidebar.selectbox(
    "ESP32 serial port",
    port_options,
    index=port_options.index("COM4")
)

st.sidebar.write(f"Baud rate: {BAUD_RATE}")

threshold = st.sidebar.slider(
    "Dip detection threshold (%)",
    min_value=50.0,
    max_value=100.0,
    value=95.0,
    step=1.0
)

st.sidebar.caption(
    "Adjust the threshold if real dips are not detected. "
    "The value should be above the dip brightness and below "
    "the normal baseline."
)

if st.sidebar.button("Clear live history"):
    for key in [
        "live_values", "live_times", "dip_times", "intervals"
    ]:
        st.session_state[key] = []
    st.session_state.in_dip = False
    st.session_state.dip_start = None
    st.session_state.last_brightness = None
    st.rerun()

if st.sidebar.button("Disconnect / reset connection"):
    try:
        open_serial.clear()
    except Exception:
        pass
    st.rerun()

# --------------------------------------------
# Read available serial data
# --------------------------------------------

try:
    ser = open_serial(selected_port, BAUD_RATE)
    st.session_state.serial_error = None

    # Read all complete lines currently available.
    for _ in range(500):
        if ser.in_waiting <= 0:
            break

        raw = ser.readline().decode("utf-8", errors="ignore").strip()
        if not raw:
            continue

        match = BRIGHTNESS_PATTERN.search(raw)
        if not match:
            continue

        brightness = float(match.group(1))
        now = time.monotonic()

        st.session_state.live_times.append(now)
        st.session_state.live_values.append(brightness)
        st.session_state.last_brightness = brightness

        # Detect the beginning of a dip.
        if not st.session_state.in_dip and brightness < threshold:
            st.session_state.in_dip = True
            st.session_state.dip_start = now

        # Hysteresis avoids counting fluctuations around threshold.
        elif st.session_state.in_dip and brightness >= threshold + 1.0:
            start = st.session_state.dip_start

            if start is not None:
                center = (start + now) / 2.0
                st.session_state.dip_times.append(center)

                dip_times = st.session_state.dip_times
                if len(dip_times) >= 2:
                    interval = dip_times[-1] - dip_times[-2]
                    if 1.0 < interval < 30.0:
                        st.session_state.intervals.append(interval)

            st.session_state.in_dip = False
            st.session_state.dip_start = None

except Exception as exc:
    st.session_state.serial_error = str(exc)

# Keep a bounded display history; retain dip history for prediction.
MAX_POINTS = 1800
st.session_state.live_times = st.session_state.live_times[-MAX_POINTS:]
st.session_state.live_values = st.session_state.live_values[-MAX_POINTS:]

# --------------------------------------------
# Connection status and metrics
# --------------------------------------------

if st.session_state.serial_error:
    st.error(
        f"Cannot read {selected_port}: "
        f"{st.session_state.serial_error}"
    )
    st.info(
        "Check the USB cable and COM port. Close Arduino Serial "
        "Monitor and Serial Plotter before connecting."
    )
else:
    st.success(f"Connected to ESP32 on {selected_port}")

last_value = st.session_state.last_brightness
intervals = st.session_state.intervals
dip_times = st.session_state.dip_times

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Latest brightness",
    f"{last_value:.1f}%" if last_value is not None else "Waiting..."
)

c2.metric("Detected dips", len(dip_times))

c3.metric(
    "Last dip interval",
    f"{intervals[-1]:.3f} s" if intervals else "Waiting..."
)

c4.metric(
    "Median interval",
    f"{np.median(intervals[-12:]):.3f} s" if intervals else "Waiting..."
)

# --------------------------------------------
# Live brightness graph
# --------------------------------------------

st.subheader("Live light curve")

if st.session_state.live_values:
    relative_times = (
        np.array(st.session_state.live_times)
        - st.session_state.live_times[0]
    )
    brightness = np.array(st.session_state.live_values)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(relative_times, brightness, linewidth=1)
    ax.axhline(
        threshold, linestyle="--",
        label=f"Dip threshold ({threshold:.0f}%)"
    )
    ax.set_xlabel("Time since dashboard started (s)")
    ax.set_ylabel("Brightness (%)")
    ax.set_title("ESP32 LDR readings")
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)
else:
    st.info("Waiting for brightness readings from the ESP32...")

# --------------------------------------------
# Timing analysis and model prediction
# --------------------------------------------

st.subheader("Next-dip prediction")

if len(intervals) >= 3:
    recent = np.array(intervals[-3:], dtype=float)

    st.write(
        "Previous three measured intervals (s):",
        ", ".join(f"{v:.3f}" for v in recent)
    )

    if MODEL_PATH.exists():
        try:
            model = joblib.load(MODEL_PATH)
            predicted_interval = float(
                model.predict(recent.reshape(1, -1))[0]
            )

            if 1.0 <= predicted_interval <= 30.0:
                st.metric(
                    "AI predicted next interval",
                    f"{predicted_interval:.3f} s"
                )

                if dip_times:
                    elapsed_since_last_dip = (
                        time.monotonic() - dip_times[-1]
                    )
                    predicted_remaining = (
                        predicted_interval - elapsed_since_last_dip
                    )

                    st.metric(
                        "Estimated time until next dip",
                        f"{max(0.0, predicted_remaining):.2f} s"
                    )

                    if predicted_remaining < -0.25:
                        st.caption(
                            "The predicted time has passed. "
                            "Wait for the next detected dip and "
                            "compare it with the prediction."
                        )
            else:
                st.warning("Model output was outside the expected range.")

        except Exception as exc:
            st.error(f"Could not load or use the saved model: {exc}")
    else:
        st.warning(
            f"Model not found at {MODEL_PATH}. "
            "Run train_ai_final.py first."
        )

    st.caption(
        "The saved model predicts the next interval, not planet size "
        "or habitability. Predictions depend on the training data."
    )

elif len(intervals) < 3:
    st.info(
        f"Need at least 3 completed dip intervals before prediction. "
        f"Currently available: {len(intervals)}."
    )

# --------------------------------------------
# Recent measurements
# --------------------------------------------

with st.expander("Show recent sensor readings"):
    if st.session_state.live_values:
        recent_df = pd.DataFrame({
            "Elapsed time (s)": np.round(
                np.array(st.session_state.live_times)
                - st.session_state.live_times[0], 3
            ),
            "Brightness (%)": st.session_state.live_values
        })
        st.dataframe(recent_df.tail(30), use_container_width=True)
    else:
        st.write("No readings received yet.")

with st.expander("Show detected dip intervals"):
    if intervals:
        st.dataframe(
            pd.DataFrame({
                "Interval number": range(1, len(intervals) + 1),
                "Interval (s)": np.round(intervals, 4)
            }).tail(50),
            use_container_width=True
        )
    else:
        st.write("No complete dip intervals detected yet.")

# Automatically refresh the page to read new serial data.
st_autorefresh(interval=250, key="live_refresh")

