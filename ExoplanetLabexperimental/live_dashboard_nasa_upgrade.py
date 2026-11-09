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
# AI POWERED EXOPLANET HUNTER - LIVE DASHBOARD
# --------------------------------------------
st.set_page_config(
    page_title="AI Powered Exoplanet Hunter - Live",
    page_icon="🔭",
    layout="wide",
)

st.markdown("""
<style>
.stApp { background: radial-gradient(ellipse at top, #14243d 0%, #08111f 48%, #050a13 100%); color: #e8f1ff; }
[data-testid="stMetric"] { background: rgba(22, 42, 67, .72); border: 1px solid rgba(103, 173, 230, .25); padding: 14px 16px; border-radius: 12px; }
[data-testid="stMetricLabel"] { color: #a9c8e8 !important; }
[data-testid="stMetricValue"] { color: #f4f9ff !important; }
section[data-testid="stSidebar"] { background: #0a1423; }
hr { border-color: rgba(103, 173, 230, .22); }
</style>
""", unsafe_allow_html=True)
st.markdown("# 🛰️ AI Powered Exoplanet Hunter")
st.markdown("### TRANSIT OBSERVATORY · ESP32-S3 / LDR")
st.caption(
    "Live simulated transit monitoring · Event analysis · Interval prediction. "
    "All derived planet values are experimental estimates, not confirmed astronomical measurements."
)

MODEL_PATH = Path("models/transit_interval_model_final.joblib")
BAUD_RATE = 115200
MAX_POINTS = 1800

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
    "run_start_time": None,
    "serial_error": None,
    "serial_buffer": "",
    "low_candidate_count": 0,
    "low_candidate_start": None,
    "low_candidate_min_brightness": None,
    "low_candidate_min_time": None,
    "recovery_count": 0,
    "dip_min_brightness": None,
    "dip_min_time": None,
    "dip_events": [],
    "recovery_start_time": None,
    "recovery_start_brightness": None,
    "pending_prediction": None,
    "prediction_history": [],
    "last_prediction_interval_count": 0,
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
    st.session_state.recovery_start_time = None
    st.session_state.recovery_start_brightness = None


# --------------------------------------------
# Sidebar controls
# --------------------------------------------
st.sidebar.header("Mission controls")
st.sidebar.caption("NASA-inspired display · LED/LDR transit simulation")
st.sidebar.divider()
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
    for key in ("live_values", "live_times", "dip_times", "intervals", "dip_events", "prediction_history"):
        st.session_state[key] = []
    st.session_state.pending_prediction = None
    st.session_state.last_prediction_interval_count = 0
    st.session_state.serial_buffer = ""
    st.session_state.last_brightness = None
    st.session_state.run_start_time = None
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
        if st.session_state.run_start_time is None:
            st.session_state.run_start_time = now
        st.session_state.live_times.append(now)
        st.session_state.live_values.append(brightness)
        st.session_state.last_brightness = brightness

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

        else:
            if (
                st.session_state.dip_min_brightness is None
                or brightness < st.session_state.dip_min_brightness
            ):
                st.session_state.dip_min_brightness = brightness
                st.session_state.dip_min_time = now

            recovery_threshold = min(threshold + 1.0, 100.0)
            if brightness >= recovery_threshold:
                if st.session_state.recovery_count == 0:
                    st.session_state.recovery_start_time = now
                    st.session_state.recovery_start_brightness = brightness
                st.session_state.recovery_count += 1
            else:
                st.session_state.recovery_count = 0
                st.session_state.recovery_start_time = None
                st.session_state.recovery_start_brightness = None

            if st.session_state.recovery_count >= 3:
                dip_time = st.session_state.dip_min_time
                dip_start = st.session_state.dip_start
                dip_min = st.session_state.dip_min_brightness
                recovery_time = st.session_state.recovery_start_time or now
                if dip_time is not None and dip_min is not None:
                    previous_dips = st.session_state.dip_times
                    if not previous_dips or dip_time > previous_dips[-1]:
                        previous_dips.append(dip_time)
                        recent_baseline = [v for v in st.session_state.live_values[-300:] if v >= threshold]
                        baseline = float(np.median(recent_baseline)) if recent_baseline else float(threshold)
                        baseline = max(baseline, 1e-9)
                        depth_fraction = max(0.0, min(1.0, (baseline - float(dip_min)) / baseline))
                        depth_percent = depth_fraction * 100.0
                        radius_ratio = float(np.sqrt(depth_fraction))
                        st.session_state.dip_events.append({
                            "Event": len(st.session_state.dip_events) + 1,
                            "Dip minimum time (s)": round(float(dip_time - (st.session_state.run_start_time or dip_time)), 3),
                            "Duration (s)": round(max(0.0, float(recovery_time - (dip_start if dip_start is not None else dip_time))), 3),
                            "Baseline (%)": round(baseline, 3),
                            "Minimum (%)": round(float(dip_min), 3),
                            "Depth (%)": round(depth_percent, 3),
                            "Radius ratio Rp/Rs": round(radius_ratio, 4),
                        })
                        if len(previous_dips) >= 2:
                            interval = previous_dips[-1] - previous_dips[-2]
                            if 1.0 < interval < 30.0:
                                st.session_state.intervals.append(float(interval))
                reset_detector()

except Exception as exc:
    st.session_state.serial_error = str(exc)

st.session_state.live_times = st.session_state.live_times[-MAX_POINTS:]
st.session_state.live_values = st.session_state.live_values[-MAX_POINTS:]


# --------------------------------------------
# Mission-control status and metrics
# --------------------------------------------
if st.session_state.serial_error:
    st.error(f"Cannot read {selected_port}: {st.session_state.serial_error}")
    st.info("Check the USB cable and COM port. Close Arduino Serial Monitor and Serial Plotter before connecting.")
else:
    st.success(f"Telemetry link active · {selected_port} · {BAUD_RATE} baud")

last_value = st.session_state.last_brightness
intervals = st.session_state.intervals
dip_times = st.session_state.dip_times
events = st.session_state.dip_events
recent_baselines = [v for v in st.session_state.live_values[-300:] if v >= threshold]
current_baseline = float(np.median(recent_baselines)) if recent_baselines else None
latest_event = events[-1] if events else None
current_depth = latest_event.get("Depth (%)") if latest_event else None
current_ratio = latest_event.get("Radius ratio Rp/Rs") if latest_event else None

m1, m2, m3, m4 = st.columns(4)
m1.metric("STAR SIGNAL", f"{last_value:.2f}%" if last_value is not None else "Awaiting data")
m2.metric("TRANSIT EVENTS", str(len(dip_times)))
m3.metric("LATEST DEPTH", f"{current_depth:.3f}%" if current_depth is not None else "Awaiting dip")
m4.metric("RADIUS RATIO · Rp/Rs", f"{current_ratio:.4f}" if current_ratio is not None else "Awaiting dip")

m5, m6, m7, m8 = st.columns(4)
m5.metric("BASELINE ESTIMATE", f"{current_baseline:.2f}%" if current_baseline is not None else "Calibrating")
m6.metric("LAST INTERVAL", f"{intervals[-1]:.3f} s" if intervals else "Awaiting 2 dips")
m7.metric("MEDIAN INTERVAL", f"{np.median(intervals[-12:]):.3f} s" if intervals else "Awaiting data")
m8.metric("MODEL INPUTS", f"{min(len(intervals), 3)}/3 intervals")

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
# Transit event table
# --------------------------------------------
st.subheader("🪐 Transit event log")
if events:
    event_df = pd.DataFrame(events)
    st.dataframe(event_df.iloc[::-1].head(50), use_container_width=True, hide_index=True)
    st.download_button(
        "Export displayed transit events (CSV)",
        data=event_df.to_csv(index=False).encode("utf-8"),
        file_name="exoplanet_transit_events_export.csv",
        mime="text/csv",
    )
else:
    st.info("No completed transit events yet. The table fills after a dip is detected and brightness recovers.")

with st.expander("How transit depth and radius ratio are estimated"):
    st.markdown(
        "For each completed dip, baseline brightness is estimated as the median of recent readings "
        "at or above the active detection threshold. Transit depth is `100 × (baseline − minimum) / baseline`. "
        "For a simplified, non-grazing transit with a uniformly bright star, `Rp/Rs ≈ sqrt(depth fraction)`. "
        "The LED/LDR rig is a simulation; sensor geometry, ambient light, noise, limb darkening, and incomplete "
        "transits can violate these assumptions. Treat the ratio as an illustrative estimate, not a real planet measurement."
    )

# --------------------------------------------
# ML prediction and measured accuracy
# --------------------------------------------
st.subheader("🤖 ML next-transit prediction")
model = None
prediction = None
model_error = None
if len(intervals) < 3:
    st.info(
        "Need 3 valid completed intervals before attempting a prediction. "
        f"Currently: {len(intervals)} interval(s) from {len(dip_times)} completed dips."
    )
elif not MODEL_PATH.exists():
    model_error = f"Prediction model not found: {MODEL_PATH.resolve()}"
else:
    try:
        model = load_interval_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
        recent = np.asarray(intervals[-3:], dtype=float).reshape(1, -1)
        expected_features = getattr(model, "n_features_in_", None)
        if expected_features is not None and expected_features != recent.shape[1]:
            model_error = (
                f"Saved model expects {expected_features} input feature(s), but the current dashboard supplies 3. "
                "The model was not changed; its original training input format must be matched."
            )
        else:
            prediction = float(model.predict(recent)[0])
            if not np.isfinite(prediction) or not (1.0 <= prediction <= 30.0):
                model_error = f"Model output {prediction:.3f} s is outside the dashboard's expected 1–30 s range."
                prediction = None
    except Exception as exc:
        model_error = f"Could not load or use the saved model: {exc}"

if model_error:
    st.error(model_error)
elif prediction is not None:
    st.metric("PREDICTED NEXT INTERVAL", f"{prediction:.3f} s")
    if dip_times:
        elapsed = time.monotonic() - dip_times[-1]
        remaining = prediction - elapsed
        st.metric("ESTIMATED TIME UNTIL NEXT DIP", f"{max(0.0, remaining):.2f} s")
        if remaining < -0.25:
            st.caption("The predicted time has passed; wait for the next detected event to score this prediction.")

    if len(intervals) > st.session_state.last_prediction_interval_count:
        pending = st.session_state.pending_prediction
        if pending is not None and len(intervals) > pending["interval_count"]:
            actual = float(intervals[-1])
            forecast = float(pending["prediction"])
            error = actual - forecast
            st.session_state.prediction_history.append({
                "Forecast based on intervals": pending["interval_count"],
                "Predicted (s)": round(forecast, 4),
                "Actual (s)": round(actual, 4),
                "Error (actual − predicted) (s)": round(error, 4),
                "Absolute error (s)": round(abs(error), 4),
                "Absolute percentage error (%)": round(abs(error) / actual * 100.0, 2) if actual else np.nan,
            })
        st.session_state.pending_prediction = {
            "interval_count": len(intervals),
            "prediction": prediction,
        }
        st.session_state.last_prediction_interval_count = len(intervals)

if st.session_state.prediction_history:
    st.markdown("#### Forecast performance")
    accuracy_df = pd.DataFrame(st.session_state.prediction_history)
    p1, p2, p3 = st.columns(3)
    p1.metric("Scored forecasts", len(accuracy_df))
    p2.metric("Mean absolute error", f"{accuracy_df['Absolute error (s)'].mean():.3f} s")
    valid_ape = accuracy_df["Absolute percentage error (%)"].dropna()
    p3.metric("Mean absolute % error", f"{valid_ape.mean():.1f}%" if not valid_ape.empty else "N/A")
    st.dataframe(accuracy_df.iloc[::-1].head(25), use_container_width=True, hide_index=True)
else:
    st.info("Prediction accuracy appears after a forecast has been made and a later measured interval becomes available.")

st.caption(
    "The model predicts the next interval from the latest three measured intervals. "
    "Accuracy metrics compare a saved forecast with the next subsequently observed interval; they are experimental and may be sparse."
)

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