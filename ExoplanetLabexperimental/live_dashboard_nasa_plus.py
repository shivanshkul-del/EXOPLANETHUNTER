import re
import time
from datetime import datetime
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
st.markdown("# 🛰️ EXOPLANET HUNTER")
st.markdown("### TRANSIT OBSERVATORY · ESP32-S3 / LDR")
st.caption(
    "Live simulated transit monitoring · Event analysis · Interval prediction. "
    "All derived planet values are experimental estimates, not confirmed astronomical measurements."
)

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "transit_interval_model_final.joblib"
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
    "last_reading_wall_time": None,
    "timeline_events": [],
    "timeline_initialized": False,
    "calibration_seen": False,
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value.copy() if isinstance(value, list) else value


if not st.session_state.timeline_initialized:
    st.session_state.timeline_events.append({
        "Time": datetime.now().strftime("%H:%M:%S"),
        "Elapsed (s)": 0.0,
        "Stage": "STARTUP",
        "Details": "Dashboard session initialized",
    })
    st.session_state.timeline_initialized = True


def add_timeline_event(stage, details, elapsed=None):
    if elapsed is None:
        start = st.session_state.run_start_time
        elapsed = max(0.0, time.monotonic() - start) if start is not None else 0.0
    st.session_state.timeline_events.append({
        "Time": datetime.now().strftime("%H:%M:%S"),
        "Elapsed (s)": round(float(elapsed), 3),
        "Stage": str(stage),
        "Details": str(details),
    })
    st.session_state.timeline_events = st.session_state.timeline_events[-250:]


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
history_window = st.sidebar.slider(
    "Light-curve display window (readings)",
    min_value=100, max_value=1800, value=900, step=100,
)
stellar_luminosity = st.sidebar.number_input(
    "Illustrative star luminosity (Sun = 1)",
    min_value=0.01, max_value=100.0, value=1.0, step=0.1,
    help="Assumption for a theoretical illustration only; the LED/LDR does not measure stellar luminosity.",
)

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
    for key in ("live_values", "live_times", "dip_times", "intervals", "dip_events", "prediction_history", "timeline_events"):
        st.session_state[key] = []
    st.session_state.pending_prediction = None
    st.session_state.last_prediction_interval_count = 0
    st.session_state.serial_buffer = ""
    st.session_state.timeline_events = []
    st.session_state.timeline_initialized = False
    st.session_state.calibration_seen = False
    st.session_state.last_reading_wall_time = None
    st.session_state.last_brightness = None
    st.session_state.run_start_time = None
    st.session_state.last_reading_wall_time = None
    st.session_state.timeline_initialized = False
    st.session_state.calibration_seen = False
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
        st.session_state.last_reading_wall_time = datetime.now().strftime("%H:%M:%S")
        if not st.session_state.calibration_seen:
            add_timeline_event("CALIBRATION", "First valid brightness reading received", elapsed=0.0)
            st.session_state.calibration_seen = True

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
                    add_timeline_event(
                        "CANDIDATE DIP",
                        f"3 consecutive readings below {threshold:.1f}% threshold",
                        elapsed=now - (st.session_state.run_start_time or now),
                    )
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
                if st.session_state.recovery_count == 0:
                    st.session_state.recovery_start_time = now
                    st.session_state.recovery_start_brightness = brightness
                st.session_state.recovery_count += 1
            else:
                st.session_state.recovery_count = 0
                st.session_state.recovery_start_time = None
                st.session_state.recovery_start_brightness = None

            # Complete a dip after 3 consecutive recovered readings.
            if st.session_state.recovery_count >= 3:
                dip_time = st.session_state.dip_min_time
                dip_start = st.session_state.dip_start
                dip_min = st.session_state.dip_min_brightness
                recovery_time = st.session_state.recovery_start_time or now
                if dip_time is not None and dip_min is not None:
                    previous_dips = st.session_state.dip_times
                    if not previous_dips or dip_time > previous_dips[-1]:
                        previous_dips.append(dip_time)
                        # Estimate baseline from recent readings above the active threshold.
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
                        add_timeline_event(
                            "TRANSIT DETECTED",
                            f"Event {len(st.session_state.dip_events)} completed; depth {depth_percent:.3f}%",
                            elapsed=float(dip_time - (st.session_state.run_start_time or dip_time)),
                        )
                        if len(previous_dips) >= 2:
                            interval = previous_dips[-1] - previous_dips[-2]
                            if 1.0 < interval < 30.0:
                                st.session_state.intervals.append(float(interval))
                                add_timeline_event(
                                    "INTERVAL MEASURED",
                                    f"Interval {len(st.session_state.intervals)} = {interval:.3f} s",
                                    elapsed=float(dip_time - (st.session_state.run_start_time or dip_time)),
                                )
                reset_detector()

except Exception as exc:
    st.session_state.serial_error = str(exc)

# Keep a bounded display history; preserve dip and interval history.
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
st.caption(
    f"Last valid reading: {st.session_state.last_reading_wall_time or 'not received yet'} · "
    f"Current-session samples: {len(st.session_state.live_values)}"
)

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
    values = np.asarray(st.session_state.live_values[-history_window:], dtype=float)
    times = np.asarray(st.session_state.live_times[-history_window:], dtype=float)
    relative_times = times - float(st.session_state.live_times[0])
    baseline_samples = [v for v in values if v >= threshold]
    chart_baseline = float(np.median(baseline_samples)) if baseline_samples else float(threshold)
    chart_df = pd.DataFrame({
        "Elapsed time (s)": relative_times,
        "Brightness (%)": values,
        "Baseline estimate (%)": np.full(len(values), chart_baseline),
        "Transit threshold (%)": np.full(len(values), threshold),
    })
    st.line_chart(
        chart_df,
        x="Elapsed time (s)",
        y=["Brightness (%)", "Baseline estimate (%)", "Transit threshold (%)"],
        height=390,
    )
    st.caption(
        f"Showing {len(chart_df)} readings. Change the history window in the sidebar. "
        "Native Streamlit charts provide basic interaction; precise mouse-wheel zoom depends on your Streamlit version."
    )
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
# Orbital period, habitability illustration & candidate score
# --------------------------------------------
st.subheader("🌍 Orbital & habitability indicators")
orbital_col, habitat_col, confidence_col = st.columns(3)

if intervals:
    median_interval = float(np.median(intervals[-12:]))
    orbital_col.metric("REPEATING INTERVAL", f"{median_interval:.3f} s")
    orbital_col.caption(
        "Observed interval between detected dips in this bench simulation. "
        "It is only an orbital-period proxy if the simulation timing is calibrated to a real-world scale."
    )
else:
    orbital_col.metric("REPEATING INTERVAL", "Awaiting repeats")
    orbital_col.caption("At least two completed dips are needed to calculate an interval.")

inner_au = 0.95 * float(np.sqrt(stellar_luminosity))
outer_au = 1.67 * float(np.sqrt(stellar_luminosity))
habitat_col.metric("ILLUSTRATIVE HZ BAND", f"{inner_au:.2f}–{outer_au:.2f} AU")
habitat_col.caption(
    f"Rough solar-scaled illustration for assumed luminosity {stellar_luminosity:.2f} L☉. "
    "The LED/LDR cannot measure stellar luminosity or planet distance; this is not a habitability determination."
)

if events:
    last_event = events[-1]
    depth_score = min(1.0, max(0.0, float(last_event.get("Depth (%)", 0.0)) / 10.0))
    duration = max(0.0, float(last_event.get("Duration (s)", 0.0)))
    duration_score = min(1.0, duration / 1.0)
    repeat_score = 0.0
    if len(intervals) >= 2:
        recent_intervals = np.asarray(intervals[-5:], dtype=float)
        mean_interval = float(np.mean(recent_intervals))
        variation = float(np.std(recent_intervals) / mean_interval) if mean_interval > 0 else 1.0
        repeat_score = max(0.0, min(1.0, 1.0 - variation))
    candidate_score = 100.0 * (
        0.45 * depth_score + 0.20 * duration_score + 0.35 * repeat_score
    )
    confidence_col.metric("CANDIDATE REVIEW SCORE", f"{candidate_score:.0f}/100")
    confidence_col.caption(
        "Heuristic based on depth, duration and repeatability—not a calibrated probability or confirmation of a planet."
    )
else:
    confidence_col.metric("CANDIDATE REVIEW SCORE", "Awaiting event")
    confidence_col.caption("Available after a completed dip.")

# --------------------------------------------
# NASA-style mission timeline
# --------------------------------------------
st.subheader("🧭 Mission timeline")
if st.session_state.timeline_events:
    timeline_df = pd.DataFrame(st.session_state.timeline_events)
    st.dataframe(
        timeline_df.iloc[::-1].head(40),
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("Timeline entries appear as telemetry and candidate events arrive.")

# --------------------------------------------
# Read-only report export & saved-run comparison
# --------------------------------------------
st.subheader("📄 Download current-session report")
report_lines = [
    "# Exoplanet Hunter — Current Session Report",
    "",
    f"- Generated: {datetime.now().astimezone().isoformat(timespec='seconds')}",
    f"- Selected serial port: {selected_port}",
    f"- Baud rate: {BAUD_RATE}",
    f"- Connection status: {'ERROR: ' + str(st.session_state.serial_error) if st.session_state.serial_error else 'No current serial exception'}",
    f"- Last valid reading: {st.session_state.last_reading_wall_time or 'not received'}",
    f"- Sensor readings retained in memory: {len(st.session_state.live_values)}",
    f"- Completed dip/transit events: {len(events)}",
    f"- Measured intervals: {len(intervals)}",
    f"- Active dip threshold: {threshold:.2f}%",
    f"- Latest brightness: {last_value if last_value is not None else 'not available'}",
    f"- Median recent interval: {float(np.median(intervals[-12:])) if intervals else 'not available'}",
    f"- Assumed luminosity for habitability illustration: {stellar_luminosity:.2f} solar luminosities",
    "",
    "## Scientific cautions",
    "- This is an LED/LDR transit simulation, not a confirmed astronomical observation.",
    "- The habitability-zone band is a rough illustration based on user-assumed luminosity.",
    "- The candidate review score is heuristic, not a probability.",
    "- Report generation does not modify any model or source CSV file.",
    "",
    "## Transit event data (CSV format)",
]
if events:
    report_lines.append(pd.DataFrame(events).to_csv(index=False))
else:
    report_lines.append("No completed events in this session.")
if intervals:
    report_lines += ["", "## Measured intervals (seconds)", ", ".join(f"{x:.4f}" for x in intervals)]
report_data = "\n".join(report_lines)
st.download_button(
    "Download session report (Markdown)",
    data=report_data.encode("utf-8"),
    file_name=f"exoplanet_session_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
    mime="text/markdown",
)

st.subheader("🗂️ Compare saved experiment runs")
saved_csvs = sorted(
    path for path in PROJECT_DIR.glob("*.csv")
    if path.is_file() and not path.name.lower().startswith("exoplanet_transit_events_export")
)
if saved_csvs:
    selected_csvs = st.multiselect(
        "Select existing run CSVs (read-only)",
        options=[path.name for path in saved_csvs],
        default=[path.name for path in saved_csvs[:2]],
    )
    comparison_rows = []
    for filename in selected_csvs:
        csv_path = PROJECT_DIR / filename
        try:
            frame = pd.read_csv(csv_path)
            numeric_columns = frame.select_dtypes(include=[np.number]).columns.tolist()
            row = {
                "File": filename,
                "Rows": len(frame),
                "Columns": len(frame.columns),
                "Numeric fields": ", ".join(numeric_columns[:6]) if numeric_columns else "none",
            }
            for col in numeric_columns[:4]:
                numbers = pd.to_numeric(frame[col], errors="coerce").dropna()
                row[f"{col} mean"] = round(float(numbers.mean()), 4) if not numbers.empty else np.nan
                row[f"{col} min"] = round(float(numbers.min()), 4) if not numbers.empty else np.nan
            comparison_rows.append(row)
        except Exception as exc:
            comparison_rows.append({"File": filename, "Read error": str(exc)})
    if comparison_rows:
        st.dataframe(pd.DataFrame(comparison_rows), use_container_width=True, hide_index=True)
    st.caption("Read-only comparison: source CSV files are never rewritten.")
else:
    st.info("No top-level experiment CSVs found beside this dashboard. Keep your run CSVs in the project folder to compare them.")

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

    # Record one forecast per interval-count; score it only after a subsequent interval arrives.
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

with st.expander("Model status & diagnostics"):
    st.write(f"**Model path:** `{MODEL_PATH}`")
    st.write(f"**Model file exists:** {'Yes' if MODEL_PATH.exists() else 'No'}")
    if MODEL_PATH.exists():
        st.write(f"**Model size:** {MODEL_PATH.stat().st_size:,} bytes")
        st.write(
            "**Last modified:** "
            + datetime.fromtimestamp(MODEL_PATH.stat().st_mtime).astimezone().isoformat(timespec="seconds")
        )
    if model is not None:
        st.success("Saved model loaded successfully for prediction.")
        feature_count = getattr(model, "n_features_in_", None)
        st.write(f"**Expected input features:** {feature_count if feature_count is not None else 'Not exposed by this estimator'}")
        feature_names = getattr(model, "feature_names_in_", None)
        if feature_names is not None:
            st.write("**Expected feature names:** " + ", ".join(map(str, feature_names)))
        st.write(f"**Prediction readiness:** {'Ready' if prediction is not None else 'Model loaded, but prediction unavailable'}")
    elif model_error:
        st.warning(model_error)
    else:
        st.info(
            f"Prediction is waiting for 3 valid intervals. Current count: {len(intervals)}. "
            "The saved model file is not changed."
        )

if st.session_state.prediction_history:
    st.markdown("#### Forecast performance")
    accuracy_df = pd.DataFrame(st.session_state.prediction_history)
    p1, p2, p3 = st.columns(3)
    p1.metric("Scored forecasts", len(accuracy_df))
    p2.metric("Mean absolute error", f"{accuracy_df['Absolute error (s)'].mean():.3f} s")
    valid_ape = accuracy_df["Absolute percentage error (%)"].dropna()
    p3.metric("Mean absolute % error", f"{valid_ape.mean():.1f}%" if not valid_ape.empty else "N/A")
    st.dataframe(accuracy_df.iloc[::-1].head(25), use_container_width=True, hide_index=True)
    st.markdown("#### Predicted vs actual interval")
    comparison_plot = accuracy_df[
        ["Forecast based on intervals", "Predicted (s)", "Actual (s)"]
    ].copy()
    st.line_chart(
        comparison_plot.set_index("Forecast based on intervals")[["Predicted (s)", "Actual (s)"]],
        height=280,
    )
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
