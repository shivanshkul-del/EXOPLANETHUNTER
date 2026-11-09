import csv
import time
from pathlib import Path
from datetime import datetime
import serial

PORT = "COM10"
BAUD_RATE = 115200
DURATION_SECONDS = 60

output_dir = Path("data")
output_dir.mkdir(exist_ok=True)
filename = output_dir / f"run_{datetime.now():%Y%m%d_%H%M%S}.csv"

print(f"Connecting to {PORT}...", flush=True)
rows = []

with serial.Serial(PORT, BAUD_RATE, timeout=2) as esp:
    time.sleep(2)
    esp.reset_input_buffer()
    start = time.monotonic()
    print(f"Recording for {DURATION_SECONDS} seconds...", flush=True)

    while time.monotonic() - start < DURATION_SECONDS:
        line = esp.readline().decode("utf-8", errors="ignore").strip()

        if not line or line.startswith("time_ms"):
            continue

        parts = line.split(",")
        if len(parts) != 3:
            continue

        try:
            rows.append([
                int(parts[0]),
                float(parts[1]),
                float(parts[2])
            ])
        except ValueError:
            continue

with filename.open("w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["time_ms", "light_mV", "brightness_pct"])
    writer.writerows(rows)

print(f"Saved {len(rows)} readings.", flush=True)
print(f"File: {filename}", flush=True)
