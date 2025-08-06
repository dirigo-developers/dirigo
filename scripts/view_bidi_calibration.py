import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ── configurable inputs ──────────────────────────────────────────────────────
csv_path = Path(r"C:\Users\MIT\AppData\Local\Dirigo\Dirigo\scanner\trigger_delay_calibration.csv")    
amp_label = "Amplitude (rad)"
y1_label = "Frequency (Hz)"
y2_label = "Phase (rad)"

# ── load data ────────────────────────────────────────────────────────────────
# skip_header=1 ignores the header line
data = np.loadtxt(csv_path, delimiter=",", skiprows=1)

amp  = data[:, 0]   # x-axis
freq = data[:, 1]   # left y-axis (Hz)
phase = data[:, 2]  # right y-axis (rad)

# ── create plot ──────────────────────────────────────────────────────────────
fig, ax1 = plt.subplots(figsize=(8, 5))

# left-hand axis
ax1.scatter(amp, freq, color="tab:blue", label=y1_label)
ax1.set_xlabel(amp_label)
ax1.set_ylabel(y1_label, color="tab:blue")
ax1.tick_params(axis="y", labelcolor="tab:blue")

# right-hand axis shares the same x
ax2 = ax1.twinx()
ax2.scatter(amp, phase, color="tab:red", label=y2_label)
ax2.set_ylabel(y2_label, color="tab:red")
ax2.tick_params(axis="y", labelcolor="tab:red")

# optional: a combined legend
lines = ax1.get_lines() + ax2.get_lines()
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc="best")

ax1.grid(True, which="both", ls="--", alpha=0.4)
fig.tight_layout()
plt.show()
