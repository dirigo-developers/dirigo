
import numpy as np
import matplotlib.pyplot as plt

from dirigo import io

"""Check the trigger delay calibration. """


csv_file = io.config_path() / "scanner" / "trigger_delay_calibration.csv"

amplitude, frequency, phase = np.loadtxt(
    csv_file,
    delimiter=",",
    skiprows=1,          # skip the header
    unpack=True,         # returns three 1-D arrays
)


fig, (ax_phase, ax_freq, ax_delay) = plt.subplots(
    3, 1, figsize=(8, 9), sharex=True, layout="constrained"
)

# Amplitude vs. Phase
ax_phase.plot(amplitude, phase, ".", lw=0.8)
ax_phase.set_ylabel("Phase (rad)")
ax_phase.set_title("Trigger-delay calibration: Phase vs. Amplitude")
ax_phase.grid(True)

# Amplitude vs. Frequency
ax_freq.plot(amplitude, frequency, ".", lw=0.8)
ax_freq.set_ylabel("Frequency (Hz)")
ax_freq.set_title("Frequency vs. Amplitude")
ax_freq.grid(True)

# Amplitude vs. Delay (time)
period = 1.0 / frequency           # seconds
delay  = (phase / (2 * np.pi)) * period

ax_delay.plot(amplitude, delay * 1e9, ".", lw=0.8)   # convert to ns for readability
ax_delay.set_xlabel("Scanner amplitude (rad)")
ax_delay.set_ylabel("Delay (ns)")
ax_delay.set_title("Time-delay vs. Amplitude")
ax_delay.grid(True)

plt.show()