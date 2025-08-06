import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def main(csv_path: str | Path) -> None:
    # ── 1. Load the data ─────────────────────────────────────────────────────
    data = np.loadtxt(csv_path, delimiter=",", comments="#")
    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError("CSV must contain exactly two numeric columns.")
    ch0, ch1 = data.T

    # ── 2. Build the x-axis (1 µm per sample, centred at 0 µm) ──────────────
    n = len(ch0)
    x = np.arange(n) - (n - 1) / 2.0        # µm

    # ── 3. Quartic fits: y = a·x⁴ + b·x³ + c·x² + d·x + e ───────────────────
    coeff0 = np.polyfit(x, ch0, deg=4)      # quartic coefficients for Ch 0
    coeff1 = np.polyfit(x, ch1, deg=4)      # quartic coefficients for Ch 1
    fit0 = np.polyval(coeff0, x)
    fit1 = np.polyval(coeff1, x)

    # ── 4. Plot raw data and fits ────────────────────────────────────────────
    plt.figure(figsize=(8, 4))

    # raw traces
    plt.plot(x, ch0, label="Channel 0", linewidth=1.2)
    plt.plot(x, ch1, label="Channel 1", linewidth=1.2)

    # quartic fits (dashed)
    plt.plot(x, fit0, "--", label="4th-order fit Ch 0")
    plt.plot(x, fit1, "--", label="4th-order fit Ch 1")

    plt.xlabel("Position (µm)")
    plt.ylabel("Signal (a.u.)")
    plt.title("Channel signals and 4th-order polynomial fits")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Uncomment to print coefficients if desired
    print("Channel 0 coefficients:", coeff0)
    print("Channel 1 coefficients:", coeff1)


if __name__ == "__main__":
    csv_path = r"C:\dirigo-data\line_gradient_data.csv"
    main(csv_path)
