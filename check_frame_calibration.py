import csv
import matplotlib.pyplot as plt
import numpy as np

CSV_PATH = r"C:\Users\MIT\AppData\Local\Dirigo\Dirigo\scanner\frame_calibration.csv"          # <-- put your file name here


# ------------------------------------------------------------------
# 1.  Read CSV  (rows → series, no pandas)
# ------------------------------------------------------------------
rows = []
with open(CSV_PATH, newline="") as f:
    rdr = csv.reader(f)
    for row in rdr:
        if not row or row[0].startswith("#"):
            continue                   # skip blanks / comments
        rows.append([float(v) for v in row])

if not rows:
    raise RuntimeError("No data loaded")

n_samples = len(rows[0])              # assumes all rows same length
x_full    = np.arange(n_samples)      # 0 … N‑1

# ------------------------------------------------------------------
# 2.  Stack all replicates into one 1‑D array
# ------------------------------------------------------------------
x_all = np.tile(x_full,  len(rows))   # [0,1,…N‑1, 0,1,…,N‑1, …]
y_all = np.concatenate(rows)          # flatten rows into single vector

# ------------------------------------------------------------------
# 3.  Global quadratic fit  (one curve for all points)
# ------------------------------------------------------------------
coeffs = np.polyfit(x_all, y_all, deg=2)     # a, b, c
a, b, c = coeffs
print(f"global fit:  y = {a:.6g}·x² + {b:.6g}·x + {c:.6g}")

# Evaluate fit on a dense grid for a smooth plot
x_fit = np.linspace(0, n_samples-1, 400)
y_fit = np.polyval(coeffs, x_fit)

# ------------------------------------------------------------------
# 4.  Plot
# ------------------------------------------------------------------
for idx, y in enumerate(rows, start=1):
    plt.plot(x_full, y, marker='o', ls='', label=f"replicate {idx}")

plt.plot(x_fit, y_fit, 'k--', lw=2, label="global quadratic fit")
plt.xlabel("sample index")
plt.ylabel("value")
plt.title("Pooled quadratic fit across all replicates")
plt.legend()
plt.tight_layout()
plt.show()

