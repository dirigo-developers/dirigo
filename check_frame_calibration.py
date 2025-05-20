import matplotlib.pyplot as plt
import numpy as np

CSV_PATH = r"C:\Users\MIT\AppData\Local\Dirigo\Dirigo\scanner\distortion_calibration.csv"
ff = 0.9

data = np.loadtxt(CSV_PATH, delimiter=',', dtype=np.float64)
ys = data[:,1:]
xs = np.tile(data[:,[0]], (1, ys.shape[1])) * ff/.5

n_samples, n_replicates = ys.shape

nan_mask = np.isnan(ys)
coeffs = np.polyfit(xs[~nan_mask].ravel(), ys[~nan_mask].ravel(), deg=2) 
a, b, c = coeffs
print(f"global fit:  y = {a:.6g}·x² + {b:.6g}·x + {c:.6g}")

# Evaluate fit on a dense grid for a smooth plot
x_fit = np.linspace(-ff, ff, 1000)
y_fit = np.polyval(coeffs, x_fit)

# Compute integral of fit 
print(
    f"Fit integral: {np.sum(y_fit-1)*(x_fit[1]-x_fit[0])}"
)

# ------------------------------------------------------------------
# 4.  Plot
# ------------------------------------------------------------------
for idx in range(n_replicates):
    plt.plot(xs[:,idx], ys[:,idx], marker='o', ls='', label=f"replicate {idx}")

plt.plot(x_fit, y_fit, 'k--', lw=2, label="global quadratic fit")
plt.xlabel("sample index")
plt.ylabel("value")
plt.title("Pooled quadratic fit across all replicates")
plt.legend()
plt.tight_layout()
plt.grid(True)  
plt.show()
