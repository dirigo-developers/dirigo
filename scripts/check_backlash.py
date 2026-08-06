import tifffile
import numpy as np
from scipy import fft, signal
import matplotlib.pyplot as plt

from dirigo import io


U = 10
zpos_path = io.data_path() / "stack_2.tif"
zneg_path = io.data_path() / "stack_3.tif"


print("reading stack 1")
with tifffile.TiffFile(zpos_path) as tif:
    zpos_stack = np.array(tif.asarray()[...,1])
    ome_xml = tif.ome_metadata

    i = ome_xml.find("PhysicalSizeZ") + len("PhysicalSizeZ= ")
    z_um = float(ome_xml[i:(i+10)].split('"')[0]) * 1e6


print("reading stack 2")
with tifffile.TiffFile(zneg_path) as tif:
    # flip the negative going z stack
    z_negstack = np.array(tif.asarray()[::-1,...,1])
    # z_negstack = np.array(tif.asarray()[...,1])

# Cross correlate
print("cross correlating")
S_p = fft.fft(zpos_stack.astype(np.float32), axis=0)
S_n = fft.fft(z_negstack.astype(np.float32), axis=0)

XC = S_p * np.conj(S_n)
XC /= np.abs(XC) + 1e-6
XC = np.sum(XC, axis=(1, 2))

corr = signal.resample(fft.ifft(XC), U * len(XC))
corr_shifted = fft.fftshift(np.abs(corr))

# Build physical lag axis
N         = len(corr)
lags      = (np.arange(N) - N // 2) / U    
lags_phys = lags * z_um  

# Find peak lag
peak_idx = np.argmax(corr_shifted)
peak_lag = lags_phys[peak_idx]

# Plot
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(lags_phys, corr_shifted, label='Cross-correlation')
ax.axvline(peak_lag, color='r', linestyle='--',
           label=f'Peak: {peak_lag:.3f} um')
ax.axvline(0, color='gray', linestyle=':', linewidth=0.8)

ax.set_xlabel(f'Z lag (um)')
ax.set_ylabel('Normalised cross-correlation magnitude')
ax.set_title('Z-stack phase cross-correlation')
ax.legend()
plt.tight_layout()
plt.show()

