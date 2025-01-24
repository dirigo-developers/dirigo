import json
import base64

import tifffile
from platformdirs import user_documents_path
import numpy as np
from matplotlib import pyplot as plt


basename = "experiment"
save_path = user_documents_path() / "Dirigo"

fn = save_path / f"{basename}_{0}.tif"

with tifffile.TiffFile(fn) as tif:
    metadata = json.loads(tif.pages[0].tags['ImageDescription'].value)
    timestamps = np.frombuffer(base64.b64decode(metadata['timestamps']), dtype=np.float64)

instanteous_frequency = 1/np.diff(timestamps)

# 2. Define the Rolling Window Size
window_size = 50

# 3. Create the Rolling Average Kernel
# A simple moving average kernel
kernel = np.ones(window_size) / window_size

# 4. Compute the Rolling Average using convolution
# 'valid' mode returns output of length (N - window_size + 1)
rolling_avg = np.convolve(instanteous_frequency, kernel, mode='valid')

# To align the rolling average with the original data,
# we can pad the beginning with NaNs or handle as desired
# Here, we'll pad with NaNs
pad_size = window_size - 1
rolling_avg_padded = np.concatenate((np.full(pad_size, np.nan), rolling_avg))

fig, ax = plt.subplots()
ax.plot(instanteous_frequency, label='Instantaneous frequency', alpha=0.5)
ax.plot(rolling_avg, label=f'{window_size}-Point Rolling Average', color='red', linewidth=2)
plt.show()
