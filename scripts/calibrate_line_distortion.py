import numpy as np
from numpy.polynomial.polynomial import Polynomial
import matplotlib.pyplot as plt

from dirigo.main import Dirigo
from dirigo import io

from dirigo.plugins.acquisitions import FrameAcquisition
from dirigo.plugins.calibrations import StageTranslationCalibration


# User-adjustable parameters
TRANSLATION = "30 um"
N_STEPS = 3

# Plotting functions
def check_calibration(file_path: str, ff: float):

    data = np.loadtxt(file_path, delimiter=',', dtype=np.float64, skiprows=1)
    ys = data[:,1:]
    xs = np.tile(data[:,[0]], (1, ys.shape[1])) 

    n_samples, n_replicates = ys.shape

    nan_mask = np.isnan(ys)
    pfit: Polynomial = Polynomial.fit(
        x=xs[~nan_mask].ravel(),
        y=ys[~nan_mask].ravel(),
        deg=2
    )
    c0, c1, c2 = pfit.convert().coef
    print(f"global fit:  y = {c2:.6g}·x² + {c1:.6g}·x + {c0:.6g}")

    x_fit = np.linspace(-ff, ff, 1000)
    y_fit = pfit(x_fit)

    print(f"Fit integral: {np.sum(y_fit-1)*(x_fit[1]-x_fit[0])}")

    for idx in range(n_replicates):
        plt.plot(xs[:,idx], ys[:,idx], marker='o', ls='', label=f"replicate {idx}")

    plt.plot(x_fit, y_fit, 'k--', lw=2, label="global fit")
    plt.xlabel("sample index")
    plt.ylabel("value")
    plt.title("Pooled fit across all replicates")
    plt.legend()
    plt.tight_layout()
    plt.grid(True)  
    plt.show()


# Calibration script
diri = Dirigo()

# Use default FrameAcquisition spec as basis for frame size, pixel size, etc
frame_spec = FrameAcquisition.get_specification()

spec = StageTranslationCalibration.Spec(
    translation=TRANSLATION,
    n_steps=N_STEPS,
    **frame_spec.to_dict()
)

name = "line_distortion_calibration"
acquisition = diri.make_acquisition("stage_translation_calibration", spec=spec)
processor   = diri.make_processor("raster_frame", upstream=acquisition)
writer      = diri.make_writer(name, upstream=processor)
# also log raw frame data so we can reprocess later to check calibration
raw_writer  = diri.make("writer", "tiff", upstream=acquisition)
raw_writer.basename = name
raw_writer.frames_per_file = -1

acquisition.start()
writer.join()

# Check initial distortion
check_calibration(writer.data_filepath, ff=spec.fill_fraction)


# Check quality of correction using saved data and reprocessing frames
loader    = diri.make_loader("raw_raster_frame", 
                      file_path=io.data_path() / f"{name}.tif",
                      spec_class=StageTranslationCalibration.Spec)
processor = diri.make_processor("raster_frame", upstream=loader)
writer    = diri.make_writer(name, upstream=processor)

loader.start()
writer.join()

check_calibration(writer.data_filepath, ff=spec.fill_fraction)