import numpy as np
from numpy.polynomial.polynomial import Polynomial
import matplotlib.pyplot as plt


from dirigo.main import Dirigo
from dirigo.sw_interfaces.logger import save_path
from dirigo.plugins.acquisitions import FrameDistortionCalibration
from dirigo.plugins.loaders import RawRasterFrameLoader
from dirigo.plugins.processors import RasterFrameProcessor
from dirigo.plugins.loggers import FrameDistortionCalibrationLogger



def check_calibration(file_path: str, ff: float) -> tuple:

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



diri = Dirigo()

# Step 1: capture data with incremental (known) stage shifts
acquisition = diri.acquisition_factory(
    type='frame_distortion_calibration', 
    spec_name='frame_distortion_calibration'
)
raw_logger = diri.logger_factory(acquisition)
raw_logger.basename = "distortion_calibration"
raw_logger.frames_per_file = float('inf')

acquisition.start()
acquisition.join(timeout=100)
raw_logger.join(timeout=100)


# Step 2: measure distortion field and fit error
RawRasterFrameLoader.SPEC_OBJECT = FrameDistortionCalibration.SPEC_OBJECT # over-ride spec object
loader = RawRasterFrameLoader(save_path() / "distortion_calibration.tif")

processor = RasterFrameProcessor(loader)
distort_logger = FrameDistortionCalibrationLogger(processor)

loader.add_subscriber(processor)
processor.add_subscriber(distort_logger)
distort_logger.start()
processor.start()

loader.start()
distort_logger.join(timeout=200)

check_calibration(distort_logger._fn, loader.spec.fill_fraction)

