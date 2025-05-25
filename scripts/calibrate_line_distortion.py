from dirigo.main import Dirigo
from dirigo import io

from dirigo.plugins.acquisitions import FrameAcquisition
from dirigo.plugins.calibrations import LineDistortionCalibration

# Adjustable parameters
TRANSLATION = "30 um"
N_STEPS = 3


# Calibration script
diri = Dirigo()

# Use default FrameAcquisition spec as basis for frame size, pixel size, etc
frame_spec = FrameAcquisition.get_specification()

spec = LineDistortionCalibration.Spec(
    translation=TRANSLATION,
    n_steps=N_STEPS,
    **frame_spec.to_dict()
)

name = "line_distortion_calibration"
acquisition = diri.make("acquisition", name, spec=spec)
processor   = diri.make("processor", "raster_frame", upstream=acquisition)
logger      = diri.make("logger", name, upstream=processor)
raw_logger  = diri.make("logger", "tiff", upstream=acquisition)
raw_logger.basename = name
raw_logger.frames_per_file = float('inf')

acquisition.start()
logger.join()


# Check quality of correction using saved data
loader    = diri.make("loader", "raw_raster_frame", 
                      file_path=io.data_path() / f"{name}.tif",
                      spec_class=LineDistortionCalibration.Spec)
processor = diri.make("processor", "raster_frame", upstream=loader)
logger    = diri.make("logger", name, upstream=processor)

loader.start()
logger.join()

