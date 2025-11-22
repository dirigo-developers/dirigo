from dirigo.main import Dirigo
from dirigo.plugins.acquisitions import FrameAcquisition
from dirigo.plugins.calibrations import StageTranslationCalibration


# User-adjustable parameters
N_STEPS = 5


# Calibration script
diri = Dirigo()

# Use default FrameAcquisition spec as basis for frame size, pixel size, etc
frame_spec = FrameAcquisition.get_specification()
#frame_spec.bidirectional_scanning = False

spec = StageTranslationCalibration.Spec(
    translation=frame_spec.line_width/3,
    n_steps=N_STEPS,
    **frame_spec.to_dict()
)

acquisition = diri.make("acquisition", "stage_translation_calibration", spec=spec)
processor   = diri.make("processor", "raster_frame", upstream=acquisition)
writer      = diri.make("writer", "stage_scanner_angle_calibration", upstream=processor)

acquisition.start()
writer.join()

