from dirigo.main import Dirigo

from dirigo.plugins.acquisitions import FrameAcquisition
from dirigo.plugins.calibrations import TriggerDelayCalibration

"""
calibrate_trigger_delay.py

Set up a continuous raster frame acquisition borrowing FrameAcquisition default
specification. Override fast axis amplitude setting and measure phase delay from
bidirectional data. Repeat for n_amplitudes across ampl_range.min to 
ampl_range.max fraction (0.0-1.0) of full scale scanner amplitude.
"""

# Adjustable parameters
N_AMPL = 100
AMPL_MIN = 0.25 # fraction of full amplitude
AMPL_MAX = 1.0


# Calibration script
diri = Dirigo()

# Use the default FrameAcquisition spec as basis
frame_spec = FrameAcquisition.get_specification()

spec = TriggerDelayCalibration.Spec(
    ampl_range={"min": AMPL_MIN, "max": AMPL_MAX},
    n_amplitudes=N_AMPL,
    **frame_spec.to_dict()
)

name = "trigger_delay_calibration"
acquisition = diri.make("acquisition", name, spec=spec)
processor   = diri.make("processor", "raster_frame", upstream=acquisition)
loggers     = diri.make("logger", name, upstream=processor)

acquisition.start()

# wait until logger finishes
loggers.join()