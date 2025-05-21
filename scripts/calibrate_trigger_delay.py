

from dirigo.main import Dirigo

from dirigo.plugins.acquisitions import FrameAcquisition
from dirigo.plugins.calibrations import TriggerDelayCalibration
from dirigo.plugins.processors import RasterFrameProcessor




diri = Dirigo()


# Use the default FrameAcquisition spec as basis
frame_spec = FrameAcquisition.get_specification()

spec = TriggerDelayCalibration.Spec(
    ampl_range={"min": 0.25, "max": 1.0},
    n_amplitudes=100,
    **frame_spec.to_dict()
)

acquisition = TriggerDelayCalibration(diri.hw, diri.system_config, spec=spec)
processor = RasterFrameProcessor(upstream=acquisition)

acquisition.start()