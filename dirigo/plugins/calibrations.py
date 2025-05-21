import time
from typing import Type

import numpy as np

from dirigo import units

from dirigo.sw_interfaces import Acquisition
from dirigo.plugins.acquisitions import FrameAcquisition

from dirigo.hw_interfaces import Digitizer, FastRasterScanner, SlowRasterScanner



class TriggerDelayCalibrationSpec(FrameAcquisition.Spec):
    def __init__(self, 
                 ampl_range: dict | units.FloatRange = units.FloatRange(0.25, 1.0),
                 n_amplitudes: str | int = 10,
                 measurement_frames_per_ampl: str | int = 10,
                 pause_time: str | units.Time = units.Time("5 s"),
                 **kwargs):
        super().__init__(**kwargs)
        
        if isinstance(ampl_range, dict):
            ampl_range = units.FloatRange(ampl_range['min'], ampl_range['max'])
        self.ampl_range = ampl_range
        self.n_amplitudes = int(n_amplitudes)
        self.measurement_frames_per_ampl = int(measurement_frames_per_ampl)
        self.pause_time = units.Time(pause_time)


class TriggerDelayCalibration(Acquisition):
    required_resources = [Digitizer, FastRasterScanner, SlowRasterScanner]
    Spec: Type[TriggerDelayCalibrationSpec] = TriggerDelayCalibrationSpec

    def __init__(self, hw, system_config, spec):
        super().__init__(hw, system_config, spec)
        self.spec: TriggerDelayCalibrationSpec

        ampl = self.hw.fast_raster_scanner.angle_limits.range
        self._amplitudes = np.linspace(
            start=self.spec.ampl_range.min * ampl,
            stop=self.spec.ampl_range.max * ampl,
            num=self.spec.n_amplitudes
        )
        self.spec.buffers_per_acquisition = self.spec.measurement_frames_per_ampl
        
        # Frame acquisition needs to exist by the end of __init__ for other 
        # Workers (e.g. FrameProcessor) to instantiate correctly
        self._frame_acquisition = FrameAcquisition(self.hw, self.system_config, self.spec)
        self._frame_acquisition.add_subscriber(self)

        self.digitizer_profile = self._frame_acquisition.digitizer_profile
        self.runtime_info = self._frame_acquisition.runtime_info

    def run(self):
        try:
            for ampl in self._amplitudes:
                # Remake the acquisition object and start it
                self._frame_acquisition = FrameAcquisition(self.hw, self.spec)
                self._frame_acquisition.add_subscriber(self)
                self._frame_acquisition.start()

                # Over-ride amplitude
                self.hw.fast_raster_scanner.amplitude = units.Angle(ampl)

                # Get measurement frames
                for _ in range(self.spec.measurement_frames_per_ampl):
                    product = self.inbox.get()
                    if product is None: return
                    with product: 
                        self.publish(product)
                
                if self.inbox.get() is not None: # receive final None
                    return 
                self._frame_acquisition.join(1)
                self._frame_acquisition = None
                time.sleep(self.spec.pause_time)

        finally:
            self.publish(None) # publish the sentinel
