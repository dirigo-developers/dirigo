import time
from typing import Type

import numpy as np
from numpy.polynomial.polynomial import Polynomial
from scipy import fft

from dirigo import units
from dirigo import io
from dirigo.sw_interfaces import Acquisition, Processor, Logger
from dirigo.sw_interfaces.worker import EndOfStream
from dirigo.plugins.acquisitions import FrameAcquisition

from dirigo.hw_interfaces import Digitizer, FastRasterScanner

"""
Calibrations are special instances of Acquisition designed to produce data for
measurement of system parameters (e.g. trigger delay, line distortion, etc.)
"""


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
    required_resources = [Digitizer, FastRasterScanner] # child FrameAcquisition will also check resources
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
            for i, ampl in enumerate(self._amplitudes):
                try:
                    print(
                        f"Calibrating amplitude: {units.Angle(ampl)} "
                        f"({i+1} of {len(self._amplitudes)})"
                    )
                    # Remake the acquisition object and start it
                    self._frame_acquisition = FrameAcquisition(
                        self.hw, self.system_config, self.spec
                    )
                    self._frame_acquisition.add_subscriber(self)
                    self._frame_acquisition.start()

                    # Over-ride amplitude
                    self.hw.fast_raster_scanner.amplitude = units.Angle(ampl)

                    # Get measurement frames
                    while True:
                        with self._receive_product() as product: 
                            self._publish(product)
                    
                except EndOfStream:
                    self._frame_acquisition.join(1)
                    self._frame_acquisition = None
                    time.sleep(self.spec.pause_time)

        finally:
            self._publish(None)


class TriggerDelayCalibrationLogger(Logger):
    """Logs measured trigger delay at amplitudes."""
    def __init__(self, upstream: Processor):
        super().__init__(upstream)

        self._amplitudes = []
        self._frequencies = []
        self._phases = []
        
    def run(self):
        try:
            while True:
                with self._receive_product() as product:
                    # store the amplitude/frequency/phase data
                    self._amplitudes.append(
                        self._acq.hw.fast_raster_scanner.amplitude
                    )
                    self._frequencies.append(product.frequency)
                    self._phases.append(product.phase)

        except EndOfStream:
            self._publish(None) # pass sentinel
            self.save_data()
            

    def save_data(self):
        amplitudes = np.unique(self._amplitudes)
        frequencies = []
        phases = []
        for ampl in amplitudes:
            matching_f = []
            matching_p = []
            for a, f, p in zip(self._amplitudes, self._frequencies, self._phases):
                if a == ampl:
                    matching_f.append(f)
                    matching_p.append(p)

            frequencies.append(np.median(matching_f))
            phases.append(np.median(matching_p))

        # stack into a 2-column array
        data = np.column_stack([amplitudes, frequencies, phases])

        # write with a header comment for units
        np.savetxt(
            io.config_path() / "scanner/trigger_delay_calibration.csv",
            data,
            delimiter=',',
            header='amplitude (rad),frequency (Hz),phase (rad)',
            comments=''    # prevent numpy from prefixing "#" on header lines
        )




class LineDistortionCalibrationSpec(FrameAcquisition.Spec):
    def __init__(self,
                 translation: units.Position | str,
                 ignore_frames: int = 10,
                 n_steps: int = 4,
                 **kwargs):
        super().__init__(**kwargs)
        self.translation = units.Position(translation)
        self.ignore_frames = ignore_frames
        self.n_steps = n_steps

        # override frame limit
        self.buffers_per_acquisition = float('inf')


class LineDistortionCalibration(Acquisition):
    Spec: Type[LineDistortionCalibrationSpec] = LineDistortionCalibrationSpec

    def __init__(self, hw, system_config, spec):
        super().__init__(hw, system_config, spec, thread_name="Line distortion calibration")
        self.spec: LineDistortionCalibrationSpec

        self._frame_acquisition = FrameAcquisition(self.hw, system_config, self.spec)
        self._frame_acquisition.add_subscriber(self)

        self.digitizer_profile = self._frame_acquisition.digitizer_profile
        self.runtime_info = self._frame_acquisition.runtime_info

        fast_axis = self.system_config.fast_raster_scanner['axis']
        if fast_axis == "x":
            self._fast_stage = self.hw.stages.x
        else:
            self._fast_stage = self.hw.stages.y
        self._original_position = self._fast_stage.position

    def run(self):
        try:
            self._frame_acquisition.start()

            # collect some sacrificial frames
            for _ in range(10 * self.spec.ignore_frames):
                with self._receive_product() as product:
                    pass
            
            for i in range(self.spec.n_steps):
                print(f"Collecting frame {i} of {self.spec.n_steps}")

                # Gather frame
                with self._receive_product() as product: 
                    self._publish(product)

                # Move 
                self._fast_stage.move_to(
                    self._fast_stage.position + self.spec.translation
                )

                # Discard frames in motion
                for _ in range(self.spec.ignore_frames):
                    with self._receive_product() as product:
                        pass

            # Move back to original position
            self._fast_stage.move_to(self._original_position)

        except EndOfStream:
            self._publish(None)

        finally:
            self._publish(None)
            self._frame_acquisition.stop()


class LineDistortionCalibrationLogger(Logger):
    """Logs apparent distortion use patches to create a local distortion field."""
    UPSAMPLE       = 20          # global phase‑corr up‑sampling
    EPS            = 1e-1
    PATCH          = 64
    STRIDE         = 8

    def __init__(self, upstream: Processor):
        super().__init__(upstream)
        self._acq: LineDistortionCalibration
        self._fn = io.config_path() / "scanner/distortion_calibration_data.csv"

    def run(self):
        self._frames, self._positions = [], [] # collect measurement frames/pos
        try:
            while True:
                with self._receive_product() as product:
                    self._frames.append(product.data[:,:,0].copy())
                    self._positions.append(product.positions)

        except EndOfStream:
            self._publish(None) # forward sentinel
            self.save_data()

    def save_data(self):
        """Process distortion field and save fit results"""
        spec = self._acq.spec

        n_f = len(self._frames)
        n_comparisons = n_f - 1
        ref_frame = self._frames[0]
        n_y, n_x = ref_frame.shape
        yc = n_y//2
        yr = n_y//8

        # Create: dx_true/dx_observed
        dx_true = round(spec.translation / spec.pixel_size)
        dx_observed = np.zeros(shape=(n_x // self.STRIDE, n_comparisons))
        field_position = np.zeros(n_x // self.STRIDE)
        
        ref_frame = self._frames[0]
        for f_idx, frame in enumerate(self._frames[1:]):

            for p_idx in range(n_x // self.STRIDE):
                p0 = (p_idx * self.STRIDE) # ref patch start pixel index
                ref_patch = ref_frame[(yc-yr):(yc+yr), p0:(p0 + self.PATCH)]

                m0 = p0 - dx_true # mov patch start pixel index
                if (m0 < 0) or (p0+self.PATCH >= n_x) or (m0+self.PATCH) >= n_x: 
                    dx_observed[p_idx, f_idx] = np.nan
                    field_position[p_idx] = np.nan
                    continue
                mov_patch = frame[(yc-yr):(yc+yr), m0:(m0 + self.PATCH)]

                _, j = self.x_corr(ref_patch, mov_patch)
                print(units.Position(j*spec.pixel_size))

                dx_observed[p_idx, f_idx] = j + dx_true
                ref_patch_center = p0 + self.PATCH//2
                mov_patch_center = m0 + self.PATCH//2
                comp_center = (ref_patch_center + mov_patch_center)/2
                field_position[p_idx] = -(comp_center - n_x/2) / n_x * 2 * spec.fill_fraction

            ref_frame = frame

        # Fit error
        field_positions = np.tile(field_position[:,np.newaxis], (1, n_comparisons)) 

        nan_mask = np.isnan(dx_observed)
        pfit: Polynomial = Polynomial.fit(
            x=field_positions[~nan_mask].ravel(),
            y=(dx_true/dx_observed)[~nan_mask].ravel(),
            deg=2
        )
        c0, c1, c2 = pfit.convert().coef

        # Save errors
        data = np.concatenate(
            (field_position[:,np.newaxis], dx_true/dx_observed),
            axis=1
        )
        np.savetxt(self._fn, data, delimiter=',',header="field position,dx_true/dx_observed")

        # If not calibrated (would have trivial Polynomial(1)), save distortion polynomial
        if self._processor._distortion_polynomial == Polynomial([1]):
            fn = io.config_path() / "optics/distortion_calibration.csv"
            calib_data = np.array([[self._acq.runtime_info.scanner_amplitude, c0, c1, c2]])
            if fn.exists(): # add to existing calibration
                data = np.loadtxt(fn, delimiter=',', dtype=np.float64, skiprows=1, ndmin=2)
                data = np.concatenate((data, calib_data), axis=0)
            else:
                data = calib_data

            np.savetxt(fn, data, delimiter=',',header="scanner amplitude (rad),coefficients (in ascending order)")

    @classmethod
    def x_corr(cls, ref_frame: np.ndarray, moving_frame: np.ndarray):
        n_y, n_x = ref_frame.shape

        xps = fft.rfft2(ref_frame, workers=-1) * np.conj(fft.rfft2(moving_frame, workers=-1))
        s = (n_y, n_x * cls.UPSAMPLE)
        corr = fft.irfft2(xps / (np.abs(xps) + cls.EPS), s, workers=-1)
        arg_max = np.argmax(corr)
        i = arg_max // corr.shape[1]
        j = arg_max %  corr.shape[1]

        if i > (s[0] // 2):  # Handle wrap-around for negative shifts
            i -= s[0]
        if j > (s[1] // 2): 
            j -= s[1]

        return i / cls.UPSAMPLE, j / cls.UPSAMPLE