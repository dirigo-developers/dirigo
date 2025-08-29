import time, threading
from functools import cached_property

import numpy as np
from numpy.polynomial.polynomial import Polynomial
from numba import njit, prange, types, int16, uint16, int32, int64, float32, complex64
from scipy import fft

from dirigo.components import units, io
from dirigo.components.profiling import timer
from dirigo.sw_interfaces.worker import EndOfStream
from dirigo.sw_interfaces.processor import Processor, ProcessorProduct
from dirigo.sw_interfaces.acquisition import Acquisition, AcquisitionProduct
from dirigo.plugins.acquisitions import (
    LineAcquisitionSpec, LineAcquisition, 
    FrameAcquisitionSpec, FrameAcquisition,
    LineCameraLineAcquisitionSpec, LineCameraLineAcquisition
)


TWO_PI = 2 * np.pi

uint8_3d_readonly   = types.Array(types.uint8, 3, 'C', readonly=True)
int8_3d_readonly   = types.Array(types.int8, 3, 'C', readonly=True)
int16_3d_readonly  = types.Array(types.int16, 3, 'C', readonly=True)
uint16_3d_readonly = types.Array(types.uint16, 3, 'C', readonly=True)

# ---------- Raster Frame Processor ----------
sigs = [
    #buffer_data         invert_mask  offset    bit_shift  gradient     resampled (out)  start_indices  nsamples_to_sum
    (int8_3d_readonly,   int16[:],    int16[:], int32,     float32[:],  int16[:,:,:],   int32[:,:],    int32[:,:]),
    (uint8_3d_readonly,  int16[:],    int16[:], int32,     float32[:],  int16[:,:,:],   int32[:,:],    int32[:,:]),
    (int16_3d_readonly,  int16[:],    int16[:], int32,     float32[:],  int16[:,:,:],   int32[:,:],    int32[:,:]),
    (uint16_3d_readonly, int16[:],    int16[:], int32,     float32[:],  int16[:,:,:],   int32[:,:],    int32[:,:])
]
@njit(sigs, parallel=True, fastmath=True, nogil=True, cache=True)
def resample_kernel(raw_data: np.ndarray, 
                    invert_mask: np.ndarray,
                    offset: np.ndarray,
                    bit_shift: int,
                    gradient: np.ndarray, 
                    resampled: np.ndarray, 
                    start_indices: np.ndarray, 
                    nsamples_to_sum: np.ndarray):
    """
    buffer_data shape: (Nrecords, Nsamples, Nchannels)
    dewarped shape:    (Ns, Nf, Nc)
    start_indices shape: (Ndirections, Nf)
    nsamples_to_sum shape: (Ndirections, Nf)
    
    Ndirections is 1 (unidirectional) or 2 (bidirectional).
    """
    Nrecords, Nsamples, Nc_in = raw_data.shape
    Ns, Nf, Nc_out = resampled.shape # Acquisition must be in channel-minor order (interleaved)
    # Nc_in may not be the same as Nc_out if there's a empty byte (e.g. BGRX format)
    Ndirections = start_indices.shape[0]

    # Clamp start_indices to [0, Nsamples); calculate final scaling factor
    scaling_factor = np.zeros_like(start_indices, np.float32)
    for d in prange(Ndirections):
        for fi in prange(Nf):
            start_indices[d, fi] = min(max(start_indices[d, fi], 0), Nsamples-1)
            scaling_factor[d, fi] = bit_shift * gradient[fi] / nsamples_to_sum[d, fi]
    
    # Main loop over slow-axis pixels
    for si in prange(Ns): 
        fwd_rvs     = si % Ndirections # 0 => forward, 1 => reverse (if Ndirections=2)
        ri          = si // Ndirections # record index 
        stride      = 1 - 2 * fwd_rvs # +1 if forward, -1 if reverse

        # Loop over fast-axis pixels
        for fi in range(Nf): 
            Nsum    = nsamples_to_sum[fwd_rvs, fi]
            sf      = scaling_factor[fwd_rvs, fi]
            start   = start_indices[fwd_rvs, fi]

            tmp_values = np.zeros(Nc_out, dtype=np.int32)

            # Step through the raw samples
            end = start + (Nsum * stride)
            for sample in range(start, end, stride):
                for c in range(Nc_out):
                    tmp_values[c] += raw_data[ri, sample, c] - offset[c]

            # Store the average back into resampled
            for c in range(Nc_out):
                resampled[si, fi, c] = invert_mask[c] * (sf * tmp_values[c]) 


sigs = [
#    data                trigger delay  samples_per_period  cropped
    (int8_3d_readonly,   int64,         float32,            float32[:,:]),
    (uint16_3d_readonly, int64,         float32,            float32[:,:]),
    (int16_3d_readonly,  int64,         float32,            float32[:,:])
]
#@njit(sigs, nogil=True, parallel=True, fastmath=True, cache=True)
def crop_bidi_data(data: np.ndarray,
                   trigger_delay: int,
                   samples_per_period: float, # exact number of samples per fast axis period (can be noninteger)
                   cropped: np.ndarray):
    """
    Select ranges, flips the reverse scan, and converts to float32 for phase
    correlation.

    Note: operates on channel 0 only.
    """
    lines_per_frame, n = cropped.shape

    # Find the nominal midpoints and endpoints of fwd and rvs scans
    mid_fwd = samples_per_period/4
    mid_rvs = mid_fwd + samples_per_period/2

    fwd0 = int(mid_fwd) - n//2 - trigger_delay
    fwd1 = int(mid_fwd) + n//2 - trigger_delay
    rvs0 = int(mid_rvs) - n//2 - trigger_delay
    rvs1 = int(mid_rvs) + n//2 - trigger_delay

    for rec in prange(lines_per_frame//2):
        cropped[2*rec, :] = data[rec, fwd0:fwd1, 0].astype(np.float32)
        cropped[2*rec+1, :] = data[rec, rvs1:rvs0:-1, 0].astype(np.float32) # flipped with the slicing
        

@njit(
    complex64[:](complex64[:,:]),
    nogil=True, parallel=True, fastmath=True, cache=True
)
def compute_cross_power_spectrum(F: np.ndarray):
    """Compute the cross-power spectrum for F."""
    n_freqs = F.shape[1]  # Number of frequency components
    n_pairs = F.shape[0] // 2  # Number of record pairs
    xps = np.zeros(n_freqs, dtype=np.complex64)  # Output array

    for i in prange(n_freqs * n_pairs):
        pair = i // n_freqs
        freq = i % n_freqs
        xps[freq] += F[2 * pair, freq] * np.conj(F[2 * pair + 1, freq])  

    # normalize
    EPS = 1e-1  # Small constant
    for freq in prange(n_freqs):
        mag = np.abs(xps[freq]) + EPS
        xps[freq] /= mag

    return xps


class RasterFrameProcessor(Processor[Acquisition]):
    def __init__(self, 
                 upstream: LineAcquisition | FrameAcquisition, # and subclasses: FrameAcquisition, StackAcquisition, etc.
                 bits_precision: int = 16):
        """
        Initialize a raster frame processor worker for the Acquisition worker.

        To change default behavior of computing average for each pixel in 16-
        bit precision, change the bits_precision argument.
        """
        super().__init__(upstream)
        self._acquisition: LineAcquisition | FrameAcquisition
        self._spec: LineAcquisitionSpec | FrameAcquisitionSpec # to refine type hinting
        
        digitizer_profile = upstream.digitizer_profile
        system_config = upstream.system_config # static parameters
        runtime_info = upstream.runtime_info # dynamic parameters

        if (not isinstance(bits_precision, int)) or (bits_precision % 2) or not (8 <= bits_precision <= 16):
            raise ValueError("Bits precision must be even integer between 8 and 16")
        elif bits_precision < runtime_info.digitizer_bit_depth:
            raise ValueError("Bit precision can't be less than the data acquisition device bit depth.")
        self._bits_precision = bits_precision
        
        # Compute shape
        if isinstance(self._spec, FrameAcquisitionSpec):
            n_lines = self._spec.lines_per_frame
        else:
            # fall-back when processing LineAcquisition as a frame
            n_lines = self._spec.lines_per_buffer
        
        n_channels = sum([c.enabled for c in digitizer_profile.channels])
        dt = np.int16 # TODO set this programatically
        self.processed_shape = (n_lines, self._spec.pixels_per_line, n_channels)

        self._invert_mask = -2 * np.array(
            [c.inverted for c in digitizer_profile.channels], dtype=dt
        ) + 1

        self.signal_offset = np.round(io.load_signal_offset()).astype(np.int16)

        # Pre-allocate array for processed image
        self._init_product_pool(n=4, shape=self.processed_shape, dtype=dt)

        # Trigger timing/delay
        self._fixed_trigger_delay = runtime_info.digitizer_trigger_delay

        if "galvo" in system_config.fast_raster_scanner['type'].lower():
            delay = units.Time(system_config.fast_raster_scanner['input_delay'])
            self._trigger_error = delay * self._sample_clock_rate
            
        elif "resonant" in system_config.fast_raster_scanner['type'].lower():
            # Set a preliminary fast axis frequency
            self._fast_scanner_frequency = units.Frequency(system_config.fast_raster_scanner['frequency'])
            try:
                # anticipate correct phase delay from calibration table
                self._initial_trigger_error = self.calibrated_trigger_delay(
                    scanner_amplitude=runtime_info.scanner_amplitude
                )
                self._trigger_error = self._initial_trigger_error

            except:
                # Calibration could not be loaded, don't use
                self._initial_trigger_error = None
                self._trigger_error = 0

            # Preallocate a cropping buffer for bidi acquisitions
            lines_per_frame = self.processed_shape[0]
            n = 2**int(np.log2(self.samples_per_period/2 - 2*self._fixed_trigger_delay))
            self._cropped = np.zeros((lines_per_frame, n), dtype=np.float32) # preallocate
        
        try:
            ampl = runtime_info.scanner_amplitude
            self._distortion_polynomial = io.load_line_distortion_calibration(ampl)
        except:
            self._distortion_polynomial = Polynomial([1])

        self._scaling_factor = 2 ** (
            self._bits_precision - runtime_info.digitizer_bit_depth
        )

        # Try loading gradient calibration
        try:
            if hasattr(self._spec, "frame_height"):
                raise Exception
            else: # line acquisition
                self._gradient = io.load_line_gradient_calibration(
                    line_width = self._spec.line_width,
                    pixel_size = self._spec.pixel_size
                )
        except:
            self._gradient = np.ones((self._spec.pixels_per_line,), np.float32)

        self._phases = np.full(shape=(10,), fill_value=np.nan, dtype=np.float32)
        self._frames_processed = 0

    def _receive_product(self, block: bool = True, timeout: float | None = None) -> AcquisitionProduct:
        return super()._receive_product(block, timeout) # type: ignore

    def _work(self):
        try:
            while True: 
                with self._receive_product() as acquisition_product:
                    processed = self._process_frame(acquisition_product)
                    self._publish(processed) # sends off to Logger and/or Display workers
                    self._frames_processed += 1

        except EndOfStream:
            self._publish(None) # forward sentinel

    def _process_frame(self, acq_product: AcquisitionProduct) -> ProcessorProduct:
        processed = self._get_free_product() # Check out a product from the pool

        # If array of timestamps are assigned (default is None)
        if isinstance(acq_product.timestamps, np.ndarray):
            # Estimate scanner frequency from timestamps
            avg_trig_period = np.mean(np.diff(acq_product.timestamps))
            self._fast_scanner_frequency = 1 / avg_trig_period
            print(self._fast_scanner_frequency)

        # Measure phase from bidi data (in uni-directional, phase is not critical)
        if self._spec.bidirectional_scanning:
            p = self._frames_processed % len(self._phases)
            self._phases[p] = self.measure_phase(acq_product.data)
            self._trigger_error = np.median(self._phases[~np.isnan(self._phases)])

        # Update resampling start indices--these can change a bit if the scanner frequency drifts
        start_indices = self.calculate_start_indices(self._trigger_error) - self._fixed_trigger_delay
        nsamples_to_sum = np.abs(np.diff(start_indices, axis=1))

        with timer("resample_kernel"):
            resample_kernel(
                raw_data=acq_product.data, 
                invert_mask=self._invert_mask,
                offset=self.signal_offset,
                bit_shift=self._scaling_factor,
                gradient=self._gradient,
                resampled=processed.data,
                start_indices=start_indices, 
                nsamples_to_sum=nsamples_to_sum
            )
        processed.timestamps = acq_product.timestamps
        processed.positions = acq_product.positions
        
        if hasattr(self, "_fast_scanner_frequency"):
            processed.phase = TWO_PI * self._trigger_error \
            / (self._acquisition.digitizer_profile.sample_clock.rate * avg_trig_period)
            processed.frequency = float(self._fast_scanner_frequency)
        
        return processed

    @cached_property
    def _temporal_edges(self):
        # Set up an array with the SPATIAL edges of pixels--we will bin samples into the pixel edges
        ff = self._spec.fill_fraction

        # Create a resampling function based on fast axis waveform type
        if 'resonant' in self._acquisition.system_config.fast_raster_scanner['type'].lower():
            # image line should be taken from center of sinusoid sweep
            pixel_edges = np.linspace(ff, -ff, self._spec.pixels_per_line + 1) 

            # Use distortion polynomial to correct spatial edges
            int_p = self._distortion_polynomial.integ()
            x = np.linspace(-1, 1, 1_000_000)
            y = int_p(x)
            modified_pixel_edges = np.interp(pixel_edges, y, x)

            # arccos inverts the cosinusoidal path, normalize scan period to 0.0 to 1.0
            temporal_edges = np.arccos(modified_pixel_edges) / TWO_PI

        else: #TODO, rename to smooth triangle or something
            # Fill fraction needs to be slightly adjusted to reflect pixel period rounding 
            ff_corrected = self._spec.pixels_per_line / round(self._spec.pixels_per_line / self._spec.fill_fraction)

            # assume we start at 0 and go to corrected fill fraction
            pixel_edges = np.linspace(0, ff_corrected, self._spec.pixels_per_line + 1)

            temporal_edges = pixel_edges # The scan should already be linearized

        if self._spec.bidirectional_scanning:
            temporal_edges_fwd = temporal_edges
            temporal_edges_rvs = 1.0 - temporal_edges
            temporal_edges = np.vstack([temporal_edges_fwd, temporal_edges_rvs]) 
        else:
            temporal_edges = np.vstack([temporal_edges])

        return temporal_edges

    def calculate_start_indices(self, trigger_phase: int = 0):
        """Calculate the start position in digitizer records for each pixel."""
        starts_exact = self._temporal_edges * self.samples_per_period + trigger_phase
        return np.ceil(starts_exact - 1e-6).astype(np.int32) 
    
    def measure_phase(self, data: np.ndarray) -> float:
        """
        Measure the apparent fast raster scanner trigger phase, in samples
        (for bidirectional scanning).
        """
        UPSAMPLE = 1        # TODO move this somewhere else
        PHASE_MAX = 320

        crop_bidi_data(
            data =                  data, 
            trigger_delay =         self._fixed_trigger_delay, 
            samples_per_period =    self.samples_per_period,
            cropped =               self._cropped
        )

        F = fft.rfft(self._cropped, axis=1, workers=4)
        xps = compute_cross_power_spectrum(F)

        n = self._cropped.shape[1] * UPSAMPLE
        corr = np.abs(fft.irfft(xps, n))

        shift = float(np.argmax(corr))
        if shift > (n//2):  # Handle wrap-around for negative shifts
            shift -= n

        # if phase is outside the pre-determined range, then return NaN (indeterminate)
        if abs(shift) > PHASE_MAX * UPSAMPLE * 2:
            return np.nan

        return shift / UPSAMPLE / 2 - 1
    
    def calibrated_trigger_delay(self, scanner_amplitude: units.Angle) -> float:
        ampls, freqs, phases = io.load_scanner_calibration()

        phase = np.interp(scanner_amplitude, ampls, phases)
        frequency = np.interp(scanner_amplitude, ampls, freqs)

        digitizer_rate = self._acquisition.digitizer_profile.sample_clock.rate
        return (phase / TWO_PI) * (digitizer_rate / frequency)
    
    @cached_property
    def _sample_clock_rate(self) -> units.SampleRate:
        return self._acquisition.digitizer_profile.sample_clock.rate
    
    @property
    def samples_per_period(self) -> float:
        """
        The exact number of digitizer samples per fast raster scanner period.
        """
        if hasattr(self, "_fast_scanner_frequency"):
            return float(self._sample_clock_rate / self._fast_scanner_frequency)
        else:
            return self._acquisition.product_shape[1]
    
    @property
    def data_range(self):
        """
        The data range after processing (resampling) has been performed.

        May be higher than the native bit depth of the data capture device.
        """
        return units.IntRange(
            min=-2**(self._bits_precision-1), 
            max=2**(self._bits_precision-1) - 1
        )



# ---------- Line Camera Processor ----------
class LineCameraLineProcessor(Processor[LineCameraLineAcquisition]):
    def __init__(self, 
                 upstream):
        """
        Initialize a camera line processor worker for the Acquisition worker.
        """
        super().__init__(upstream)
        self._spec: LineCameraLineAcquisitionSpec
        self._acquisition: LineCameraLineAcquisition
        
        camera_profile = self._acquisition.camera_profile # TODO, do we need this?
        system_config = self._acquisition.system_config
        runtime_info = self._acquisition.runtime_info

        # Compute shape & datatype. Pre-allocate Products
        n_lines = self._spec.lines_per_buffer
        n_channels = 3 if runtime_info.camera_bit_depth > 16 else 1
        self.processed_shape = (n_lines, self._spec.pixels_per_line, n_channels)

        self._init_product_pool(n=4, shape=self.processed_shape, dtype=np.int16)

        if runtime_info.camera_bit_depth == 24: # RGB24
            d = 8  # uint8
        else:
            d = runtime_info.camera_bit_depth  # unsigned integer
        # scaling factor: factor to take full-scale camera value (unsigned) to
        # full-scale int16 (32768)
        self._scaling_factor = 2 ** (15 - d)

        # Load distortion calibration
        try:
            raise Exception # temporary
            self._distortion_polynomial = io.load_line_distortion_calibration()
        except:
            self._distortion_polynomial = Polynomial([1])

        # Load illumination calibration
        try:
            raise Exception # temporary
            self._gradient = io.load_gradient_calibration()
        except:
            self._gradient = np.ones((self._spec.pixels_per_line,), np.float32)

        self._buffers_processed = 0

    def _receive_product(self, block: bool = True, timeout: float | None = None) -> AcquisitionProduct:
        return super()._receive_product(block, timeout) # type: ignore
    
    def _product_stream(self): # TODO factor up into a base class
        """Yield acquisition products until EndOfStream is raised."""
        while True:
            try:
                with self._receive_product() as prod:
                    yield prod
            except EndOfStream:
                break
    
    def _work(self):
        try:
            for acq_prod in self._product_stream():
                proc_product = self._get_free_product()
                self._process_frame(acq_prod, proc_product)
                self._publish(proc_product) 
        finally: 
            # always send sentinel to gracefully shutdown downstream Workers
            self._publish(None)  

    def _process_frame(self, 
                       in_product: AcquisitionProduct, 
                       out_product: ProcessorProduct) -> None:
        """Actual processing work for each buffer/frame"""

        Nc = out_product.data.shape[2]
        resample_kernel(
            raw_data=in_product.data, 
            invert_mask=np.ones(shape=(Nc,), dtype=np.int16), # never invert
            offset=np.zeros(shape=(Nc,), dtype=np.int16), # no offsets
            bit_shift=self._scaling_factor,
            gradient=self._gradient,
            resampled=out_product.data,
            start_indices=self.start_indices, 
            nsamples_to_sum=self.nsamples_to_sum
        )
        out_product.positions = in_product.positions

        self._buffers_processed += 1
        
    @cached_property # these won't change over the course of the acquisition
    def start_indices(self) -> np.ndarray:
        """Returns array of the pixel start indexes for resampling (dewarping)"""
        N_x = self._spec.pixels_per_line

        x_sensor = np.arange(N_x+1) # sensor's coordinates (distorted relative to true spatial coordinates)

        # Use distortion polynomial to correct spatial edges
        int_p = self._distortion_polynomial.integ()
        x = np.linspace(0, N_x, 1_000_000)
        y = int_p(x)
        x_space = np.interp(x_sensor, y, x)

        x_space = x_space[np.newaxis, :]

        return np.ceil(x_space).astype(np.int32)
    
    @cached_property
    def nsamples_to_sum(self):
        return np.abs(np.diff(self.start_indices, axis=1))
    
    @property
    def data_range(self):
        """
        The data range after processing (resampling) has been performed.

        May be higher than the native bit depth of the data capture device.
        """
        return units.IntRange(
            min=-2**15, 
            max=2**15 - 1
        )



# ---------- Rolling Average Processor ----------
sigs = [
#    ring_buffer       frame_index  averaged
    (uint16[:,:,:,:],  int64,       uint16[:,:,:]),
    (int16[:,:,:,:],   int64,       int16[:,:,:] ),
]
@njit(sigs, nogil=True, parallel=True, fastmath=True, cache=True)
def _rolling_average_kernel(ring_buffer: np.ndarray, 
                           frame_index: int, 
                           averaged: np.ndarray) -> np.ndarray:
    Nf, Ny, Nx, Nc = ring_buffer.shape
    d = min(Nf, frame_index + 1)

    for iy in prange(Ny):
        for ix in prange(Nx):
            for ic in range(Nc):
                tmp = 0
                for iframe in range(d):
                    tmp += ring_buffer[iframe, iy, ix, ic]
                averaged[iy, ix, ic] = tmp // d

    return averaged


class RollingAverageProcessor(Processor[RasterFrameProcessor]):
    def __init__(self,
                 upstream):
        super().__init__(upstream)
        self._data_range = upstream.data_range
        self._acquisition: FrameAcquisition | LineAcquisition 

        self._init_product_pool(
            n       = 2, 
            shape   = upstream.product_shape, 
            dtype   = upstream.product_dtype
        )
        
        pixel_size = self._acquisition.spec.pixel_size
        self._reset_average_xy_cutoff_sqr = pixel_size ** 2 # TODO, may require some tuning to remove unwanted resets
        self._reset_average_z_cutoff = pixel_size # TODO, dido

        self._prev_position = None
        self._n_frame_average = 1
        self._average_buffer = None
        self._i: int = 0 # tracks rolling average index
        self._average_buffer_lock = threading.Lock() # Use lock for thread safety

    @property
    def n_frame_average(self) -> int:
        return self._n_frame_average
    
    @n_frame_average.setter
    def n_frame_average(self, n_frames: int):
        if not isinstance(n_frames, int) or n_frames < 1:
            raise ValueError("Rolling frame average must be an integer >= 1.")
        with self._average_buffer_lock:
            self._n_frame_average = n_frames
            self._average_buffer = None # Triggers reset of the average buffer 

    def _receive_product(self, 
                         block: bool = True, 
                         timeout: float | None = None
                         ) -> AcquisitionProduct | ProcessorProduct:
        return super()._receive_product(block, timeout) # type: ignore
    
    def _work(self):
        try:
            while True:
                with self._receive_product() as in_product:

                    # Check whether position has changed
                    if in_product.positions is not None:

                        if isinstance(in_product.positions, np.ndarray):
                            current_positions = in_product.positions[-1,:] # use final position only
                        
                        elif isinstance(in_product.positions, tuple):
                            current_positions = np.array(in_product.positions)

                        if (self._prev_position is not None):

                            dr2 = (current_positions[0] - self._prev_position[0])**2  \
                                + (current_positions[1] - self._prev_position[1])**2
                            
                            if dr2 > self._reset_average_xy_cutoff_sqr: 
                                # if the Euclidean exceeds from last frame exceeds pixel size, then reset the rolling average index
                                # Setting _i = 0, will cause this frame to be sliced into the ring buffer at the first position
                                # and the averaging kernel will only average that 1 frame (effectively no averaging)
                                # this allows reusing the _average_buffer object.
                                self._i = 0

                            elif len(current_positions) > 2: # Z axis also included
                                dz = abs(current_positions[2] - self._prev_position[2])
                                if dz > self._reset_average_z_cutoff:
                                    # Similarly, reset _i if z position changes
                                    self._i = 0
                        
                        self._prev_position = current_positions

                    with self._average_buffer_lock:
                        # _average_buffer is initially None. First frame establishes height, width, etc
                        # Averaging can also be reset or changed by setting this to None, prompting reallocation
                        if self._average_buffer is None:
                            frame_shape = in_product.data.shape
                            ring_frame_buffer_shape = (self.n_frame_average,) + frame_shape
                            self._average_buffer = np.zeros(
                                shape=ring_frame_buffer_shape, 
                                dtype=self.data_range.recommended_dtype
                            )
                            self._i = 0

                        a = self._i % self.n_frame_average
                        self._average_buffer[a] = in_product.data

                        out_product = self._get_free_product()

                        _rolling_average_kernel(
                            self._average_buffer, 
                            self._i, 
                            out_product.data
                        )

                        self._publish(out_product)
                        
                self._i += 1

        except EndOfStream:
            self._publish(None)

    @property
    def data_range(self) -> units.IntRange:
        return self._data_range