import time
from functools import cached_property

import numpy as np
from numpy.polynomial.polynomial import Polynomial
from numba import njit, prange, int16, uint16, int32, int64, float32, complex64
from scipy import fft

from dirigo.components import units
from dirigo.components import io
from dirigo.sw_interfaces.processor import Processor
from dirigo.sw_interfaces.acquisition import AcquisitionProduct
from dirigo.plugins.acquisitions import (
    LineAcquisitionSpec, LineAcquisition, 
    FrameAcquisitionSpec, FrameAcquisition
)


TWO_PI = 2 * np.pi

# issues:
# need to support uint16 and uint8 (rarer)
sigs = [
    #buffer_data    invert_mask  offset    bit_shift  gradient     resampled (out)  start_indices  nsamples_to_sum
    (int16[:,:,:],  int16[:],    int16[:], int32,     float32[:],  int16[:,:,:],    int32[:,:],    int32[:,:]),
    (uint16[:,:,:], int16[:],    int16[:], int32,     float32[:],  uint16[:,:,:],   int32[:,:],    int32[:,:])
]
@njit(sigs, parallel=True, fastmath=True, nogil=True, cache=True)
def resample_kernel(raw_data: np.ndarray, 
                    invert_mask: np.ndarray,
                    offset: np.ndarray,
                    bit_shift: int,
                    gradient: np.ndarray, # should this have seperate rows for fwd and rvs?
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
    Nrecords, Nsamples, Nchannels = raw_data.shape
    Ns, Nf, Nc = resampled.shape # Acquisition must be in channel-minor order (interleaved)
    Ndirections = start_indices.shape[0]
    

    # Clamp start_indices to [0, Nsamples); calculate final scaling factor
    scaling_factor = np.zeros_like(start_indices, np.float32)
    for d in prange(Ndirections):
        for fi in prange(Nf):
            start_indices[d, fi] = min(max(start_indices[d, fi], 0), Nsamples-1)
            scaling_factor[d, fi] = bit_shift * gradient[fi] / nsamples_to_sum[d, fi]
    
    # Main loop over slow-axis pixels
    for si in prange(Ns): 
        fwd_rvs = si % Ndirections # 0 => forward, 1 => reverse (if Ndirections=2)
        ri = si // Ndirections # record index 
        stride = 1 - 2 * fwd_rvs # +1 if forward, -1 if reverse

        # Loop over fast-axis pixels
        for fi in range(Nf): 
            Nsum = nsamples_to_sum[fwd_rvs, fi]
            sf = scaling_factor[fwd_rvs, fi]
            start = start_indices[fwd_rvs, fi]

            tmp_values = np.zeros(Nchannels, dtype=np.int32)

            # Step through the raw samples
            end = start + (Nsum * stride)
            for sample in range(start, end, stride):
                for c in range(Nchannels):
                    tmp_values[c] += raw_data[ri, sample, c] - offset[c]

            # Store the average back into resampled
            for c in range(Nchannels):
                resampled[si, fi, c] = invert_mask[c] * (sf * tmp_values[c]) 


sigs = [
    float32[:,:](uint16[:,:,:], int64, int64, float32),
    float32[:,:](int16[:,:,:],  int64, int64, float32)
]
@njit(sigs, nogil=True, parallel=True, fastmath=True, cache=True)
def crop_bidi_data(data: np.ndarray, lines_per_frame: int, trigger_delay: int, samples_per_period: float):
    """
    Select ranages, flips the reverse scan, and converts to float32 for phase
    correlation.

    Note: operates on channel 0 only.
    """

    # Largest power of two window
    n = 2**int(np.log2(samples_per_period/2 - 2*trigger_delay))

    # Find the nominal midpoints and endpoints of fwd and rvs scans
    mid_fwd = samples_per_period/4
    mid_rvs = mid_fwd + samples_per_period/2

    fwd0 = int(mid_fwd) - n//2 - trigger_delay
    fwd1 = int(mid_fwd) + n//2 - trigger_delay
    rvs0 = int(mid_rvs) - n//2 - trigger_delay
    rvs1 = int(mid_rvs) + n//2 - trigger_delay

    # Make fp32 output array
    out_array = np.zeros((lines_per_frame, n), dtype=np.float32)

    for rec in prange(lines_per_frame//2):
        out_array[2*rec, :] = data[rec, fwd0:fwd1, 0].astype(np.float32)
        out_array[2*rec+1, :] = data[rec, rvs1:rvs0:-1, 0].astype(np.float32) # flipped with the slicing

    return out_array
        

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


class RasterFrameProcessor(Processor):
    def __init__(self, 
                 upstream: LineAcquisition | FrameAcquisition, # and subclasses: FrameAcquisition, StackAcquisition, etc.
                 bits_precision: int = 16):
        """
        Initialize a raster frame processor worker for the Acquisition worker.

        To change default behavior of computing average for each pixel in 16-
        bit precision, change the bits_precision argument.
        """
        super().__init__(upstream)
        self._acq: LineAcquisition | FrameAcquisition
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
        try:
            n_lines = self._spec.lines_per_frame
        except:
            # fall-back when processing LineAcquisition as a frame
            n_lines = self._spec.lines_per_buffer
        
        n_channels = sum([c.enabled for c in digitizer_profile.channels])
        dt = np.int16 # TODO set this programatically
        self.processed_shape = (n_lines, self._spec.pixels_per_line, n_channels)

        self._invert_mask = -2 * np.array(
            [c.inverted for c in digitizer_profile.channels], dtype=dt
        ) + 1

        self.signal_offset = np.zeros_like(self._invert_mask)

        # Pre-allocate array for processed image
        self.init_product_pool(n=4, shape=self.processed_shape, dtype=dt)

        # Trigger timing
        self._fixed_trigger_delay = runtime_info.digitizer_trigger_delay

        if "galvo" in system_config.fast_raster_scanner['type'].lower():
            delay = units.Time(system_config.fast_raster_scanner['input_delay'])
            self._trigger_error = delay * self._sample_clock_rate
        elif "resonant" in system_config.fast_raster_scanner['type'].lower():
            # use a calibration table
            try:
                scanner_amplitude = runtime_info.scanner_amplitude
                ampls, freqs, phases = io.load_scanner_calibration()
                phase = np.interp(scanner_amplitude, ampls, phases)
                frequency = np.interp(scanner_amplitude, ampls, freqs)
                self._initial_trigger_error = \
                    (phase / TWO_PI) * (digitizer_profile.sample_clock.rate / frequency)
                #print("INITIAL TRIGGER PHASE", self._initial_trigger_error)
                self._trigger_error = self._initial_trigger_error

            except:
                # Calibration could not be loaded, don't use
                self._initial_trigger_error = None
                self._trigger_error = 0
        
        try:
            ampl = runtime_info.scanner_amplitude
            self._distortion_polynomial = io.load_distortion_calibration(ampl)
        except:
            self._distortion_polynomial = Polynomial([1])

        self._scaling_factor = 2 ** (
            self._bits_precision - runtime_info.digitizer_bit_depth
        )                   # should we bit shift instead of scale by multiply?

        # Try loading gradient calibration
        try:
            self._gradient = io.load_gradient_calibration()
        except:
            self._gradient = np.ones((self._spec.pixels_per_line,), np.float32)

        self._frames_processed = 0


    def run(self):
        
        while True: # Loops until receives sentinel None
            acq_prod: AcquisitionProduct = self.inbox.get() 

            if acq_prod is None: # Check for sentinel None
                self.publish(None) # pass along sentinel to indicate end
                print('Exiting processing thread ')
                return # concludes run() - this thread ends
            
            with acq_prod:
                proc_product = self.get_free_product() # Request a product object
                t0 = time.perf_counter()

                # If array of timestamps are assigned (default is None)
                if isinstance(acq_prod.timestamps, np.ndarray):
                    # Estimate scanner frequency from timestamps
                    avg_trig_period = np.mean(np.diff(acq_prod.timestamps))
                    self._fast_scanner_frequency = 1 / avg_trig_period
                    #print("FAST SCANNER FREQ", self._fast_scanner_frequency)

                # Measure phase from bidi data (in uni-directional, phase is not critical)
                if self._spec.bidirectional_scanning:
                    trigger_phase = self.measure_phase(acq_prod.data)
                    #print("TRIGGER SAMP", trigger_phase)

                    # quality check
                    if (self._initial_trigger_error is None 
                        or abs(trigger_phase - self._initial_trigger_error) < 10): 
                        self._trigger_error = trigger_phase
                        
                # Update resampling start indices--these can change a bit if the scanner frequency drifts
                start_indices = self.calculate_start_indices(self._trigger_error) - self._fixed_trigger_delay
                nsamples_to_sum = np.abs(np.diff(start_indices, axis=1))
                t1 = time.perf_counter()

                resample_kernel(
                    raw_data=acq_prod.data, 
                    invert_mask=self._invert_mask,
                    offset=self.signal_offset,
                    bit_shift=self._scaling_factor,
                    gradient=self._gradient,
                    resampled=proc_product.data,
                    start_indices=start_indices, 
                    nsamples_to_sum=nsamples_to_sum
                )
                proc_product.timestamps = acq_prod.timestamps
                proc_product.positions = acq_prod.positions
                proc_product.phase = TWO_PI * self._trigger_error \
                    / (self._acq.digitizer_profile.sample_clock.rate * avg_trig_period)
                proc_product.frequency = self._fast_scanner_frequency

                self.publish(proc_product) # sends off to Logger and/or Display workers
                self._frames_processed += 1
                t2 = time.perf_counter()

                #print(f"RECALC: {1000*(t1-t0):.03f} | FRAME {1000*(t2-t1):.03f} ")

    @cached_property
    def _temporal_edges(self):
        # Set up an array with the SPATIAL edges of pixels--we will bin samples into the pixel edges
        ff = self._spec.fill_fraction

        # Create a resampling function based on fast axis waveform type
        if 'resonant' in self._acq.system_config.fast_raster_scanner['type'].lower():
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
        """Calculate the start position in digitizer recrods for each pixel."""
        starts_exact = self._temporal_edges * self.samples_per_period + trigger_phase
        return np.ceil(starts_exact - 1e-6).astype(np.int32) 
    
    def measure_phase(self, data: np.ndarray) -> units.Angle:
        """
        Measure the apparent fast raster scanner trigger phase, in samples
        (for bidirectional scanning).
        """
        UPSAMPLE = 2 # TODO move this somewhere else
        
        data_window = crop_bidi_data(
            data=data, 
            lines_per_frame=self._spec.lines_per_frame, 
            trigger_delay=self._fixed_trigger_delay, 
            samples_per_period=self.samples_per_period
        )                                         # TODO preallocate data_window

        F = fft.rfft(data_window, axis=1, workers=4)
        xps = compute_cross_power_spectrum(F)

        n = data_window.shape[1] * UPSAMPLE
        corr = np.abs(fft.irfft(xps, n))

        shift = np.argmax(corr)
        if shift > (n//2):  # Handle wrap-around for negative shifts
            shift -= n

        return shift / UPSAMPLE / 2 - 1
    
    @cached_property
    def _sample_clock_rate(self) -> units.SampleRate:
        return self._acq.digitizer_profile.sample_clock.rate
    
    @property
    def samples_per_period(self) -> float:
        """
        The exact number of digitizer samples per fast raster scanner period.
        """
        return float(self._sample_clock_rate / self._fast_scanner_frequency)
    
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

