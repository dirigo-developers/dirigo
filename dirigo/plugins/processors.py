import time
from functools import cached_property

import numpy as np
from numba import njit, prange, types
from scipy import fft

from dirigo import units
from dirigo.sw_interfaces.processor import Processor, ProcessedFrame
from dirigo.sw_interfaces.acquisition import Acquisition, AcquisitionBuffer
from dirigo.hw_interfaces.digitizer import Digitizer
from dirigo.plugins.acquisitions import FrameAcquisitionSpec


TWO_PI = 2 * np.pi

# issues:
# need to support uint16 and uint8 (rarer)
sigs = [
    #buffer_data          scaling_factor  resampled            start_indices     nsamples_to_sum
    (types.int16[:,:,:],  types.int32,    types.int16[:,:,:],  types.int32[:,:], types.int32[:,:]),
    (types.uint16[:,:,:], types.int32,    types.uint16[:,:,:], types.int32[:,:], types.int32[:,:])
]
@njit(sigs, parallel=True, fastmath=True, nogil=True, cache=True)
def resample_kernel(buffer_data: np.ndarray, 
                    scaling_factor: int, # faster/better to bit shift than multiply (scale)
                    resampled: np.ndarray, 
                    start_indices: np.ndarray, 
                    nsamples_to_sum: np.ndarray) -> np.ndarray:
    """
    buffer_data shape: (Nrecords, Nsamples, Nchannels)
    dewarped shape:    (Ns, Nf, Nc)
    start_indices shape: (Ndirections, Nf)
    nsamples_to_sum shape: (Ndirections, Nf)
    
    Ndirections is 1 (unidirectional) or 2 (bidirectional).
    """
    Nrecords, Nsamples, Nchannels = buffer_data.shape
    Ns, Nf, Nc = resampled.shape # Acquisition must be in channel-minor order (interleaved)
    Ndirections = start_indices.shape[0]

    # Clamp start_indices so they don't go out of [0, Nsamples]    
    for d in prange(Ndirections):
        for fi in prange(Nf):
            start_indices[d, fi] = min(max(start_indices[d, fi], 0), Nsamples)
    
    # Main loop over slow-axis pixels
    for si in prange(Ns): 
        fwd_rvs = si % Ndirections # 0 => forward, 1 => reverse (if Ndirections=2)
        ri = si // Ndirections # record index 
        stride = 1 - 2 * fwd_rvs # +1 if forward, -1 if reverse

        # Loop over fast-axis pixels
        for fi in range(Nf): 
            Nsum = nsamples_to_sum[fwd_rvs, fi]
            start = start_indices[fwd_rvs, fi]

            tmp_values = np.zeros(Nchannels, dtype=np.int32)

            # Step through the raw samples
            end = start + (Nsum * stride)
            for sample in range(start, end, stride):
                for c in range(Nchannels):
                    tmp_values[c] += buffer_data[ri, sample, c]

            # Store the average back into resampled
            for c in range(Nchannels):
                resampled[si, fi, c] = (scaling_factor * tmp_values[c]) // Nsum

    return resampled


sigs = [
    types.float32[:,:](types.uint16[:,:,:], types.int64, types.int64, types.float64),
    types.float32[:,:](types.int16[:,:,:],  types.int64, types.int64, types.float64)
]
@njit(sigs, nogil=True, parallel=True, fastmath=True, cache=True)
def crop_bidi_data(data: np.ndarray, lines_per_frame: int, trigger_delay:int, samples_per_period: float):
    """Select ranages and convert to float32.

    Note: operates on channel 0 only.
    """
    Nrec, Ns, Nc = data.shape # records, samples, channels

    # Largest power of two window
    n = 2**int(np.log2(Ns / 2 - trigger_delay))

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
    types.complex64[:](types.complex64[:,:]),
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
                 acquisition: Acquisition,
                 bits_precision: int = 16):
        """
        Initialize a raster frame processor worker for the Acquisition worker.

        To change default behavior of computing average for each pixel in 16-
        bit precision, change the bits_precision argument.
        """
        super().__init__(acquisition)

        if (not isinstance(bits_precision, int)) or (bits_precision % 2) or not (8 <= bits_precision <= 16):
            raise ValueError("Bits precision must be even integer between 8 and 16")
        elif bits_precision < self._acq.data_acquisition_device.bit_depth:
            raise ValueError("Bit precision can't be less than the data acquisition device bit depth.")
        self._bits_precision = bits_precision
        
        self._spec: FrameAcquisitionSpec # to refine type hinting
        if isinstance(acquisition.data_acquisition_device, Digitizer):
            n_channels = sum(
                [c.enabled for c in acquisition.data_acquisition_device.channels]
            )
        else:
            raise NotImplementedError(
                f"RasterFrameProcessor implemented for use with Digitizer, "
                f"got type: {type(acquisition.data_acquisition_device)}"
            )
        self.processed_shape = ( # Final shape after processing
                self._spec.lines_per_frame,
                self._spec.pixels_per_line,
                n_channels
        )
        
        # Initialize with the nominal rate, will measure with data timestamps
        self._fast_scanner_frequency = self._acq.hw.fast_raster_scanner.frequency

        # Pre-allocate array for processed image
        if self._acq.data_acquisition_device.data_range.min < 0:
            # If the data range min is negative, then we will need SIGNED integer data
            self.processed = np.zeros(self.processed_shape, dtype=np.int16)
        else:
            self.processed = np.zeros(self.processed_shape, dtype=np.uint16)

        # Trigger timing
        self._fixed_trigger_delay = self._acq.hw.digitizer.acquire.trigger_delay_samples
        self._trigger_phase = 0 # This value can be adjusted
        if hasattr(self._acq.hw.fast_raster_scanner, 'input_delay'):
            self._trigger_phase = self._acq.hw.fast_raster_scanner.input_delay * self._acq.hw.digitizer.sample_clock.rate

        self._scaling_factor = 2**(self._bits_precision - self._acq.data_acquisition_device.bit_depth) # should we bit shift instead of scale with multiply?

        # for testing
        #self.calculate_start_indices(self._trigger_phase)
        
    def run(self):
        # Calculate preliminary start indices
        start_indices = self.calculate_start_indices(self._trigger_phase) - self._fixed_trigger_delay
        nsamples_to_sum = np.abs(np.diff(start_indices, axis=1))
        
        while True: # Loops until receives sentinel None
            buf: AcquisitionBuffer = self.inbox.get(block=True) # we may want to add a timeout

            if buf is None: # Check for sentinel None
                self.publish(None) # pass along sentinel to indicate end
                print('Exiting processing thread')
                return # concludes run() - this thread ends

            t0 = time.perf_counter()
            resample_kernel(buf.data, self._scaling_factor, self.processed, start_indices, nsamples_to_sum)
            
            self.publish(ProcessedFrame(
                data=self.processed, 
                timestamps=buf.timestamps,
                positions=buf.positions
            )) # sends off to Logger and/or Display workers
            t1 = time.perf_counter()
            
            print(f"{self.native_id} Processed a frame in {1000*(t1-t0):.02f} ms")

            # If timestamps are assigned (default is None)
            if isinstance(buf.timestamps, np.ndarray):
                # Measure frequency from timestamps
                self._fast_scanner_frequency = 1 / np.mean(np.diff(buf.timestamps))

            # Measure phase from bidi data (in uni-directional, phase is not critical)
            if self._spec.bidirectional_scanning:
                self._trigger_phase = self.measure_phase(buf.data)

            # Update resampling start indices--these can change a bit if the scanner frequency drifts
            start_indices = self.calculate_start_indices(self._trigger_phase) - self._fixed_trigger_delay
            nsamples_to_sum = np.abs(np.diff(start_indices, axis=1))


    def calculate_start_indices(self, trigger_phase: int = 0):
        # Set up an array with the SPATIAL edges of pixels--we will bin samples into the pixel edges
        ff = self._spec.fill_fraction

        # Create a resampling function based on fast axis waveform type
        waveform = self._acq.hw.fast_raster_scanner.waveform
        if waveform == "sinusoid": # resonant scanner
            # image line should be taken from center of sinusoid sweep
            pixel_edges = np.linspace(ff, -ff, self._spec.pixels_per_line + 1)

            # arccos inverts the cosinusoidal path, normalize scan period to 0.0 to 1.0
            temporal_edges = np.arccos(pixel_edges) / TWO_PI

        elif waveform == "sawtooth":
            pass # polygon scanner
        elif waveform == "triangle":
            pass # galvo running birectional
        elif waveform == "asymmetric triangle": #TODO, rename to smooth triangle or something
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

        # TODO compute only once: Up to here is completely the same each iteration 
        
        starts_exact = temporal_edges * self.samples_per_period + trigger_phase

        start_indices = np.ceil(starts_exact - 1e-6).astype(np.int32) 

        return start_indices
    
    def measure_phase(self, data: np.ndarray) -> units.Angle:
        """Measure the apparent fast raster scanner trigger phase for bidirectional acquisitions."""
        UPSAMPLE = 8 # TODO move this somewhere else
        
        data_window = crop_bidi_data(data, self._spec.lines_per_frame, self._acq.hw.digitizer.acquire.trigger_delay_samples, self.samples_per_period) # todo preallocate data_window

        F = fft.rfft(data_window, axis=1, workers=4)
        xps = compute_cross_power_spectrum(F)

        n = data_window.shape[1] * UPSAMPLE
        corr = np.abs(fft.irfft(xps, n))

        shift = np.argmax(corr)
        print("N", n, "SHIFT", shift)
        if shift > (n//2):  # Handle wrap-around for negative shifts
            shift -= n

        print(f"Estimated shift (samples): {shift / UPSAMPLE}")

        #return phase
        return shift / UPSAMPLE / 2
    
    @cached_property
    def _sample_clock_rate(self) -> units.SampleRate:
        return self._acq.hw.digitizer.sample_clock.rate
    
    @property
    def samples_per_period(self) -> float:
        """
        The exact number of digitizer samples per fast raster scanner period.
        """
        return self._sample_clock_rate / self._fast_scanner_frequency 
    
    @property
    def data_range(self):
        """
        The data range after processing (resampling) has been performed.

        May be higher than the native bit depth of the data capture device.
        """
        if self._acq.data_acquisition_device.data_range.min < 0:
            return units.ValueRange(
                min=-2**(self._bits_precision-1), 
                max=2**(self._bits_precision-1) - 1
            )
        else: # unsigned data
            return units.ValueRange(
                min=0, 
                max=2**self._bits_precision - 1
            )
