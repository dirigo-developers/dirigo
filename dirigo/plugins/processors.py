import time
from functools import cached_property

import numpy as np
from numba import njit, prange, types
from scipy import fft

from dirigo import units
from dirigo.hw_interfaces import digitizer
from dirigo.sw_interfaces import Processor
from dirigo.plugins.acquisitions import FrameAcquisitionSpec


TWO_PI = 2 * np.pi

# issues:
# need to support uint16 and uint8 (rarer)

@njit((types.uint16[:,:,:], types.uint16[:,:,:], types.int32[:,:], types.int32[:,:]),
      nogil=True, parallel=True, fastmath=True, cache=True)
def dewarp_kernel(buffer_data: np.ndarray, dewarped: np.ndarray, 
                  start_indices: np.ndarray, nsamples_to_sum: np.ndarray) -> np.ndarray:
    Nrecords, Nsamples, Nchannels = buffer_data.shape
    # Nf = Number Fast axis pixels (pixels per line)
    # Ns = Number Slow axis pixels (lines per frame)
    # Nc = Number of channels (colors)
    Ns, Nf, Nc = dewarped.shape # Acquisition must be in channel-minor order (interleaved)
    
    # For non-bidi, start_indices is a single row; for bidi, it is two rows
    # For non-bidi, records per buffer is same as number of lines in final image
    # For bidid, there will be half as many records per buffer as lines
    Ndirections = start_indices.shape[0]
    
    for si in prange(Ns): # si = Slow pixel Index
        fwd_rvs = si % Ndirections # denotes 'forward' (0) or 'reverse' (1) scan (unidirectional scanning is only 'forward')
        ri = si // Ndirections # record index (for unidirectional scanning ri=si)
        stride = 1 - 2 * fwd_rvs # for stride in range later, 0 -> 1 and 1 -> -1 

        for fi in prange(Nf): # fi = Fast pixel index
            Nsum = nsamples_to_sum[fwd_rvs, fi]
            start = start_indices[fwd_rvs, fi]

            #Requesting indices outside sampled data: skip
            if (start < 0) or (start > Nsamples):
                dewarped[si, fi, 0] = 0
                dewarped[si, fi, 1] = 0
                continue

            tmp0 = types.int32(0)
            tmp1 = types.int32(0)
            for sample in range(start, start + Nsum * stride, stride):
                tmp0 += buffer_data[ri, sample, 0] # unrolled channels
                tmp1 += buffer_data[ri, sample, 1]

            dewarped[si, fi, 0] = tmp0 // Nsum
            dewarped[si, fi, 1] = tmp1 // Nsum

    return dewarped



@njit(
    types.float32[:,:](types.uint16[:,:,:], types.int64, types.int64, types.float64), 
    nogil=True, parallel=True, fastmath=True, cache=True
)
def window_bidi_data(data: np.ndarray, lines_per_frame: int, trigger_delay:int, samples_per_period: float):
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
    def __init__(self, acquisition):
        super().__init__(acquisition)
        self._spec: FrameAcquisitionSpec # to refine type hinting
        self.dewarped_shape = (
                self._spec.lines_per_frame,
                self._spec.pixels_per_line,
                self._spec.nchannels
            )
        
        # Initialize with the nominal rate, will measure with data timestamps
        self._fast_scanner_frequency = self._acq.hw.fast_raster_scanner.frequency

        # Pre-allocate array for dewarped image
        self.dewarped = np.zeros(self.dewarped_shape, dtype=np.uint16)
        
    def run(self):
        trigger_delay = self._acq.hw.digitizer.acquire.trigger_delay_samples # not adjustable during acquisition
        trigger_phase = 0
        start_indices = self.calculate_start_indices(trigger_phase) - trigger_delay
        nsamples_to_sum = np.abs(np.diff(start_indices, axis=1))
        
        while True: # Loops until receives sentinel None
            buf: digitizer.DigitizerBuffer = self.inbox.get(block=True) # we may want to add a timeout

            if buf is None: # Check for sentinel None
                self.publish(None) # pass along sentinel to indicate end
                print('Exiting processing thread')
                return # concludes run() - this thread ends

            # if buf.positions is not None:
            #     print(f'processor got positions: {buf.positions[0]}')

            t0 = time.perf_counter()
            dewarp_kernel(buf.data, self.dewarped, start_indices, nsamples_to_sum)
            self.publish(self.dewarped) # sends off to Logger or Display workers
            t1 = time.perf_counter()
            
            #print(f"{self.native_id} [max sum: {np.max(nsamples_to_sum)}] Processed a frame in {1000*(t1-t0):.02f} ms")

            # If timestamps are assigned (default is None)
            if isinstance(buf.timestamps, np.ndarray):
                # Measure frequency from timestamps
                self._fast_scanner_frequency = 1 / np.mean(np.diff(buf.timestamps))

            # Measure phase from bidi data (in uni-directional, phase is not critical)
            if self._spec.bidirectional_scanning:
                trigger_phase = self.measure_phase(buf.data)

            # Update dewarping start indices
            start_indices = self.calculate_start_indices(trigger_phase) - trigger_delay
            nsamples_to_sum = np.abs(np.diff(start_indices, axis=1))


    def calculate_start_indices(self, trigger_phase: int = 0):
        # Set up an array with the SPATIAL edges of pixels--we will bin samples into the pixel edges
        ff = self._spec.fill_fraction
        pixel_edges = np.linspace(ff, -ff, self._spec.pixels_per_line + 1)

        # Create a dewarping function based on fast axis waveform type
        waveform = self._acq.hw.fast_raster_scanner.waveform
        if waveform == "sinusoid":
            # arccos inverts the cosinusoidal path, normalize scan period to 0.0 to 1.0
            temporal_edges = np.arccos(pixel_edges) / TWO_PI
        elif waveform == "sawtooth":
            pass # like a polygon scanner
        elif waveform == "triangle":
            pass # galvo running birectional
        elif waveform == "asymmetric triangle":
            pass # galvo normal raster

        if self._spec.bidirectional_scanning:
            temporal_edges_fwd = temporal_edges
            temporal_edges_rvs = 1.0 - temporal_edges
            temporal_edges = np.vstack([temporal_edges_fwd, temporal_edges_rvs]) 
        else:
            temporal_edges = np.vstack([temporal_edges])

        # TODO compute only once: Up to here is completely the same each iteration 
        
        starts_exact = temporal_edges * self.samples_per_period + trigger_phase

        start_indices = np.ceil(starts_exact).astype(np.int32) # Forward scan should be ceil, Reverse scan should be floor

        return start_indices
    
    def measure_phase(self, data: np.ndarray) -> units.Angle:
        """Measure the apparent fast raster scanner trigger phase for bidirectional acquisitions."""
        UPSAMPLE = 8
        
        data_window = window_bidi_data(data, self._spec.lines_per_frame, self._acq.hw.digitizer.acquire.trigger_delay_samples, self.samples_per_period) # todo preallocate data_window

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
        """The exact number of digitizer samples per fast raster scanner period.
        """
        return self._sample_clock_rate / self._fast_scanner_frequency 
