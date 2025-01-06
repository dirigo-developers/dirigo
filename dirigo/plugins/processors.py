import numpy as np
from numba import njit, prange, types

from dirigo import units
from dirigo.sw_interfaces import Processor
from dirigo.plugins.acquisitions import FrameAcquisitionSpec


TWO_PI = 2 * np.pi

# issues:
# need to support uint16 and uint8 (rarer)

@njit(
    types.uint16[:,:,:](
        types.uint16[:,:,:], 
        types.UniTuple(types.int64, 3), 
        types.int32[:,:], 
        types.int32[:,:]
    ),
    nogil=True, 
    parallel=True, 
    fastmath=True, 
    cache=True
)
def dewarp_kernel(
    buffer_data: np.ndarray, 
    dewarped_frame_shape: tuple, 
    start_indices: np.ndarray, 
    nsamples_to_sum: np.ndarray
) -> np.ndarray:
    
    Nrecords, Nsamples, Nchannels = buffer_data.shape
    # Nf = Number Fast axis pixels (pixels per line)
    # Ns = Number Slow axis pixels (lines per frame)
    # Nc = Number of channels (colors)
    Ns, Nf, Nc = dewarped_frame_shape # Acquisition must be in channel-minor order (interleaved)
    if Nchannels != Nc:
        raise ValueError("Dimension error: buffer channels does not equal dewarped shape channels")
    
    dewarped = np.zeros(shape=dewarped_frame_shape, dtype=np.uint16) # Is this OK to hardcode to 16-bit?

    # For non-bidi, start_indices is a single row; for bidi, it is two rows
    # For non-bidi, records per buffer is same as number of lines in final image
    # For bidid, there will be half as many records per buffer as lines
    Ndirections = start_indices.shape[0]

    for si in prange(Ns): # si = Slow pixel Index
        fwd_rvs = si % Ndirections # denotes whether this line is 'forward' or 'reverse' scan (unidirectional scanning is only 'forward')
        ri = si // Ndirections # record index (for unidirectional scanning ri=si)
        stride = 1 if fwd_rvs == 0 else -1

        for fi in prange(Nf): # fi = Fast pixel index
            Nsum = nsamples_to_sum[fwd_rvs, fi]

            if Nsum == 0: # No samples exist in pixel range
                continue # could interpolate a value here?

            i = start_indices[fwd_rvs, fi]

            if (i < 0) or (i > Nsamples): # Requesting indices outside sampled data
                continue

            if Nsum == 1: # if there is only one sample in the pixel range, then set dewarped value directly
                for ci in range(Nc):
                    dewarped[si, fi, ci] = buffer_data[ri, i, ci] #- 2**15
            else: # average a number of pixels
                for ci in range(Nc):
                    tmp = types.int32(0)
                    for sample in range(0, Nsum * stride, stride):
                        tmp += buffer_data[ri, i + sample, ci]
                    dewarped[si, fi, ci] = tmp // Nsum #- 2**15

    return dewarped
        

class RasterFrameProcessor(Processor):
    def __init__(self, acquisition):
        super().__init__(acquisition)
        self._spec: FrameAcquisitionSpec # to refine type hinting
        self.dewarped_shape = (
                self._spec.lines_per_frame,
                self._spec.pixels_per_line,
                self._spec.nchannels
            )
        
    def run(self):
        while True: # Loops until receives sentinel None
            data: np.ndarray = self.inbox.get(block=True) # we may want to add a timeout

            if data is None: # Check for sentinel None
                # execute any cleanup code here
                self.publish(None) # pass sentinel
                print('exiting processing thread')
                return # concludes run() - this thread ends         

            start_indices = self.calculate_start_indices() - self._acq.hw.digitizer.acquire.trigger_delay_samples
            nsamples_to_sum = np.abs(np.diff(start_indices, axis=1)) # put inside kernel

            dewarped = dewarp_kernel(data, self.dewarped_shape, start_indices, nsamples_to_sum)
            print(f"{self.native_id} Processed a frame")

            self.publish(dewarped)


    def calculate_start_indices(self, trigger_phase: units.Angle = units.Angle(0.0)):
        ff = self._spec.fill_fraction
        pixel_edges = np.linspace(ff, -ff, self._spec.pixels_per_line + 1)

        # Create a dewarping function based on fast axis waveform type
        waveform = self._acq.hw.fast_raster_scanner.waveform
        if waveform == "sinusoid":
            # arccos dewarps the sinusoidal path
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
            #temporal_edges_rvs = np.roll(temporal_edges_rvs, shift=-1) # why?
            temporal_edges = np.vstack([temporal_edges_fwd, temporal_edges_rvs]) 
        else:
            temporal_edges = np.vstack([temporal_edges])

        period = self._acq.hw.digitizer.sample_clock.rate \
            / self._acq.hw.fast_raster_scanner.frequency # best guess. TODO: use record timestamps
        
        starts_exact = (temporal_edges + trigger_phase / TWO_PI) * period 

        start_indices = np.ceil(starts_exact).astype(np.int32)

        return start_indices