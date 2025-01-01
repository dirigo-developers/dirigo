import queue

import numpy as np
from numba import njit, prange, uint16, int32

import dirigo
from dirigo.sw_interfaces import Processor


class FrameProcessor(Processor):
    pass


# issues:
# need to support uint16 and uint8 (rarer)
# switch on bidi?
# switch on interleaved, multichannel?

#@njit((uint16[:,:], int32[:,:], uint16[:,:,:]), nogil=True, parallel=True, fastmath=True, cache=True) 
def dewarp_kernel(buffer_data, dewarped_frame_shape, start_indices, nsamples_to_sum):
    Nrecords,Nsamples = buffer_data.shape
    # Nf = Number Fast axis pixels (pixels per line)
    # Ns = Number Slow axis pixels (lines per frame)
    # Nc = Number of channels (colors)
    Ns, Nf, Nc = dewarped_frame_shape # Acquisition must be interleaved (channel-minor axis)
    dewarped = np.zeros(shape=dewarped_frame_shape, dtype=np.int16)

    # For non-bidi, start_indices is a single row; for bidi, it is two rows
    # For non-bidi, records per buffer is same as number of lines in final image
    # For bidid, there will be half as many records per buffer as lines
    Ndirections = start_indices.shape[0]


    for si in prange(Ns): # si = Slow pixel Index
        fwd_rvs = si % Ndirections # denotes whether this line is 'forward' or 'reverse' scan (unidirectional scanning is only 'forward')
        ri = si // Ndirections # record index (for unidirectional scanning ri=si)
        stride = Nc if fwd_rvs == 0 else -Nc

        for fi in prange(Nf): # fi = Fast pixel index
            Nsum = nsamples_to_sum[fwd_rvs, fi]

            if Nsum == 0: # No samples exist in pixel range
                continue # could interpolate a value here?

            i = start_indices[fwd_rvs, fi] * 2
            if (i < 0) or (i > Nsamples):   
                continue

          
        

class ResonantScanProcessor(Processor):
    def __init__(self, raw_queue, processed_queue):
        super().__init__(raw_queue, processed_queue) # privately stores queues available

    def run(self):
        # while loop
        try:
            data: np.ndarray = self._raw_queue.get_nowait() #
            if data is None:
                self.stop_task()
                return
            
            # processing kernel

        except queue.Empty:
            pass