from functools import cached_property
from base64 import b64encode
import json
from pathlib import Path

import tifffile
import numpy as np
from scipy import fft
from platformdirs import user_config_dir

from dirigo.sw_interfaces.processor import Processor, ProcessorProduct
from dirigo.sw_interfaces import Logger
from dirigo.sw_interfaces.acquisition import Acquisition, AcquisitionProduct
from dirigo.plugins.acquisitions import FrameAcquisitionSpec, BidiCalibration



class TiffLogger(Logger):

    def __init__(self, 
                 upstream: Acquisition | Processor,
                 max_frames_per_file: int = 1,
                 **kwargs):
        super().__init__(upstream, **kwargs)

        self.frames_per_file = max_frames_per_file # TODO add validation
        
        self._fn = None
        self._writer = None # generated upon attempt to save first frame
        self.frames_saved = 0
        self.files_saved = 0

        self.timestamps = []
        self.positions = []
        self.metadata = None
         
    def run(self):
        try:
            while True:
                prod: AcquisitionProduct | ProcessorProduct = self.inbox.get(block=True)

                if prod is None: # Check for sentinel None
                    self.publish(None) # pass sentinel
                    print('Exiting TiffLogger thread')
                    return # thread ends
                
                self.save_data(prod)
                prod._release()

        finally:
            self._close_and_write_metadata()

    def save_data(self, frame: AcquisitionProduct | ProcessorProduct):
        """Save data and metadata to a TIFF file"""

        # Create the writer object if necessary
        if self._writer is None:
            self.metadata = {}
            if self.frames_per_file == float('inf'):
                # if we are writing an indeterminately long file, defer saving metadata
                pass
            else:
                # Otherwise, if we can predict the metadata size a priori, 
                # generate some blank (serialized) metadata to overwrite later
                if frame.timestamps is not None:
                    # For multi-line timestamps
                    timestamps_shape = (self.frames_per_file,) + frame.timestamps.shape
                    timestamps = np.zeros(timestamps_shape, dtype=np.float64)
                    self.metadata['timestamps'] = b64encode(timestamps.tobytes()).decode('ascii')

                if frame.positions is not None:
                    if isinstance(frame.positions, np.ndarray):
                        positions_shape = (self.frames_per_file,) + frame.positions.shape
                    else:
                        positions_shape = (self.frames_per_file, len(frame.positions))
                    positions = np.zeros(positions_shape, dtype=np.float64)
                    self.metadata['positions'] = b64encode(positions.tobytes()).decode('ascii')

            self._fn = self.save_path / f"{self.basename}_{self.files_saved}.tif"
            self._writer = tifffile.TiffWriter(self._fn, bigtiff=self._use_big_tiff)

        # Accumulate metadata
        self.timestamps.append(frame.timestamps)
        self.positions.append(frame.positions)

        self._writer.write(
                frame.data, 
                photometric='minisblack',
                planarconfig='contig',
                resolution=(self._x_dpi, self._y_dpi),
                metadata=self.metadata,
                contiguous=True # The dataset size must not change
            )
        self.frames_saved += 1

        # when number of frames per file reached, close writer & write metadata
        if self.frames_saved % self.frames_per_file == 0:
            self._close_and_write_metadata()
           
    def _close_and_write_metadata(self):
        if self._writer:
            self._writer.close()
            self._writer = None
            self.files_saved += 1

            metadata = {}
            if self.frames_per_file == float('inf'):
                # If the total metadata size is not known a priori, skip
                # TODO, re-write the file
                pass
            else:
                # Re-open and overwite blank metadata
                if self.timestamps[0] is not None:
                    metadata['timestamps'] = b64encode(np.array(self.timestamps).tobytes()).decode('ascii')
                if self.positions[0] is not None:
                    metadata['positions'] = b64encode(np.array(self.positions).tobytes()).decode('ascii')

                with tifffile.TiffFile(self._fn, mode="r+b") as tif:
                    tif.pages[0].tags['ImageDescription'].overwrite(json.dumps(metadata))

            # Clear accumulants
            self.timestamps = []
            self.positions = []

    @cached_property
    def _use_big_tiff(self) -> bool:
        """Returns False (don't use BigTiff) when frames per file is 1.
        
        This strategy was chosen because little disadvantage to using BigTiff.
        Subclass and overwrite to provide different logic
        """
        if self.frames_per_file == 1:
            return False
        else:
            return True 
        
    # This uses the acquisition or processor references to retrieve resolution
    # An alternative would be to pass resolution (and other metadata) in the queue
    @property
    def _fast_axis_dpi(self) -> float:
        acq = self._acquisition if self._acquisition else self._processor._acq
        spec: FrameAcquisitionSpec = acq.spec
        
        pixel_width_inches = ((spec.pixel_size * 1000) / 25.4)
        return 1 / pixel_width_inches
        
    @property
    def _slow_axis_dpi(self) -> float:
        acq = self._acquisition if self._acquisition else self._processor._acq
        spec: FrameAcquisitionSpec = acq.spec

        if hasattr(spec, 'pixel_height'):
            pixel_height = spec.pixel_height
        else:
            # fallback in case we are processing LineAcquisition data
            pixel_height = spec.pixel_size
        pixel_height_inches = ((pixel_height * 1000) / 25.4)
        return 1 / pixel_height_inches
    
    @cached_property
    def _x_dpi(self) -> float:
        acq = self._acquisition if self._acquisition else self._processor._acq
        fast_axis =  acq.hw.fast_raster_scanner.axis
        return self._fast_axis_dpi if fast_axis == 'x' else self._slow_axis_dpi
    
    @cached_property
    def _y_dpi(self) -> float:
        acq = self._acquisition if self._acquisition else self._processor._acq
        slow_axis =  acq.hw.slow_raster_scanner.axis
        return self._slow_axis_dpi if slow_axis == 'y' else self._fast_axis_dpi


class BidiCalibrationLogger(Logger):
    """Logs bidirecetional phase at amplitudes."""
    def __init__(self, upstream: Processor):
        super().__init__(upstream)

        self._amplitudes = []
        self._frequencies = []
        self._phases = []
        
    def run(self):
        try:
            while True:
                product: ProcessorProduct = self.inbox.get(block=True)
                if product is None: return # Check for sentinel None

                with product:
                    self._amplitudes.append(
                        self._acquisition.hw.fast_raster_scanner.amplitude
                    )
                    self._frequencies.append(product.frequency)
                    self._phases.append(product.phase)
                    
        finally:
            self.publish(None) # pass sentinel
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
            Path(user_config_dir('Dirigo')) / "scanner/calibration.csv",
            data,
            delimiter=',',
            header='amplitude (rad),frequency(Hz),phase (rad)',
            comments=''    # prevent numpy from prefixing "#" on header lines
        )




class FrameSizeCalibrationLogger(Logger):
    """Logs apparent translation."""
    def __init__(self, upstream: Processor):
        super().__init__(upstream)

    def run(self):
        try:
            # Get reference frame
            product: ProcessorProduct = self.inbox.get(block=True)
            if product is None: return # Check for sentinel None
            with product:
                self._ref_frame = product.data.copy()

            # Get translated frame
            product: ProcessorProduct = self.inbox.get(block=True)
            if product is None: return # Check for sentinel None
            with product:
                self._translated_frame = product.data.copy()

        finally:
            self.publish(None) # pass sentinel
            self.save_data()

    def save_data(self):
        UPSAMPLE = 1
        EPS = 1e-1  # Small constant

        # calculate cross correlation
        ref = self._ref_frame[:,:,0]
        mov = self._translated_frame[:,:,0]

        xps = fft.rfft2(ref) * np.conj(fft.rfft2(mov))

        s = (ref.shape[0] * UPSAMPLE, ref.shape[1] * UPSAMPLE)
        corr = fft.irfft2(xps / np.abs(xps), s)

        arg_max = np.argmax(corr)
        i = arg_max // corr.shape[1]
        j = arg_max %  corr.shape[1]

        if i > (s[0] // 2):  # Handle wrap-around for negative shifts
            i -= s[0]
        if j > (s[1] // 2): 
            j -= s[1]

        data = np.array([[0,i,j]])

        # write with a header comment for units
        np.savetxt(
            Path(user_config_dir('Dirigo')) / "scanner/frame_calibration.csv",
            data,
            delimiter=',',
            header='frame_size (m),i,j',
            comments=''    # prevent numpy from prefixing "#" on header lines
        )