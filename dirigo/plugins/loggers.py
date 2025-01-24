from functools import cached_property
import base64
import json

import tifffile
import numpy as np
from platformdirs import user_documents_path

from dirigo.sw_interfaces.processor import Processor, ProcessedFrame
from dirigo.sw_interfaces import Logger, Acquisition
from dirigo.plugins.acquisitions import FrameAcquisitionSpec



class TiffLogger(Logger):

    def __init__(self, acquisition: Acquisition = None, processor: Processor = None):
        super().__init__(acquisition, processor)

        self.frames_per_file = 10 # add validation
        self.basename = "experiment"
        self.save_path = user_documents_path() / "Dirigo"

        self._fn = None
        self._writer = None # generated upon attempt to save first frame
        self.frames_saved = 0
        self.files_saved = 0

        self.timestamps = []
        self.positions = []
        self.blank_metadata = None
         
    def run(self):
        try:
            while True:
                frame: ProcessedFrame = self.inbox.get(block=True)

                if frame is None: # Check for sentinel None
                    self.publish(None) # pass sentinel
                    print('Exiting TiffLogger thread')
                    return # thread ends
                
                self.save_data(frame)

        finally:
            self._close_and_write_metadata()

    def save_data(self, frame: ProcessedFrame):
        """Save data and metadata to a TIFF file"""

        # Create the writer object if necessary
        if self._writer is None:
            
            if self.frames_per_file == float('inf'):
                # if we are writing an indeterminately long file, defer saving metadata
                self.blank_metadata = None
            else:
                # Generate some blank metadata to overwrite later
                if frame.timestamps is not None:
                    # For line by line timestamps
                    timestamps_shape = (self.frames_per_file,) + frame.timestamps.shape
                    timestamps = np.zeros(timestamps_shape, dtype=np.float64)
                if frame.positions is not None:
                    pass
                self.blank_metadata = {
                    'timestamps': base64.b64encode(timestamps.tobytes()).decode('ascii'),
                }
                
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
                metadata=self.blank_metadata,
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

            if self.frames_per_file == float('inf'):
                # If the total metadata size is not known a priori, skip
                # TODO, re-write the file
                pass
            else:
                # Re-open and overwite blank metadata
                serialized_timestamps = base64.b64encode(np.array(self.timestamps).tobytes()).decode('ascii')
                with tifffile.TiffFile(self._fn, mode="r+b") as tif:
                    tif.pages[0].tags['ImageDescription'].overwrite(
                        json.dumps({'timestamps': serialized_timestamps})
                    )

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

        pixel_height_inches = ((spec.pixel_height * 1000) / 25.4)
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

