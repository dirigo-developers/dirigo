from functools import cached_property

import tifffile
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

        self._writer = None # generated upon attempt to save first frame
        self.frames_saved = 0
        self.files_saved = 0
         
    def run(self):
        try:
            while True:
                buf: ProcessedFrame = self.inbox.get(block=True)

                if buf is None: # Check for sentinel None
                    self.publish(None) # pass sentinel
                    print('Exiting TiffLogger thread')
                    return # thread ends
                
                self.save_data(buf.data)

        finally:
            if self._writer:
                self._writer.close()
                self._writer = None

    def save_data(self, data):
        """Save data to a TIFF file"""
        if self._writer is None:
            fn = self.save_path / f"{self.basename}_{self.files_saved}.tif"
            self._writer = tifffile.TiffWriter(fn, bigtiff=self._use_big_tiff)

        self._writer.write(
                data, 
                photometric='minisblack',
                planarconfig='contig',
                resolution=(self._x_dpi, self._y_dpi),
                contiguous=True # The dataset size must not change
            )
        self.frames_saved += 1

        if self.frames_saved % self.frames_per_file == 0:
            # when reaching number of frames per file, close writer
            self._writer.close()
            self._writer = None
            self.files_saved += 1

    @cached_property
    def _use_big_tiff(self) -> bool:
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

