from datetime import datetime

import numpy as np
import tifffile

from dirigo.sw_interfaces import Logger, Acquisition, Processor




class TiffLogger(Logger):

    def __init__(self, acquisition: Acquisition = None, processor: Processor = None):
        super().__init__(acquisition, processor)

        self.append = True
        self.save_path = "C:/github/data/"
        self._writer = tifffile.TiffWriter(self.save_path + "append.tif") \
            if self.append else None

    def run(self):
        try:
            while True:
                data: np.ndarray = self.inbox.get(block=True)

                if data is None: # Check for sentinel None
                    self.publish(None) # pass sentinel
                    print('Exiting TiffLogger thread')
                    return # thread ends
                
                self.save_data(data)

        finally:
            if self._writer:
                self._writer.close()
                self._writer = None


    def save_data(self, data):
        """Save the data to a TIFF file."""
        if self.append:
            # Append to an existing multi-page TIFF file
            self._writer.write(
                data, 
                photometric='minisblack',
                planarconfig='contig',
                contiguous=True # The dataset size must not change
            )
        else:
            # Save as a new file (each frame is a separate file)
            file_name = self._generate_filename()
            tifffile.imwrite(
                self.save_path + file_name, 
                data, 
                photometric='minisblack',
                planarconfig='contig' 
            )
    
    def _generate_filename(self) -> str:
        """Generate a unique filename for new files."""
        # Use a timestamp or sequential naming for uniqueness
        return f"{datetime.now().strftime('%Y%m%d_%H%M%S%f')}.tif"