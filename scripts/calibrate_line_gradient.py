from dirigo.main import Dirigo
from dirigo.plugins.acquisitions import FrameAcquisition


"""
To measure signal gradient across a line, place a vat of fluorescence dye under
the microscope. The script will start a RasterFrameAcquisition, and the average
line signal is fit to a polynomial.
"""


diri = Dirigo()

# Use default FrameAcquisition spec as basis for frame size, pixel size, etc
spec = FrameAcquisition.get_specification()
# strink frame height, so acquisition is essentially a line
spec.frame_height = spec.line_width / 10 
spec.pixel_height = spec.pixel_size / 10


acquisition = diri.make("acquisition", "raster_frame", spec=spec)
processor   = diri.make("processor", "raster_frame", upstream=acquisition)
logger      = diri.make("logger", "line_gradient_calibration", upstream=processor)

acquisition.start()
logger.join()

