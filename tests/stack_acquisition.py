from dirigo.main import Dirigo
from dirigo import units, io
from dirigo.plugins.acquisitions import StepAndHoldStackAcquisition, AxialDirection


MICROSTEP = units.Position("0.1905 um") # Zaber X-LSMXXXB stage

# Make spec
spec = StepAndHoldStackAcquisition.get_specification() # Get the base spec as template
spec.bidirectional_scanning = False
spec.line_width = units.Position("1 mm")
spec.frame_height = units.Position("1 mm")
spec.pixel_size = round(units.Position("0.5 um")/MICROSTEP)*MICROSTEP
spec.pixel_height = spec.pixel_size
spec.z_range = units.PositionRange(
    min=round(units.Position("-30 um")/spec.pixel_size)*spec.pixel_size,
    max=round(units.Position("70 um")/spec.pixel_size)*spec.pixel_size
)
spec.depth_spacing = spec.pixel_size # ~isotropic sampling
spec.published_frames_per_step = 1
spec.discarded_frames_per_step = 1
spec.z_direction = AxialDirection.NEGATIVE



# Set up dirigo and run acquisition
diri = Dirigo()


acquisition = diri.make_acquisition(
    name = "step_stack", 
    spec = spec)
processor = diri.make_processor(
    name = "raster_frame", 
    upstream = acquisition
)
averager = diri.make_processor(
    name = "rolling_average", 
    upstream = processor, 
    n_frame_average = spec.published_frames_per_step,
    skip_n_frames = spec.published_frames_per_step - 1
)
writer = diri.make_writer(
    name        = "tiff",
    upstream    = averager,
    mode        = "z-stack"
)
writer.basename = "stack"
writer.save_path = io.data_path()
writer.frames_per_file = float('inf')

acquisition.start()


