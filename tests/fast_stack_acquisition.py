from dirigo.main import Dirigo
from dirigo import units, io
from dirigo.plugins.acquisitions import ContinuousStackAcquisition, AxialDirection


# Make spec
spec = ContinuousStackAcquisition.get_specification() # Get the base spec as template
spec.bidirectional_scanning = True
spec.line_width = units.Position("1 mm")
spec.frame_height = units.Position("1 mm")
spec.pixel_size = units.Position("5 um")
spec.pixel_height = spec.pixel_size
spec.z_range = units.PositionRange(
    min = units.Position("-50 um"),
    max = units.Position("50 um")
)
spec.depth_spacing = units.Position("1 um")
spec.z_direction = AxialDirection.POSITIVE
spec.n_volumes = 3



# Set up dirigo and run acquisition
diri = Dirigo()


acquisition = diri.make_acquisition(
    name = "continuous_stack", 
    spec = spec
)
processor = diri.make_processor(
    name = "raster_frame",
    upstream = acquisition
)
# writer = diri.make_writer(
#     name        = "tiff",
#     upstream    = processor,
#     mode        = "z-stack"
# )
# writer.basename = "fast_stack"
# writer.save_path = io.data_path()
# writer.frames_per_file = float('inf')

# # Experimental processor
# peak_finder = diri.make_processor(
#     name = "axial_peak_finder",
#     upstream = processor
# )


acquisition.start()