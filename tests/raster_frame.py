from dirigo.main import Dirigo


save_raw = True


diri = Dirigo()

acquisition = diri.make_acquisition("raster_frame")
if save_raw:
    raw_logger = diri.make_logger("tiff", upstream=acquisition)
processor = diri.make_processor("raster_frame", upstream=acquisition)
averager = diri.make_processor("rolling_average", upstream=processor)
display = diri.make_display_processor(
    name                    = "frame", 
    upstream                = averager,
    color_vector_names      = ["green", "magenta"],
    transfer_function_name  = "gamma"
)
if not save_raw:
    logger = diri.make_logger("tiff", upstream=processor)

acquisition.start()
acquisition.join(timeout=100.0)

print("Acquisition complete")
