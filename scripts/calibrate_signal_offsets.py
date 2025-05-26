from dirigo.main import Dirigo


"""
To measure detector signal offsets, run a line acquisition, but leave slow
axis parked (alterantively: keep shutter closed), so there is no real signal
"""


diri = Dirigo()

acquisition = diri.make("acquisition", "raster_line")
logger      = diri.make("logger", "signal_offset_calibration", upstream=acquisition)

acquisition.start()
logger.join()

