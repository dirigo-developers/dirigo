from dirigo.main import Dirigo


"""
To measure detector signal offsets, run a line acquisition, but leave slow
axis parked (alterantively: keep shutter closed), so there is no real signal
"""


diri = Dirigo()

acquisition = diri.make("acquisition", "raster_line")
writer      = diri.make("writer", "signal_offset_calibration", upstream=acquisition)

acquisition.start()
writer.join()

