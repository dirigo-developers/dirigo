import nidaqmx

from dirigo.components.io import ScannerConfig


class ScannerSettings:
    """
    Manages the scanner parameters (fixed configuration and transient settings)
    """
    def __init__(self, default_config:ScannerConfig):
        self.config = default_config # this should remain fixed

        # following parameters may change
        self.scanner_frequency = self.config.nominal_scanner_frequency


# Comment: in the long-run it would best to abstact this class to allow 
# multiple DAQ platforms
class Scanner:
    """
    Operates the laser scanner, primarily through an NI DAQ card.
    """
    def __init__(self, scanner_config):
        self.settings = ScannerSettings(scanner_config)


    # METHODS
    # start_fast_scanner(line_width)
    # center_slow_scanner()
    # park_scanners()

    # TASKS
    # FastScannerFrequencyMonitor(SingleUseTask):
    # FrameClock(SingleUseTask)
    # SlowScanner(SingleUseTask)
    # MotorPositionEncoder(SingleUseTask):
    # XYMotorPositionEncoder():
    # EncoderDerivedTrigger(SingleUseTask):
    # LineIllumination(SingleUseTask):


