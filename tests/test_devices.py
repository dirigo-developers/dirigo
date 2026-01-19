from dirigo.plugins.illuminators import LEDViaNIConfig, LEDViaNI



cfg = LEDViaNIConfig(
    vendor="Advanced Illumination",
    model="LL163",
    enable_channel="Dev1/port0/line15"
)

led = LEDViaNI(cfg)

led.connect()

led.enabled = True
led.enabled = False

led.close()
