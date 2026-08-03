from dirigo.main import Dirigo


diri = Dirigo()


acquisition = diri.make_acquisition("continuous_stack")

acquisition.start()