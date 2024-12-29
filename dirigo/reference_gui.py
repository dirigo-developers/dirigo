import tkinter as tk
from tkinter import ttk

import dirigo
from dirigo.main import Dirigo
from dirigo.hw_interfaces.digitizer import SampleClock, Channel, Trigger
from dirigo.hw_interfaces.scanner import FastRasterScanner
from dirigo.hw_interfaces.stage import MultiAxisStage



class ReferenceGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Dirigo Reference GUI")

        self.diri = Dirigo()

        # Gather some references for brevity below
        hardware = self.diri.hw
        digitizer = hardware.digitizer
        scanner = hardware.fast_raster_scanner
        stage = hardware.stage

        # Create UI elements
       
        self.sample_clock_frame = SampleClockFrame(self, digitizer.sample_clock)
        self.sample_clock_frame.grid(row=0, column=0, padx=10, pady=10)

        self.channels_frame = ChannelsNotebook(self, digitizer.channels)
        self.channels_frame.grid(row=1, column=0, padx=10, pady=10)

        self.trigger_frame = TriggerFrame(self, digitizer.trigger, self.channels_frame)
        self.trigger_frame.grid(row=2, column=0, padx=10, pady=10)

        self.scanner_frame = FastRasterScannerFrame(self, scanner)
        self.scanner_frame.grid(row=3, column=0, padx=10, pady=10)

        self.stage_frame = StageFrame(self, stage)
        self.stage_frame.grid(row=4, column=0, padx=10, pady=10)
  

class SampleClockFrame(ttk.LabelFrame):
    def __init__(self, parent, sample_clock:SampleClock):
        super().__init__(parent, text="Sample Clock")
        self._sample_clock = sample_clock # not sure we need to hold this ref?

        # Sample Clock Source
        # No dependencies
        label = ttk.Label(self, text="Clock Source")
        label.grid(row=0, column=0, sticky="w", padx=10, pady=5)
        self.source_var = tk.StringVar(value=sample_clock.source)
        self.source_menu = ttk.OptionMenu(
            self, 
            self.source_var, 
            sample_clock.source,
            *sorted(sample_clock.source_options, key=lambda x: len(x)), # this sorting puts Internal Clock at the top
            command=self._source_callback
        )
        self.source_menu.grid(row=0, column=1, sticky="e", padx=10, pady=5)

        # Sample Rate
        # Depends on clock source
        label = ttk.Label(self, text="Sample Rate")
        label.grid(row=1, column=0, sticky="w", padx=10, pady=5)
        self.rate_menu_choice = None
        self._refresh_rate_options()

        # Sample Edge
        # No dependencies
        label = ttk.Label(self, text="Sample Edge")
        label.grid(row=2, column=0, sticky="w", padx=10, pady=5)
        self.edge_var = tk.StringVar(value=sample_clock.edge)
        self.edge_menu = ttk.OptionMenu(
            self, 
            self.edge_var, 
            sample_clock.edge, 
            *sample_clock.edge_options,
            command=lambda value: setattr(sample_clock, 'edge', value))
        self.edge_menu.grid(row=2, column=1, sticky="e", padx=10, pady=5)

    def _source_callback(self, value):
        self._sample_clock.source = value
        self._refresh_rate_options()

    def _refresh_rate_options(self):
        if self.rate_menu_choice:
            self.rate_menu_choice.destroy()

        if not self._sample_clock.source:
            self.rate_menu_choice = ttk.Label(self, text="Select source")

        elif "internal" in self._sample_clock.source.lower():
            self.rate_var = tk.StringVar(value=self._sample_clock.rate)
            self.rate_menu_choice = ttk.OptionMenu(
                self, 
                self.rate_var,
                self._sample_clock.rate,
                *[str(x) for x in sorted(self._sample_clock.rate_options)], # sorts and formats the rate options
                command=lambda value: setattr(self._sample_clock, 'rate', value)
            )

        elif "external" in self._sample_clock.source.lower():
            self._last_valid_ext_rate = ""
            validate_ext_rate_cmd = self.register(self._validate_ext_rate)
            invalid_ext_rate_cmd = self.register(self._invalid_ext_rate)
            self.rate_var = tk.StringVar(value=self._sample_clock.rate)
            self.rate_menu_choice = ttk.Entry(
                self, 
                textvariable=self.rate_var,
                validate="focusout",
                validatecommand=(validate_ext_rate_cmd, "%P"),
                invalidcommand=invalid_ext_rate_cmd,
            )    

        self.rate_menu_choice.grid(row=1, column=1, sticky="e", padx=10, pady=5)

    def _validate_ext_rate(self, new_value):
        if new_value == "" or new_value == "-":  # Allow empty or negative sign during typing
            return True

        try:
            value = float(new_value)
            
            rate_range = self._sample_clock.rate_options
            if not (rate_range.min < value < rate_range.max):
                print("Warning: Proposed external clock rate out of range")
                return False

            # Valid input 
            self._last_valid_ext_rate = new_value
            self._sample_clock.rate = new_value
            return True
        
        except ValueError:
            return False
        
    def _invalid_ext_rate(self):
        self.rate_menu_choice.delete(0, tk.END)
        self.rate_menu_choice.insert(0, self._last_valid_ext_rate)

        
class ChannelsNotebook(ttk.LabelFrame):
    def __init__(self, parent, channels:list[Channel]):
        super().__init__(parent, text="Channels")
        self._channels = channels
        
        # Create Channels tabs (aka Notebook)
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True)

        self.frames:list[ChannelFrame] = []
        for channel in self._channels:
            frame = ChannelFrame(self.notebook, channel)
            self.notebook.add(frame, text=f"{channel.index + 1}")
            self.frames.append(frame)


class ChannelFrame(ttk.Frame):
    def __init__(self, parent, channel:Channel):
        super().__init__(parent)
        self._channel = channel

        # Enabled Checkbox
        self.enabled_var = tk.BooleanVar(value=channel.enabled)
        check_button = ttk.Checkbutton(
            self, 
            text="Enabled", 
            variable=self.enabled_var,
            command=lambda: setattr(channel, 'enabled', self.enabled_var.get())
        )
        check_button.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        #self.enabled_var.trace_add('write', )

        # Coupling Dropdown
        self.coupling_var = tk.StringVar(value=channel.coupling)
        label = ttk.Label(self, text="Coupling")
        label.grid(row=1, column=0, padx=5, pady=5, sticky="w")
        coupling_menu = ttk.OptionMenu(
            self, 
            self.coupling_var, 
            channel.coupling, 
            *channel.coupling_options,
            command=lambda value: setattr(channel, 'coupling', value)
        )
        coupling_menu.grid(row=1, column=1, padx=5, pady=5)

        # Impedance Dropdown
        self.impedance_var = tk.StringVar(value=channel.impedance)
        label = ttk.Label(self, text="Impedance")
        label.grid(row=2, column=0, padx=5, pady=5, sticky="w")
        impedance_menu = ttk.OptionMenu(
            self, 
            self.impedance_var, 
            channel.impedance, 
            *channel.impedance_options,
            command=self._impedance_callback
        )
        impedance_menu.grid(row=2, column=1, padx=5, pady=5)

        # Range Dropdown
        self.range_var= tk.StringVar(value=channel.range)
        label = ttk.Label(self, text="Range")
        label.grid(row=3, column=0, padx=5, pady=5, sticky="w")
        self.range_menu = None
        self._refresh_range_options()       

    def _impedance_callback(self, value):
        self._channel.impedance = value
        self._refresh_range_options()

    def _refresh_range_options(self):
        if self.range_menu:
            self.range_menu.destroy()
        
        if not self._channel.impedance:
            self.range_menu = ttk.Label(self, text="Select impedance")

        else:
            self.range_menu = ttk.OptionMenu(
                self,
                self.range_var,
                self._channel.range,
                *sorted(self._channel.range_options, key=lambda x: -len(x)), # bit hacky: sorting strategy
                command=lambda value: setattr(self._channel, 'range', value)
            )

        self.range_menu.grid(row=3, column=1, padx=5, pady=5)


class TriggerFrame(ttk.LabelFrame):
    def __init__(self, parent, trigger:Trigger, channels_notebook:ChannelsNotebook):
        super().__init__(parent, text="Trigger")
        self._trigger = trigger 
        self._channels_notebook = channels_notebook

        # Trigger Source
        # Depends on enabled input channels
        label = ttk.Label(self, text="Source")
        label.grid(row=0, column=0, sticky="w", padx=10, pady=5)
        self.source_menu = None
        self._refresh_source_options()
        for channel_frame in channels_notebook.frames:
            channel_frame.enabled_var.trace_add('write', self._refresh_source_options)

        # Trigger Slope
        # No dependencies
        label = ttk.Label(self, text="Slope")
        label.grid(row=1, column=0, sticky="w", padx=10, pady=5)
        self.slope_var = tk.StringVar(value=trigger.slope)
        self.slope_menu = ttk.OptionMenu(
            self, 
            self.slope_var, 
            trigger.slope,
            *trigger.slope_options,
            command=lambda value: setattr(trigger, 'slope', value)
        )
        self.slope_menu.grid(row=1, column=1, sticky="e", padx=10, pady=5)

        # Trigger Level TODO, react to source
        # Very dependent
        label = ttk.Label(self, text="Level")
        label.grid(row=2, column=0, sticky="w", padx=10, pady=5)
        self.level_var = tk.StringVar(value=trigger.level)
        self.level_spinbox = ttk.Spinbox(
            self, 
            from_=None if trigger.level_limits is None else trigger.level_limits.min,
            to=None if trigger.level_limits is None else trigger.level_limits.max,
            increment=0.1,
            textvariable=self.level_var,
            wrap=False,
            width=8,
            command=lambda: setattr(trigger, 'level', float(self.level_var.get()))
        )
        self.level_spinbox.grid(row=2, column=1, sticky="e", padx=10, pady=5)

    def _refresh_source_options(self, *args):
        if self.source_menu:
            self.source_menu.destroy()

        # Update channels enable property before getting options
        for channel_frame in self._channels_notebook.frames:
            channel_frame._channel.enabled = channel_frame.enabled_var.get()

        self.source_var = tk.StringVar(value=self._trigger.source)
        self.source_menu = ttk.OptionMenu(
            self, 
            self.source_var, 
            self._trigger.source,
            *self._trigger.source_options,
            command=lambda value: setattr(self._trigger, 'source', value)
        )
        self.source_menu.grid(row=0, column=1, sticky="e", padx=10, pady=5)


class FastRasterScannerFrame(ttk.LabelFrame):
    def __init__(self, parent, scanner:FastRasterScanner):
        super().__init__(parent, text="Fast Raster Axis")
        self._scanner = scanner # not sure we need to hold this ref?

        # Enabled Checkbox
        self.enabled_var = tk.BooleanVar(value=scanner.enabled)
        check_button = ttk.Checkbutton(
            self, 
            text="Enabled", 
            variable=self.enabled_var,
            command=lambda: setattr(scanner, 'enabled', self.enabled_var.get())
        )
        check_button.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        
        # Amplitude
        label = ttk.Label(self, text="Amplitude (deg.)")
        label.grid(row=2, column=0, sticky="w", padx=10, pady=5)
        self.level_var = tk.StringVar(value=scanner.amplitude)
        self.level_spinbox = ttk.Spinbox(
            self, 
            from_=0,
            to=scanner.angle_limits.max_degrees - scanner.angle_limits.min_degrees,
            increment=2.0, # arbitrary, may want to define by interface?
            textvariable=self.level_var,
            wrap=False,
            width=8,
            command=lambda: setattr(
                scanner, 'amplitude', dirigo.Angle(f"{self.level_var.get()} deg")
            )
        )
        self.level_spinbox.grid(row=2, column=1, sticky="e", padx=10, pady=5)


class StageFrame(ttk.LabelFrame):
    def __init__(self, parent, stage:MultiAxisStage):
        super().__init__(parent, text="Stage")

        # X Axis
        # Position (interface in meters, but more convenient to use mm here)
        label = ttk.Label(self, text="X position (mm)")
        label.grid(row=0, column=0, sticky="w", padx=10, pady=5)
        self.x_pos_var = tk.StringVar(value=1000 * stage.x.position)
        self.x_pos_spinbox = ttk.Spinbox(
            self, 
            from_=1000 * stage.x.position_limits.min,
            to=1000 * stage.x.position_limits.max,
            increment=10.0, # arbitrary, may want to define by interface?
            textvariable=self.x_pos_var,
            wrap=False,
            width=8,
            command=lambda: stage.x.move_to(
                dirigo.Position(float(self.x_pos_var.get()) / 1000) # convert from mm to meters
            )
        )
        self.x_pos_spinbox.grid(row=0, column=1, sticky="e", padx=10, pady=5)

        # Y Axis
        # Position (interface in meters, but more convenient to use mm here)
        label = ttk.Label(self, text="Y position (mm)")
        label.grid(row=1, column=0, sticky="w", padx=10, pady=5)
        self.y_pos_var = tk.StringVar(value=1000 * stage.y.position)
        self.y_pos_spinbox = ttk.Spinbox(
            self, 
            from_=1000 * stage.y.position_limits.min,
            to=1000 * stage.y.position_limits.max,
            increment=10.0, # arbitrary, may want to define by interface?
            textvariable=self.y_pos_var,
            wrap=False,
            width=8,
            command=lambda: stage.y.move_to(
                dirigo.Position(float(self.y_pos_var.get()) / 1000) # convert to meters
            )
        )
        self.y_pos_spinbox.grid(row=1, column=1, sticky="e", padx=10, pady=5)


class AcquireFrame():
    pass
    


if __name__ == "__main__":
    app = ReferenceGUI()
    app.mainloop()
