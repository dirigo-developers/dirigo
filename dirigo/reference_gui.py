import tkinter as tk
from tkinter import ttk

from dirigo.components import digitizer
from dirigo.plugins.alazar import AlazarDigitizer



class MainApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Dirigo Reference GUI")

        self.digitizer = AlazarDigitizer() # TODO make this a factor function return

        # Create UI elements
        self.sample_clock_frame = SampleClockFrame(self, self.digitizer.sample_clock)
        self.sample_clock_frame.grid(row=0, column=0, padx=10, pady=10)

        self.channels_frame = ChannelsFrame(self, self.digitizer.channels)
        self.channels_frame.grid(row=1, column=0, padx=10, pady=10)

        self.trigger_frame = TriggerFrame(self, self.digitizer.trigger)
        self.trigger_frame.grid(row=2, column=0, padx=10, pady=10)
  

class SampleClockFrame(ttk.LabelFrame):
    def __init__(self, parent, sample_clock:digitizer.SampleClock):
        super().__init__(parent, text="Sample Clock")
        self._sample_clock = sample_clock # not sure we need to hold this ref?

        # Sample Clock Source
        label = ttk.Label(self, text="Clock Source")
        label.grid(row=0, column=0, sticky="w", padx=10, pady=5)
        self.source_var = tk.StringVar(value=sample_clock.source)
        self.source_menu = ttk.OptionMenu(
            self, 
            self.source_var, 
            sample_clock.source,
            *sample_clock.source_options,
            command=self._source_callback
        )
        self.source_menu.grid(row=0, column=1, padx=10, pady=5)

        # Sample Rate
        label = ttk.Label(self, text="Sample Rate")
        label.grid(row=1, column=0, sticky="w", padx=10, pady=5)
        self.rate_menu_choice = None
        self._refresh_rate_options()

        # Sample Edge
        label = ttk.Label(self, text="Sample Edge")
        label.grid(row=2, column=0, sticky="w", padx=10, pady=5)
        self.edge_var = tk.StringVar(value=sample_clock.edge)
        self.edge_menu = ttk.OptionMenu(
            self, 
            self.edge_var, 
            sample_clock.edge, 
            *sample_clock.edge_options,
            command=lambda value: setattr(sample_clock, 'edge', value))
        self.edge_menu.grid(row=2, column=1, padx=10, pady=5)

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
                *self._sample_clock.rate_options,
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

        self.rate_menu_choice.grid(row=1, column=1, padx=10, pady=5)

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

        
class ChannelsFrame(ttk.LabelFrame):
    def __init__(self, parent, channels:list[digitizer.Channel]):
        super().__init__(parent, text="Channels")
        self._channels = channels
        
        # Create Channels tabs (aka Notebook)
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True)

        self.channel_settings = {}
        for channel in channels:
            self.create_channel_frame(cid=channel.index)

    def create_channel_frame(self, cid:int):
        frame = ttk.Frame(self.notebook)
        channel = self._channels[cid]
        self.notebook.add(frame, text=f"{cid+1}")

        # Store channel settings
        self.channel_settings[cid] = {
            "enabled": tk.BooleanVar(value=channel.enabled),
            "coupling": tk.StringVar(value=channel.coupling),
            "impedance": tk.StringVar(value=channel.impedance),
            "range": tk.StringVar(value=channel.range),
        }

        # Enabled Checkbox
        enabled_var:tk.BooleanVar = self.channel_settings[cid]["enabled"]
        check_button = ttk.Checkbutton(
            frame, 
            text="Enabled", 
            variable=enabled_var,
            command=lambda: setattr(channel, 'enabled', enabled_var.get())
        )
        check_button.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        # Coupling Dropdown
        coupling_var:tk.StringVar = self.channel_settings[cid]["coupling"]
        label = ttk.Label(frame, text="Coupling")
        label.grid(row=1, column=0, padx=5, pady=5, sticky="w")
        coupling_menu = ttk.OptionMenu(
            frame, 
            coupling_var, 
            channel.coupling, 
            *channel.coupling_options,
            command=lambda value: setattr(channel, 'coupling', value)
        )
        coupling_menu.grid(row=1, column=1, padx=5, pady=5)

        # Impedance Dropdown
        impedance_var:tk.StringVar = self.channel_settings[cid]["impedance"]
        label = ttk.Label(frame, text="Impedance")
        label.grid(row=2, column=0, padx=5, pady=5, sticky="w")
        impedance_menu = ttk.OptionMenu(
            frame, 
            impedance_var, 
            channel.impedance, 
            *channel.impedance_options,
            command=lambda value: setattr(channel, 'impedance', value)
        )
        impedance_menu.grid(row=2, column=1, padx=5, pady=5)

        # Range Dropdown
        range_var:tk.StringVar = self.channel_settings[cid]["range"]
        label = ttk.Label(frame, text="Range")
        label.grid(row=3, column=0, padx=5, pady=5, sticky="w")
        range_menu = ttk.OptionMenu(
            frame, 
            range_var, 
            channel.range, 
            *channel.range_options,
            command=lambda value: setattr(channel, 'range', value)
        )
        range_menu.grid(row=3, column=1, padx=5, pady=5)


class TriggerFrame(ttk.LabelFrame):
    def __init__(self, parent, trigger:digitizer.Trigger):
        super().__init__(parent, text="Trigger")
        self._trigger = trigger # not sure we need to hold this ref?

        # Trigger Source
        label = ttk.Label(self, text="Source")
        label.grid(row=0, column=0, sticky="w", padx=10, pady=5)
        self.source_var = tk.StringVar(value=trigger.source)
        self.source_menu = ttk.OptionMenu(
            self, 
            self.source_var, 
            trigger.source,
            *trigger.source_options,
            command=lambda value: setattr(trigger, 'source', value)
        )
        self.source_menu.grid(row=0, column=1, padx=10, pady=5)

        # Trigger Slope
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
        self.slope_menu.grid(row=1, column=1, padx=10, pady=5)

        # Trigger Level
        label = ttk.Label(self, text="Level")
        label.grid(row=2, column=0, sticky="w", padx=10, pady=5)
        self.level_var = tk.StringVar(value=trigger.level)
        self.level_spinbox = ttk.Spinbox(
            self, 
            from_=trigger.level_min,
            to=trigger.level_max,
            increment=0.1,
            textvariable=self.level_var,
            wrap=False,
            command=lambda value: setattr(trigger, 'level', value)
        )
        self.level_spinbox.grid(row=2, column=1, padx=10, pady=5)


class AcquireFrame():
    pass
        


if __name__ == "__main__":
    app = MainApp()
    app.mainloop()
