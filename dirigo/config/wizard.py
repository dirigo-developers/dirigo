"""System configuration wizard for Dirigo acquisition software with dynamic form generation."""

import customtkinter as ctk
from pathlib import Path
from datetime import datetime
from tkinter import messagebox, filedialog
import toml
import json

from schema import SystemConfig, SystemMetadata, DeviceDef, SYSTEM_CONFIG_SCHEMA_VERSION
from form_generator import PydanticFormGenerator


class SystemConfigWizard(ctk.CTk):
    """Main wizard window for creating system configurations."""
    
    def __init__(self):
        super().__init__()
        
        # Window setup
        self.title("Dirigo System Configuration Wizard")
        self.geometry("700x600")
        
        # Set theme
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # Current config being built
        self.current_config = SystemConfig()
        
        # Form generators
        self.form_generators = {}
        
        # Current step
        self.current_step = 0
        self.steps = [
            ("System Metadata", SystemMetadata),
            # Future: ("Add Devices", None),  # Custom step for device management
        ]
        
        # Build UI
        self._create_widgets()
        
    def _create_widgets(self):
        """Create and layout all widgets."""
        
        # Main container with padding
        main_frame = ctk.CTkFrame(self, fg_color="transparent")
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Title
        title = ctk.CTkLabel(
            main_frame, 
            text="System Configuration Wizard",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        title.pack(pady=(0, 20))
        
        # Step indicator
        self.step_label = ctk.CTkLabel(
            main_frame,
            text="",
            font=ctk.CTkFont(size=16)
        )
        self.step_label.pack(pady=(0, 20))
        
        # Content frame for current step (scrollable)
        content_container = ctk.CTkFrame(main_frame)
        content_container.pack(fill="both", expand=True, pady=(0, 20))
        
        self.content_frame = ctk.CTkScrollableFrame(content_container, fg_color="transparent")
        self.content_frame.pack(fill="both", expand=True)
        
        # Validation feedback
        self.validation_label = ctk.CTkLabel(
            main_frame,
            text="",
            text_color="red",
            wraplength=650
        )
        self.validation_label.pack(fill="x", pady=(0, 10))
        
        # Navigation buttons
        nav_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        nav_frame.pack(fill="x")
        
        self.back_button = ctk.CTkButton(
            nav_frame,
            text="Back",
            command=self._on_back,
            width=100
        )
        self.back_button.pack(side="left", padx=(0, 10))
        
        self.next_button = ctk.CTkButton(
            nav_frame,
            text="Next",
            command=self._on_next,
            width=100
        )
        self.next_button.pack(side="right", padx=(10, 0))
        
        self.save_button = ctk.CTkButton(
            nav_frame,
            text="Save Configuration",
            command=self._save_config,
            fg_color="green",
            hover_color="darkgreen",
            width=150
        )
        # Save button will appear when ready
        
        # Initialize with first step
        self._show_current_step()
        
    def _show_current_step(self):
        """Display the current step."""
        if self.current_step >= len(self.steps):
            self._show_summary()
            return
        
        step_name, model_class = self.steps[self.current_step]
        
        # Update UI
        self.step_label.configure(text=f"Step {self.current_step + 1}: {step_name}")
        self.validation_label.configure(text="")
        
        # Update button states
        self.back_button.configure(state="normal" if self.current_step > 0 else "disabled")
        self.next_button.configure(text="Next" if self.current_step < len(self.steps) - 1 else "Finish")
        
        # Clear content frame
        for widget in self.content_frame.winfo_children():
            widget.destroy()
        
        if model_class:
            # Create form using the dynamic generator
            self._create_dynamic_form(step_name, model_class)
    
    def _create_dynamic_form(self, step_name: str, model_class):
        """Create a form dynamically from a Pydantic model."""
        
        # Create info label
        info_label = ctk.CTkLabel(
            self.content_frame,
            text=f"Configure {step_name}",
            font=ctk.CTkFont(size=14),
            anchor="w"
        )
        info_label.pack(fill="x", pady=(0, 20))
        
        # Create form container
        form_frame = ctk.CTkFrame(self.content_frame, fg_color="transparent")
        form_frame.pack(fill="both", expand=True, padx=20)
        
        # Initialize form generator
        generator = PydanticFormGenerator(form_frame)
        
        # Get initial data if we have it
        initial_data = None
        if step_name == "System Metadata" and hasattr(self.current_config, 'system'):
            if self.current_config.system:
                initial_data = self.current_config.system.model_dump()
        
        # Generate the form
        generator.create_form(model_class, initial_data)
        
        # Store generator reference
        self.form_generators[self.current_step] = generator
    
    def _validate_current_step(self) -> tuple[bool, str]:
        """Validate the current step's form data.
        
        Returns:
            (is_valid, error_message)
        """
        if self.current_step not in self.form_generators:
            return True, ""
        
        generator = self.form_generators[self.current_step]
        is_valid, model_instance, error_msg = generator.validate()
        
        if is_valid and model_instance:
            # Update the configuration with validated data
            step_name, _ = self.steps[self.current_step]
            if step_name == "System Metadata":
                self.current_config.system = model_instance
        
        return is_valid, error_msg or ""
    
    def _on_next(self):
        """Handle next button click."""
        # Validate current step
        is_valid, error_msg = self._validate_current_step()
        
        if not is_valid:
            self.validation_label.configure(text=f"⚠ {error_msg}")
            return
        
        self.validation_label.configure(text="")
        
        # Move to next step
        self.current_step += 1
        
        if self.current_step >= len(self.steps):
            # We're done, show summary
            self._show_summary()
        else:
            self._show_current_step()
    
    def _on_back(self):
        """Handle back button click."""
        if self.current_step > 0:
            self.current_step -= 1
            self._show_current_step()
    
    def _show_summary(self):
        """Display a summary of the configuration."""
        # Update UI
        self.step_label.configure(text="Configuration Complete")
        self.next_button.configure(state="disabled")
        self.save_button.pack(side="right")
        
        # Clear content frame
        for widget in self.content_frame.winfo_children():
            widget.destroy()
        
        summary_label = ctk.CTkLabel(
            self.content_frame,
            text="Configuration Summary",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        summary_label.pack(pady=(10, 20))
        
        # Create tabs for different views
        tabview = ctk.CTkTabview(self.content_frame)
        tabview.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        
        # Formatted view tab
        tabview.add("Formatted View")
        formatted_text = ctk.CTkTextbox(tabview.tab("Formatted View"), wrap="word")
        formatted_text.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Format the configuration for display
        summary = f"System Configuration (Version {SYSTEM_CONFIG_SCHEMA_VERSION})\n"
        summary += "=" * 50 + "\n\n"
        
        summary += "SYSTEM METADATA\n"
        summary += "-" * 30 + "\n"
        if self.current_config.system.name:
            summary += f"Name: {self.current_config.system.name}\n"
        
        if self.current_config.system.notes:
            summary += f"Notes:\n{self.current_config.system.notes}\n"
        
        if self.current_config.system.created_at:
            summary += f"Created: {self.current_config.system.created_at.isoformat()}\n"
        
        summary += f"\nDEVICES\n"
        summary += "-" * 30 + "\n"
        summary += f"Total configured: {len(self.current_config.components)}\n"
        
        for device in self.current_config.components:
            summary += f"\n• {device.name} ({device.kind})\n"
            summary += f"  Plugin: {device.plugin_id}\n"
            if device.config:
                summary += f"  Config: {json.dumps(device.config, indent=4)}\n"
        
        formatted_text.insert("1.0", summary)
        formatted_text.configure(state="disabled")
        
        # TOML preview tab
        tabview.add("TOML Preview")
        toml_text = ctk.CTkTextbox(tabview.tab("TOML Preview"), wrap="word")
        toml_text.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Convert to dict for TOML
        config_dict = self.current_config.model_dump(exclude_none=True)
        
        # Convert datetime to string for TOML
        if "system" in config_dict and "created_at" in config_dict["system"]:
            if isinstance(config_dict["system"]["created_at"], datetime):
                config_dict["system"]["created_at"] = config_dict["system"]["created_at"].isoformat()
        
        # Convert to TOML format for preview
        import io
        toml_buffer = io.StringIO()
        toml.dump(config_dict, toml_buffer)
        toml_str = toml_buffer.getvalue()
        
        toml_text.insert("1.0", toml_str)
        toml_text.configure(state="disabled")
    
    def _save_config(self):
        """Save the configuration to a TOML file."""
        # Ask user where to save
        file_path = filedialog.asksaveasfilename(
            defaultextension=".toml",
            filetypes=[("TOML files", "*.toml"), ("All files", "*.*")],
            initialfile=f"system_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.toml"
        )
        
        if not file_path:
            return
        
        try:
            # Convert to dict and save
            config_dict = self.current_config.model_dump(exclude_none=True)
            
            # Convert datetime to string for TOML
            if "system" in config_dict and "created_at" in config_dict["system"]:
                if isinstance(config_dict["system"]["created_at"], datetime):
                    config_dict["system"]["created_at"] = config_dict["system"]["created_at"].isoformat()
            
            with open(file_path, "w") as f:
                toml.dump(config_dict, f)
            
            messagebox.showinfo(
                "Success",
                f"Configuration saved to:\n{file_path}"
            )
            
            # Optionally close the wizard
            if messagebox.askyesno("Continue?", "Would you like to create another configuration?"):
                # Reset the wizard
                self.current_config = SystemConfig()
                self.current_step = 0
                self.form_generators.clear()
                self.save_button.pack_forget()
                self.next_button.configure(state="normal")
                self._show_current_step()
            else:
                self.destroy()
                
        except Exception as e:
            messagebox.showerror(
                "Save Error",
                f"Failed to save configuration:\n{str(e)}"
            )


def main():
    """Run the wizard application."""
    app = SystemConfigWizard()
    app.mainloop()


if __name__ == "__main__":
    main()