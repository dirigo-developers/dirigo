import customtkinter as ctk
from typing import Any, Optional, Type, get_args, get_origin, Union
from datetime import datetime, date, time
from pydantic import BaseModel
from pydantic.fields import FieldInfo
import json


class PydanticFormGenerator:
    """Generates CustomTkinter forms from Pydantic models."""
    
    def __init__(self, parent_widget: ctk.CTkFrame):
        """Initialize the form generator.
        
        Args:
            parent_widget: The CTk widget to add form elements to
        """
        self.parent = parent_widget
        self.widgets = {}  # Store references to all created widgets
        self.field_frames = {}  # Store frame containers for each field
        
    def create_form(self, model_class: Type[BaseModel], initial_data: Optional[dict] = None) -> dict:
        """Create a form from a Pydantic model.
        
        Args:
            model_class: The Pydantic model class to generate form from
            initial_data: Optional dictionary of initial values
            
        Returns:
            Dictionary mapping field names to their widget references
        """
        self.model_class = model_class
        self.widgets.clear()
        self.field_frames.clear()
        
        # Clear existing widgets
        for widget in self.parent.winfo_children():
            widget.destroy()
        
        # Get model fields
        fields = model_class.model_fields
        
        # Create form elements for each field
        for field_name, field_info in fields.items():
            self._create_field_widget(field_name, field_info, initial_data)
        
        return self.widgets
    
    def _create_field_widget(self, field_name: str, field_info: FieldInfo, initial_data: Optional[dict] = None):
        """Create appropriate widget for a field based on its type."""
        
        # Skip fields we want to handle specially
        if field_name in ['config_version']:  # Auto-managed fields
            return
        
        # Create container frame for this field
        field_frame = ctk.CTkFrame(self.parent, fg_color="transparent")
        field_frame.pack(fill="x", pady=(0, 15))
        self.field_frames[field_name] = field_frame
        
        # Get field metadata
        field_type = field_info.annotation
        default_value = field_info.default
        description = field_info.description or field_name.replace('_', ' ').title()
        
        # Handle Optional types
        is_optional = False
        if get_origin(field_type) in [Union, type(Union)]:
            args = get_args(field_type)
            if type(None) in args:
                is_optional = True
                # Get the non-None type
                field_type = next((arg for arg in args if arg != type(None)), str)
        
        # Create label
        label_text = description
        if not is_optional and default_value is None:
            label_text += " *"  # Mark required fields
        
        label = ctk.CTkLabel(field_frame, text=label_text, anchor="w")
        label.pack(fill="x", pady=(0, 5))
        
        # Get initial value
        initial_value = None
        if initial_data and field_name in initial_data:
            initial_value = initial_data[field_name]
        elif default_value is not None:
            if hasattr(default_value, '__call__'):  # Handle factory functions
                initial_value = default_value()
            else:
                initial_value = default_value
        
        # Create appropriate widget based on type
        widget = self._create_widget_for_type(
            field_frame, field_name, field_type, initial_value, is_optional
        )
        
        if widget:
            self.widgets[field_name] = widget
    
    def _create_widget_for_type(self, parent: ctk.CTkFrame, field_name: str, 
                                field_type: Type, initial_value: Any, is_optional: bool) -> Optional[ctk.CTkBaseClass]:
        """Create the appropriate widget based on field type."""
        
        # Handle datetime
        if field_type == datetime:
            return self._create_datetime_widget(parent, field_name, initial_value)
        
        # Handle date
        elif field_type == date:
            return self._create_date_widget(parent, field_name, initial_value)
        
        # Handle bool
        elif field_type == bool:
            return self._create_checkbox_widget(parent, field_name, initial_value)
        
        # Handle int
        elif field_type == int:
            return self._create_number_widget(parent, field_name, initial_value, is_int=True)
        
        # Handle float
        elif field_type == float:
            return self._create_number_widget(parent, field_name, initial_value, is_int=False)
        
        # Handle str
        elif field_type == str:
            # Use textbox for "notes" or "description" fields
            if any(keyword in field_name.lower() for keyword in ['notes', 'description', 'comment']):
                return self._create_textbox_widget(parent, field_name, initial_value)
            else:
                return self._create_entry_widget(parent, field_name, initial_value)
        
        # Handle list
        elif get_origin(field_type) == list:
            return self._create_list_widget(parent, field_name, initial_value)
        
        # Handle dict
        elif get_origin(field_type) == dict:
            return self._create_dict_widget(parent, field_name, initial_value)
        
        # Handle Literal (enum-like)
        elif get_origin(field_type) == Literal:
            options = get_args(field_type)
            return self._create_dropdown_widget(parent, field_name, options, initial_value)
        
        # Handle nested Pydantic models
        elif isinstance(field_type, type) and issubclass(field_type, BaseModel):
            return self._create_nested_model_widget(parent, field_name, field_type, initial_value)
        
        # Default to entry widget
        else:
            return self._create_entry_widget(parent, field_name, str(initial_value) if initial_value else "")
    
    def _create_entry_widget(self, parent: ctk.CTkFrame, field_name: str, initial_value: Any) -> ctk.CTkEntry:
        """Create a simple text entry widget."""
        entry = ctk.CTkEntry(parent, placeholder_text=f"Enter {field_name.replace('_', ' ')}")
        entry.pack(fill="x")
        
        if initial_value:
            entry.insert(0, str(initial_value))
        
        return entry
    
    def _create_textbox_widget(self, parent: ctk.CTkFrame, field_name: str, initial_value: Any) -> ctk.CTkTextbox:
        """Create a multi-line text box widget."""
        textbox = ctk.CTkTextbox(parent, height=100, wrap="word")
        textbox.pack(fill="x")
        
        if initial_value:
            textbox.insert("1.0", str(initial_value))
        
        return textbox
    
    def _create_datetime_widget(self, parent: ctk.CTkFrame, field_name: str, initial_value: Any) -> ctk.CTkFrame:
        """Create a datetime display/edit widget."""
        dt_frame = ctk.CTkFrame(parent, fg_color="transparent")
        dt_frame.pack(fill="x")
        
        # Use current time if no initial value
        if not initial_value:
            initial_value = datetime.now()
        
        # Create string var for display
        dt_str = initial_value.isoformat() if isinstance(initial_value, datetime) else str(initial_value)
        dt_var = ctk.StringVar(value=dt_str)
        
        # Display label
        dt_label = ctk.CTkLabel(
            dt_frame,
            textvariable=dt_var,
            fg_color=("gray85", "gray20"),
            corner_radius=6,
            anchor="w"
        )
        dt_label.pack(side="left", fill="x", expand=True, padx=5, pady=5)
        
        # Update button to set to current time
        update_btn = ctk.CTkButton(
            dt_frame,
            text="Set to Now",
            width=100,
            command=lambda: dt_var.set(datetime.now().isoformat())
        )
        update_btn.pack(side="right", padx=(10, 0))
        
        # Store the variable for value retrieval
        dt_frame.dt_var = dt_var
        return dt_frame
    
    def _create_checkbox_widget(self, parent: ctk.CTkFrame, field_name: str, initial_value: Any) -> ctk.CTkCheckBox:
        """Create a checkbox widget."""
        checkbox = ctk.CTkCheckBox(parent, text="")
        checkbox.pack(anchor="w")
        
        if initial_value:
            checkbox.select()
        
        return checkbox
    
    def _create_number_widget(self, parent: ctk.CTkFrame, field_name: str, initial_value: Any, is_int: bool) -> ctk.CTkEntry:
        """Create a number entry widget."""
        entry = ctk.CTkEntry(parent, placeholder_text=f"Enter {'integer' if is_int else 'number'}")
        entry.pack(fill="x")
        
        if initial_value is not None:
            entry.insert(0, str(initial_value))
        
        # Add validation (could be enhanced with actual validation callback)
        return entry
    
    def _create_dropdown_widget(self, parent: ctk.CTkFrame, field_name: str, options: tuple, initial_value: Any) -> ctk.CTkOptionMenu:
        """Create a dropdown widget for Literal types."""
        # Convert options to strings
        str_options = [str(opt) for opt in options]
        
        # Set initial value
        if initial_value and str(initial_value) in str_options:
            default = str(initial_value)
        else:
            default = str_options[0]
        
        dropdown = ctk.CTkOptionMenu(parent, values=str_options)
        dropdown.set(default)
        dropdown.pack(fill="x")
        
        return dropdown
    
    def _create_list_widget(self, parent: ctk.CTkFrame, field_name: str, initial_value: Any) -> ctk.CTkTextbox:
        """Create a widget for list input (simple JSON for now)."""
        info_label = ctk.CTkLabel(parent, text="Enter JSON array:", font=("", 10), anchor="w")
        info_label.pack(fill="x")
        
        textbox = ctk.CTkTextbox(parent, height=60)
        textbox.pack(fill="x")
        
        if initial_value:
            textbox.insert("1.0", json.dumps(initial_value, indent=2))
        else:
            textbox.insert("1.0", "[]")
        
        return textbox
    
    def _create_dict_widget(self, parent: ctk.CTkFrame, field_name: str, initial_value: Any) -> ctk.CTkTextbox:
        """Create a widget for dict input (simple JSON for now)."""
        info_label = ctk.CTkLabel(parent, text="Enter JSON object:", font=("", 10), anchor="w")
        info_label.pack(fill="x")
        
        textbox = ctk.CTkTextbox(parent, height=80)
        textbox.pack(fill="x")
        
        if initial_value:
            textbox.insert("1.0", json.dumps(initial_value, indent=2))
        else:
            textbox.insert("1.0", "{}")
        
        return textbox
    
    def _create_nested_model_widget(self, parent: ctk.CTkFrame, field_name: str, 
                                   model_class: Type[BaseModel], initial_value: Any) -> ctk.CTkFrame:
        """Create a widget for nested Pydantic models."""
        # For now, create a sub-frame with nested fields
        # This could be enhanced to support collapsible sections
        nested_frame = ctk.CTkFrame(parent)
        nested_frame.pack(fill="x", padx=10)
        
        # Create sub-generator for nested model
        sub_generator = PydanticFormGenerator(nested_frame)
        initial_data = initial_value.model_dump() if isinstance(initial_value, BaseModel) else None
        nested_widgets = sub_generator.create_form(model_class, initial_data)
        
        # Store reference to nested widgets
        nested_frame.nested_widgets = nested_widgets
        nested_frame.sub_generator = sub_generator
        
        return nested_frame
    
    def get_values(self) -> dict:
        """Extract values from all form widgets.
        
        Returns:
            Dictionary of field names to values
        """
        values = {}
        
        for field_name, widget in self.widgets.items():
            value = self._get_widget_value(widget)
            if value is not None:  # Only include non-None values
                values[field_name] = value
        
        return values
    
    def _get_widget_value(self, widget: Any) -> Any:
        """Get value from a widget based on its type."""
        
        if isinstance(widget, ctk.CTkEntry):
            text = widget.get().strip()
            return text if text else None
        
        elif isinstance(widget, ctk.CTkTextbox):
            text = widget.get("1.0", "end-1c").strip()
            
            # Try to parse as JSON if it looks like JSON
            if text.startswith(('[', '{')):
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    return text
            
            return text if text else None
        
        elif isinstance(widget, ctk.CTkCheckBox):
            return widget.get() == 1
        
        elif isinstance(widget, ctk.CTkOptionMenu):
            return widget.get()
        
        elif isinstance(widget, ctk.CTkFrame):
            # Handle datetime frame
            if hasattr(widget, 'dt_var'):
                dt_str = widget.dt_var.get()
                try:
                    return datetime.fromisoformat(dt_str)
                except:
                    return None
            
            # Handle nested model frame
            elif hasattr(widget, 'sub_generator'):
                return widget.sub_generator.get_values()
        
        return None
    
    def validate(self) -> tuple[bool, Optional[BaseModel], Optional[str]]:
        """Validate the form data against the Pydantic model.
        
        Returns:
            Tuple of (is_valid, model_instance or None, error_message or None)
        """
        try:
            values = self.get_values()
            model_instance = self.model_class(**values)
            return True, model_instance, None
        except Exception as e:
            return False, None, str(e)