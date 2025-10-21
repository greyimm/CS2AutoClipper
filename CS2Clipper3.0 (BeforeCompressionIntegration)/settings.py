"""
CS2 Clipper Settings GUI and Configuration Management

This module contains:
1. Configuration management functions (loading, saving, defaults)
2. GUI for editing settings
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import json
import os
import subprocess
import platform


# ========================================
# CONFIGURATION MANAGEMENT FUNCTIONS
# ========================================

def get_config_file_path():
    """
    Get the path to the configuration file.
    Always looks in the same directory as this script.
    
    Returns:
        str: Full path to settings_config.json
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, "settings_config.json")


def get_default_settings():
    """
    Get the default configuration settings.
    
    Returns:
        dict: Default configuration dictionary
    """
    return {
        # === FOLDER SETTINGS ===
        "video_folder": r"D:\Videos\Counter-strike 2",
        "output_folder": r"D:\Videos\Counter-strike 2\CS2Clipper\trimmedvideos",
        "unsorted_folder": r"D:\Videos\Counter-strike 2\CS2Clipper\unsortedfiles",
        
        # === TRIM SETTINGS ===
        "trim_start_seconds": 4,
        "trim_end_seconds": 3,
        
        # === OUTPUT VIDEO SETTINGS ===
        "output_resolution_width": 1920,
        "output_resolution_height": 1080,
        "output_fps": 0,
        
        # === PROCESSING SETTINGS ===
        "num_videos_to_process": -1,
        "show_preview": False,
        "multiprocessing_mode": "Auto"
    }


def load_config(gui_mode=False):
    """
    Load configuration from file or return defaults.
    
    Args:
        gui_mode (bool): Whether running in GUI mode (affects error handling)
    
    Returns:
        dict: Configuration dictionary with settings
    """
    config_file = get_config_file_path()
    default_config = get_default_settings()
    
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                loaded_config = json.load(f)
                
                # Migrate old settings to new naming (backward compatibility)
                if "resolution_width" in loaded_config and "output_resolution_width" not in loaded_config:
                    loaded_config["output_resolution_width"] = loaded_config.pop("resolution_width")
                if "resolution_height" in loaded_config and "output_resolution_height" not in loaded_config:
                    loaded_config["output_resolution_height"] = loaded_config.pop("resolution_height")
                
                # Merge with defaults (defaults provide any missing keys)
                config = {**default_config, **loaded_config}
                
                return config
                
        except Exception as e:
            if gui_mode:
                raise
            
            return default_config.copy()
    else:
        return default_config.copy()


def save_config(config_dict, gui_mode=False):
    """
    Save configuration to file.
    
    Args:
        config_dict (dict): Configuration dictionary to save
        gui_mode (bool): Whether running in GUI mode (affects error handling)
    
    Returns:
        tuple: (success: bool, error_message: str or None)
    """
    config_file = get_config_file_path()
    
    try:
        with open(config_file, 'w') as f:
            json.dump(config_dict, f, indent=4)
        return (True, None)
    except Exception as e:
        error_msg = f"Failed to save settings: {str(e)}"
        return (False, error_msg)


def print_config_summary(config):
    """
    Print a formatted summary of the current configuration.
    Call this explicitly when you want to display config info.
    
    Args:
        config (dict): Configuration dictionary
    """
    print(f"\nLoaded settings:")
    print(f"  Config file location: {get_config_file_path()}")
    print(f"  Video folder: {config['video_folder']}")
    print(f"  Output folder: {config['output_folder']}")
    print(f"  Unsorted folder: {config['unsorted_folder']}")
    print(f"  Trim start: {config['trim_start_seconds']} seconds before first kill")
    print(f"  Trim end: {config['trim_end_seconds']} seconds after last kill")
    print(f"  Output resolution: {config['output_resolution_width']}x{config['output_resolution_height']}")
    print(f"  Output FPS: {'Source FPS' if config['output_fps'] == 0 else f'{config['output_fps']} FPS'}")
    print(f"  Videos to process: {'All' if config['num_videos_to_process'] == -1 else config['num_videos_to_process']}")
    print(f"  Show preview: {'Enabled' if config['show_preview'] else 'Disabled (Optimized)'}")
    print(f"  Multiprocessing mode: {config['multiprocessing_mode']}")


# ========================================
# GUI APPLICATION CLASS
# ========================================

class CS2ClipperSettings:
    """
    CS2 Clipper Settings GUI
    
    This application provides a user-friendly interface for configuring
    the CS2 Auto Clipper video processing settings.
    """
    
    # ========================================
    # HELP TOOLTIP DEFINITIONS
    # ========================================
    
    HELP_TOOLTIPS = {
        # === FOLDER SETTINGS ===
        'video_folder': "Source folder containing your CS2 video recordings to be processed",
        'output_folder': "Destination folder where trimmed/processed videos will be saved",
        'unsorted_folder': "Folder for videos that couldn't be processed (0 kills, >5 kills, or errors)",
        
        # === PROCESSING SETTINGS ===
        'process_all': (
            "When enabled: Processes all videos in the video folder\n\n"
            "When disabled: Only processes the specified number of videos"
        ),
        'show_preview': (
            "When enabled: Shows live video preview during processing (slower, useful for debugging)\n\n"
            "When disabled: Processes videos without display (faster, optimized for batch processing)\n\n"
            "Does not work with Multiprocessing."
        ),
        'multiprocessing_mode': (
            "Multiprocessing allows processing multiple videos simultaneously for faster batch processing.\n\n"
            "Auto: Automatically detects your hardware and enables multiprocessing if you have sufficient RAM and CPU cores. "
            "Disables if specs are too low (<6GB RAM).\n\n"
            "On: Forces multiprocessing based on your available hardware (2-4 parallel processes). Use only if you have good hardware.\n\n"
            "Off: Processes videos one at a time (sequential). Safest option for older/slower systems."
        )
    }
    
    def __init__(self, root):
        self.root = root
        self.root.title("CS2 Clipper Settings")
        self.root.geometry("430x660")
        self.root.minsize(430, 500)

        # Use shared config functions
        self.config_file = get_config_file_path()
        self.default_settings = get_default_settings()
        
        # Load existing settings
        self.settings = self.load_settings()
        
        # Create GUI
        self.create_widgets()
    
    
    # ========================================
    # HELPER METHODS FOR REPEATED GUI ELEMENTS
    # ========================================
    
    def _create_section_header(self, parent, current_row, header_text):
        """Create a standardized bold section header"""
        ttk.Label(parent, text=header_text, font=('Arial', 10, 'bold')).grid(
            row=current_row, column=0, sticky=tk.W, pady=(10, 5)
        )
        return current_row + 1
    
    
    def _create_standard_frame(self, parent, current_row):
        """Create a standardized frame with consistent styling"""
        frame = ttk.Frame(parent)
        frame.grid(row=current_row, column=0, sticky=(tk.W, tk.E), pady=(0, 15))
        return frame, current_row + 1
    
    
    def _create_labeled_spinbox(self, parent, row, label_text, var_name, settings_key, 
                                from_val, to_val, unit_text, width=10):
        """Create a labeled spinbox with unit label"""
        ttk.Label(parent, text=label_text).grid(row=row, column=0, sticky=tk.W, pady=5)
        
        int_var = tk.IntVar(value=self.settings.get(settings_key))
        setattr(self, var_name, int_var)
        
        ttk.Spinbox(parent, from_=from_val, to=to_val, textvariable=int_var, width=width).grid(
            row=row, column=1, padx=(10, 0), sticky=tk.W
        )
        ttk.Label(parent, text=unit_text).grid(row=row, column=2, padx=(5, 0), sticky=tk.W)
        
        return int_var
    
    
    def _create_checkbox_with_info(self, parent, checkbox_text, var_name, tooltip_key, 
                                   default_value, command=None):
        """Create a checkbox with an info button"""
        frame = ttk.Frame(parent)
        
        bool_var = tk.BooleanVar(value=default_value)
        setattr(self, var_name, bool_var)
        
        checkbox = ttk.Checkbutton(frame, text=checkbox_text, variable=bool_var, command=command)
        checkbox.pack(side=tk.LEFT)
        
        self._create_info_button(frame, tooltip_key)
        
        return frame, bool_var, checkbox
    
    
    def _create_info_button(self, parent, tooltip_key, padx=(5, 0)):
        """Create a standardized info button [?] that shows a tooltip"""
        info_btn = ttk.Label(parent, text=" [?]", foreground="blue", cursor="hand2")
        info_btn.pack(side=tk.LEFT, padx=padx)
        info_btn.bind("<Button-1>", lambda e: self.show_tooltip(self.HELP_TOOLTIPS[tooltip_key]))
        return info_btn
    
    
    def _create_folder_input(self, parent, current_row, label_text, var_name, tooltip_key, browse_command, label_padx):
        """Create a standardized folder input section"""
        # Create label
        ttk.Label(parent, text=label_text, font=('Arial', 10, 'bold')).grid(
            row=current_row, column=0, sticky=tk.W, pady=(0, 5)
        )
        
        # Create info button
        info_btn = ttk.Label(parent, text=" [?]", foreground="blue", cursor="hand2")
        info_btn.grid(row=current_row, column=0, sticky=tk.W, padx=label_padx)
        info_btn.bind("<Button-1>", lambda e: self.show_tooltip(self.HELP_TOOLTIPS[tooltip_key]))
        current_row += 1
        
        # Create input frame
        input_frame = ttk.Frame(parent)
        input_frame.grid(row=current_row, column=0, sticky=(tk.W, tk.E), pady=(0, 15))
        input_frame.columnconfigure(0, weight=1)
        
        # Create StringVar and store it as instance variable
        string_var = tk.StringVar(value=self.settings[tooltip_key])
        setattr(self, var_name, string_var)
        
        # Create entry
        entry = ttk.Entry(input_frame, textvariable=string_var, width=60)
        entry.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 10))
        
        # Create browse button
        ttk.Button(input_frame, text="Browse...", command=browse_command).grid(row=0, column=1)
        
        current_row += 1
        return current_row
    
    
    def _browse_folder(self, var, title):
        """Generic folder browsing method"""
        folder = filedialog.askdirectory(initialdir=var.get(), title=title)
        if folder:
            var.set(folder)
    
    
    def _on_mousewheel(self, event):
        # Get current scroll position (0.0 = top, 1.0 = bottom)
        current_top, current_bottom = self.canvas.yview()
        
        scroll_amount = int(-1*(event.delta/120))
        
        # ADJUST THIS VALUE to control how far up you can scroll
        scroll_limit = 0.0  # Change this to adjust upward scroll limit
        
        # Prevent scrolling up if already at or above the limit
        if scroll_amount < 0 and current_top <= scroll_limit:
            return  # Don't allow scrolling up
        
        self.canvas.yview_scroll(scroll_amount, "units")
    
    
    def _on_configure(self, event):
        """Update scroll region when canvas is resized or content changes"""
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    
    
    # ========================================
    # GUI CREATION METHOD
    # ========================================
    
    def create_widgets(self):
        """Create all GUI widgets with scrolling support"""
        # Configure root window grid
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)  # Scrollable content area expands
        self.root.rowconfigure(1, weight=0)  # Footer stays fixed
        
        # Create canvas and scrollbar for scrollable content
        canvas_container = ttk.Frame(self.root)
        canvas_container.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.E, tk.W))
        canvas_container.columnconfigure(0, weight=1)
        canvas_container.rowconfigure(0, weight=1)
        
        self.canvas = tk.Canvas(canvas_container, highlightthickness=0)
        scrollbar = ttk.Scrollbar(canvas_container, orient="vertical", command=self.canvas.yview)
        
        self.canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack canvas and scrollbar
        self.canvas.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.E, tk.W))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Create main frame inside canvas
        main_frame = ttk.Frame(self.canvas, padding="20")
        canvas_window = self.canvas.create_window((0, 0), window=main_frame, anchor="nw")
        
        # Bind events for scrolling
        main_frame.bind("<Configure>", self._on_configure)
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfig(canvas_window, width=e.width))
        
        # Bind mousewheel scrolling (cross-platform)
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)  # Windows/Mac
        self.canvas.bind_all("<Button-4>", lambda e: self.canvas.yview_scroll(-1, "units"))  # Linux
        self.canvas.bind_all("<Button-5>", lambda e: self.canvas.yview_scroll(1, "units"))   # Linux
        
        current_row = 0
        
        # ===== FOLDER SETTINGS SECTION =====
        current_row = self._create_folder_settings_section(main_frame, current_row)
        
        # ===== TRIM SETTINGS SECTION =====
        current_row = self._create_trim_settings_section(main_frame, current_row)
        
        # ===== OUTPUT VIDEO SETTINGS SECTION =====
        current_row = self._create_output_video_settings_section(main_frame, current_row)
        
        # ===== PROCESSING SETTINGS SECTION =====
        current_row = self._create_processing_settings_section(main_frame, current_row)
        
        # Configure grid weights for main frame
        main_frame.columnconfigure(0, weight=1)
        
        # ===== FOOTER WITH BUTTONS =====
        self._create_footer()
    
    
    def _create_footer(self):
        """Create footer with buttons anchored to bottom right"""
        # Footer frame - stays at bottom
        footer_frame = ttk.Frame(self.root, padding="10")
        footer_frame.grid(row=1, column=0, sticky=(tk.E, tk.W))
        
        # Configure footer to push buttons to the right
        footer_frame.columnconfigure(0, weight=1)  # Left side expands (pushes buttons right)
        
        # Button container aligned to the right
        button_container = ttk.Frame(footer_frame)
        button_container.grid(row=0, column=1, sticky=tk.E)
        
        # Create buttons
        ttk.Button(
            button_container, 
            text="Save Settings", 
            command=self.save_settings, 
            width=15
        ).grid(row=0, column=0, padx=(0, 5))
        
        ttk.Button(
            button_container, 
            text="Reset to Defaults", 
            command=self.reset_defaults, 
            width=15
        ).grid(row=0, column=1, padx=5)
        
        ttk.Button(
            button_container, 
            text="Run", 
            # Change this button to Run and add the following method:  
            command=self.run_script,
            width=15
        ).grid(row=0, column=2, padx=(5, 0))
    
    
    # ========================================
    # FOLDER SETTINGS SECTION
    # ========================================
    
    def _create_folder_settings_section(self, main_frame, current_row):
        """Create the folder settings section (Video, Output, Unsorted folders)"""
        
        current_row = self._create_folder_input(
            main_frame, current_row, 
            "Video Folder:", 
            "video_folder_var", 
            "video_folder", 
            lambda: self._browse_folder(self.video_folder_var, "Select Video Folder"),
            (100, 0)
        )
        
        current_row = self._create_folder_input(
            main_frame, current_row, 
            "Output Folder:", 
            "output_folder_var", 
            "output_folder", 
            lambda: self._browse_folder(self.output_folder_var, "Select Output Folder"),
            (110, 0)
        )
        
        current_row = self._create_folder_input(
            main_frame, current_row, 
            "Unsorted Folder:", 
            "unsorted_folder_var", 
            "unsorted_folder", 
            lambda: self._browse_folder(self.unsorted_folder_var, "Select Unsorted Folder"),
            (125, 0)
        )
        
        return current_row
    
    
    # ========================================
    # TRIM SETTINGS SECTION
    # ========================================
    
    def _create_trim_settings_section(self, main_frame, current_row):
        """Create the trim settings section"""
        
        current_row = self._create_section_header(main_frame, current_row, "Trim Settings:")
        
        trim_frame, current_row = self._create_standard_frame(main_frame, current_row)
        
        self._create_labeled_spinbox(
            trim_frame, 0,
            "Seconds before first kill (Start):",
            "trim_start_var",
            "trim_start_seconds",
            0, 30,
            "seconds"
        )
        
        self._create_labeled_spinbox(
            trim_frame, 1,
            "Seconds after last kill (End):",
            "trim_end_var",
            "trim_end_seconds",
            0, 30,
            "seconds"
        )
        
        return current_row
    
    
    # ========================================
    # OUTPUT VIDEO SETTINGS SECTION
    # ========================================
    
    def _create_output_video_settings_section(self, main_frame, current_row):
        """Create the output video settings section (Resolution, FPS)"""
        
        current_row = self._create_section_header(main_frame, current_row, "Output Video Settings:")
        
        output_settings_frame, current_row = self._create_standard_frame(main_frame, current_row)
        
        # === RESOLUTION SETTINGS ===
        ttk.Label(output_settings_frame, text="Output Resolution:").grid(row=0, column=0, sticky=tk.W, pady=5)
        
        resolution_input_frame = ttk.Frame(output_settings_frame)
        resolution_input_frame.grid(row=0, column=1, sticky=tk.W, padx=(10, 0))
        
        self.output_resolution_width_var = tk.IntVar(value=self.settings.get("output_resolution_width", 1920))
        ttk.Spinbox(resolution_input_frame, from_=640, to=7680, textvariable=self.output_resolution_width_var, width=10).grid(
            row=0, column=0, padx=(0, 5), sticky=tk.W
        )
        ttk.Label(resolution_input_frame, text="x").grid(row=0, column=1, padx=(0, 5))
        
        self.output_resolution_height_var = tk.IntVar(value=self.settings.get("output_resolution_height", 1080))
        ttk.Spinbox(resolution_input_frame, from_=480, to=4320, textvariable=self.output_resolution_height_var, width=10).grid(
            row=0, column=2, padx=(0, 5), sticky=tk.W
        )
        ttk.Label(resolution_input_frame, text="pixels").grid(row=0, column=3, padx=(5, 0), sticky=tk.W)
        
        # === FPS SETTING ===
        ttk.Label(output_settings_frame, text="Output FPS:").grid(row=1, column=0, sticky=tk.W, pady=(10, 5))
        
        fps_input_frame = ttk.Frame(output_settings_frame)
        fps_input_frame.grid(row=1, column=1, sticky=tk.W, padx=(10, 0), pady=(10, 0))
        
        self.output_fps_var = tk.IntVar(value=self.settings.get("output_fps", 0))
        ttk.Spinbox(fps_input_frame, from_=0, to=240, textvariable=self.output_fps_var, width=10).grid(
            row=0, column=0, padx=(0, 5), sticky=tk.W
        )
        ttk.Label(fps_input_frame, text="(0 = use source FPS)").grid(row=0, column=1, sticky=tk.W)
        
        return current_row
    
    
    # ========================================
    # PROCESSING SETTINGS SECTION
    # ========================================
    
    def _create_processing_settings_section(self, main_frame, current_row):
        """Create the processing settings section (Num videos, Preview, Multiprocessing)"""
        
        current_row = self._create_section_header(main_frame, current_row, "Processing Settings:")
        
        processing_frame, current_row = self._create_standard_frame(main_frame, current_row)
        
        # === NUMBER OF VIDEOS TO PROCESS ===
        ttk.Label(processing_frame, text="Number of videos to process:").grid(
            row=0, column=0, sticky=tk.W, pady=5
        )
        
        self.num_videos_var = tk.IntVar(
            value=self.settings["num_videos_to_process"] if self.settings["num_videos_to_process"] != -1 else 5
        )
        self.num_videos_spinbox = ttk.Spinbox(
            processing_frame, from_=1, to=1000, textvariable=self.num_videos_var, width=10
        )
        self.num_videos_spinbox.grid(row=0, column=1, padx=(10, 10), sticky=tk.W)
        
        # Process All checkbox with info button
        process_all_frame, self.process_all_var, self.process_all_checkbox = self._create_checkbox_with_info(
            processing_frame,
            "Process All",
            "process_all_var",
            "process_all",
            self.settings["num_videos_to_process"] == -1,
            command=self.toggle_process_all
        )
        process_all_frame.grid(row=0, column=2, padx=(10, 0), sticky=tk.W)
        
        if self.process_all_var.get():
            self.num_videos_spinbox.config(state=tk.DISABLED)
        
        # === SHOW VIDEO PREVIEW ===
        preview_frame, self.show_preview_var, self.show_preview_checkbox = self._create_checkbox_with_info(
            processing_frame,
            "Show Video Preview",
            "show_preview_var",
            "show_preview",
            self.settings.get("show_preview", False)
        )
        preview_frame.grid(row=1, column=0, sticky=tk.W, pady=(10, 0))
        
        # === MULTIPROCESSING MODE ===
        multiprocessing_label_frame = ttk.Frame(processing_frame)
        multiprocessing_label_frame.grid(row=2, column=0, sticky=tk.W, pady=(10, 0), columnspan=3)

        ttk.Label(multiprocessing_label_frame, text="Multiprocessing Mode:").pack(side=tk.LEFT)

        self.multiprocessing_var = tk.StringVar(value=self.settings.get("multiprocessing_mode", "Auto"))
        multiprocessing_dropdown = ttk.Combobox(
            multiprocessing_label_frame,
            textvariable=self.multiprocessing_var,
            values=["Auto", "On", "Off"],
            state="readonly",
            width=10
        )
        multiprocessing_dropdown.pack(side=tk.LEFT, padx=(10, 5))

        self._create_info_button(multiprocessing_label_frame, 'multiprocessing_mode', padx=(0, 0))
        
        return current_row
    
    
    def toggle_process_all(self):
        """Toggle the number of videos spinbox based on 'Process All' checkbox"""
        if self.process_all_var.get():
            self.num_videos_spinbox.config(state=tk.DISABLED)
        else:
            self.num_videos_spinbox.config(state=tk.NORMAL)
    
    
    # ========================================
    # UTILITY METHODS
    # ========================================
    
    def show_tooltip(self, message):
        """Show an info message box"""
        messagebox.showinfo("Information", message)

    def run_script(self):
        script_path = os.path.join(os.path.dirname(__file__), 'run.py')
        subprocess.run(['python', script_path])

    
    # ========================================
    # SETTINGS MANAGEMENT METHODS
    # ========================================
    
    def load_settings(self):
        """Load settings from config file using module functions"""
        try:
            return load_config(gui_mode=True)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load settings: {str(e)}\nUsing defaults.")
            return self.default_settings.copy()
    
    
    def save_settings(self):
        """Save current settings to config file using module functions"""
        current_settings = {
            # === FOLDER SETTINGS ===
            "video_folder": self.video_folder_var.get(),
            "output_folder": self.output_folder_var.get(),
            "unsorted_folder": self.unsorted_folder_var.get(),
            
            # === TRIM SETTINGS ===
            "trim_start_seconds": self.trim_start_var.get(),
            "trim_end_seconds": self.trim_end_var.get(),
            
            # === OUTPUT VIDEO SETTINGS ===
            "output_resolution_width": self.output_resolution_width_var.get(),
            "output_resolution_height": self.output_resolution_height_var.get(),
            "output_fps": self.output_fps_var.get(),
            
            # === PROCESSING SETTINGS ===
            "num_videos_to_process": -1 if self.process_all_var.get() else self.num_videos_var.get(),
            "show_preview": self.show_preview_var.get(),
            "multiprocessing_mode": self.multiprocessing_var.get()
        }
        
        success, error_msg = save_config(current_settings, gui_mode=True)
        
        if success:
            messagebox.showinfo("Success", "Settings saved successfully!")
            self.settings = current_settings
        else:
            messagebox.showerror("Error", error_msg)
    
    
    def reset_defaults(self):
        """Reset all settings to default values"""
        if messagebox.askyesno("Confirm Reset", "Are you sure you want to reset all settings to defaults?"):
            # === FOLDER SETTINGS ===
            self.video_folder_var.set(self.default_settings["video_folder"])
            self.output_folder_var.set(self.default_settings["output_folder"])
            self.unsorted_folder_var.set(self.default_settings["unsorted_folder"])
            
            # === TRIM SETTINGS ===
            self.trim_start_var.set(self.default_settings["trim_start_seconds"])
            self.trim_end_var.set(self.default_settings["trim_end_seconds"])
            
            # === OUTPUT VIDEO SETTINGS ===
            self.output_resolution_width_var.set(self.default_settings["output_resolution_width"])
            self.output_resolution_height_var.set(self.default_settings["output_resolution_height"])
            self.output_fps_var.set(self.default_settings["output_fps"])
            
            # === PROCESSING SETTINGS ===
            self.num_videos_var.set(5)
            self.process_all_var.set(True)
            self.show_preview_var.set(False)
            self.multiprocessing_var.set("Auto")
            self.num_videos_spinbox.config(state=tk.DISABLED)
            
            messagebox.showinfo("Reset", "Settings reset to defaults. Click 'Save Settings' to persist changes.")


# ========================================
# MAIN ENTRY POINT
# ========================================

def main():
    """Main entry point for the GUI application"""
    root = tk.Tk()
    app = CS2ClipperSettings(root)
    root.mainloop()

if __name__ == "__main__":
    main()