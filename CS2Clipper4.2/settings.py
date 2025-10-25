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
import threading
import time

# Import shared utilities and constants
from constants import (
    SCRIPT_DIR,
    CONFIG_FILE_NAME,
    DEFAULT_SETTINGS,
    CONTROL_FILE_PATH
)

from file_utils import (
    get_unique_filepath,
    cleanup_incomplete_files,
    write_control_state,
    read_control_state,
    remove_control_file,
    validate_resolution,
    validate_fps
)


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
    return os.path.join(SCRIPT_DIR, CONFIG_FILE_NAME)


def get_default_settings():
    """
    Get the default configuration settings.
    
    Returns:
        dict: Default configuration dictionary
    """
    return DEFAULT_SETTINGS.copy()


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
                
                # Merge with defaults (defaults provide any missing keys)
                config = {**default_config, **loaded_config}
                
                return config
                
        except Exception as e:
            if gui_mode:
                raise
            
            print(f"Warning: Could not load config file: {e}")
            print("Using default settings.")
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
    print(f"  Compress for Discord: {'Enabled' if config['compress_for_discord'] else 'Disabled'}")


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
        ),
        'compress_for_discord': (
            "When enabled: Creates ONLY Discord-compressed versions (<8MB) with '_discord' suffix. Your Output Video Settings will be ignored.\n\n"
            "When disabled: Creates regular processed videos using your Output Video Settings.\n\n"
            "⚠️ IMPORTANT: This is an either/or choice:\n"
            "✓ Checked = Discord-optimized videos only (automatic settings)\n"
            "✓ Unchecked = Regular videos only (your custom settings)\n\n"
            "Discord compression uses automatic settings based on video duration:\n"
            "• 10s video → 720p @ 60fps, 2600kbps\n"
            "• 60s video → 540p @ 60fps, 874kbps\n"
            "• 120s video → 360p @ 60fps, 394kbps\n\n"
            "Requires HandBrakeCLI to be installed (handbrake.fr)."
        ),
        'save_versions_mode': (
            "Controls which video versions are saved:\n\n"
            "• Trimmed and Compressed: Saves trimmed file and compressed trimmed file\n"
            "• Trimmed and Original: Saves original source file and trimmed file\n"
            "• Original and Compressed: Saves compressed trimmed file and original source file\n"
            "• Save All: Saves original source file, compressed trimmed file, AND trimmed file"
        )
    }
    
    def __init__(self, root):
        self.root = root
        self.root.title("CS2 Clipper Settings")
        self.root.geometry("430x680")
        self.root.minsize(430, 500)

        # Register window close handler
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Load existing settings
        self.settings = load_config(gui_mode=True)
        
        # Create GUI
        self.create_widgets()

        # Process tracking variables
        self.process = None
        self.process_thread = None
    
    
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
    
    
    def _create_checkbox_with_info(self, parent, checkbox_text, var_name, tooltip_key, 
                                   default_value, command=None):
        """Create a checkbox with an info button - returns frame containing checkbox"""
        frame = ttk.Frame(parent)
        
        bool_var = tk.BooleanVar(value=default_value)
        setattr(self, var_name, bool_var)
        
        ttk.Checkbutton(frame, text=checkbox_text, variable=bool_var, command=command).pack(side=tk.LEFT)
        self._create_info_button(frame, tooltip_key)
        
        return frame
    
    
    def _create_info_button(self, parent, tooltip_key, padx=(5, 0)):
        """Create a standardized info button [?] that shows a tooltip"""
        info_btn = ttk.Label(parent, text=" [?]", foreground="blue", cursor="hand2")
        info_btn.pack(side=tk.LEFT, padx=padx)
        info_btn.bind("<Button-1>", lambda e: self.show_tooltip(self.HELP_TOOLTIPS[tooltip_key]))
        return info_btn
    
    
    def _create_folder_input(self, parent, current_row, label_text, var_name, tooltip_key, browse_command):
        """Create a standardized folder input section"""
        # Create label with info button in a frame
        label_frame = ttk.Frame(parent)
        label_frame.grid(row=current_row, column=0, sticky=tk.W, pady=(0, 5))
        
        ttk.Label(label_frame, text=label_text, font=('Arial', 10, 'bold')).pack(side=tk.LEFT)
        self._create_info_button(label_frame, tooltip_key, padx=(5, 0))
        
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
        """Handle mouse wheel scrolling"""
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
    
    
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
        
        # Store the Run button as an instance variable
        self.run_btn = ttk.Button(
            button_container, 
            text="Run", 
            command=self.run_script,
            width=15
        )
        self.run_btn.grid(row=0, column=2, padx=(5, 0))
    
    
    def on_closing(self):
        """Handle window close event (X button)"""
        if self.process is not None and self.process.poll() is None:
            # Process is still running
            response = messagebox.askyesno(
                "Processing Active",
                "Video processing is currently running.\n\n"
                "Closing this window will stop processing and cleanup incomplete files.\n\n"
                "Are you sure you want to stop?"
            )
            
            if not response:
                return  # User chose not to close
            
            # User confirmed - stop the process
            print("\n" + "="*60)
            print("STOPPING PROCESS AND CLEANING UP")
            print("="*60)
            
            # Write 'stopped' to control file
            write_control_state('stopped')
            print("[OK] Stop signal sent to subprocess")
            
            # Give process time to cleanup
            print("Waiting for subprocess to cleanup... (3 seconds)")
            for i in range(6):
                time.sleep(0.5)
                if self.process.poll() is not None:
                    print("[OK] Subprocess exited gracefully")
                    break
                if i % 2 == 0:
                    print(".", end="", flush=True)
            else:
                print("\nSubprocess didn't exit, forcing termination...")
            
            # Terminate the subprocess if still running
            if self.process.poll() is None:
                print("Sending SIGTERM to subprocess...")
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                    print("[OK] Subprocess terminated")
                except subprocess.TimeoutExpired:
                    print("Subprocess didn't respond to SIGTERM, forcing kill...")
                    self.process.kill()
                    try:
                        self.process.wait(timeout=2)
                        print("[OK] Subprocess killed")
                    except:
                        print("[WARNING] Subprocess may still be running")
            
            # Clean up control file
            remove_control_file()
            print("[OK] Control file removed")
            
            # Cleanup incomplete output files
            print("\nChecking for incomplete files...")
            output_folder = self.output_folder_var.get()
            unsorted_folder = self.unsorted_folder_var.get()
            
            folders = [f for f in [output_folder, unsorted_folder] if os.path.exists(f)]
            incomplete_cleaned = cleanup_incomplete_files(folders, silent=True)
            
            if incomplete_cleaned > 0:
                print(f"[OK] Cleaned up {incomplete_cleaned} incomplete file(s)")
            else:
                print("No incomplete files found")
            
            print("="*60 + "\n")
        
        # Close the window
        self.root.destroy()


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
            lambda: self._browse_folder(self.video_folder_var, "Select Video Folder")
        )
        
        current_row = self._create_folder_input(
            main_frame, current_row, 
            "Output Folder:", 
            "output_folder_var", 
            "output_folder", 
            lambda: self._browse_folder(self.output_folder_var, "Select Output Folder")
        )
        
        current_row = self._create_folder_input(
            main_frame, current_row, 
            "Unsorted Folder:", 
            "unsorted_folder_var", 
            "unsorted_folder", 
            lambda: self._browse_folder(self.unsorted_folder_var, "Select Unsorted Folder")
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
        """Create the processing settings section"""
        
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
        self.num_videos_spinbox.grid(row=0, column=1, padx=(10, 0), sticky=tk.W)
        
        # === PROCESS ALL ===
        process_all_frame = self._create_checkbox_with_info(
            processing_frame,
            "Process All",
            "process_all_var",
            "process_all",
            self.settings["num_videos_to_process"] == -1,
            command=self.toggle_process_all
        )
        process_all_frame.grid(row=1, column=0, sticky=tk.W, pady=(5, 0))
        
        if self.process_all_var.get():
            self.num_videos_spinbox.config(state=tk.DISABLED)
        
        # === SHOW VIDEO PREVIEW ===
        preview_frame = self._create_checkbox_with_info(
            processing_frame,
            "Show Video Preview",
            "show_preview_var",
            "show_preview",
            self.settings.get("show_preview", False)
        )
        preview_frame.grid(row=2, column=0, sticky=tk.W, pady=(10, 0))
        
        # === MULTIPROCESSING MODE ===
        multiprocessing_label_frame = ttk.Frame(processing_frame)
        multiprocessing_label_frame.grid(row=3, column=0, sticky=tk.W, pady=(10, 0), columnspan=3)

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
        
        # === SAVE VERSIONS MODE ===
        save_versions_label_frame = ttk.Frame(processing_frame)
        save_versions_label_frame.grid(row=4, column=0, sticky=tk.W, pady=(10, 0), columnspan=3)

        ttk.Label(save_versions_label_frame, text="Save Versions:").pack(side=tk.LEFT)

        self.save_versions_mode_var = tk.StringVar(value=self.settings.get("save_versions_mode", "Trimmed and Compressed"))
        save_versions_dropdown = ttk.Combobox(
            save_versions_label_frame,
            textvariable=self.save_versions_mode_var,
            values=["Trimmed and Compressed", "Trimmed and Original", "Original and Compressed", "Save All"],
            state="readonly",
            width=25
        )
        save_versions_dropdown.pack(side=tk.LEFT, padx=(10, 5))

        self._create_info_button(save_versions_label_frame, 'save_versions_mode', padx=(0, 0))
        
        # === COMPRESS FOR DISCORD ===
        discord_frame = self._create_checkbox_with_info(
            processing_frame,
            "Compress for Discord",
            "compress_for_discord_var",
            "compress_for_discord",
            self.settings.get("compress_for_discord", False)
        )
        discord_frame.grid(row=5, column=0, sticky=tk.W, pady=(10, 0))

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
        """Run the processing script with pause/resume capability"""
        # Check current state
        if self.run_btn.cget("text") == "Run":
            # Start the process
            script_path = os.path.join(SCRIPT_DIR, 'run.py')
            
            try:
                # Create control file with "running" state
                write_control_state('running')
                
                # Start subprocess
                if platform.system() == 'Windows':
                    self.process = subprocess.Popen(
                        ['python', script_path],
                        creationflags=subprocess.CREATE_NEW_CONSOLE
                    )
                else:
                    self.process = subprocess.Popen(['python', script_path])
                
                # Change button to Pause
                self.run_btn.config(text="Pause")
                
                # Start monitoring thread
                self.process_thread = threading.Thread(target=self.monitor_process, daemon=True)
                self.process_thread.start()
                
                messagebox.showinfo(
                    "Processing Started", 
                    "Video processing has started.\n\n"
                    "Click 'Pause' to pause processing.\n"
                    "You can continue using the settings window."
                )
                
            except Exception as e:
                # Clean up control file on error
                remove_control_file()
                messagebox.showerror("Error", f"Failed to start processing:\n{str(e)}")
        
        elif self.run_btn.cget("text") == "Pause":
            # Pause the process
            try:
                write_control_state('paused')
                self.run_btn.config(text="Resume")
                messagebox.showinfo("Paused", "Processing will pause after the current video completes.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to pause:\n{str(e)}")
        
        elif self.run_btn.cget("text") == "Resume":
            # Resume the process
            try:
                write_control_state('running')
                self.run_btn.config(text="Pause")
                messagebox.showinfo("Resumed", "Processing resumed.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to resume:\n{str(e)}")


    def monitor_process(self):
        """Monitor the subprocess and reset button when it finishes"""
        if self.process:
            self.process.wait()  # Wait for process to complete
            
            # Clean up control file
            remove_control_file()
            
            # Reset button to Run (use after() to update from main thread)
            self.root.after(0, self.reset_run_button)


    def reset_run_button(self):
        """Reset the run button to initial state"""
        self.run_btn.config(text="Run")
        self.process = None
        
    # ========================================
    # SETTINGS MANAGEMENT METHODS
    # ========================================
        
    def save_settings(self):
        """Save current settings to config file"""
        # Validate resolution
        width = self.output_resolution_width_var.get()
        height = self.output_resolution_height_var.get()
        valid, error_msg = validate_resolution(width, height)
        if not valid:
            messagebox.showerror("Invalid Resolution", error_msg)
            return
        
        # Validate FPS
        fps = self.output_fps_var.get()
        valid, error_msg = validate_fps(fps)
        if not valid:
            messagebox.showerror("Invalid FPS", error_msg)
            return
        
        # Build settings dictionary
        current_settings = {
            # === FOLDER SETTINGS ===
            "video_folder": self.video_folder_var.get(),
            "output_folder": self.output_folder_var.get(),
            "unsorted_folder": self.unsorted_folder_var.get(),
            
            # === TRIM SETTINGS ===
            "trim_start_seconds": self.trim_start_var.get(),
            "trim_end_seconds": self.trim_end_var.get(),
            
            # === OUTPUT VIDEO SETTINGS ===
            "output_resolution_width": width,
            "output_resolution_height": height,
            "output_fps": fps,
            
            # === PROCESSING SETTINGS ===
            "num_videos_to_process": -1 if self.process_all_var.get() else self.num_videos_var.get(),
            "show_preview": self.show_preview_var.get(),
            "multiprocessing_mode": self.multiprocessing_var.get(),
            "compress_for_discord": self.compress_for_discord_var.get(),
            "save_versions_mode": self.save_versions_mode_var.get()
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
            defaults = get_default_settings()
            
            # === FOLDER SETTINGS ===
            self.video_folder_var.set(defaults["video_folder"])
            self.output_folder_var.set(defaults["output_folder"])
            self.unsorted_folder_var.set(defaults["unsorted_folder"])
            
            # === TRIM SETTINGS ===
            self.trim_start_var.set(defaults["trim_start_seconds"])
            self.trim_end_var.set(defaults["trim_end_seconds"])
            
            # === OUTPUT VIDEO SETTINGS ===
            self.output_resolution_width_var.set(defaults["output_resolution_width"])
            self.output_resolution_height_var.set(defaults["output_resolution_height"])
            self.output_fps_var.set(defaults["output_fps"])
            
            # === PROCESSING SETTINGS ===
            self.num_videos_var.set(5)
            self.process_all_var.set(True)
            self.show_preview_var.set(False)
            self.multiprocessing_var.set("Auto")
            self.compress_for_discord_var.set(False)
            self.save_versions_mode_var.set("Trimmed and Compressed")
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