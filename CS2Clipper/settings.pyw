import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import json
import os


class CS2ClipperSettings:
    """
    CS2 Clipper Settings GUI
    
    This application provides a user-friendly interface for configuring
    the CS2 Auto Clipper video processing settings.
    """
    
    def __init__(self, root):
        self.root = root
        self.root.title("CS2 Clipper Settings")
        self.root.geometry("700x800")
        
        # Default config file path - always in the same directory as this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.config_file = os.path.join(script_dir, "settings_config.json")
        
        # Default values
        self.default_settings = {
            "video_folder": r"D:\Videos\Counter-strike 2",
            "output_folder": r"C:\Users\dgkbr\OneDrive\Documents\Desktop\Misc\Projects\CS2Clipper\trimmedvideos",
            "unsorted_folder": r"C:\Users\dgkbr\OneDrive\Documents\Desktop\Misc\Projects\CS2Clipper\unsortedfiles",
            "trim_start_seconds": 4,
            "trim_end_seconds": 3,
            "output_resolution_width": 1920,
            "output_resolution_height": 1080,
            "output_fps": 0,
            "num_videos_to_process": -1,
            "show_preview": False,
            "multiprocessing_mode": "Auto"
        }
        
        # Load existing settings or use defaults
        self.settings = self.load_settings()
        
        # Create GUI
        self.create_widgets()
    
    
    # ========================================
    # GUI CREATION METHODS
    # ========================================
    
    def create_widgets(self):
        """Create all GUI widgets"""
        # Main frame with padding and scrollbar support
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure root window to expand
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        current_row = 0
        
        # ===== VIDEO FOLDER =====
        ttk.Label(main_frame, text="Video Folder:", font=('Arial', 10, 'bold')).grid(
            row=current_row, column=0, sticky=tk.W, pady=(0, 5)
        )
        info_btn1 = ttk.Label(main_frame, text=" [?]", foreground="blue", cursor="hand2")
        info_btn1.grid(row=current_row, column=0, sticky=tk.W, padx=(100, 0))
        info_btn1.bind("<Button-1>", lambda e: self.show_tooltip(
            "Source folder containing your CS2 video recordings to be processed"
        ))
        current_row += 1
        
        video_frame = ttk.Frame(main_frame)
        video_frame.grid(row=current_row, column=0, sticky=(tk.W, tk.E), pady=(0, 15))
        video_frame.columnconfigure(0, weight=1)
        
        self.video_folder_var = tk.StringVar(value=self.settings["video_folder"])
        video_entry = ttk.Entry(video_frame, textvariable=self.video_folder_var, width=60)
        video_entry.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 10))
        ttk.Button(video_frame, text="Browse...", command=self.browse_video_folder).grid(row=0, column=1)
        current_row += 1
        
        # ===== OUTPUT FOLDER =====
        ttk.Label(main_frame, text="Output Folder:", font=('Arial', 10, 'bold')).grid(
            row=current_row, column=0, sticky=tk.W, pady=(0, 5)
        )
        info_btn2 = ttk.Label(main_frame, text=" [?]", foreground="blue", cursor="hand2")
        info_btn2.grid(row=current_row, column=0, sticky=tk.W, padx=(110, 0))
        info_btn2.bind("<Button-1>", lambda e: self.show_tooltip(
            "Destination folder where trimmed/processed videos will be saved"
        ))
        current_row += 1
        
        output_frame = ttk.Frame(main_frame)
        output_frame.grid(row=current_row, column=0, sticky=(tk.W, tk.E), pady=(0, 15))
        output_frame.columnconfigure(0, weight=1)
        
        self.output_folder_var = tk.StringVar(value=self.settings["output_folder"])
        output_entry = ttk.Entry(output_frame, textvariable=self.output_folder_var, width=60)
        output_entry.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 10))
        ttk.Button(output_frame, text="Browse...", command=self.browse_output_folder).grid(row=0, column=1)
        current_row += 1
        
        # ===== UNSORTED FOLDER =====
        ttk.Label(main_frame, text="Unsorted Folder:", font=('Arial', 10, 'bold')).grid(
            row=current_row, column=0, sticky=tk.W, pady=(0, 5)
        )
        info_btn3 = ttk.Label(main_frame, text=" [?]", foreground="blue", cursor="hand2")
        info_btn3.grid(row=current_row, column=0, sticky=tk.W, padx=(125, 0))
        info_btn3.bind("<Button-1>", lambda e: self.show_tooltip(
            "Folder for videos that couldn't be processed (0 kills, >5 kills, or errors)"
        ))
        current_row += 1
        
        unsorted_frame = ttk.Frame(main_frame)
        unsorted_frame.grid(row=current_row, column=0, sticky=(tk.W, tk.E), pady=(0, 15))
        unsorted_frame.columnconfigure(0, weight=1)
        
        self.unsorted_folder_var = tk.StringVar(value=self.settings["unsorted_folder"])
        unsorted_entry = ttk.Entry(unsorted_frame, textvariable=self.unsorted_folder_var, width=60)
        unsorted_entry.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 10))
        ttk.Button(unsorted_frame, text="Browse...", command=self.browse_unsorted_folder).grid(row=0, column=1)
        current_row += 1
        
        # ===== TRIM SETTINGS =====
        ttk.Label(main_frame, text="Trim Settings:", font=('Arial', 10, 'bold')).grid(
            row=current_row, column=0, sticky=tk.W, pady=(10, 5)
        )
        current_row += 1
        
        trim_frame = ttk.Frame(main_frame)
        trim_frame.grid(row=current_row, column=0, sticky=(tk.W, tk.E), pady=(0, 15))
        
        ttk.Label(trim_frame, text="Seconds before first kill (Start):").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.trim_start_var = tk.IntVar(value=self.settings["trim_start_seconds"])
        ttk.Spinbox(trim_frame, from_=0, to=30, textvariable=self.trim_start_var, width=10).grid(
            row=0, column=1, padx=(10, 0), sticky=tk.W
        )
        ttk.Label(trim_frame, text="seconds").grid(row=0, column=2, padx=(5, 0), sticky=tk.W)
        
        ttk.Label(trim_frame, text="Seconds after last kill (End):").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.trim_end_var = tk.IntVar(value=self.settings["trim_end_seconds"])
        ttk.Spinbox(trim_frame, from_=0, to=30, textvariable=self.trim_end_var, width=10).grid(
            row=1, column=1, padx=(10, 0), sticky=tk.W
        )
        ttk.Label(trim_frame, text="seconds").grid(row=1, column=2, padx=(5, 0), sticky=tk.W)
        current_row += 1
        
        # ===== OUTPUT VIDEO SETTINGS =====
        ttk.Label(main_frame, text="Output Video Settings:", font=('Arial', 10, 'bold')).grid(
            row=current_row, column=0, sticky=tk.W, pady=(10, 5)
        )
        current_row += 1
        
        output_settings_frame = ttk.Frame(main_frame)
        output_settings_frame.grid(row=current_row, column=0, sticky=(tk.W, tk.E), pady=(0, 15))
        
        # Resolution settings
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
        
        # FPS setting
        ttk.Label(output_settings_frame, text="Output FPS:").grid(row=1, column=0, sticky=tk.W, pady=(10, 5))
        
        fps_input_frame = ttk.Frame(output_settings_frame)
        fps_input_frame.grid(row=1, column=1, sticky=tk.W, padx=(10, 0), pady=(10, 0))
        
        self.output_fps_var = tk.IntVar(value=self.settings.get("output_fps", 0))
        ttk.Spinbox(fps_input_frame, from_=0, to=240, textvariable=self.output_fps_var, width=10).grid(
            row=0, column=0, padx=(0, 5), sticky=tk.W
        )
        ttk.Label(fps_input_frame, text="(0 = use source FPS)").grid(row=0, column=1, sticky=tk.W)
        
        current_row += 1
        
        # ===== PROCESSING SETTINGS =====
        ttk.Label(main_frame, text="Processing Settings:", font=('Arial', 10, 'bold')).grid(
            row=current_row, column=0, sticky=tk.W, pady=(10, 5)
        )
        current_row += 1
        
        processing_frame = ttk.Frame(main_frame)
        processing_frame.grid(row=current_row, column=0, sticky=(tk.W, tk.E), pady=(0, 15))
        
        # Number of videos
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
        process_all_frame = ttk.Frame(processing_frame)
        process_all_frame.grid(row=0, column=2, padx=(10, 0), sticky=tk.W)
        
        self.process_all_var = tk.BooleanVar(value=self.settings["num_videos_to_process"] == -1)
        self.process_all_checkbox = ttk.Checkbutton(
            process_all_frame, text="Process All", variable=self.process_all_var, command=self.toggle_process_all
        )
        self.process_all_checkbox.pack(side=tk.LEFT)
        
        # Info icon for Process All
        info_process_all = ttk.Label(process_all_frame, text=" [?]", foreground="blue", cursor="hand2")
        info_process_all.pack(side=tk.LEFT, padx=(5, 0))
        info_process_all.bind("<Button-1>", lambda e: self.show_tooltip(
            "When enabled: Processes all videos in the video folder\n\n"
            "When disabled: Only processes the specified number of videos"
        ))
        
        if self.process_all_var.get():
            self.num_videos_spinbox.config(state=tk.DISABLED)
        
        # SHOW PREVIEW CHECKBOX
        preview_label_frame = ttk.Frame(processing_frame)
        preview_label_frame.grid(row=1, column=0, sticky=tk.W, pady=(10, 0))

        self.show_preview_var = tk.BooleanVar(value=self.settings.get("show_preview", False))
        self.show_preview_checkbox = ttk.Checkbutton(
            preview_label_frame, 
            text="Show Video Preview",
            variable=self.show_preview_var
        )
        self.show_preview_checkbox.pack(side=tk.LEFT)

        # Info icon for preview
        info_preview = ttk.Label(preview_label_frame, text=" [?]", foreground="blue", cursor="hand2")
        info_preview.pack(side=tk.LEFT, padx=(5, 0))
        info_preview.bind("<Button-1>", lambda e: self.show_tooltip(
            "When enabled: Shows live video preview during processing (slower, useful for debugging)\n\n"
            "When disabled: Processes videos without display (faster, optimized for batch processing)\n\n"
            "Does not work with Multiprocessing."
        ))
        
        # MULTIPROCESSING MODE DROPDOWN
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

        # Info icon for multiprocessing
        info_multiprocessing = ttk.Label(multiprocessing_label_frame, text=" [?]", foreground="blue", cursor="hand2")
        info_multiprocessing.pack(side=tk.LEFT)
        info_multiprocessing.bind("<Button-1>", lambda e: self.show_tooltip(
            "Multiprocessing allows processing multiple videos simultaneously for faster batch processing.\n\n"
            "Auto: Automatically detects your hardware and enables multiprocessing if you have sufficient RAM and CPU cores. "
            "Disables if specs are too low (<6GB RAM).\n\n"
            "On: Forces multiprocessing based on your available hardware (2-4 parallel processes). Use only if you have good hardware.\n\n"
            "Off: Processes videos one at a time (sequential). Safest option for older/slower systems."
        ))

        current_row += 1
        
        # ===== BUTTONS =====
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=current_row, column=0, pady=(20, 0))
        
        ttk.Button(button_frame, text="Save Settings", command=self.save_settings, width=15).grid(row=0, column=0, padx=5)
        ttk.Button(button_frame, text="Reset to Defaults", command=self.reset_defaults, width=15).grid(row=0, column=1, padx=5)
        ttk.Button(button_frame, text="Close", command=self.root.quit, width=15).grid(row=0, column=2, padx=5)
        
        # Configure grid weights
        main_frame.columnconfigure(0, weight=1)
    
    
    # ========================================
    # BROWSE FOLDER METHODS
    # ========================================
    
    def browse_video_folder(self):
        """Browse for video input folder"""
        folder = filedialog.askdirectory(initialdir=self.video_folder_var.get(), title="Select Video Folder")
        if folder:
            self.video_folder_var.set(folder)
    
    
    def browse_output_folder(self):
        """Browse for output folder"""
        folder = filedialog.askdirectory(initialdir=self.output_folder_var.get(), title="Select Output Folder")
        if folder:
            self.output_folder_var.set(folder)
    
    
    def browse_unsorted_folder(self):
        """Browse for unsorted folder"""
        folder = filedialog.askdirectory(initialdir=self.unsorted_folder_var.get(), title="Select Unsorted Folder")
        if folder:
            self.unsorted_folder_var.set(folder)
    
    
    # ========================================
    # UTILITY METHODS
    # ========================================
    
    def toggle_process_all(self):
        """Toggle the number of videos spinbox based on 'Process All' checkbox"""
        if self.process_all_var.get():
            self.num_videos_spinbox.config(state=tk.DISABLED)
        else:
            self.num_videos_spinbox.config(state=tk.NORMAL)
    
    
    def show_tooltip(self, message):
        """Show an info message box"""
        messagebox.showinfo("Information", message)
    
    
    # ========================================
    # SETTINGS MANAGEMENT METHODS
    # ========================================
    
    def load_settings(self):
        """Load settings from config file or return defaults"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    loaded_settings = json.load(f)
                    # Migrate old settings to new naming
                    if "resolution_width" in loaded_settings and "output_resolution_width" not in loaded_settings:
                        loaded_settings["output_resolution_width"] = loaded_settings.pop("resolution_width")
                    if "resolution_height" in loaded_settings and "output_resolution_height" not in loaded_settings:
                        loaded_settings["output_resolution_height"] = loaded_settings.pop("resolution_height")
                    return {**self.default_settings, **loaded_settings}
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load settings: {str(e)}\nUsing defaults.")
                return self.default_settings.copy()
        return self.default_settings.copy()
    
    
    def save_settings(self):
        """Save current settings to config file"""
        current_settings = {
            "video_folder": self.video_folder_var.get(),
            "output_folder": self.output_folder_var.get(),
            "unsorted_folder": self.unsorted_folder_var.get(),
            "trim_start_seconds": self.trim_start_var.get(),
            "trim_end_seconds": self.trim_end_var.get(),
            "output_resolution_width": self.output_resolution_width_var.get(),
            "output_resolution_height": self.output_resolution_height_var.get(),
            "output_fps": self.output_fps_var.get(),
            "num_videos_to_process": -1 if self.process_all_var.get() else self.num_videos_var.get(),
            "show_preview": self.show_preview_var.get(),
            "multiprocessing_mode": self.multiprocessing_var.get()
        }
        
        try:
            with open(self.config_file, 'w') as f:
                json.dump(current_settings, f, indent=4)
            messagebox.showinfo("Success", "Settings saved successfully!")
            self.settings = current_settings
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save settings: {str(e)}")
    
    
    def reset_defaults(self):
        """Reset all settings to default values"""
        if messagebox.askyesno("Confirm Reset", "Are you sure you want to reset all settings to defaults?"):
            self.video_folder_var.set(self.default_settings["video_folder"])
            self.output_folder_var.set(self.default_settings["output_folder"])
            self.unsorted_folder_var.set(self.default_settings["unsorted_folder"])
            self.trim_start_var.set(self.default_settings["trim_start_seconds"])
            self.trim_end_var.set(self.default_settings["trim_end_seconds"])
            self.output_resolution_width_var.set(self.default_settings["output_resolution_width"])
            self.output_resolution_height_var.set(self.default_settings["output_resolution_height"])
            self.output_fps_var.set(self.default_settings["output_fps"])
            self.num_videos_var.set(5)
            self.process_all_var.set(True)
            self.show_preview_var.set(False)
            self.num_videos_spinbox.config(state=tk.DISABLED)
            self.multiprocessing_var.set("Auto")
            messagebox.showinfo("Reset", "Settings reset to defaults. Click 'Save Settings' to persist changes.")


# ========================================
# MAIN ENTRY POINT
# ========================================

def main():
    """Main entry point for the application"""
    root = tk.Tk()
    app = CS2ClipperSettings(root)
    root.mainloop()


if __name__ == "__main__":
    main()