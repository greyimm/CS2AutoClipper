import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import json
import os

class CS2ClipperSettings:
    def __init__(self, root):
        self.root = root
        self.root.title("CS2 Clipper Settings")
        self.root.geometry("600x650")
        
        # Default config file path
        self.config_file = "cs2clipper_config.json"
        
        # Default values
        self.default_settings = {
            "video_folder": r"D:\Videos\Counter-strike 2",
            "output_folder": r"C:\Users\dgkbr\OneDrive\Documents\Desktop\Misc\Projects\CS2Clipper\trimmedvideos",
            "unsorted_folder": r"C:\Users\dgkbr\OneDrive\Documents\Desktop\Misc\Projects\CS2Clipper\unsortedfiles",
            "trim_start_seconds": 4,
            "trim_end_seconds": 3,
            "resolution_width": 1920,
            "resolution_height": 1080,
            "num_videos_to_process": -1  # -1 means process all videos
        }
        
        # Load existing settings or use defaults
        self.settings = self.load_settings()
        
        # Create GUI
        self.create_widgets()
        
    def create_widgets(self):
        # Main frame with padding
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Video Folder Section
        ttk.Label(main_frame, text="Video Folder:", font=('Arial', 10, 'bold')).grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        
        folder_frame = ttk.Frame(main_frame)
        folder_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 15))
        
        self.video_folder_var = tk.StringVar(value=self.settings["video_folder"])
        video_entry = ttk.Entry(folder_frame, textvariable=self.video_folder_var, width=50)
        video_entry.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 10))
        
        ttk.Button(folder_frame, text="Browse...", command=self.browse_video_folder).grid(row=0, column=1)
        
        # Output Folder Section
        ttk.Label(main_frame, text="Output Folder:", font=('Arial', 10, 'bold')).grid(row=2, column=0, sticky=tk.W, pady=(0, 5))
        
        output_frame = ttk.Frame(main_frame)
        output_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 15))
        
        self.output_folder_var = tk.StringVar(value=self.settings["output_folder"])
        output_entry = ttk.Entry(output_frame, textvariable=self.output_folder_var, width=50)
        output_entry.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 10))
        
        ttk.Button(output_frame, text="Browse...", command=self.browse_output_folder).grid(row=0, column=1)
        
        # Unsorted Folder Section
        ttk.Label(main_frame, text="Unsorted Folder:", font=('Arial', 10, 'bold')).grid(row=4, column=0, sticky=tk.W, pady=(0, 5))
        
        unsorted_frame = ttk.Frame(main_frame)
        unsorted_frame.grid(row=5, column=0, sticky=(tk.W, tk.E), pady=(0, 15))
        
        self.unsorted_folder_var = tk.StringVar(value=self.settings["unsorted_folder"])
        unsorted_entry = ttk.Entry(unsorted_frame, textvariable=self.unsorted_folder_var, width=50)
        unsorted_entry.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 10))
        
        ttk.Button(unsorted_frame, text="Browse...", command=self.browse_unsorted_folder).grid(row=0, column=1)
        
        # Trim Settings Section
        ttk.Label(main_frame, text="Trim Settings:", font=('Arial', 10, 'bold')).grid(row=6, column=0, sticky=tk.W, pady=(10, 5))
        
        trim_frame = ttk.Frame(main_frame)
        trim_frame.grid(row=7, column=0, sticky=(tk.W, tk.E), pady=(0, 15))
        
        # Trim Start
        ttk.Label(trim_frame, text="Seconds before first kill:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.trim_start_var = tk.IntVar(value=self.settings["trim_start_seconds"])
        trim_start_spinbox = ttk.Spinbox(trim_frame, from_=0, to=30, textvariable=self.trim_start_var, width=10)
        trim_start_spinbox.grid(row=0, column=1, padx=(10, 0), sticky=tk.W)
        ttk.Label(trim_frame, text="seconds").grid(row=0, column=2, padx=(5, 0), sticky=tk.W)
        
        # Trim End
        ttk.Label(trim_frame, text="Seconds after last kill:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.trim_end_var = tk.IntVar(value=self.settings["trim_end_seconds"])
        trim_end_spinbox = ttk.Spinbox(trim_frame, from_=0, to=30, textvariable=self.trim_end_var, width=10)
        trim_end_spinbox.grid(row=1, column=1, padx=(10, 0), sticky=tk.W)
        ttk.Label(trim_frame, text="seconds").grid(row=1, column=2, padx=(5, 0), sticky=tk.W)
        
        # Resolution Settings Section
        ttk.Label(main_frame, text="Video Resolution:", font=('Arial', 10, 'bold')).grid(row=8, column=0, sticky=tk.W, pady=(10, 5))
        
        resolution_frame = ttk.Frame(main_frame)
        resolution_frame.grid(row=9, column=0, sticky=(tk.W, tk.E), pady=(0, 15))
        
        # Resolution Width
        ttk.Label(resolution_frame, text="Width:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.resolution_width_var = tk.IntVar(value=self.settings["resolution_width"])
        width_spinbox = ttk.Spinbox(resolution_frame, from_=640, to=7680, textvariable=self.resolution_width_var, width=10)
        width_spinbox.grid(row=0, column=1, padx=(10, 5), sticky=tk.W)
        ttk.Label(resolution_frame, text="pixels").grid(row=0, column=2, padx=(5, 20), sticky=tk.W)
        
        # Resolution Height
        ttk.Label(resolution_frame, text="Height:").grid(row=0, column=3, sticky=tk.W, pady=5)
        self.resolution_height_var = tk.IntVar(value=self.settings["resolution_height"])
        height_spinbox = ttk.Spinbox(resolution_frame, from_=480, to=4320, textvariable=self.resolution_height_var, width=10)
        height_spinbox.grid(row=0, column=4, padx=(10, 5), sticky=tk.W)
        ttk.Label(resolution_frame, text="pixels").grid(row=0, column=5, padx=(5, 0), sticky=tk.W)
        
        # Processing Settings Section
        ttk.Label(main_frame, text="Processing Settings:", font=('Arial', 10, 'bold')).grid(row=10, column=0, sticky=tk.W, pady=(10, 5))
        
        processing_frame = ttk.Frame(main_frame)
        processing_frame.grid(row=11, column=0, sticky=(tk.W, tk.E), pady=(0, 15))
        
        ttk.Label(processing_frame, text="Number of videos to process:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.num_videos_var = tk.IntVar(value=self.settings["num_videos_to_process"] if self.settings["num_videos_to_process"] != -1 else 5)
        self.num_videos_spinbox = ttk.Spinbox(processing_frame, from_=1, to=1000, textvariable=self.num_videos_var, width=10)
        self.num_videos_spinbox.grid(row=0, column=1, padx=(10, 10), sticky=tk.W)
        
        self.process_all_var = tk.BooleanVar(value=self.settings["num_videos_to_process"] == -1)
        self.process_all_checkbox = ttk.Checkbutton(
            processing_frame, 
            text="Process All", 
            variable=self.process_all_var,
            command=self.toggle_process_all
        )
        self.process_all_checkbox.grid(row=0, column=2, padx=(10, 0), sticky=tk.W)
        
        # Disable spinbox if "Process All" is checked
        if self.process_all_var.get():
            self.num_videos_spinbox.config(state=tk.DISABLED)
        
        # Buttons Section
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=12, column=0, pady=(20, 0))
        
        ttk.Button(button_frame, text="Save Settings", command=self.save_settings, width=15).grid(row=0, column=0, padx=5)
        ttk.Button(button_frame, text="Reset to Defaults", command=self.reset_defaults, width=15).grid(row=0, column=1, padx=5)
        ttk.Button(button_frame, text="Close", command=self.root.quit, width=15).grid(row=0, column=2, padx=5)
        
        # Configure grid weights
        main_frame.columnconfigure(0, weight=1)
        folder_frame.columnconfigure(0, weight=1)
        output_frame.columnconfigure(0, weight=1)
        unsorted_frame.columnconfigure(0, weight=1)
        
    def browse_video_folder(self):
        folder = filedialog.askdirectory(initialdir=self.video_folder_var.get(), title="Select Video Folder")
        if folder:
            self.video_folder_var.set(folder)
    
    def browse_output_folder(self):
        folder = filedialog.askdirectory(initialdir=self.output_folder_var.get(), title="Select Output Folder")
        if folder:
            self.output_folder_var.set(folder)
    
    def browse_unsorted_folder(self):
        folder = filedialog.askdirectory(initialdir=self.unsorted_folder_var.get(), title="Select Unsorted Folder")
        if folder:
            self.unsorted_folder_var.set(folder)
    
    def toggle_process_all(self):
        """Toggle the number of videos spinbox based on 'Process All' checkbox"""
        if self.process_all_var.get():
            self.num_videos_spinbox.config(state=tk.DISABLED)
        else:
            self.num_videos_spinbox.config(state=tk.NORMAL)
    
    def load_settings(self):
        """Load settings from config file or return defaults"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    loaded_settings = json.load(f)
                    # Merge with defaults in case new settings were added
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
            "resolution_width": self.resolution_width_var.get(),
            "resolution_height": self.resolution_height_var.get(),
            "num_videos_to_process": -1 if self.process_all_var.get() else self.num_videos_var.get()
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
            self.resolution_width_var.set(self.default_settings["resolution_width"])
            self.resolution_height_var.set(self.default_settings["resolution_height"])
            self.num_videos_var.set(5)
            self.process_all_var.set(True)
            self.num_videos_spinbox.config(state=tk.DISABLED)
            messagebox.showinfo("Reset", "Settings reset to defaults. Click 'Save Settings' to persist changes.")

def main():
    root = tk.Tk()
    app = CS2ClipperSettings(root)
    root.mainloop()

if __name__ == "__main__":
    main()