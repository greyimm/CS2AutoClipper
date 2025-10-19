import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import subprocess
import os

class VideoCompressor:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Compressor")
        self.root.geometry("600x600")
        
        self.video_file = None
        self.original_size = 0
        self.estimated_size = 0
        
        self.create_widgets()
        
    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # File Selection Zone
        select_frame = tk.Frame(main_frame, bg="#e0e0e0", relief=tk.RIDGE, bd=2, height=100)
        select_frame.pack(fill=tk.X, pady=(0, 20))
        select_frame.pack_propagate(False)
        
        select_label = tk.Label(
            select_frame, 
            text="Click to Select Video File",
            bg="#e0e0e0",
            font=("Arial", 12),
            fg="#666666",
            cursor="hand2"
        )
        select_label.pack(expand=True)
        
        # Make it clickable
        select_frame.bind("<Button-1>", lambda e: self.browse_file())
        select_label.bind("<Button-1>", lambda e: self.browse_file())
        
        # File info section
        self.info_frame = ttk.LabelFrame(main_frame, text="File Information", padding="10")
        self.info_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.filename_label = ttk.Label(self.info_frame, text="No file selected", wraplength=500)
        self.filename_label.pack(anchor=tk.W)
        
        self.filesize_label = ttk.Label(self.info_frame, text="Original size: --")
        self.filesize_label.pack(anchor=tk.W, pady=(5, 0))
        
        self.estimated_label = ttk.Label(self.info_frame, text="Estimated size: --", foreground="blue")
        self.estimated_label.pack(anchor=tk.W, pady=(5, 0))
        
        # Compression settings
        settings_frame = ttk.LabelFrame(main_frame, text="Compression Settings", padding="10")
        settings_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Quality slider
        quality_frame = ttk.Frame(settings_frame)
        quality_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(quality_frame, text="Quality (CRF):").pack(side=tk.LEFT)
        self.quality_var = tk.IntVar(value=23)
        self.quality_scale = ttk.Scale(
            quality_frame, 
            from_=18, 
            to=35, 
            variable=self.quality_var,
            orient=tk.HORIZONTAL,
            command=self.update_estimate
        )
        self.quality_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
        self.quality_label = ttk.Label(quality_frame, text="23 (Balanced)")
        self.quality_label.pack(side=tk.LEFT)
        
        ttk.Label(settings_frame, text="Lower = Better quality, Larger file | Higher = Lower quality, Smaller file", 
                 font=("Arial", 8), foreground="gray").pack()
        
        # Preset buttons for common quality levels
        preset_frame = ttk.Frame(settings_frame)
        preset_frame.pack(fill=tk.X, pady=(10, 5))
        
        ttk.Label(preset_frame, text="Quick Presets:").pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(preset_frame, text="High Quality", command=lambda: self.set_preset("high"), width=12).pack(side=tk.LEFT, padx=2)
        ttk.Button(preset_frame, text="Balanced", command=lambda: self.set_preset("balanced"), width=12).pack(side=tk.LEFT, padx=2)
        ttk.Button(preset_frame, text="High Compression", command=lambda: self.set_preset("compressed"), width=15).pack(side=tk.LEFT, padx=2)
        
        # Trim settings
        trim_frame = ttk.Frame(settings_frame)
        trim_frame.pack(fill=tk.X, pady=(15, 5))
        
        ttk.Label(trim_frame, text="Start Time (seconds):").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.trim_start_var = tk.IntVar(value=0)
        ttk.Spinbox(trim_frame, from_=0, to=3600, textvariable=self.trim_start_var, width=10).grid(row=0, column=1)
        
        ttk.Label(trim_frame, text="End Time (seconds):").grid(row=1, column=0, sticky=tk.W, pady=(5, 0), padx=(0, 10))
        self.trim_end_var = tk.IntVar(value=0)
        ttk.Spinbox(trim_frame, from_=0, to=3600, textvariable=self.trim_end_var, width=10).grid(row=1, column=1, pady=(5, 0))
        
        ttk.Label(settings_frame, text="(0 = no trimming)", font=("Arial", 8), foreground="gray").pack()
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=(10, 0))
        
        self.estimate_btn = ttk.Button(button_frame, text="Estimate Size", command=self.estimate_size, state=tk.DISABLED)
        self.estimate_btn.pack(side=tk.LEFT, padx=5)
        
        self.compress_btn = ttk.Button(button_frame, text="Compress Video", command=self.compress_video, state=tk.DISABLED)
        self.compress_btn.pack(side=tk.LEFT, padx=5)
        
        # Progress
        self.progress_label = ttk.Label(main_frame, text="", foreground="green")
        self.progress_label.pack(pady=(10, 0))
    
    def browse_file(self):
        """Browse for video file"""
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video files", "*.mp4 *.avi *.mkv *.mov *.flv"), ("All files", "*.*")]
        )
        if file_path:
            self.load_file(file_path)
    
    def load_file(self, file_path):
        """Load video file and display info"""
        self.video_file = file_path
        self.original_size = os.path.getsize(file_path)
        
        # Update UI
        self.filename_label.config(text=f"File: {os.path.basename(file_path)}")
        self.filesize_label.config(text=f"Original size: {self.format_size(self.original_size)}")
        
        # Enable buttons
        self.estimate_btn.config(state=tk.NORMAL)
        self.compress_btn.config(state=tk.NORMAL)
        
        # Auto-estimate
        self.estimate_size()
    
    def format_size(self, size_bytes):
        """Format bytes to human readable"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} TB"
    
    def update_estimate(self, *args):
        """Update quality label and re-estimate"""
        crf = int(self.quality_var.get())
        if crf <= 20:
            quality_text = f"{crf} (High Quality)"
        elif crf <= 25:
            quality_text = f"{crf} (Balanced)"
        else:
            quality_text = f"{crf} (Compressed)"
        self.quality_label.config(text=quality_text)
        
        if self.video_file:
            self.estimate_size()
    
    def set_preset(self, preset_type):
        """Set compression preset"""
        if preset_type == "high":
            self.quality_var.set(20)
        elif preset_type == "balanced":
            self.quality_var.set(23)
        elif preset_type == "compressed":
            self.quality_var.set(28)
        self.update_estimate()
    
    def get_audio_bitrate(self, crf):
        """Calculate audio bitrate based on CRF value"""
        # CRF 18-22: 192k (high quality)
        # CRF 23-26: 128k (balanced)
        # CRF 27-35: 96k (compressed but still serviceable)
        if crf <= 22:
            return "192k"
        elif crf <= 26:
            return "128k"
        else:
            return "96k"  # Minimum serviceable quality
    
    def estimate_size(self):
        """Estimate compressed file size"""
        if not self.video_file:
            return
        
        # Rough estimation based on CRF value
        # Lower CRF = better quality = larger file
        # Higher CRF = lower quality = smaller file
        crf = self.quality_var.get()
        # Invert the ratio: CRF 18 = 70% of original, CRF 35 = 30% of original
        compression_ratio = 1.0 - (crf - 18) * 0.024  # Ranges from ~70% to ~30%
        
        self.estimated_size = int(self.original_size * compression_ratio)
        reduction = ((self.original_size - self.estimated_size) / self.original_size) * 100
        
        self.estimated_label.config(
            text=f"Estimated size: {self.format_size(self.estimated_size)} (~{reduction:.1f}% reduction)"
        )
    
    def compress_video(self):
        """Compress the video using FFmpeg"""
        if not self.video_file:
            messagebox.showerror("Error", "No video file selected")
            return
        
        # Check if FFmpeg is installed
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        except FileNotFoundError:
            messagebox.showerror(
                "FFmpeg Not Found",
                "FFmpeg is required but not installed.\n\n"
                "Please download from: https://ffmpeg.org/download.html\n"
                "or install via: winget install ffmpeg (Windows)"
            )
            return
        
        # Get output filename
        base_name = os.path.splitext(self.video_file)[0]
        output_file = filedialog.asksaveasfilename(
            defaultextension=".mp4",
            initialfile=f"{os.path.basename(base_name)}_compressed.mp4",
            filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")]
        )
        
        if not output_file:
            return
        
        # Build FFmpeg command
        crf = self.quality_var.get()
        audio_bitrate = self.get_audio_bitrate(crf)
        
        cmd = ['ffmpeg', '-i', self.video_file]
        
        # Add trim parameters if set
        trim_start = self.trim_start_var.get()
        trim_end = self.trim_end_var.get()
        
        if trim_start > 0:
            cmd.extend(['-ss', str(trim_start)])
        if trim_end > 0:
            cmd.extend(['-to', str(trim_end)])
        
        # Compression settings with variable audio bitrate
        cmd.extend([
            '-c:v', 'libx264',
            '-crf', str(crf),
            '-preset', 'medium',
            '-c:a', 'aac',
            '-b:a', audio_bitrate,
            '-y',  # Overwrite output file
            output_file
        ])
        
        # Show progress
        self.progress_label.config(text="Compressing... Please wait...")
        self.compress_btn.config(state=tk.DISABLED)
        self.root.update()
        
        try:
            # Run FFmpeg
            subprocess.run(cmd, check=True, capture_output=True)
            
            # Get actual output size
            actual_size = os.path.getsize(output_file)
            
            self.progress_label.config(
                text=f"✓ Compression complete! Final size: {self.format_size(actual_size)}"
            )
            messagebox.showinfo("Success", f"Video compressed successfully!\n\nSaved to: {output_file}")
            
        except subprocess.CalledProcessError as e:
            self.progress_label.config(text="✗ Compression failed", foreground="red")
            messagebox.showerror("Error", f"Compression failed:\n{e.stderr.decode()}")
        finally:
            self.compress_btn.config(state=tk.NORMAL)

def main():
    root = tk.Tk()
    app = VideoCompressor(root)
    root.mainloop()

if __name__ == "__main__":
    main()