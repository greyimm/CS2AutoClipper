"""
Discord Video Compression Module

This module provides:
1. Programmatic compression function for use in run.py
2. GUI application for standalone compression
3. Shared compression settings and utilities
"""

import subprocess
import os
import json

# Import shared utilities and constants
from constants import (
    DISCORD_COMPRESSION_TABLE,
    RESOLUTION_DIMENSIONS,
    DISCORD_MAX_SIZE_BYTES
)

from file_utils import (
    get_tool_path,
    check_tool_available,
    get_unique_filepath,
    get_video_duration,
    get_video_fps,
    format_file_size,
    format_duration
)


# ========================================
# COMPRESSION SETTINGS
# ========================================

def get_resolution_dimensions(resolution):
    """
    Convert resolution string to dimensions.
    
    Args:
        resolution (str): Resolution name (e.g., "720p")
    
    Returns:
        tuple: (width, height) in pixels
    """
    return RESOLUTION_DIMENSIONS.get(resolution, (1280, 720))


def get_compression_settings(duration_seconds):
    """
    Get compression settings based on video duration.
    
    Args:
        duration_seconds (float): Video duration in seconds
    
    Returns:
        dict: Compression settings (video_bitrate, audio_bitrate, resolution, fps, mono)
    """
    for max_seconds, video_br, audio_br, resolution, fps in DISCORD_COMPRESSION_TABLE:
        if duration_seconds <= max_seconds:
            return {
                'video_bitrate': video_br,
                'audio_bitrate': audio_br,
                'resolution': resolution,
                'fps': fps,
                'mono': audio_br == 32 and duration_seconds >= 360
            }
    
    # If video is longer than 10 minutes, use the last setting
    return {
        'video_bitrate': 63,
        'audio_bitrate': 32,
        'resolution': "240p",
        'fps': 60,
        'mono': True
    }


# ========================================
# PROGRAMMATIC COMPRESSION FUNCTION
# ========================================

def compress_video_for_discord(input_file, output_file=None, silent=False):
    """
    Compress a video file for Discord (under 8MB) programmatically.
    
    Args:
        input_file (str): Path to input video file
        output_file (str): Path to output file (optional, will auto-generate if None)
        silent (bool): If True, suppress print statements
    
    Returns:
        tuple: (success: bool, output_path: str or None, error_message: str or None)
    """
    try:
        # Check if HandBrakeCLI is available
        handbrake_path = get_tool_path('HandBrakeCLI', silent=True)
        if handbrake_path is None:
            return (False, None, "HandBrakeCLI is not installed or not in PATH")
        
        # Get video duration and FPS
        try:
            duration = get_video_duration(input_file)
            source_fps = get_video_fps(input_file)
        except Exception as e:
            return (False, None, f"Could not read video metadata: {str(e)}")
        
        # Get compression settings
        settings = get_compression_settings(duration)
        
        # Generate output filename if not provided
        if output_file is None:
            base_name = os.path.splitext(input_file)[0]
            output_file = f"{base_name}_discord.mp4"
        
        # Ensure unique filename
        output_file = get_unique_filepath(output_file)
        
        # Get resolution dimensions
        width, height = get_resolution_dimensions(settings['resolution'])
        
        # Use source FPS if under 60, otherwise cap at 60
        actual_fps = min(source_fps, 60)
        
        if not silent:
            print(f"  Compressing for Discord: {os.path.basename(input_file)}")
            print(f"    Duration: {duration:.1f}s, Settings: {settings['resolution']} @ {actual_fps:.0f}fps, {settings['video_bitrate']}kbps")
        
        # Build HandBrakeCLI command
        cmd = [
            handbrake_path,
            '-i', input_file,
            '-o', output_file,
            '--format', 'av_mp4',
            '--encoder', 'x264',
            '--vb', str(settings['video_bitrate']),
            '--encoder-preset', 'medium',
            '--encoder-profile', 'high',
            '--width', str(width),
            '--height', str(height),
            '--rate', str(actual_fps),
            '--cfr',
            '--aencoder', 'av_aac',
            '--ab', str(settings['audio_bitrate'])
        ]
        
        # Add mono mixdown if needed
        if settings['mono']:
            cmd.extend(['--mixdown', 'mono'])
        else:
            cmd.extend(['--mixdown', 'stereo'])
        
        # Run HandBrakeCLI
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        
        # Capture output for debugging
        output_lines = []
        for line in process.stdout:
            output_lines.append(line)
        
        process.wait()
        
        if process.returncode == 0:
            # Verify output file exists and has size
            if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
                return (False, None, "HandBrakeCLI did not produce valid output file")
            
            output_size = os.path.getsize(output_file)
            output_size_mb = output_size / (1024 * 1024)
            
            if not silent:
                if output_size_mb <= 8.0:
                    print(f"    âœ“ Success! Final size: {output_size_mb:.2f} MB")
                else:
                    print(f"    âš  Warning: File exceeds 8MB ({output_size_mb:.2f} MB)")
            
            return (True, output_file, None)
        else:
            # Show the actual error from HandBrakeCLI
            error_output = ''.join(output_lines[-10:])
            return (False, None, f"HandBrakeCLI compression failed:\n{error_output}")
    
    except Exception as e:
        return (False, None, str(e))


# ========================================
# GUI APPLICATION CLASS
# ========================================

class DiscordVideoCompressor:
    """GUI application for Discord video compression"""
    
    def __init__(self, root):
        # Import tkinter here to avoid issues when only using compression function
        import tkinter as tk
        from tkinter import ttk, filedialog, messagebox
        
        # Store tkinter modules as instance variables
        self.tk = tk
        self.ttk = ttk
        self.filedialog = filedialog
        self.messagebox = messagebox
        
        self.root = root
        self.root.title("Discord Video Compressor")
        self.root.geometry("600x500")
        
        self.video_file = None
        self.video_duration = 0
        self.source_fps = 60
        
        self.create_widgets()
        
    def create_widgets(self):
        """Create all GUI widgets"""
        # Main frame
        main_frame = self.tself.tk.Frame(self.root, padding="20")
        main_frame.pack(fill=self.tk.BOTH, expand=True)
        
        # Title
        title_label = self.tk.Label(
            main_frame,
            text="Discord Video Compressor",
            font=("Arial", 16, "bold")
        )
        title_label.pack(pady=(0, 10))
        
        subtitle_label = self.tk.Label(
            main_frame,
            text="Automatically compress videos to under 8MB for Discord",
            font=("Arial", 10),
            fg="#666666"
        )
        subtitle_label.pack(pady=(0, 20))
        
        # File selection zone
        select_frame = self.tk.Frame(main_frame, bg="#e0e0e0", relief=self.tk.RIDGE, bd=2, height=100)
        select_frame.pack(fill=self.tk.X, pady=(0, 20))
        select_frame.pack_propagate(False)
        
        select_label = self.tk.Label(
            select_frame,
            text="Click to Select Video File",
            bg="#e0e0e0",
            font=("Arial", 12),
            fg="#666666",
            cursor="hand2"
        )
        select_label.pack(expand=True)
        
        select_frame.bind("<Button-1>", lambda e: self.browse_file())
        select_label.bind("<Button-1>", lambda e: self.browse_file())
        
        # File info section
        self.info_frame = self.self.tself.tk.LabelFrame(main_frame, text="Video Information", padding="10")
        self.info_frame.pack(fill=self.tk.X, pady=(0, 15))
        
        self.filename_label = self.tself.tk.Label(self.info_frame, text="No file selected", wraplength=500)
        self.filename_label.pack(anchor=self.tk.W)
        
        self.filesize_label = self.tself.tk.Label(self.info_frame, text="Original size: --")
        self.filesize_label.pack(anchor=self.tk.W, pady=(5, 0))
        
        self.duration_label = self.tself.tk.Label(self.info_frame, text="Duration: --")
        self.duration_label.pack(anchor=self.tk.W, pady=(5, 0))
        
        # Compression settings preview
        self.settings_frame = self.self.tself.tk.LabelFrame(main_frame, text="Compression Settings (Auto)", padding="10")
        self.settings_frame.pack(fill=self.tk.X, pady=(0, 15))
        
        self.settings_label = self.tself.tk.Label(
            self.settings_frame,
            text="Settings will be determined automatically based on video duration",
            foreground="gray"
        )
        self.settings_label.pack(anchor=self.tk.W)
        
        # Compress button
        self.compress_btn = self.ttk.Button(
            main_frame,
            text="Compress Video",
            command=self.compress_video,
            state=self.tk.DISABLED
        )
        self.compress_btn.pack(pady=(10, 10))
        
        # Progress label
        self.progress_label = self.tself.tk.Label(main_frame, text="", foreground="green")
        self.progress_label.pack(pady=(10, 0))
        
    def browse_file(self):
        """Browse for video file"""
        file_path = self.filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mkv *.mov *.flv *.webm"),
                ("All files", "*.*")
            ]
        )
        if file_path:
            self.load_file(file_path)
    
    def load_file(self, file_path):
        """Load video file and display info"""
        try:
            self.video_file = file_path
            original_size = os.path.getsize(file_path)
            
            # Get duration and FPS
            self.video_duration = get_video_duration(file_path)
            self.source_fps = get_video_fps(file_path)
            
            # Get compression settings
            settings = get_compression_settings(self.video_duration)
            
            # Determine actual FPS to use (source or 60fps max)
            actual_fps = min(self.source_fps, 60)
            
            # Update UI
            self.filename_label.config(text=f"File: {os.path.basename(file_path)}")
            self.filesize_label.config(text=f"Original size: {format_file_size(original_size)}")
            self.duration_label.config(text=f"Duration: {format_duration(self.video_duration)}")
            
            # Display settings
            settings_text = (
                f"Resolution: {settings['resolution']} @ {actual_fps:.0f}fps (source: {self.source_fps:.0f}fps)\n"
                f"Video Bitrate: {settings['video_bitrate']} kbps\n"
                f"Audio Bitrate: {settings['audio_bitrate']} kbps"
            )
            if settings['mono']:
                settings_text += " (Mono)"
            else:
                settings_text += " (Stereo)"
            
            self.settings_label.config(text=settings_text, foreground="black")
            
            # Enable compress button
            self.compress_btn.config(state=self.tk.NORMAL)
            
        except Exception as e:
            self.messagebox.showerror("Error", str(e))
    
    def compress_video(self):
        """Compress video using HandBrakeCLI"""
        if not self.video_file:
            self.messagebox.showerror("Error", "No video file selected")
            return
        
        # Check if HandBrakeCLI is installed
        if not check_tool_available('HandBrakeCLI'):
            self.messagebox.showerror(
                "HandBrakeCLI Not Found",
                "HandBrakeCLI is required but not installed.\n\n"
                "Please download from: https://handbrake.fr/downloads.php\n\n"
                "Windows: Use the installer and ensure HandBrakeCLI is in your PATH\n"
                "Mac: brew install handbrake\n"
                "Linux: sudo apt install handbrake-cli"
            )
            return
        
        # Get output filename
        base_name = os.path.splitext(self.video_file)[0]
        output_file = self.filedialog.asksaveasfilename(
            defaultextension=".mp4",
            initialfile=f"{os.path.basename(base_name)}_discord.mp4",
            filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")]
        )
        
        if not output_file:
            return
        
        # Show progress
        self.progress_label.config(text="Compressing... This may take several minutes...")
        self.compress_btn.config(state=self.tk.DISABLED)
        self.root.update()
        
        # Compress using shared function
        success, result_path, error_msg = compress_video_for_discord(
            self.video_file,
            output_file,
            silent=True
        )
        
        if success:
            # Get output file size
            output_size = os.path.getsize(result_path)
            output_size_mb = output_size / (1024 * 1024)
            
            # Check if under 8MB
            if output_size_mb <= 8.0:
                status = "âœ“ Success!"
                color = "green"
            else:
                status = "âš  Warning: File exceeds 8MB"
                color = "orange"
            
            self.progress_label.config(
                text=f"{status} Final size: {format_file_size(output_size)} ({output_size_mb:.2f} MB)",
                foreground=color
            )
            
            self.messagebox.showinfo(
                "Compression Complete",
                f"Video compressed successfully!\n\n"
                f"Final size: {format_file_size(output_size)} ({output_size_mb:.2f} MB)\n"
                f"Saved to: {result_path}"
            )
        else:
            self.progress_label.config(text="âœ— Compression failed", foreground="red")
            self.messagebox.showerror("Error", f"Compression failed:\n{error_msg}")
        
        # Re-enable button
        self.compress_btn.config(state=self.tk.NORMAL)


# ========================================
# MAIN ENTRY POINT
# ========================================

def main():
    """Main entry point for the GUI application"""
    import tkinter as tk
    root = tk.Tk()
    app = DiscordVideoCompressor(root)
    root.mainloop()

if __name__ == "__main__":
    main()