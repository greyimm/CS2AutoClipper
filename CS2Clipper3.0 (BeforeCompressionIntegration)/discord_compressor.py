import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import subprocess
import os
import json

class DiscordVideoCompressor:
    def __init__(self, root):
        self.root = root
        self.root.title("Discord Video Compressor")
        self.root.geometry("600x500")
        
        self.video_file = None
        self.video_duration = 0
        self.source_fps = 60
        
        # Compression settings table from DiscordCompression.txt
        self.settings_table = [
            # (max_seconds, video_bitrate, audio_bitrate, resolution, fps)
            (10, 2600, 96, "720p", 60),
            (20, 2600, 96, "720p", 60),
            (30, 1834, 96, "720p", 60),
            (40, 1354, 96, "720p", 60),
            (50, 1066, 96, "540p", 60),
            (60, 874, 96, "540p", 60),
            (70, 736, 96, "540p", 60),
            (80, 634, 96, "540p", 60),
            (90, 554, 96, "360p", 60),
            (100, 490, 96, "360p", 60),
            (110, 437, 96, "360p", 60),
            (120, 394, 96, "360p", 60),
            (180, 262, 64, "240p", 60),
            (240, 182, 64, "240p", 60),
            (300, 127, 64, "240p", 60),
            (360, 124, 32, "240p", 60),
            (420, 102, 32, "240p", 60),
            (480, 86, 32, "240p", 60),
            (540, 74, 32, "240p", 60),
            (600, 63, 32, "240p", 60),
        ]
        
        self.create_widgets()
        
    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = tk.Label(
            main_frame,
            text="Discord Video Compressor",
            font=("Arial", 16, "bold")
        )
        title_label.pack(pady=(0, 10))
        
        subtitle_label = tk.Label(
            main_frame,
            text="Automatically compress videos to under 8MB for Discord",
            font=("Arial", 10),
            fg="#666666"
        )
        subtitle_label.pack(pady=(0, 20))
        
        # File selection zone
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
        
        select_frame.bind("<Button-1>", lambda e: self.browse_file())
        select_label.bind("<Button-1>", lambda e: self.browse_file())
        
        # File info section
        self.info_frame = ttk.LabelFrame(main_frame, text="Video Information", padding="10")
        self.info_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.filename_label = ttk.Label(self.info_frame, text="No file selected", wraplength=500)
        self.filename_label.pack(anchor=tk.W)
        
        self.filesize_label = ttk.Label(self.info_frame, text="Original size: --")
        self.filesize_label.pack(anchor=tk.W, pady=(5, 0))
        
        self.duration_label = ttk.Label(self.info_frame, text="Duration: --")
        self.duration_label.pack(anchor=tk.W, pady=(5, 0))
        
        # Compression settings preview
        self.settings_frame = ttk.LabelFrame(main_frame, text="Compression Settings (Auto)", padding="10")
        self.settings_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.settings_label = ttk.Label(
            self.settings_frame,
            text="Settings will be determined automatically based on video duration",
            foreground="gray"
        )
        self.settings_label.pack(anchor=tk.W)
        
        # Compress button
        self.compress_btn = ttk.Button(
            main_frame,
            text="Compress Video",
            command=self.compress_video,
            state=tk.DISABLED
        )
        self.compress_btn.pack(pady=(10, 10))
        
        # Progress label
        self.progress_label = ttk.Label(main_frame, text="", foreground="green")
        self.progress_label.pack(pady=(10, 0))
        
    def browse_file(self):
        """Browse for video file"""
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mkv *.mov *.flv *.webm"),
                ("All files", "*.*")
            ]
        )
        if file_path:
            self.load_file(file_path)
    
    def format_size(self, size_bytes):
        """Format bytes to human readable"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} TB"
    
    def format_duration(self, seconds):
        """Format seconds to MM:SS"""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"
    
    def get_video_duration(self, file_path):
        """Get video duration using ffprobe"""
        try:
            cmd = [
                'ffprobe',
                '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'json',
                file_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)
            duration = float(data['format']['duration'])
            return duration
        except Exception as e:
            messagebox.showerror("Error", f"Could not read video duration:\n{str(e)}")
            return None
    
    def get_video_fps(self, file_path):
        """Get video framerate using ffprobe"""
        try:
            cmd = [
                'ffprobe',
                '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=r_frame_rate',
                '-of', 'json',
                file_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)
            fps_str = data['streams'][0]['r_frame_rate']
            # Parse fraction like "60/1" or "30000/1001"
            num, denom = map(int, fps_str.split('/'))
            fps = num / denom
            return fps
        except Exception as e:
            # Default to 60 if we can't detect
            return 60.0
    
    def get_compression_settings(self, duration_seconds):
        """Get compression settings based on video duration"""
        for max_seconds, video_br, audio_br, resolution, fps in self.settings_table:
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
    
    def get_resolution_dimensions(self, resolution):
        """Convert resolution string to dimensions"""
        resolutions = {
            "720p": (1280, 720),
            "540p": (960, 540),
            "360p": (640, 360),
            "240p": (426, 240)
        }
        return resolutions.get(resolution, (1280, 720))
    
    def load_file(self, file_path):
        """Load video file and display info"""
        self.video_file = file_path
        original_size = os.path.getsize(file_path)
        
        # Get duration
        duration = self.get_video_duration(file_path)
        if duration is None:
            return
        
        self.video_duration = duration
        
        # Get source FPS
        self.source_fps = self.get_video_fps(file_path)
        
        # Get compression settings
        settings = self.get_compression_settings(duration)
        
        # Determine actual FPS to use (source or 60fps max)
        actual_fps = min(self.source_fps, 60)
        
        # Update UI
        self.filename_label.config(text=f"File: {os.path.basename(file_path)}")
        self.filesize_label.config(text=f"Original size: {self.format_size(original_size)}")
        self.duration_label.config(text=f"Duration: {self.format_duration(duration)}")
        
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
        self.compress_btn.config(state=tk.NORMAL)
    
    def check_handbrake(self):
        """Check if HandBrakeCLI is installed"""
        try:
            subprocess.run(['HandBrakeCLI', '--version'], capture_output=True, check=True)
            return True
        except FileNotFoundError:
            messagebox.showerror(
                "HandBrakeCLI Not Found",
                "HandBrakeCLI is required but not installed.\n\n"
                "Please download from: https://handbrake.fr/downloads.php\n\n"
                "Windows: Use the installer and ensure HandBrakeCLI is in your PATH\n"
                "Mac: brew install handbrake\n"
                "Linux: sudo apt install handbrake-cli"
            )
            return False
    
    def compress_video(self):
        """Compress video using HandBrakeCLI"""
        if not self.video_file:
            messagebox.showerror("Error", "No video file selected")
            return
        
        # Check if HandBrakeCLI is installed
        if not self.check_handbrake():
            return
        
        # Get output filename
        base_name = os.path.splitext(self.video_file)[0]
        output_file = filedialog.asksaveasfilename(
            defaultextension=".mp4",
            initialfile=f"{os.path.basename(base_name)}_discord.mp4",
            filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")]
        )
        
        if not output_file:
            return
        
        # Get compression settings
        settings = self.get_compression_settings(self.video_duration)
        width, height = self.get_resolution_dimensions(settings['resolution'])
        
        # Use source FPS if under 60, otherwise cap at 60
        actual_fps = min(self.source_fps, 60)
        
        # Build HandBrakeCLI command
        cmd = [
            'HandBrakeCLI',
            '-i', self.video_file,
            '-o', output_file,
            '--format', 'av_mp4',
            '--encoder', 'x264',
            '--vb', str(settings['video_bitrate']),
            '--two-pass',
            '--turbo',
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
        
        # Show progress
        self.progress_label.config(text="Compressing... This may take several minutes...")
        self.compress_btn.config(state=tk.DISABLED)
        self.root.update()
        
        try:
            # Run HandBrakeCLI
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )
            
            # Wait for completion
            for line in process.stdout:
                # HandBrake outputs progress, but we'll just wait
                pass
            
            process.wait()
            
            if process.returncode == 0:
                # Get output file size
                output_size = os.path.getsize(output_file)
                output_size_mb = output_size / (1024 * 1024)
                
                # Check if under 8MB
                if output_size_mb <= 8.0:
                    status = "✓ Success!"
                    color = "green"
                else:
                    status = "⚠ Warning: File exceeds 8MB"
                    color = "orange"
                
                self.progress_label.config(
                    text=f"{status} Final size: {self.format_size(output_size)} ({output_size_mb:.2f} MB)",
                    foreground=color
                )
                
                messagebox.showinfo(
                    "Compression Complete",
                    f"Video compressed successfully!\n\n"
                    f"Final size: {self.format_size(output_size)} ({output_size_mb:.2f} MB)\n"
                    f"Saved to: {output_file}"
                )
            else:
                raise Exception("HandBrakeCLI returned an error")
            
        except Exception as e:
            self.progress_label.config(text="✗ Compression failed", foreground="red")
            messagebox.showerror("Error", f"Compression failed:\n{str(e)}")
        
        finally:
            self.compress_btn.config(state=tk.NORMAL)

def main():
    root = tk.Tk()
    app = DiscordVideoCompressor(root)
    root.mainloop()

if __name__ == "__main__":
    main()