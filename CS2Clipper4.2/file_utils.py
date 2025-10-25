"""
CS2 Clipper - File Utilities
Shared file operations and tool path management
"""

import os
import subprocess
import json
import time
import glob
import platform
from pathlib import Path

from constants import (
    SCRIPT_DIR,
    INCOMPLETE_FILE_MAX_AGE_SECONDS,
    INCOMPLETE_FILE_MAX_SIZE_BYTES,
    CONTROL_FILE_PATH
)


# ========================================
# UNIQUE FILENAME GENERATION
# ========================================

def get_unique_filepath(base_path, suffix=''):
    """
    Generate unique filename by adding (#) counter if file exists.
    
    Args:
        base_path (str): Base file path (e.g., "folder/video.mp4")
        suffix (str): Optional suffix before extension (e.g., "_trimmed")
    
    Returns:
        str: Unique file path
    
    Example:
        get_unique_filepath("video.mp4", "_trimmed")
        Returns: "video_trimmed.mp4" or "video_trimmed(2).mp4" if exists
    """
    directory = os.path.dirname(base_path)
    filename = os.path.basename(base_path)
    name, ext = os.path.splitext(filename)
    
    # Add suffix if provided
    if suffix:
        base_output = os.path.join(directory, f"{name}{suffix}{ext}")
    else:
        base_output = base_path
    
    # If file doesn't exist, use it
    if not os.path.exists(base_output):
        return base_output
    
    # Generate numbered versions
    counter = 1
    while True:
        if suffix:
            numbered_path = os.path.join(directory, f"{name}{suffix}({counter}){ext}")
        else:
            numbered_path = os.path.join(directory, f"{name}({counter}){ext}")
        
        if not os.path.exists(numbered_path):
            return numbered_path
        counter += 1


# ========================================
# FILE CLEANUP
# ========================================

def cleanup_incomplete_files(folders, max_age=INCOMPLETE_FILE_MAX_AGE_SECONDS, 
                            max_size=INCOMPLETE_FILE_MAX_SIZE_BYTES, silent=False):
    """
    Search for and remove incomplete/temporary files in specified folders.
    
    Args:
        folders (list): List of folder paths to check
        max_age (int): Maximum file age in seconds
        max_size (int): Maximum file size in bytes (files smaller are suspect)
        silent (bool): Suppress output messages
    
    Returns:
        int: Number of files cleaned up
    """
    cleaned_count = 0
    
    # Patterns for potentially incomplete files
    incomplete_patterns = [
        '*_trimmed.mp4',
        '*_processed.mp4',
        '*_discord.mp4',
    ]
    
    if not silent:
        print("Checking for incomplete files from previous runs...")
    
    for folder in folders:
        if not os.path.exists(folder):
            continue
        
        for pattern in incomplete_patterns:
            full_pattern = os.path.join(folder, pattern)
            
            for file_path in glob.glob(full_pattern):
                try:
                    # Check file age
                    file_age = time.time() - os.path.getmtime(file_path)
                    file_size = os.path.getsize(file_path)
                    
                    # Remove if: recent AND suspiciously small
                    if file_age < max_age and file_size < max_size:
                        os.remove(file_path)
                        cleaned_count += 1
                        
                        if not silent:
                            print(f"  Removed incomplete: {os.path.basename(file_path)}")
                
                except Exception as e:
                    # Silently skip files we can't check/remove
                    pass
    
    if not silent:
        if cleaned_count > 0:
            print(f"✓ Cleaned up {cleaned_count} incomplete file(s)\n")
        else:
            print("  No incomplete files found\n")
    
    return cleaned_count


# ========================================
# TOOL PATH RESOLUTION
# ========================================

def get_tool_path(tool_name, executable_name=None, silent=False):
    """
    Find path to external tool (HandBrakeCLI, ffprobe, etc.).
    Checks local directory first, then system PATH.
    
    Args:
        tool_name (str): Tool name for messages
        executable_name (str): Executable filename (defaults to tool_name)
        silent (bool): Suppress messages
    
    Returns:
        str: Path to executable or None if not found
    """
    if executable_name is None:
        executable_name = tool_name
    
    # Add .exe extension on Windows
    if platform.system() == 'Windows' and not executable_name.endswith('.exe'):
        executable_name = f"{executable_name}.exe"
    
    # Check local directory first
    local_path = os.path.join(SCRIPT_DIR, executable_name)
    if os.path.exists(local_path):
        if not silent:
            print(f"Using local {tool_name}: {local_path}")
        return local_path
    
    # Check if in system PATH
    try:
        # Try to run with --version to verify it exists
        subprocess.run(
            [tool_name, '--version'],
            capture_output=True,
            check=True,
            timeout=5
        )
        if not silent:
            print(f"Using system {tool_name}")
        return tool_name
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return None


def check_tool_available(tool_name, executable_name=None):
    """
    Check if a tool is available without verbose output.
    
    Args:
        tool_name (str): Tool name
        executable_name (str): Executable filename
    
    Returns:
        bool: True if tool is available
    """
    return get_tool_path(tool_name, executable_name, silent=True) is not None


# ========================================
# CONTROL FILE OPERATIONS (Pause/Resume)
# ========================================

def read_control_state(control_file=CONTROL_FILE_PATH):
    """
    Read the current state from control file.
    
    Args:
        control_file (str): Path to control file
    
    Returns:
        str: 'running', 'paused', or 'stopped' (if file doesn't exist)
    """
    if not os.path.exists(control_file):
        return 'stopped'
    
    try:
        with open(control_file, 'r') as f:
            state = f.read().strip()
        return state if state in ['running', 'paused'] else 'running'
    except Exception:
        return 'running'


def write_control_state(state, control_file=CONTROL_FILE_PATH):
    """
    Write state to control file.
    
    Args:
        state (str): State to write ('running', 'paused', 'stopped')
        control_file (str): Path to control file
    
    Returns:
        bool: Success status
    """
    try:
        with open(control_file, 'w') as f:
            f.write(state)
        return True
    except Exception as e:
        print(f"Warning: Could not write control state: {e}")
        return False


def remove_control_file(control_file=CONTROL_FILE_PATH):
    """
    Remove control file.
    
    Args:
        control_file (str): Path to control file
    
    Returns:
        bool: Success status
    """
    if os.path.exists(control_file):
        try:
            os.remove(control_file)
            return True
        except Exception as e:
            print(f"Warning: Could not remove control file: {e}")
            return False
    return True


def wait_while_paused(control_file=CONTROL_FILE_PATH):
    """
    Block execution while process is paused.
    
    Args:
        control_file (str): Path to control file
    
    Returns:
        bool: True if should continue, False if should stop
    """
    print("\n" + "="*60)
    print("⏸️  PROCESSING PAUSED")
    print("="*60)
    print("Waiting for resume signal...")
    print("(Close the console window or delete control file to stop)")
    print("="*60 + "\n")
    
    while True:
        state = read_control_state(control_file)
        
        if state == 'stopped':
            print("\nℹ️  Process control file deleted. Stopping...")
            return False
        elif state == 'running':
            print("\n▶️  PROCESSING RESUMED")
            print("="*60 + "\n")
            return True
        
        time.sleep(1)  # Check every second


# ========================================
# VIDEO METADATA
# ========================================

def get_video_duration(file_path, ffprobe_path=None):
    """
    Get video duration in seconds using ffprobe.
    
    Args:
        file_path (str): Path to video file
        ffprobe_path (str): Path to ffprobe (auto-detected if None)
    
    Returns:
        float: Duration in seconds
    
    Raises:
        Exception: If ffprobe not found or video cannot be read
    """
    if ffprobe_path is None:
        ffprobe_path = get_tool_path('ffprobe', silent=True)
    
    if ffprobe_path is None:
        raise Exception("ffprobe not found. Please install FFmpeg.")
    
    try:
        cmd = [
            ffprobe_path,
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'json',
            file_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=10)
        data = json.loads(result.stdout)
        duration = float(data['format']['duration'])
        return duration
    
    except subprocess.TimeoutExpired:
        raise Exception("ffprobe timed out reading video file")
    except json.JSONDecodeError:
        raise Exception("Could not parse ffprobe output")
    except KeyError:
        raise Exception("Video file has no duration metadata")
    except Exception as e:
        raise Exception(f"Could not read video duration: {str(e)}")


def get_video_fps(file_path, ffprobe_path=None, default=60.0):
    """
    Get video framerate using ffprobe.
    
    Args:
        file_path (str): Path to video file
        ffprobe_path (str): Path to ffprobe (auto-detected if None)
        default (float): Default FPS if detection fails
    
    Returns:
        float: Frames per second
    """
    if ffprobe_path is None:
        ffprobe_path = get_tool_path('ffprobe', silent=True)
    
    if ffprobe_path is None:
        print(f"  Warning: ffprobe not found, assuming {default} FPS")
        return default
    
    try:
        cmd = [
            ffprobe_path,
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=r_frame_rate',
            '-of', 'json',
            file_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=10)
        data = json.loads(result.stdout)
        fps_str = data['streams'][0]['r_frame_rate']
        
        # Parse fraction like "60/1" or "30000/1001"
        num, denom = map(int, fps_str.split('/'))
        fps = num / denom
        return fps
    
    except Exception:
        # Return default on any error
        return default


# ========================================
# FORMAT HELPERS
# ========================================

def format_file_size(size_bytes):
    """
    Format bytes to human-readable string.
    
    Args:
        size_bytes (int/float): Size in bytes
    
    Returns:
        str: Formatted size (e.g., "1.23 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def format_duration(seconds):
    """
    Format seconds to MM:SS or HH:MM:SS.
    
    Args:
        seconds (float): Duration in seconds
    
    Returns:
        str: Formatted duration
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes:02d}:{secs:02d}"


# ========================================
# VALIDATION
# ========================================

def validate_resolution(width, height):
    """
    Validate output resolution.
    
    Args:
        width (int): Width in pixels
        height (int): Height in pixels
    
    Returns:
        tuple: (is_valid: bool, error_message: str or None)
    """
    from constants import (
        MIN_OUTPUT_WIDTH, MIN_OUTPUT_HEIGHT,
        MAX_OUTPUT_WIDTH, MAX_OUTPUT_HEIGHT,
        MIN_ASPECT_RATIO, MAX_ASPECT_RATIO
    )
    
    if width < MIN_OUTPUT_WIDTH or height < MIN_OUTPUT_HEIGHT:
        return False, f"Resolution too small (minimum {MIN_OUTPUT_WIDTH}x{MIN_OUTPUT_HEIGHT})"
    
    if width > MAX_OUTPUT_WIDTH or height > MAX_OUTPUT_HEIGHT:
        return False, f"Resolution too large (maximum {MAX_OUTPUT_WIDTH}x{MAX_OUTPUT_HEIGHT})"
    
    aspect_ratio = width / height if height > 0 else 0
    if aspect_ratio < MIN_ASPECT_RATIO or aspect_ratio > MAX_ASPECT_RATIO:
        return False, f"Invalid aspect ratio ({aspect_ratio:.2f})"
    
    return True, None


def validate_fps(fps):
    """
    Validate FPS value.
    
    Args:
        fps (int): Frames per second
    
    Returns:
        tuple: (is_valid: bool, error_message: str or None)
    """
    from constants import MIN_FPS, MAX_FPS
    
    if fps < MIN_FPS or fps > MAX_FPS:
        return False, f"FPS must be between {MIN_FPS} and {MAX_FPS}"
    
    return True, None