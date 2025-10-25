"""
CS2 Auto Clipper - Video Processing and Kill Detection

This module handles:
1. Video processing and frame analysis
2. Kill detection (red rectangle tracking in killfeed)
3. Video trimming and output
4. Optional Discord compression
"""

import cv2
import numpy as np
import os
import multiprocessing
from multiprocessing import Pool, cpu_count
import shutil
import platform
import time

# Import configuration management
from settings import load_config, print_config_summary

# Import shared utilities and constants
from constants import (
    CONTROL_FILE_PATH,
    PAUSE_CHECK_INTERVAL_FRAMES,
    MAX_KILLS_FOR_AUTO_PROCESSING,
    REFERENCE_WIDTH, REFERENCE_HEIGHT,
    RECT_MIN_WIDTH, RECT_MIN_HEIGHT, RECT_MAX_HEIGHT, RECT_MIN_AREA,
    RECT_MIN_WIDTH_THRESHOLD, RECT_MIN_WHITE_RATIO,
    RECT_DISAPPEAR_THRESHOLD_FRAMES, RECT_CONFIRMATION_FRAMES,
    HORIZONTAL_MATCH_THRESHOLD, VERTICAL_MATCH_MAX, VERTICAL_MATCH_MIN,
    PENDING_HORIZONTAL_THRESHOLD, PENDING_VERTICAL_MAX, PENDING_VERTICAL_MIN,
    PENDING_CLEANUP_FRAMES, POSITION_MATCH_THRESHOLD,
    KILLFEED_TOP, KILLFEED_BOTTOM, KILLFEED_LEFT,
    RAM_THRESHOLD_MINIMUM_GB, RAM_THRESHOLD_MODERATE_GB,
    RAM_DEFAULT_FALLBACK_GB, MAX_PARALLEL_WORKERS
)

from file_utils import (
    cleanup_incomplete_files,
    get_unique_filepath,
    read_control_state,
    write_control_state,
    wait_while_paused,
    remove_control_file
)

# Import Discord compression
from discord_compressor import compress_video_for_discord

# Import weapon icon detection
from icondetection import WeaponIconDetector, diagnose_detection_issue


# ========================================
# HARDWARE DETECTION
# ========================================

def get_available_ram_gb():
    """
    Get available RAM in GB using built-in methods - cross-platform.
    
    Returns:
        float: Available RAM in GB
    """
    system = platform.system()
    
    if system == "Windows":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            c_ulonglong = ctypes.c_ulonglong
            
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", c_ulonglong),
                    ("ullAvailPhys", c_ulonglong),
                    ("ullTotalPageFile", c_ulonglong),
                    ("ullAvailPageFile", c_ulonglong),
                    ("ullTotalVirtual", c_ulonglong),
                    ("ullAvailVirtual", c_ulonglong),
                    ("ullAvailExtendedVirtual", c_ulonglong),
                ]
            
            mem_status = MEMORYSTATUSEX()
            mem_status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            kernel32.GlobalMemoryStatusEx(ctypes.byref(mem_status))
            
            available_bytes = mem_status.ullAvailPhys
            return available_bytes / (1024**3)
        
        except Exception as e:
            print(f"  Warning: Could not detect RAM on Windows ({e}). Using fallback.")
            return RAM_DEFAULT_FALLBACK_GB
    
    elif system == "Linux":
        try:
            with open('/proc/meminfo', 'r') as f:
                meminfo = {}
                for line in f:
                    parts = line.split(':')
                    if len(parts) == 2:
                        key = parts[0].strip()
                        value = int(parts[1].strip().split()[0])
                        meminfo[key] = value
                
                available_kb = meminfo.get('MemAvailable', 
                    meminfo.get('MemFree', 0) + meminfo.get('Buffers', 0) + meminfo.get('Cached', 0))
                
                return available_kb / (1024**2)
        
        except Exception as e:
            print(f"  Warning: Could not detect RAM on Linux ({e}). Using fallback.")
            return RAM_DEFAULT_FALLBACK_GB
    
    elif system == "Darwin":
        try:
            import subprocess
            vm_stat = subprocess.check_output(['vm_stat']).decode('utf-8')
            page_size = int(subprocess.check_output(['pagesize']).decode('utf-8').strip())
            
            stats = {}
            for line in vm_stat.split('\n'):
                if ':' in line:
                    key, value = line.split(':')
                    stats[key.strip()] = value.strip().rstrip('.')
            
            free_pages = int(stats.get('Pages free', '0'))
            inactive_pages = int(stats.get('Pages inactive', '0'))
            available_pages = free_pages + inactive_pages
            
            return (available_pages * page_size) / (1024**3)
        
        except Exception as e:
            print(f"  Warning: Could not detect RAM on macOS ({e}). Using fallback.")
            return RAM_DEFAULT_FALLBACK_GB
    
    else:
        print(f"  Warning: Unknown OS ({system}). Using fallback RAM value.")
        return RAM_DEFAULT_FALLBACK_GB


def determine_worker_count(mode):
    """
    Determine how many parallel workers to use based on mode and hardware.
    
    Args:
        mode (str): Multiprocessing mode ('Auto', 'On', 'Off')
    
    Returns:
        int: Number of workers to use
    """
    if mode == "Off":
        return 1
    
    available_ram_gb = get_available_ram_gb()
    cpu_cores = cpu_count()
    
    if mode == "Auto":
        if available_ram_gb < RAM_THRESHOLD_MINIMUM_GB:
            print(f"  Auto mode: Insufficient RAM ({available_ram_gb:.1f}GB). Using sequential processing.")
            return 1
        elif available_ram_gb < RAM_THRESHOLD_MODERATE_GB:
            workers = min(2, max(1, cpu_cores - 1))
            print(f"  Auto mode: Moderate RAM ({available_ram_gb:.1f}GB). Using {workers} worker(s).")
            return workers
        else:
            workers = min(MAX_PARALLEL_WORKERS, max(1, cpu_cores - 1))
            print(f"  Auto mode: Sufficient RAM ({available_ram_gb:.1f}GB). Using {workers} workers.")
            return workers
    
    elif mode == "On":
        if available_ram_gb < RAM_THRESHOLD_MINIMUM_GB:
            print(f"  WARNING: Low RAM ({available_ram_gb:.1f}GB). Forcing 2 workers (may cause slowdown).")
            return 2
        else:
            workers = min(MAX_PARALLEL_WORKERS, max(2, cpu_cores - 1))
            print(f"  Multiprocessing enabled: Using {workers} workers.")
            return workers
    
    return 1


# ========================================
# DETECTION THRESHOLD CALCULATION
# ========================================

# Cache for detection thresholds by resolution to avoid recalculation
_threshold_cache = {}

def calculate_detection_thresholds(video_width, video_height):
    """
    Calculate and cache detection thresholds based on video resolution.
    All thresholds are calibrated for 1080p and scaled proportionally.
    
    Args:
        video_width (int): Video width in pixels
        video_height (int): Video height in pixels
    
    Returns:
        dict: Dictionary containing all scaled thresholds
    """
    # Check cache first
    cache_key = (video_width, video_height)
    if cache_key in _threshold_cache:
        return _threshold_cache[cache_key]
    
    # Calculate scale factors
    width_scale = video_width / REFERENCE_WIDTH
    height_scale = video_height / REFERENCE_HEIGHT
    
    # Build threshold dictionary
    thresholds = {
        # Rectangle detection (scaled)
        'MIN_WIDTH': int(round(RECT_MIN_WIDTH * width_scale)),
        'MIN_HEIGHT': int(round(RECT_MIN_HEIGHT * height_scale)),
        'MAX_HEIGHT': int(round(RECT_MAX_HEIGHT * height_scale)),
        'MIN_RECT_AREA': int(round(RECT_MIN_AREA * width_scale * height_scale)),
        'MIN_WIDTH_THRESHOLD': int(round(RECT_MIN_WIDTH_THRESHOLD * width_scale)),
        
        # Frame-based thresholds (no scaling)
        'DISAPPEAR_THRESHOLD': RECT_DISAPPEAR_THRESHOLD_FRAMES,
        'CONFIRMATION_FRAMES': RECT_CONFIRMATION_FRAMES,
        
        # Tracking thresholds (scaled)
        'HORIZONTAL_MATCH_THRESHOLD': int(round(HORIZONTAL_MATCH_THRESHOLD * width_scale)),
        'VERTICAL_MATCH_MAX': int(round(VERTICAL_MATCH_MAX * height_scale)),
        'VERTICAL_MATCH_MIN': int(round(VERTICAL_MATCH_MIN * height_scale)),
        
        # Pending rectangle thresholds (scaled)
        'PENDING_HORIZONTAL_THRESHOLD': int(round(PENDING_HORIZONTAL_THRESHOLD * width_scale)),
        'PENDING_VERTICAL_MAX': int(round(PENDING_VERTICAL_MAX * height_scale)),
        'PENDING_VERTICAL_MIN': int(round(PENDING_VERTICAL_MIN * height_scale)),
        'PENDING_CLEANUP_FRAMES': PENDING_CLEANUP_FRAMES,
        
        # Repeat detection (scaled)
        'POSITION_MATCH_THRESHOLD': int(round(POSITION_MATCH_THRESHOLD * min(width_scale, height_scale))),
        
        # Killfeed region (scaled)
        'killfeed_top': int(round(KILLFEED_TOP * height_scale)),
        'killfeed_bottom': int(round(KILLFEED_BOTTOM * height_scale)),
        'killfeed_left': int(round(KILLFEED_LEFT * width_scale)),
        
        # Scale factors for logging
        'width_scale': width_scale,
        'height_scale': height_scale
    }
    
    # Cache and return
    _threshold_cache[cache_key] = thresholds
    return thresholds


# ========================================
# VIDEO ANALYSIS HELPERS
# ========================================

def check_video_repeat(confirmed_rectangles_data, position_threshold):
    """
    Check if video contains repeated content by comparing first and second halves.
    
    Args:
        confirmed_rectangles_data (list): List of confirmed kill data
        position_threshold (int): Position matching threshold
    
    Returns:
        tuple: (is_repeating: bool, half_count: int)
    """
    rectangle_count = len(confirmed_rectangles_data)
    
    if rectangle_count < 6 or rectangle_count % 2 != 0:
        return False, 0
    
    half = rectangle_count // 2
    first_half_clips = confirmed_rectangles_data[:half]
    second_half_clips = confirmed_rectangles_data[half:]
    
    matches = 0
    for i in range(half):
        rect1_area = first_half_clips[i]['rect_area']
        rect2_area = second_half_clips[i]['rect_area']
        pos1 = first_half_clips[i]['position']
        pos2 = second_half_clips[i]['position']
        
        area_diff_ratio = abs(rect1_area - rect2_area) / max(rect1_area, rect2_area)
        position_match = (abs(pos1[0] - pos2[0]) < position_threshold and 
                        abs(pos1[1] - pos2[1]) < position_threshold)
        
        if area_diff_ratio < 0.05 and position_match:
            matches += 1
    
    is_repeating = matches >= half * 0.8
    return is_repeating, half


def determine_trim_points(rectangle_count, confirmed_rectangles_data, is_repeating, half, 
                         total_frames, fps, trim_start_seconds, trim_end_seconds):
    """
    Determine trim start and end points based on detected kills.
    
    Args:
        rectangle_count (int): Number of confirmed kills
        confirmed_rectangles_data (list): Kill data
        is_repeating (bool): Whether video is repeating
        half (int): Half point for repeating videos
        total_frames (int): Total video frames
        fps (float): Video FPS
        trim_start_seconds (int): Seconds before first kill
        trim_end_seconds (int): Seconds after last kill
    
    Returns:
        tuple: (should_trim: bool, trim_start: int, trim_end: int, status: str)
    """
    if rectangle_count == 0:
        return False, 0, total_frames, 'unsorted'
    
    if rectangle_count > MAX_KILLS_FOR_AUTO_PROCESSING and not is_repeating:
        return False, 0, total_frames, 'unsorted'
    
    should_trim = False
    trim_start = 0
    trim_end = total_frames
    
    first_5_seconds_frames = int(5 * fps)
    last_4_seconds_frames = total_frames - int(4 * fps)
    
    if is_repeating:
        # Use second half for trimming
        second_half_first_frame = confirmed_rectangles_data[half]['frame']
        second_half_last_frame = confirmed_rectangles_data[-1]['frame']
        
        if second_half_first_frame > first_5_seconds_frames:
            trim_start = max(0, second_half_first_frame - int(trim_start_seconds * fps))
            should_trim = True
        
        if second_half_last_frame < last_4_seconds_frames:
            trim_end = min(total_frames, second_half_last_frame + int(trim_end_seconds * fps))
            should_trim = True
    else:
        # Use first and last kills for trimming
        first_rectangle_frame = confirmed_rectangles_data[0]['frame']
        last_rectangle_frame = confirmed_rectangles_data[-1]['frame']
        
        if first_rectangle_frame > first_5_seconds_frames:
            trim_start = max(0, first_rectangle_frame - int(trim_start_seconds * fps))
            should_trim = True
        
        if last_rectangle_frame < last_4_seconds_frames:
            trim_end = min(total_frames, last_rectangle_frame + int(trim_end_seconds * fps))
            should_trim = True
    
    return should_trim, trim_start, trim_end, 'processed'


# ========================================
# MAIN VIDEO PROCESSING FUNCTION
# ========================================

def process_single_video(args):
    """
    Process a single video file - designed to be called by multiprocessing.
    
    Handles:
    - Kill detection using red rectangle tracking in killfeed region
    - Rectangle confirmation and tracking logic
    - Video repeat detection
    - Trimming and output based on detected kills
    
    Args:
        args (tuple): (test_file, video_index, num_videos_to_test, config_params)
    
    Returns:
        tuple: (status, source_file, output_file)
            status: 'processed', 'unsorted', 'error', 'stopped'
    """
    test_file, video_index, num_videos_to_test, config_params = args
    
    # Extract config parameters
    output_folder = config_params['output_folder']
    unsorted_folder = config_params['unsorted_folder']
    trim_start_seconds = config_params['trim_start_seconds']
    trim_end_seconds = config_params['trim_end_seconds']
    output_width = config_params['output_resolution_width']
    output_height = config_params['output_resolution_height']
    output_fps_setting = config_params['output_fps']
    show_preview = config_params['show_preview']
    worker_count = config_params['worker_count']
    
    # Error handling wrapper
    try:
        print(f"\n{'='*60}")
        print(f"Processing video {video_index + 1}/{num_videos_to_test}: {os.path.basename(test_file)}")
        print(f"{'='*60}\n")
        
        cap = cv2.VideoCapture(test_file)
        
        if not cap.isOpened():
            print(f"Error opening {test_file}")
            return ('error', test_file, None)
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video resolution: {video_width}x{video_height}")
        
        # Calculate detection thresholds (cached by resolution)
        thresholds = calculate_detection_thresholds(video_width, video_height)
        
        print(f"Detection scaling: {thresholds['width_scale']:.2f}x width, {thresholds['height_scale']:.2f}x height")
        print(f"  Rectangle size: w>{thresholds['MIN_WIDTH']}px, {thresholds['MIN_HEIGHT']}<h<{thresholds['MAX_HEIGHT']}px")
        print(f"  Minimum area: {thresholds['MIN_RECT_AREA']}pxÂ²")
        print(f"  Killfeed region: top={thresholds['killfeed_top']}, bottom={thresholds['killfeed_bottom']}, left={thresholds['killfeed_left']}")
        
        # Initialize weapon detector
        weapon_detector = WeaponIconDetector(debug=True)
        
        # Tracking variables
        frame_count = 0
        rectangle_count = 0
        recent_rectangles = []
        pending_rectangles = []
        confirmed_rectangles_data = []
        
        # Extract killfeed coordinates
        killfeed_top = thresholds['killfeed_top']
        killfeed_bottom = thresholds['killfeed_bottom']
        killfeed_left = thresholds['killfeed_left']
        
        # Main frame processing loop
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Check for stop signal periodically
            if frame_count % PAUSE_CHECK_INTERVAL_FRAMES == 0:
                state = read_control_state()
                if state == 'stopped':
                    cap.release()
                    if show_preview and worker_count == 1:
                        cv2.destroyAllWindows()
                    print(f"\nâ„¹ï¸  Processing stopped during analysis at frame {frame_count}")
                    return ('stopped', test_file, None)
            
            # Progress reporting (sequential mode only)
            if frame_count % 30 == 0 or frame_count == 1:
                if worker_count == 1:
                    current_progress = (frame_count / total_frames) * 100
                    overall_progress = ((video_index + (frame_count / total_frames)) / num_videos_to_test) * 100
                    print(f"\r  Video {video_index + 1}/{num_videos_to_test}: {current_progress:.1f}% | Overall: {overall_progress:.1f}%", 
                          end='', flush=True)
            
            # Extract killfeed region
            killfeed_region = frame[killfeed_top:killfeed_bottom, killfeed_left:video_width]
            
            # Color detection for red kill indicators
            hsv = cv2.cvtColor(killfeed_region, cv2.COLOR_BGR2HSV)
            
            lower_red1 = np.array([0, 60, 70])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([170, 120, 100])
            upper_red2 = np.array([180, 255, 255])
            
            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            red_mask = mask1 + mask2
            
            # Find contours
            contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            current_rectangles = []
            
            # Analyze each contour
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Size filter
                if w > thresholds['MIN_WIDTH'] and thresholds['MIN_HEIGHT'] < h < thresholds['MAX_HEIGHT']:
                    roi_mask = red_mask[y:y+h, x:x+w]
                    red_pixel_count = cv2.countNonZero(roi_mask)
                    
                    rect_area = w * h
                    fill_ratio = red_pixel_count / rect_area if rect_area > 0 else 0
                    
                    # Area filter
                    if rect_area > thresholds['MIN_RECT_AREA']:
                        cv2.rectangle(killfeed_region, (x, y), (x+w, y+h), (0, 255, 255), 1)
                        
                        # Fill ratio filter
                        if 0.03 < fill_ratio < 0.25:
                            # Additional validation checks
                            roi_bgr = killfeed_region[y:y+h, x:x+w]
                            roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
                            
                            # Check 1: Minimum width
                            width_check = w >= thresholds['MIN_WIDTH_THRESHOLD']
                            
                            # Check 2: White pixel percentage (weapon icon)
                            _, white_mask = cv2.threshold(roi_gray, 200, 255, cv2.THRESH_BINARY)
                            white_pixel_count = cv2.countNonZero(white_mask)
                            white_ratio = white_pixel_count / rect_area if rect_area > 0 else 0
                            white_check = white_ratio >= RECT_MIN_WHITE_RATIO
                            
                            # Accept if passes both checks
                            if width_check and white_check:
                                current_rectangles.append((x, y, w, h))
                                cv2.rectangle(killfeed_region, (x, y), (x+w, y+h), (0, 255, 0), 2)
                            elif show_preview and worker_count == 1:
                                # Show rejected rectangles in preview
                                reason = []
                                if not width_check:
                                    reason.append(f"W:{w}<{thresholds['MIN_WIDTH_THRESHOLD']}")
                                if not white_check:
                                    reason.append(f"White:{white_ratio:.1%}<{RECT_MIN_WHITE_RATIO:.1%}")
                                
                                cv2.rectangle(killfeed_region, (x, y), (x+w, y+h), (0, 165, 255), 1)
                                cv2.putText(killfeed_region, f"REJ: {', '.join(reason)}", 
                                          (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 165, 255), 1)
            
            # Rectangle tracking and confirmation logic
            for x, y, w, h in current_rectangles:
                is_new = True
                matched_index = None
                
                center_x = x + w // 2
                center_y = y + h // 2
                
                # Check against recent rectangles
                for i, (rx, ry, rw, rh, rframe) in enumerate(recent_rectangles):
                    if (frame_count - rframe) <= thresholds['DISAPPEAR_THRESHOLD']:
                        prev_center_x = rx + rw // 2
                        prev_center_y = ry + rh // 2
                        
                        width_ratio = abs(w - rw) / max(w, rw) if max(w, rw) > 0 else 0
                        height_ratio = abs(h - rh) / max(h, rh) if max(h, rh) > 0 else 0
                        size_match = width_ratio < 0.5 and height_ratio < 0.25
                        
                        horizontal_match = abs(center_x - prev_center_x) < thresholds['HORIZONTAL_MATCH_THRESHOLD']
                        
                        vertical_diff = center_y - prev_center_y
                        vertical_match = -thresholds['VERTICAL_MATCH_MAX'] <= vertical_diff <= thresholds['VERTICAL_MATCH_MIN']
                        
                        if size_match and horizontal_match and vertical_match:
                            is_new = False
                            matched_index = i
                            break
                
                if is_new:
                    # Check pending rectangles for confirmation
                    found_pending = False
                    for p_idx, (px, py, pw, ph, p_first_frame, p_count) in enumerate(pending_rectangles):
                        p_center_x = px + pw // 2
                        p_center_y = py + ph // 2
                        
                        p_width_ratio = abs(w - pw) / max(w, pw) if max(w, pw) > 0 else 0
                        p_height_ratio = abs(h - ph) / max(h, ph) if max(h, ph) > 0 else 0
                        p_size_match = p_width_ratio < 0.35 and p_height_ratio < 0.2
                        
                        p_horizontal_match = abs(center_x - p_center_x) < thresholds['PENDING_HORIZONTAL_THRESHOLD']
                        
                        p_vertical_diff = center_y - p_center_y
                        p_vertical_match = -thresholds['PENDING_VERTICAL_MAX'] <= p_vertical_diff <= thresholds['PENDING_VERTICAL_MIN']
                        
                        if p_size_match and p_horizontal_match and p_vertical_match:
                            pending_rectangles[p_idx] = (x, y, w, h, frame_count, p_count + 1)
                            
                            if show_preview and worker_count == 1:
                                print(f"  âœ“ Pending matched: {p_count + 1}/{thresholds['CONFIRMATION_FRAMES']} confirmations (frame {frame_count})")
                            
                            # Confirm after threshold frames
                            if p_count + 1 >= thresholds['CONFIRMATION_FRAMES']:
                                rectangle_count += 1
                                recent_rectangles.append((x, y, w, h, frame_count))
                                
                                rect_area = w * h
                                roi_mask = red_mask[y:y+h, x:x+w]
                                red_pixel_count = cv2.countNonZero(roi_mask)
                                fill_ratio = red_pixel_count / rect_area if rect_area > 0 else 0
                                
                                # Weapon detection
                                rect_region = killfeed_region[y:y+h, x:x+w]
                                weapon_name, weapon_confidence, debug_info = weapon_detector.detect_weapon_in_region(rect_region)
                                
                                # Visual debug in preview
                                if show_preview and worker_count == 1:
                                    if weapon_name:
                                        cv2.rectangle(killfeed_region, (x, y), (x+w, y+h), (0, 255, 0), 3)
                                        label = f"{weapon_name} {weapon_confidence:.2f}"
                                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                                        cv2.rectangle(killfeed_region, (x, y - label_size[1] - 8), 
                                                    (x + label_size[0] + 4, y - 2), (0, 255, 0), -1)
                                        cv2.putText(killfeed_region, label, (x + 2, y - 5), 
                                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                                    else:
                                        cv2.rectangle(killfeed_region, (x, y), (x+w, y+h), (0, 255, 255), 2)
                                        cv2.putText(killfeed_region, "Unknown", (x + 2, y - 5), 
                                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                                
                                # Log detection
                                print(f"\nFrame {frame_count}: NEW kill #{rectangle_count} (confirmed after {p_count + 1} frames)")
                                print(f"  Position: ({x}, {y}), Center: ({center_x}, {center_y})")
                                print(f"  Size: w={w}, h={h}, area={rect_area}")
                                print(f"  Red pixels: {red_pixel_count}, fill={fill_ratio:.3f}")
                                
                                if weapon_name:
                                    print(f"  ðŸŽ¯ Weapon: {weapon_name} (confidence: {weapon_confidence:.3f})")
                                else:
                                    print(f"  âŒ No weapon detected (best: {weapon_confidence:.3f})")
                                
                                # Store data
                                confirmed_rectangles_data.append({
                                    'rect_area': rect_area,
                                    'position': (x, y),
                                    'frame': frame_count,
                                    'weapon': weapon_name,
                                    'weapon_confidence': weapon_confidence
                                })
                                
                                pending_rectangles.pop(p_idx)
                            
                            found_pending = True
                            break
                    
                    if not found_pending:
                        pending_rectangles.append((x, y, w, h, frame_count, 1))
                        print(f"  Pending rectangle at frame {frame_count} (needs {thresholds['CONFIRMATION_FRAMES']} confirmations)")
                
                else:
                    # Update existing rectangle
                    recent_rectangles[matched_index] = (x, y, w, h, frame_count)
            
            # Display rectangle count
            cv2.putText(killfeed_region, f"Total: {rectangle_count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Show preview if enabled (sequential mode only)
            if show_preview and worker_count == 1:
                cv2.imshow('Killfeed Region', killfeed_region)
                cv2.imshow('Red Mask', red_mask)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    response = input("\nAre you sure you want to quit? (yes/no): ").strip().lower()
                    if response in ['yes', 'y']:
                        print("Terminating program...")
                        cap.release()
                        cv2.destroyAllWindows()
                        exit()
                    else:
                        print("Continuing...")
        
        # Release resources
        cap.release()
        if show_preview and worker_count == 1:
            cv2.destroyAllWindows()
        
        print("\r" + " " * 80)
        print(f"\rFinished processing!")
        print(f"Total frames: {frame_count}")
        print(f"Total kills detected: {rectangle_count}")
        
        # Detect video repeats
        is_repeating, half = check_video_repeat(confirmed_rectangles_data, thresholds['POSITION_MATCH_THRESHOLD'])
        
        if is_repeating:
            print(f"\nVIDEO REPEAT DETECTED with {half} unique kills!")
        
        # Determine trimming strategy
        should_trim, trim_start, trim_end, status = determine_trim_points(
            rectangle_count, confirmed_rectangles_data, is_repeating, half,
            total_frames, fps, trim_start_seconds, trim_end_seconds
        )
        
        if status == 'unsorted':
            if rectangle_count == 0:
                print(f"No kills detected. Moving to unsorted.")
            else:
                print(f"More than {MAX_KILLS_FOR_AUTO_PROCESSING} kills. Moving to unsorted.")
            return ('unsorted', test_file, None)
        
        # Log trimming decision
        if should_trim:
            print(f"\nWill trim from frame {trim_start} to {trim_end}")
        else:
            print(f"\nNo trimming needed")
        
        # Output processing
        if should_trim:
            filename = os.path.basename(test_file)
            output_file = get_unique_filepath(os.path.join(output_folder, filename), '_trimmed')
            
            print(f"\nStarting video trimming from frame {trim_start} to {trim_end}...")
            
            needs_resize = (output_width != video_width or output_height != video_height)
            final_fps = fps if output_fps_setting == 0 else output_fps_setting
            
            if needs_resize:
                print(f"Output will be resized from {video_width}x{video_height} to {output_width}x{output_height}")
            if output_fps_setting != 0 and output_fps_setting != fps:
                print(f"Output FPS will be changed from {fps} to {final_fps}")
            
            cap = cv2.VideoCapture(test_file)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_file, fourcc, final_fps, (output_width, output_height))
            
            try:
                cap.set(cv2.CAP_PROP_POS_FRAMES, trim_start)
                current_frame = trim_start
                
                while current_frame < trim_end:
                    # Check for stop signal
                    if current_frame % PAUSE_CHECK_INTERVAL_FRAMES == 0:
                        state = read_control_state()
                        if state == 'stopped':
                            raise InterruptedError("Processing stopped by user")
                    
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    if needs_resize:
                        frame = cv2.resize(frame, (output_width, output_height))
                    
                    out.write(frame)
                    current_frame += 1
                
                cap.release()
                out.release()
                print(f"Trimmed video saved to: {os.path.basename(output_file)}")
                return ('processed', test_file, output_file)
            
            except InterruptedError:
                # Clean up incomplete file
                cap.release()
                out.release()
                if os.path.exists(output_file):
                    os.remove(output_file)
                    print(f"Deleted incomplete file: {os.path.basename(output_file)}")
                return ('stopped', test_file, None)
            
            except Exception as e:
                # Clean up on error
                cap.release()
                if 'out' in locals():
                    out.release()
                if os.path.exists(output_file):
                    os.remove(output_file)
                raise
        
        elif rectangle_count > 0 and rectangle_count <= MAX_KILLS_FOR_AUTO_PROCESSING and not should_trim:
            filename = os.path.basename(test_file)
            
            needs_resize = (output_width != video_width or output_height != video_height)
            needs_fps_change = (output_fps_setting != 0 and output_fps_setting != fps)
            
            if needs_resize or needs_fps_change:
                output_file = get_unique_filepath(os.path.join(output_folder, filename), '_processed')
                
                print(f"No trimming needed, but processing video...")
                if needs_resize:
                    print(f"  Resizing from {video_width}x{video_height} to {output_width}x{output_height}")
                if needs_fps_change:
                    print(f"  Changing FPS from {fps} to {output_fps_setting}")
                
                cap = cv2.VideoCapture(test_file)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                final_fps = fps if output_fps_setting == 0 else output_fps_setting
                out = cv2.VideoWriter(output_file, fourcc, final_fps, (output_width, output_height))
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    if needs_resize:
                        frame = cv2.resize(frame, (output_width, output_height))
                    
                    out.write(frame)
                
                cap.release()
                out.release()
                print(f"Processed video saved to: {os.path.basename(output_file)}")
                return ('processed', test_file, output_file)
            else:
                # Copy file with unique name
                output_file = get_unique_filepath(os.path.join(output_folder, filename))
                
                print(f"No processing needed. Copying to output folder...")
                shutil.copy2(test_file, output_file)
                print(f"Video copied to: {os.path.basename(output_file)}")
                return ('processed', test_file, output_file)
        
        return ('processed', test_file, None)
    
    except KeyboardInterrupt:
        print(f"\nâš ï¸ Processing interrupted by user for: {os.path.basename(test_file)}")
        raise
    
    except Exception as e:
        print(f"\nâŒ ERROR processing {os.path.basename(test_file)}")
        print(f"   Error type: {type(e).__name__}")
        print(f"   Error message: {str(e)}")
        print(f"   Moving to unsorted folder...\n")
        return ('error', test_file, None)


# ========================================
# MAIN EXECUTION
# ========================================

def main():
    """Main entry point for video processing."""
    multiprocessing.freeze_support()
    
    # Load configuration
    config = load_config()
    
    # Extract settings
    video_folder = config["video_folder"]
    output_folder = config["output_folder"]
    unsorted_folder = config["unsorted_folder"]
    trim_start_seconds = config["trim_start_seconds"]
    trim_end_seconds = config["trim_end_seconds"]
    output_resolution_width = config["output_resolution_width"]
    output_resolution_height = config["output_resolution_height"]
    output_fps = config["output_fps"]
    num_videos_setting = config["num_videos_to_process"]
    show_preview = config["show_preview"]
    multiprocessing_mode = config["multiprocessing_mode"]
    compress_for_discord = config["compress_for_discord"]
    save_versions_mode = config.get("save_versions_mode", "Trimmed and Compressed")
    
    # Determine worker count
    worker_count = determine_worker_count(multiprocessing_mode)
    
    # Print configuration
    print_config_summary(config)
    print(f"  Parallel workers: {worker_count}")
    
    # Create output folders
    for folder in [output_folder, unsorted_folder]:
        if not os.path.exists(folder):
            os.makedirs(folder)
    
    # Cleanup incomplete files from previous runs
    cleanup_incomplete_files([output_folder, unsorted_folder])
    
    # Find video files
    video_files = []
    for filename in os.listdir(video_folder):
        filepath = os.path.join(video_folder, filename)
        if os.path.isfile(filepath) and filename.lower().endswith('.mp4'):
            video_files.append(filepath)
    
    num_videos_to_test = len(video_files) if num_videos_setting == -1 else min(num_videos_setting, len(video_files))
    
    print(f"\nFound {len(video_files)} video files. Processing {num_videos_to_test} videos.\n")
    
    if num_videos_to_test == 0:
        print("No video files found!")
        return
    
    # Build config params
    config_params = {
        'output_folder': output_folder,
        'unsorted_folder': unsorted_folder,
        'trim_start_seconds': trim_start_seconds,
        'trim_end_seconds': trim_end_seconds,
        'output_resolution_width': output_resolution_width,
        'output_resolution_height': output_resolution_height,
        'output_fps': output_fps,
        'show_preview': show_preview,
        'worker_count': worker_count,
        'compress_for_discord': compress_for_discord,
        'save_versions_mode': save_versions_mode
    }
    
    # Result tracking
    unsorted_files = []
    error_files = []
    processed_output_files = []
    processed_to_source_map = {}
    
    # Process videos
    if worker_count > 1:
        print(f"Starting multiprocessing with {worker_count} workers...\n")
        
        batch_size = worker_count * 2
        videos_processed = 0
        
        while videos_processed < num_videos_to_test:
            # Check pause state
            state = read_control_state()
            if state == 'paused':
                should_continue = wait_while_paused()
                if not should_continue:
                    print("Processing stopped by user.")
                    break
            elif state == 'stopped':
                print("\nâ„¹ï¸  Control file not found. Stopping...")
                break
            
            # Process batch
            batch_end = min(videos_processed + batch_size, num_videos_to_test)
            current_batch = [
                (video_files[i], i, num_videos_to_test, config_params)
                for i in range(videos_processed, batch_end)
            ]
            
            print(f"Processing batch: videos {videos_processed + 1}-{batch_end} of {num_videos_to_test}")
            
            with Pool(processes=worker_count) as pool:
                batch_results = pool.map(process_single_video, current_batch)
            
            # Collect results
            for idx, result in enumerate(batch_results):
                source_file = video_files[videos_processed + idx]
                
                if result[0] == 'unsorted':
                    unsorted_files.append(result[1])
                elif result[0] == 'error':
                    error_files.append(result[1])
                elif result[0] == 'processed' and result[2] is not None:
                    processed_output_files.append(result[2])
                    processed_to_source_map[result[2]] = source_file
            
            videos_processed = batch_end
    
    else:
        print("Processing videos sequentially...\n")
        
        for i in range(num_videos_to_test):
            # Check pause state
            state = read_control_state()
            if state == 'paused':
                should_continue = wait_while_paused()
                if not should_continue:
                    print("Processing stopped by user.")
                    break
            elif state == 'stopped':
                print("\nâ„¹ï¸  Control file not found. Stopping...")
                break
            
            # Process video
            source_file = video_files[i]
            args = (source_file, i, num_videos_to_test, config_params)
            result = process_single_video(args)
            
            if result[0] == 'unsorted':
                unsorted_files.append(result[1])
            elif result[0] == 'error':
                error_files.append(result[1])
            elif result[0] == 'processed' and result[2] is not None:
                processed_output_files.append(result[2])
                processed_to_source_map[result[2]] = source_file
    
    # Combine unsorted and error files
    all_unsorted = unsorted_files + error_files
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"  Total videos processed: {num_videos_to_test}")
    print(f"  Successfully processed: {num_videos_to_test - len(all_unsorted)}")
    print(f"  Unsorted (0 or >{MAX_KILLS_FOR_AUTO_PROCESSING} kills): {len(unsorted_files)}")
    print(f"  Errors (corrupted/failed): {len(error_files)}")
    print(f"{'='*60}\n")
    
    # Discord compression and file management
    unsorted_compressed = []
    
    # Step 1: Compress unsorted files
    if compress_for_discord and len(all_unsorted) > 0:
        print(f"\n{'='*60}")
        print(f"DISCORD COMPRESSION - UNSORTED FILES")
        print(f"{'='*60}")
        print(f"Compressing {len(all_unsorted)} unsorted video(s)...\n")
        
        if not os.path.exists(unsorted_folder):
            os.makedirs(unsorted_folder)
        
        for i, unsorted_file in enumerate(all_unsorted, 1):
            if os.path.isfile(unsorted_file):
                base_name = os.path.splitext(os.path.basename(unsorted_file))[0]
                discord_output = os.path.join(unsorted_folder, f"{base_name}_discord.mp4")
                
                print(f"\n[{i}/{len(all_unsorted)}] Compressing: {os.path.basename(unsorted_file)}")
                
                success, output_path, error_msg = compress_video_for_discord(unsorted_file, discord_output, silent=False)
                
                if success:
                    unsorted_compressed.append(unsorted_file)
                    print(f"  âœ“ Compressed saved to: {os.path.basename(output_path)}")
                else:
                    print(f"  âŒ Compression failed: {error_msg}")
    
    # Step 2: Move uncompressed unsorted files
    files_to_move = [f for f in all_unsorted if f not in unsorted_compressed]
    if len(files_to_move) > 0:
        if not os.path.exists(unsorted_folder):
            os.makedirs(unsorted_folder)
        
        print(f"\nMoving {len(files_to_move)} uncompressed file(s) to unsorted folder...")
        
        for unsorted_file in files_to_move:
            if os.path.isfile(unsorted_file):
                filename = os.path.basename(unsorted_file)
                destination = get_unique_filepath(os.path.join(unsorted_folder, filename))
                shutil.copy2(unsorted_file, destination)
                
                if unsorted_file in error_files:
                    print(f"  Moved ERROR file: {filename}")
                else:
                    print(f"  Moved UNSORTED file: {filename}")
    
    # Step 3: Compress processed files
    successfully_compressed_files = []
    
    if compress_for_discord and len(processed_output_files) > 0:
        print(f"\n{'='*60}")
        print(f"DISCORD COMPRESSION - PROCESSED FILES")
        print(f"{'='*60}")
        print(f"Compressing {len(processed_output_files)} processed video(s)...\n")
        
        successful_compressions = 0
        failed_compressions = 0
        
        for i, video_file in enumerate(processed_output_files, 1):
            # Check stop signal
            state = read_control_state()
            if state == 'stopped':
                print(f"\nâ„¹ï¸  Compression interrupted by user.")
                break
            elif state == 'paused':
                should_continue = wait_while_paused()
                if not should_continue:
                    print("Compression stopped by user.")
                    break
            
            print(f"\n[{i}/{len(processed_output_files)}] Compressing: {os.path.basename(video_file)}")
            
            base_name = os.path.splitext(video_file)[0]
            discord_output = f"{base_name}_discord.mp4"
            
            success, output_path, error_msg = compress_video_for_discord(video_file, discord_output, silent=False)
            
            if success:
                successful_compressions += 1
                successfully_compressed_files.append(video_file)
                
                # Delete trimmed file unless saving all versions
                if save_versions_mode != "Save All":
                    try:
                        os.remove(video_file)
                        print(f"  ðŸ—‘ï¸  Removed trimmed file: {os.path.basename(video_file)}")
                    except Exception as e:
                        print(f"  âš ï¸  Warning: Could not delete trimmed file: {e}")
                else:
                    print(f"  ðŸ’¾ Keeping trimmed version")
            else:
                failed_compressions += 1
                print(f"  âŒ Failed: {error_msg}")
                
                # Cleanup incomplete compression
                if os.path.exists(discord_output):
                    try:
                        os.remove(discord_output)
                        print(f"  ðŸ—‘ï¸  Removed incomplete compression")
                    except:
                        pass
        
        print(f"\n{'='*60}")
        print(f"COMPRESSION SUMMARY")
        print(f"{'='*60}")
        print(f"  Processed files compressed: {successful_compressions}/{len(processed_output_files)}")
        if len(unsorted_compressed) > 0:
            print(f"  Unsorted files compressed: {len(unsorted_compressed)}/{len(all_unsorted)}")
        print(f"  Total compressed: {successful_compressions + len(unsorted_compressed)}")
        if failed_compressions > 0:
            print(f"  Failed compressions: {failed_compressions}")
        print(f"{'='*60}\n")
    
    # Step 4: Save original source files (if Save All mode)
    if save_versions_mode == "Save All" and len(processed_output_files) > 0:
        print(f"\n{'='*60}")
        print(f"SAVING ORIGINAL SOURCE FILES")
        print(f"{'='*60}")
        print(f"Copying {len(processed_output_files)} original source file(s)...\n")
        
        saved_count = 0
        failed_count = 0
        
        for processed_file in processed_output_files:
            source_file = processed_to_source_map.get(processed_file)
            
            if source_file and os.path.exists(source_file):
                try:
                    source_basename = os.path.basename(source_file)
                    name, ext = os.path.splitext(source_basename)
                    original_output = get_unique_filepath(
                        os.path.join(output_folder, f"{name}_original{ext}")
                    )
                    
                    shutil.copy2(source_file, original_output)
                    saved_count += 1
                    print(f"  ðŸ’¾ Saved original: {os.path.basename(original_output)}")
                
                except Exception as e:
                    failed_count += 1
                    print(f"  âš ï¸  Failed to save original for {os.path.basename(processed_file)}: {e}")
            else:
                failed_count += 1
                print(f"  âš ï¸  Source file not found for: {os.path.basename(processed_file)}")
        
        print(f"\n{'='*60}")
        print(f"ORIGINAL FILES SAVED: {saved_count}/{len(processed_output_files)}")
        if failed_count > 0:
            print(f"  Failed: {failed_count}")
        print(f"{'='*60}\n")


if __name__ == '__main__':
    main()