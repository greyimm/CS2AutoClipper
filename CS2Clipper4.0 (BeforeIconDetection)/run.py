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

# Import configuration management from settings module
from settings import load_config, print_config_summary

# Import Discord compression function
from discord_compressor import compress_video_for_discord

# ========================================
# PAUSE/RESUME CONTROL FUNCTIONS
# ========================================

def check_pause_state(control_file):
    """
    Check if the process should be paused.
    Returns: 'running', 'paused', or 'stopped' (if file doesn't exist)
    """
    if not os.path.exists(control_file):
        return 'stopped'
    
    try:
        with open(control_file, 'r') as f:
            state = f.read().strip()
        return state
    except:
        return 'running'


def wait_while_paused(control_file):
    """
    Wait in a loop while the process is paused.
    Returns True if should continue, False if should stop.
    """
    print("\n" + "="*60)
    print("⏸️  PROCESSING PAUSED")
    print("="*60)
    print("Waiting for resume signal...")
    print("(Close the console window or delete process_control.txt to stop)")
    print("="*60 + "\n")
    
    while True:
        state = check_pause_state(control_file)
        
        if state == 'stopped':
            print("\n⏹️  Process control file deleted. Stopping...")
            return False
        elif state == 'running':
            print("\n▶️  PROCESSING RESUMED")
            print("="*60 + "\n")
            return True
        
        time.sleep(1)  # Check every second

# Load configuration
config = load_config()

# Extract settings from config
video_folder = config["video_folder"]
output_folder = config["output_folder"]
trim_start_seconds = config["trim_start_seconds"]
trim_end_seconds = config["trim_end_seconds"]
output_resolution_width = config["output_resolution_width"]
output_resolution_height = config["output_resolution_height"]
output_fps = config["output_fps"]
num_videos_setting = config["num_videos_to_process"]
unsorted_folder = config["unsorted_folder"]
show_preview = config["show_preview"]
multiprocessing_mode = config["multiprocessing_mode"]
compress_for_discord = config["compress_for_discord"]
save_all_versions = config["save_all_versions"]


# ========================================
# HARDWARE DETECTION AND MULTIPROCESSING
# ========================================

def get_available_ram_gb():
    """Get available RAM in GB using built-in methods - cross-platform"""
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
            available_gb = available_bytes / (1024**3)
            return available_gb
        except Exception as e:
            print(f"  Warning: Could not detect RAM on Windows ({e}). Assuming 8GB available.")
            return 8.0
    
    elif system == "Linux":
        try:
            with open('/proc/meminfo', 'r') as f:
                lines = f.readlines()
                meminfo = {}
                for line in lines:
                    parts = line.split(':')
                    if len(parts) == 2:
                        key = parts[0].strip()
                        value = parts[1].strip().split()[0]
                        meminfo[key] = int(value)
                
                if 'MemAvailable' in meminfo:
                    available_kb = meminfo['MemAvailable']
                else:
                    available_kb = meminfo.get('MemFree', 0) + meminfo.get('Buffers', 0) + meminfo.get('Cached', 0)
                
                available_gb = available_kb / (1024**2)
                return available_gb
        except Exception as e:
            print(f"  Warning: Could not detect RAM on Linux ({e}). Assuming 8GB available.")
            return 8.0
    
    elif system == "Darwin":
        try:
            import subprocess
            vm_stat = subprocess.check_output(['vm_stat']).decode('utf-8')
            
            lines = vm_stat.split('\n')
            stats = {}
            for line in lines:
                if ':' in line:
                    key, value = line.split(':')
                    stats[key.strip()] = value.strip().rstrip('.')
            
            page_size = int(subprocess.check_output(['pagesize']).decode('utf-8').strip())
            
            free_pages = int(stats.get('Pages free', '0'))
            inactive_pages = int(stats.get('Pages inactive', '0'))
            available_pages = free_pages + inactive_pages
            
            available_bytes = available_pages * page_size
            available_gb = available_bytes / (1024**3)
            return available_gb
        except Exception as e:
            print(f"  Warning: Could not detect RAM on macOS ({e}). Assuming 8GB available.")
            return 8.0
    
    else:
        print(f"  Warning: Unknown operating system ({system}). Assuming 8GB available.")
        return 8.0


def determine_worker_count(mode):
    """Determine how many parallel workers to use based on mode and hardware"""
    available_ram_gb = get_available_ram_gb()
    cpu_cores = cpu_count()
    
    if mode == "Off":
        return 1
    
    elif mode == "Auto":
        if available_ram_gb < 6:
            print("  Auto mode: Insufficient RAM (<6GB available). Using sequential processing.")
            return 1
        elif available_ram_gb < 10:
            workers = min(2, max(1, cpu_cores - 1))
            print(f"  Auto mode: Moderate RAM ({available_ram_gb:.1f}GB). Using {workers} parallel process(es).")
            return workers
        else:
            workers = min(4, max(1, cpu_cores - 1))
            print(f"  Auto mode: Sufficient RAM ({available_ram_gb:.1f}GB). Using {workers} parallel processes.")
            return workers
    
    elif mode == "On":
        if available_ram_gb < 6:
            print("  WARNING: Low RAM detected. Forcing 2 workers (may cause system slowdown).")
            return 2
        else:
            workers = min(4, max(2, cpu_cores - 1))
            print(f"  Multiprocessing enabled: Using {workers} parallel processes.")
            return workers
    
    return 1


worker_count = determine_worker_count(multiprocessing_mode)

# Print configuration summary
print_config_summary(config)
print(f"  Parallel workers: {worker_count}")

# Create output folders if they don't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

if not os.path.exists(unsorted_folder):
    os.makedirs(unsorted_folder)


# ========================================
# VIDEO PROCESSING AND OBJECT DETECTION
# ========================================

# Cache for detection thresholds by resolution to avoid recalculation
_threshold_cache = {}

def calculate_detection_thresholds(video_width, video_height):
    """
    Calculate and cache detection thresholds based on video resolution.
    All thresholds are calibrated for 1080p and scaled proportionally.
    
    Returns:
        dict: Dictionary containing all scaled thresholds
    """
    # Check cache first
    cache_key = (video_width, video_height)
    if cache_key in _threshold_cache:
        return _threshold_cache[cache_key]
    
    # Reference resolution (1080p)
    REFERENCE_WIDTH = 1920
    REFERENCE_HEIGHT = 1080
    
    width_scale = video_width / REFERENCE_WIDTH
    height_scale = video_height / REFERENCE_HEIGHT
    
    # Calculate all thresholds
    thresholds = {
        # Rectangle detection
        'MIN_WIDTH': int(round(10 * width_scale)),
        'MIN_HEIGHT': int(round(10 * height_scale)),
        'MAX_HEIGHT': int(round(36 * height_scale)),
        'MIN_RECT_AREA': int(round(3000 * width_scale * height_scale)),
        'DISAPPEAR_THRESHOLD': 60,  # Frame-based, no scaling
        'CONFIRMATION_FRAMES': 3,   # Frame-based, no scaling
        
        # Tracking thresholds
        'HORIZONTAL_MATCH_THRESHOLD': int(round(90 * width_scale)),
        'VERTICAL_MATCH_MAX': int(round(110 * height_scale)),
        'VERTICAL_MATCH_MIN': int(round(10 * height_scale)),
        
        # Pending rectangle thresholds
        'PENDING_HORIZONTAL_THRESHOLD': int(round(60 * width_scale)),
        'PENDING_VERTICAL_MAX': int(round(80 * height_scale)),
        'PENDING_VERTICAL_MIN': int(round(10 * height_scale)),
        'PENDING_CLEANUP_FRAMES': 60,  # Frame-based, no scaling
        
        # Repeat detection
        'POSITION_MATCH_THRESHOLD': int(round(5 * min(width_scale, height_scale))),
        
        # Killfeed region (scaled from 1080p reference)
        'killfeed_top': int(round(60 * height_scale)),
        'killfeed_bottom': int(round(295 * height_scale)),
        'killfeed_left': int(round(1536 * width_scale)),
        
        # Scale factors for logging
        'width_scale': width_scale,
        'height_scale': height_scale
    }
    
    # Cache the result
    _threshold_cache[cache_key] = thresholds
    return thresholds


def check_video_repeat(confirmed_rectangles_data, position_threshold):
    """
    Check if video contains repeated content by comparing first and second halves.
    
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
    
    Returns:
        tuple: (should_trim: bool, trim_start: int, trim_end: int, status: str)
    """
    if rectangle_count == 0:
        return False, 0, total_frames, 'unsorted'
    
    if rectangle_count > 5 and not is_repeating:
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


def generate_unique_output_path(output_folder, filename, suffix='_trimmed'):
    """
    Generate unique output filename with (#) counter if file exists.
    
    Returns:
        str: Unique output file path
    """
    base_output_file = os.path.join(output_folder, filename.replace('.mp4', f'{suffix}.mp4'))
    
    if not os.path.exists(base_output_file):
        return base_output_file
    
    counter = 1
    while True:
        output_file = base_output_file.replace(f'{suffix}.mp4', f'{suffix}({counter}).mp4')
        if not os.path.exists(output_file):
            return output_file
        counter += 1

def process_single_video(args):
    """
    Process a single video file - designed to be called by multiprocessing
    
    This function handles:
    - Kill detection using red rectangle tracking in killfeed region
    - Rectangle confirmation and tracking logic
    - Video repeat detection
    - Trimming and output based on detected kills
    """
    test_file, video_index, num_videos_to_test, config_params = args
    
    # ========================================
    # ERROR HANDLING WRAPPER
    # ========================================
    try:
        output_folder = config_params['output_folder']
        unsorted_folder = config_params['unsorted_folder']
        trim_start_seconds = config_params['trim_start_seconds']
        trim_end_seconds = config_params['trim_end_seconds']
        output_width = config_params['output_resolution_width']
        output_height = config_params['output_resolution_height']
        output_fps_setting = config_params['output_fps']
        show_preview = config_params['show_preview']
        worker_count = config_params['worker_count']
        
        print(f"\n{'='*60}")
        print(f"Processing video {video_index + 1}/{num_videos_to_test}: {test_file}")
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
        
        print(f"Detection scaling factors: {thresholds['width_scale']:.2f}x width, {thresholds['height_scale']:.2f}x height (reference: 1920x1080)")
        print(f"  Rectangle size range: w>{thresholds['MIN_WIDTH']}px, {thresholds['MIN_HEIGHT']}<h<{thresholds['MAX_HEIGHT']}px")
        print(f"  Minimum rectangle area: {thresholds['MIN_RECT_AREA']}px²")
        print(f"  Killfeed region: top={thresholds['killfeed_top']}px, bottom={thresholds['killfeed_bottom']}px, left={thresholds['killfeed_left']}px")
        
        # Object detection tracking variables
        frame_count = 0
        rectangle_count = 0
        recent_rectangles = []
        pending_rectangles = []
        confirmed_rectangles_data = []
        
        first_rectangle_frame = None
        last_rectangle_frame = None
        
        # Extract killfeed region coordinates from thresholds
        killfeed_top = thresholds['killfeed_top']
        killfeed_bottom = thresholds['killfeed_bottom']
        killfeed_left = thresholds['killfeed_left']
        
        # Main frame processing loop - object detection happens here
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # CHECK FOR STOP SIGNAL DURING FRAME ANALYSIS (NEW)
            if frame_count % 60 == 0:
                control_file = config_params.get('control_file')
                if control_file:
                    state = check_pause_state(control_file)
                    if state == 'stopped':
                        cap.release()
                        if show_preview and worker_count == 1:
                            cv2.destroyAllWindows()
                        print(f"\n⏹️  Processing stopped during analysis at frame {frame_count}")
                        return ('stopped', test_file, None)
            
            # Progress reporting (only in sequential mode)
            if frame_count % 30 == 0 or frame_count == 1:
                if worker_count == 1:
                    current_progress = (frame_count / total_frames) * 100
                    overall_progress = ((video_index + (frame_count / total_frames)) / num_videos_to_test) * 100
                    print(f"\r  Video {video_index + 1}/{num_videos_to_test}: {current_progress:.1f}% | Overall: {overall_progress:.1f}%", end='', flush=True)
            
            # Extract killfeed region using pre-calculated scaled values
            killfeed_region = frame[killfeed_top:killfeed_bottom, killfeed_left:video_width]
            
            # Color detection for red kill indicators
            hsv = cv2.cvtColor(killfeed_region, cv2.COLOR_BGR2HSV)
            
            lower_red1 = np.array([0, 80, 80])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([165, 80, 80])
            upper_red2 = np.array([180, 255, 255])
            
            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            red_mask = mask1 + mask2
            
            # Find contours - potential kill indicators
            contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            current_rectangles = []

            # Analyze each detected contour
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Use scaled thresholds from cached dict
                if w > thresholds['MIN_WIDTH'] and thresholds['MIN_HEIGHT'] < h < thresholds['MAX_HEIGHT']:
                    roi_mask = red_mask[y:y+h, x:x+w]
                    red_pixel_count = cv2.countNonZero(roi_mask)
                    
                    rect_area = w * h
                    fill_ratio = red_pixel_count / rect_area if rect_area > 0 else 0
                    
                    # Use scaled minimum area threshold
                    if rect_area > thresholds['MIN_RECT_AREA']:
                        cv2.rectangle(killfeed_region, (x, y), (x+w, y+h), (0, 255, 255), 1)
                        
                        if 0.05 < fill_ratio < 0.22:
                            current_rectangles.append((x, y, w, h))
                            cv2.rectangle(killfeed_region, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Rectangle tracking and confirmation logic
            for x, y, w, h in current_rectangles:
                is_new = True
                matched_index = None
                
                center_x = x + w // 2
                center_y = y + h // 2
                
                # Check if rectangle matches recent rectangles (using cached thresholds)
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
                    # Check pending rectangles for confirmation (using cached thresholds)
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
                            
                            # Confirm rectangle after sufficient frames
                            if p_count + 1 >= thresholds['CONFIRMATION_FRAMES']:
                                rectangle_count += 1
                                recent_rectangles.append((x, y, w, h, frame_count))
                                
                                rect_area = w * h
                                roi_mask = red_mask[y:y+h, x:x+w]
                                red_pixel_count = cv2.countNonZero(roi_mask)
                                fill_ratio = red_pixel_count / rect_area if rect_area > 0 else 0
                                
                                print(f"\nFrame {frame_count}: NEW red rectangle #{rectangle_count} (confirmed after {p_count + 1} frames)")
                                print(f"  Position: ({x}, {y}), Center: ({center_x}, {center_y})")
                                print(f"  Size: w={w}, h={h}, rect_area={rect_area}")
                                print(f"  Red pixels: {red_pixel_count}, fill_ratio={fill_ratio:.3f}")
                                
                                confirmed_rectangles_data.append({
                                    'rect_area': rect_area,
                                    'position': (x, y),
                                    'frame': frame_count
                                })
                                
                                if first_rectangle_frame is None:
                                    first_rectangle_frame = p_first_frame
                                last_rectangle_frame = frame_count
                                
                                pending_rectangles.pop(p_idx)
                            
                            found_pending = True
                            break
                    
                    if not found_pending:
                        pending_rectangles.append((x, y, w, h, frame_count, 1))
                        print(f"  Pending rectangle at frame {frame_count} (needs {thresholds['CONFIRMATION_FRAMES']} confirmations)")
                
                else:
                    # Update existing rectangle
                    recent_rectangles[matched_index] = (x, y, w, h, frame_count)
            
            # Clean up old pending and recent rectangles
            pending_rectangles = [(x, y, w, h, f, c) for (x, y, w, h, f, c) in pending_rectangles
                                  if frame_count - f <= thresholds['PENDING_CLEANUP_FRAMES']]
            recent_rectangles = [(x, y, w, h, f) for (x, y, w, h, f) in recent_rectangles 
                                 if frame_count - f <= thresholds['DISAPPEAR_THRESHOLD']]

            # Display rectangle count
            cv2.putText(killfeed_region, f"Total: {rectangle_count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Show preview if enabled (only in sequential mode)
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

        cap.release()
        if show_preview and worker_count == 1:
            cv2.destroyAllWindows()
        
        print("\r" + " " * 80)
        print(f"\rFinished processing!")
        print(f"Total frames processed: {frame_count}")
        print(f"Total unique red rectangles detected: {rectangle_count}")
        
        # Detect video repeats using helper function
        is_repeating, half = check_video_repeat(confirmed_rectangles_data, thresholds['POSITION_MATCH_THRESHOLD'])
        
        if is_repeating:
            print(f"\nVIDEO REPEAT DETECTED with {half} unique kills!")
        
        # Determine trimming strategy using helper function
        should_trim, trim_start, trim_end, status = determine_trim_points(
            rectangle_count, confirmed_rectangles_data, is_repeating, half,
            total_frames, fps, trim_start_seconds, trim_end_seconds
        )
        
        if status == 'unsorted':
            if rectangle_count == 0:
                print(f"No rectangles detected. Moving to unsorted_files.")
            else:
                print(f"More than 5 kills detected. Moving to unsorted_files.")
            return ('unsorted', test_file, None)
        
        # Log trimming decisions
        if should_trim:
            print(f"\nWill trim video from frame {trim_start} to {trim_end}")
        else:
            print(f"\nNo trimming needed")
        
        # Output processing
        if should_trim:
            filename = os.path.basename(test_file)
            output_file = generate_unique_output_path(output_folder, filename, '_trimmed')
            
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
                
                # Get control file path
                control_file = config_params.get('control_file')
                
                while current_frame < trim_end:
                    # Check for stop signal periodically
                    if control_file and current_frame % 30 == 0:
                        state = check_pause_state(control_file)
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
                print(f"Trimmed video saved to: {output_file}")
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
                # Clean up on any error
                cap.release()
                if 'out' in locals():
                    out.release()
                if os.path.exists(output_file):
                    os.remove(output_file)
                raise
            
        elif rectangle_count > 0 and rectangle_count <= 5 and not should_trim:
            filename = os.path.basename(test_file)
            
            needs_resize = (output_width != video_width or output_height != video_height)
            needs_fps_change = (output_fps_setting != 0 and output_fps_setting != fps)
            
            if needs_resize or needs_fps_change:
                output_file = generate_unique_output_path(output_folder, filename, '_processed')
                
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
                print(f"Processed video saved to: {output_file}")
                return ('processed', test_file, output_file)
            else:
                # Generate unique filename for copy
                base_filename = os.path.basename(test_file)
                base_output_file = os.path.join(output_folder, base_filename)
                
                output_file = base_output_file
                if os.path.exists(output_file):
                    name, ext = os.path.splitext(base_filename)
                    counter = 1
                    while True:
                        output_file = os.path.join(output_folder, f"{name}({counter}){ext}")
                        if not os.path.exists(output_file):
                            break
                        counter += 1
                
                print(f"No processing needed. Copying to output folder...")
                
                shutil.copy2(test_file, output_file)
                print(f"Video copied to: {output_file}")
                return ('processed', test_file, output_file)
        
        return ('processed', test_file, None)
    
    except KeyboardInterrupt:
        # Allow user to interrupt gracefully
        print(f"\n⚠️ Processing interrupted by user for: {os.path.basename(test_file)}")
        raise  # Re-raise to stop the entire batch
        
    except Exception as e:
        # Catch all other errors
        print(f"\n❌ ERROR processing {os.path.basename(test_file)}")
        print(f"   Error type: {type(e).__name__}")
        print(f"   Error message: {str(e)}")
        print(f"   Moving to unsorted folder...\n")
        return ('error', test_file, None)

# ========================================
# Pause / Resume Helper Functions
# ========================================

def check_pause_state(control_file):
    """
    Check if the process should be paused.
    Returns: 'running', 'paused', or 'stopped' (if file doesn't exist)
    """
    if not os.path.exists(control_file):
        return 'stopped'
    
    try:
        with open(control_file, 'r') as f:
            state = f.read().strip()
        return state
    except:
        return 'running'


def wait_while_paused(control_file):
    """
    Wait in a loop while the process is paused.
    Returns True if should continue, False if should stop.
    """
    print("\n" + "="*60)
    print("⏸️  PROCESSING PAUSED")
    print("="*60)
    print("Waiting for resume signal...")
    print("(Close the console window or delete process_control.txt to stop)")
    print("="*60 + "\n")
    
    while True:
        state = check_pause_state(control_file)
        
        if state == 'stopped':
            print("\n⏹️  Process control file deleted. Stopping...")
            return False
        elif state == 'running':
            print("\n▶️  PROCESSING RESUMED")
            print("="*60 + "\n")
            return True
        
        time.sleep(1)  # Check every second

def cleanup_incomplete_files(output_folder, unsorted_folder):
    """
    Clean up any incomplete/temporary files that might be left behind.
    Call this at the start of the script.
    """
    import glob
    
    print("Checking for incomplete files from previous runs...")
    
    cleaned_count = 0
    
    # Check both output and unsorted folders
    folders_to_check = []
    if os.path.exists(output_folder):
        folders_to_check.append(("output", output_folder))
    if os.path.exists(unsorted_folder):
        folders_to_check.append(("unsorted", unsorted_folder))
    
    for folder_name, folder_path in folders_to_check:
        # Look for suspicious patterns
        patterns = [
            ('*_trimmed.mp4', 300),      # Files modified in last 5 min
            ('*_processed.mp4', 300),    # Files modified in last 5 min  
            ('*_discord.mp4', 300),      # Files modified in last 5 min
        ]
        
        for pattern, max_age_seconds in patterns:
            full_pattern = os.path.join(folder_path, pattern)
            
            for file_path in glob.glob(full_pattern):
                try:
                    # Check if file was modified recently (likely from interrupted session)
                    file_age = time.time() - os.path.getmtime(file_path)
                    file_size = os.path.getsize(file_path)
                    
                    # Remove if: recent AND suspiciously small
                    if file_age < max_age_seconds and file_size < 100000:  # < 100KB
                        os.remove(file_path)
                        cleaned_count += 1
                        print(f"  Removed incomplete {folder_name} file: {os.path.basename(file_path)}")
                except Exception as e:
                    pass  # Skip files we can't process
    
    if cleaned_count > 0:
        print(f"✓ Cleaned up {cleaned_count} incomplete file(s)\n")
    else:
        print("  No incomplete files found\n")


# ========================================
# MAIN EXECUTION
# ========================================

if __name__ == '__main__':
    multiprocessing.freeze_support()
    
    # Setup control file for pause/resume
    control_file = os.path.join(os.path.dirname(__file__), "process_control.txt")
    
    # CLEANUP INCOMPLETE FILES FROM PREVIOUS RUNS (NEW)
    cleanup_incomplete_files(output_folder, unsorted_folder)
    
    video_files = []
    
    for filename in os.listdir(video_folder):
        filepath = os.path.join(video_folder, filename)
        if os.path.isfile(filepath) and filename.lower().endswith('.mp4'):
            video_files.append(filepath)
    
    num_videos_to_test = len(video_files) if num_videos_setting == -1 else min(num_videos_setting, len(video_files))
    
    print(f"\nFound {len(video_files)} video files. Processing {num_videos_to_test} videos.\n")
    
    if num_videos_to_test == 0:
        print("No video files found!")
    else:

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
        'save_all_versions': save_all_versions,
        'control_file': control_file
    }
        
        unsorted_files = []
        error_files = []
        processed_output_files = []
        
        processed_to_source_map = {}

        if worker_count > 1:
            print(f"Starting multiprocessing with {worker_count} workers...\n")
            
            # Process videos in batches to allow pausing
            batch_size = worker_count * 2  # Process 2 batches at a time per worker
            videos_processed = 0
            
            while videos_processed < num_videos_to_test:
                # Check for pause before each batch
                state = check_pause_state(control_file)
                if state == 'paused':
                    should_continue = wait_while_paused(control_file)
                    if not should_continue:
                        print("Processing stopped by user.")
                        break
                elif state == 'stopped':
                    print("\n⏹️  Process control file not found. Stopping...")
                    break
                
                # Calculate batch range
                batch_end = min(videos_processed + batch_size, num_videos_to_test)
                current_batch = [
                    (video_files[i], i, num_videos_to_test, config_params)
                    for i in range(videos_processed, batch_end)
                ]
                
                print(f"Processing batch: videos {videos_processed + 1}-{batch_end} of {num_videos_to_test}")
                
                # Process current batch
                with Pool(processes=worker_count) as pool:
                    batch_results = pool.map(process_single_video, current_batch)
                
                # Collect results and build mapping
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
                # Check for pause before each video
                state = check_pause_state(control_file)
                if state == 'paused':
                    should_continue = wait_while_paused(control_file)
                    if not should_continue:
                        print("Processing stopped by user.")
                        break
                elif state == 'stopped':
                    print("\n⏹️  Process control file not found. Stopping...")
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

    
    # Combine unsorted and error files for moving
    all_unsorted = unsorted_files + error_files
    
    print(f"\n{'='*60}")
    print(f"PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"  Total videos processed: {num_videos_to_test}")
    print(f"  Successfully processed: {num_videos_to_test - len(all_unsorted)}")
    print(f"  Unsorted (0 or >5 kills): {len(unsorted_files)}")
    print(f"  Errors (corrupted/failed): {len(error_files)}")
    print(f"{'='*60}\n")
    


    # ========================================
    # DISCORD COMPRESSION
    # ========================================
    
    unsorted_compressed = []  # Track which unsorted files were compressed
    
    # STEP 1: Compress unsorted files if Discord compression is enabled
    if compress_for_discord and len(all_unsorted) > 0:
        print(f"\n{'='*60}")
        print(f"DISCORD COMPRESSION - UNSORTED FILES")
        print(f"{'='*60}")
        print(f"Compressing {len(all_unsorted)} unsorted video(s) for Discord...\n")
        
        if not os.path.exists(unsorted_folder):
            os.makedirs(unsorted_folder)
        
        for i, unsorted_file in enumerate(all_unsorted, 1):
            if os.path.isfile(unsorted_file):
                filename = os.path.basename(unsorted_file)
                base_name = os.path.splitext(filename)[0]
                discord_output = os.path.join(unsorted_folder, f"{base_name}_discord.mp4")
                
                print(f"\n[{i}/{len(all_unsorted)}] Compressing unsorted: {filename}")
                
                success, output_path, error_msg = compress_video_for_discord(unsorted_file, discord_output, silent=False)
                
                if success:
                    unsorted_compressed.append(unsorted_file)
                    is_error = unsorted_file in error_files
                    file_type = "ERROR" if is_error else "UNSORTED"
                    print(f"  ✓ Compressed {file_type} file saved to: {os.path.basename(output_path)}")
                else:
                    print(f"  ❌ Compression failed: {error_msg}")
                    print(f"  ℹ️  Will copy original file instead...")
    
    # STEP 2: Move unsorted files that weren't successfully compressed
    if len(all_unsorted) > 0:
        files_to_move = [f for f in all_unsorted if f not in unsorted_compressed]
        
        if len(files_to_move) > 0:
            if not os.path.exists(unsorted_folder):
                os.makedirs(unsorted_folder)
            
            print(f"\nMoving {len(files_to_move)} uncompressed file(s) to unsorted folder...")
            
            for unsorted_file in files_to_move:
                if os.path.isfile(unsorted_file):
                    filename = os.path.basename(unsorted_file)
                    base_destination = os.path.join(unsorted_folder, filename)
                    
                    # Use unique filename generation
                    destination = base_destination
                    if os.path.exists(destination):
                        name, ext = os.path.splitext(filename)
                        counter = 1
                        while True:
                            destination = os.path.join(unsorted_folder, f"{name}({counter}){ext}")
                            if not os.path.exists(destination):
                                break
                            counter += 1
                    
                    shutil.copy2(unsorted_file, destination)
                    
                    # Indicate if it was an error vs unsorted
                    if unsorted_file in error_files:
                        print(f"  Moved ERROR file: {filename}")
                    else:
                        print(f"  Moved UNSORTED file: {filename}")
        elif len(unsorted_compressed) > 0:
            print(f"\nAll unsorted files were compressed for Discord (no originals to move).")
    else:
        print("No files to move to unsorted folder.")
    
    # STEP 3: Compress processed output files
    successfully_compressed_files = []

    if compress_for_discord and len(processed_output_files) > 0:
        print(f"\n{'='*60}")
        print(f"DISCORD COMPRESSION - PROCESSED FILES")
        print(f"{'='*60}")
        print(f"Compressing {len(processed_output_files)} processed video(s) for Discord...\n")
        
        successful_compressions = 0
        failed_compressions = 0
        
        for i, video_file in enumerate(processed_output_files, 1):
            # CHECK FOR STOP/PAUSE BEFORE EACH COMPRESSION (NEW)
            state = check_pause_state(control_file)
            if state == 'stopped':
                print(f"\n⏹️  Compression interrupted by user. Stopping...")
                break
            elif state == 'paused':
                should_continue = wait_while_paused(control_file)
                if not should_continue:
                    print("Compression stopped by user.")
                    break
            
            print(f"\n[{i}/{len(processed_output_files)}] Compressing: {os.path.basename(video_file)}")
            
            # Generate output filename
            base_name = os.path.splitext(video_file)[0]
            discord_output = f"{base_name}_discord.mp4"
            
            # Compress the video
            success, output_path, error_msg = compress_video_for_discord(video_file, discord_output, silent=False)
            
            if success:
                successful_compressions += 1
                successfully_compressed_files.append(video_file)
                
                # Only delete trimmed file if save_all_versions is False
                if not save_all_versions:
                    try:
                        os.remove(video_file)
                        print(f"  🗑️  Removed trimmed file: {os.path.basename(video_file)}")
                    except Exception as e:
                        print(f"  ⚠️  Warning: Could not delete trimmed file: {e}")
                else:
                    print(f"  💾 Keeping trimmed version: {os.path.basename(video_file)}")
            else:
                failed_compressions += 1
                print(f"  ✗ Failed: {error_msg}")
                print(f"  ℹ️  Keeping original file: {os.path.basename(video_file)}")
                
                # CLEANUP INCOMPLETE COMPRESSION FILE
                if os.path.exists(discord_output):
                    try:
                        os.remove(discord_output)
                        print(f"  🗑️  Removed incomplete compression: {os.path.basename(discord_output)}")
                    except Exception as e:
                        pass  # Ignore cleanup errors
        
        print(f"\n{'='*60}")
        print(f"DISCORD COMPRESSION SUMMARY")
        print(f"{'='*60}")
        print(f"  Processed files compressed: {successful_compressions}/{len(processed_output_files)}")
        if len(unsorted_compressed) > 0:
            print(f"  Unsorted files compressed: {len(unsorted_compressed)}/{len(all_unsorted)}")
        print(f"  Total compressed: {successful_compressions + len(unsorted_compressed)}")
        if failed_compressions > 0 or len(unsorted_compressed) < len(all_unsorted):
            total_failed = failed_compressions + (len(all_unsorted) - len(unsorted_compressed))
            print(f"  Failed compressions: {total_failed}")
        print(f"{'='*60}\n")
    elif compress_for_discord and len(processed_output_files) == 0 and len(unsorted_compressed) > 0:
        print(f"\n{'='*60}")
        print(f"DISCORD COMPRESSION SUMMARY")
        print(f"{'='*60}")
        print(f"  Unsorted files compressed: {len(unsorted_compressed)}/{len(all_unsorted)}")
        print(f"{'='*60}\n")
    elif compress_for_discord:
        print("\nDiscord compression enabled, but no videos were successfully processed or unsorted.")
    
    # ========================================
    # NEW: STEP 4 - SAVE ORIGINAL SOURCE FILES
    # ========================================
    
    if save_all_versions and len(processed_output_files) > 0:
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
                    # Generate unique filename for original
                    source_basename = os.path.basename(source_file)
                    name, ext = os.path.splitext(source_basename)
                    original_output = os.path.join(output_folder, f"{name}_original{ext}")
                    
                    # Ensure unique filename
                    if os.path.exists(original_output):
                        counter = 1
                        while True:
                            original_output = os.path.join(output_folder, f"{name}_original({counter}){ext}")
                            if not os.path.exists(original_output):
                                break
                            counter += 1
                    
                    shutil.copy2(source_file, original_output)
                    saved_count += 1
                    print(f"  💾 Saved original: {os.path.basename(original_output)}")
                    
                except Exception as e:
                    failed_count += 1
                    print(f"  ⚠️  Failed to save original for {os.path.basename(processed_file)}: {e}")
            else:
                failed_count += 1
                print(f"  ⚠️  Source file not found for: {os.path.basename(processed_file)}")
        
        print(f"\n{'='*60}")
        print(f"ORIGINAL FILES SAVED: {saved_count}/{len(processed_output_files)}")
        if failed_count > 0:
            print(f"  Failed: {failed_count}")
        print(f"{'='*60}\n")