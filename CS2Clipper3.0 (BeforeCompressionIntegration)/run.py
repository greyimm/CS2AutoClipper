"""
CS2 Auto Clipper - Video Processing and Kill Detection

This module handles:
1. Video processing and frame analysis
2. Kill detection (red rectangle tracking in killfeed)
3. Video trimming and output
"""

import cv2
import numpy as np
import os
import multiprocessing
from multiprocessing import Pool, cpu_count
import shutil
import platform

# Import configuration management from settings module
from settings import load_config, print_config_summary

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
            return ('error', test_file)
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video resolution: {video_width}x{video_height}")
        
        # ========================================
        # RESOLUTION-ADAPTIVE SCALING
        # ========================================
        # All detection thresholds are calibrated for 1080p (1920x1080)
        # Scale them proportionally based on actual video dimensions
        REFERENCE_WIDTH = 1920
        REFERENCE_HEIGHT = 1080
        
        width_scale = video_width / REFERENCE_WIDTH
        height_scale = video_height / REFERENCE_HEIGHT
        
        # Scaled detection thresholds for rectangle detection
        MIN_WIDTH = int(round(10 * width_scale))
        MIN_HEIGHT = int(round(10 * height_scale))
        MAX_HEIGHT = int(round(36 * height_scale))
        MIN_RECT_AREA = int(round(3000 * width_scale * height_scale))  # Area scales with both dimensions
        DISAPPEAR_THRESHOLD = 60  # Frame-based, doesn't need scaling
        CONFIRMATION_FRAMES = 3   # Frame-based, doesn't need scaling
        
        # Matching thresholds for tracking (horizontal uses width scale, vertical uses height scale)
        HORIZONTAL_MATCH_THRESHOLD = int(round(90 * width_scale))
        VERTICAL_MATCH_MAX = int(round(110 * height_scale))
        VERTICAL_MATCH_MIN = int(round(10 * height_scale))
        
        # Pending rectangle thresholds
        PENDING_HORIZONTAL_THRESHOLD = int(round(60 * width_scale))
        PENDING_VERTICAL_MAX = int(round(80 * height_scale))
        PENDING_VERTICAL_MIN = int(round(10 * height_scale))
        PENDING_CLEANUP_FRAMES = 60  # Frame-based, doesn't need scaling
        
        # Position match threshold for repeat detection (used later)
        POSITION_MATCH_THRESHOLD = int(round(5 * min(width_scale, height_scale)))
        
        print(f"Detection scaling factors: {width_scale:.2f}x width, {height_scale:.2f}x height (reference: 1920x1080)")
        print(f"  Rectangle size range: w>{MIN_WIDTH}px, {MIN_HEIGHT}<h<{MAX_HEIGHT}px")
        print(f"  Minimum rectangle area: {MIN_RECT_AREA}px²")
        
        # Object detection tracking variables
        frame_count = 0
        rectangle_count = 0
        recent_rectangles = []
        pending_rectangles = []
        confirmed_rectangles_data = []
        
        first_rectangle_frame = None
        last_rectangle_frame = None
        
        # ========================================
        # KILLFEED REGION DEFINITION (Resolution-Adaptive)
        # ========================================
        # Reference values for 1080p (1920x1080):
        # - Top edge: 60 pixels from top
        # - Bottom edge: 295 pixels from top (approximately 3/11 of height)
        # - Left edge: 1536 pixels from left (4/5 of width, i.e., 80%)
        # - Right edge: full width
        
        KILLFEED_TOP_REF = 60        # pixels from top at 1080p
        KILLFEED_BOTTOM_REF = 295    # pixels from top at 1080p
        KILLFEED_LEFT_REF = 1536     # pixels from left at 1920p
        
        # Scale these reference values to current resolution
        killfeed_top = int(round(KILLFEED_TOP_REF * height_scale))
        killfeed_bottom = int(round(KILLFEED_BOTTOM_REF * height_scale))
        killfeed_left = int(round(KILLFEED_LEFT_REF * width_scale))
        
        print(f"  Killfeed region: top={killfeed_top}px, bottom={killfeed_bottom}px, left={killfeed_left}px")
        
        # Main frame processing loop - object detection happens here
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
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
                
                # Use scaled thresholds
                if w > MIN_WIDTH and MIN_HEIGHT < h < MAX_HEIGHT:
                    roi_mask = red_mask[y:y+h, x:x+w]
                    red_pixel_count = cv2.countNonZero(roi_mask)
                    
                    rect_area = w * h
                    fill_ratio = red_pixel_count / rect_area if rect_area > 0 else 0
                    
                    # Use scaled minimum area threshold
                    if rect_area > MIN_RECT_AREA:
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
                
                # Check if rectangle matches recent rectangles (using scaled thresholds)
                for i, (rx, ry, rw, rh, rframe) in enumerate(recent_rectangles):
                    if (frame_count - rframe) <= DISAPPEAR_THRESHOLD:
                        prev_center_x = rx + rw // 2
                        prev_center_y = ry + rh // 2
                        
                        width_ratio = abs(w - rw) / max(w, rw) if max(w, rw) > 0 else 0
                        height_ratio = abs(h - rh) / max(h, rh) if max(h, rh) > 0 else 0
                        size_match = width_ratio < 0.5 and height_ratio < 0.25
                        
                        horizontal_match = abs(center_x - prev_center_x) < HORIZONTAL_MATCH_THRESHOLD
                        
                        vertical_diff = center_y - prev_center_y
                        vertical_match = -VERTICAL_MATCH_MAX <= vertical_diff <= VERTICAL_MATCH_MIN
                        
                        if size_match and horizontal_match and vertical_match:
                            is_new = False
                            matched_index = i
                            break

                if is_new:
                    # Check pending rectangles for confirmation (using scaled thresholds)
                    found_pending = False
                    for p_idx, (px, py, pw, ph, p_first_frame, p_count) in enumerate(pending_rectangles):
                        p_center_x = px + pw // 2
                        p_center_y = py + ph // 2
                        
                        p_width_ratio = abs(w - pw) / max(w, pw) if max(w, pw) > 0 else 0
                        p_height_ratio = abs(h - ph) / max(h, ph) if max(h, ph) > 0 else 0
                        p_size_match = p_width_ratio < 0.35 and p_height_ratio < 0.2
                        
                        p_horizontal_match = abs(center_x - p_center_x) < PENDING_HORIZONTAL_THRESHOLD
                        
                        p_vertical_diff = center_y - p_center_y
                        p_vertical_match = -PENDING_VERTICAL_MAX <= p_vertical_diff <= PENDING_VERTICAL_MIN
                        
                        if p_size_match and p_horizontal_match and p_vertical_match:
                            pending_rectangles[p_idx] = (x, y, w, h, frame_count, p_count + 1)
                            
                            # Confirm rectangle after sufficient frames
                            if p_count + 1 >= CONFIRMATION_FRAMES:
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
                        print(f"  Pending rectangle at frame {frame_count} (needs {CONFIRMATION_FRAMES} confirmations)")
                
                else:
                    # Update existing rectangle
                    recent_rectangles[matched_index] = (x, y, w, h, frame_count)
            
            # Clean up old pending and recent rectangles
            pending_rectangles = [(x, y, w, h, f, c) for (x, y, w, h, f, c) in pending_rectangles
                                  if frame_count - f <= PENDING_CLEANUP_FRAMES]
            recent_rectangles = [(x, y, w, h, f) for (x, y, w, h, f) in recent_rectangles 
                                 if frame_count - f <= DISAPPEAR_THRESHOLD]

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
        
        # Detect video repeats
        is_repeating = False
        half = 0
        
        if rectangle_count >= 6 and rectangle_count % 2 == 0:
            print(f"\nChecking for video repeats...")
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
                # Use scaled position threshold for matching
                position_match = (abs(pos1[0] - pos2[0]) < POSITION_MATCH_THRESHOLD and 
                                abs(pos1[1] - pos2[1]) < POSITION_MATCH_THRESHOLD)
                
                if area_diff_ratio < 0.05 and position_match:
                    matches += 1
                    print(f"  Rectangle {i+1} matches {i+half+1}: area {rect1_area} ~= {rect2_area}, pos {pos1} ~= {pos2}")
            
            if matches >= half * 0.8:
                is_repeating = True
                print(f"\nVIDEO REPEAT DETECTED: {matches}/{half} rectangles match!")
        
        # Determine trimming strategy
        should_trim = False
        trim_start = 0
        trim_end = total_frames
        
        if is_repeating:
            print(f"Processing repeating clip with {half} unique kills...")
            
            second_half_first_frame = confirmed_rectangles_data[half]['frame']
            second_half_last_frame = confirmed_rectangles_data[-1]['frame']
            
            first_5_seconds_frames = int(5 * fps)
            if second_half_first_frame > first_5_seconds_frames:
                trim_start = max(0, second_half_first_frame - int(trim_start_seconds * fps))
                print(f"Will trim beginning: starting at frame {trim_start} ({trim_start_seconds} sec before 2nd half start)")
                should_trim = True
            else:
                print(f"Second half rectangle present in first 5 seconds. Not trimming beginning.")
            
            last_4_seconds_frames = total_frames - int(4 * fps)
            if second_half_last_frame < last_4_seconds_frames:
                trim_end = min(total_frames, second_half_last_frame + int(trim_end_seconds * fps))
                print(f"Will trim end: ending at frame {trim_end} ({trim_end_seconds} sec after 2nd half end)")
                should_trim = True
            else:
                print(f"Second half rectangle present in last 4 seconds. Not trimming end.")
                
        elif rectangle_count > 5:
            print(f"More than 5 kills detected. Moving to unsorted_files.")
            return ('unsorted', test_file)
            
        elif rectangle_count > 0 and rectangle_count <= 5:
            print(f"Processing clip with {rectangle_count} kills...")
            
            first_5_seconds_frames = int(5 * fps)
            if first_rectangle_frame > first_5_seconds_frames:
                trim_start = max(0, first_rectangle_frame - int(trim_start_seconds * fps))
                print(f"Will trim beginning: starting at frame {trim_start} ({trim_start_seconds} sec before first kill)")
                should_trim = True
            else:
                print(f"Rectangle present in first 5 seconds. Not trimming beginning.")
            
            last_4_seconds_frames = total_frames - int(4 * fps)
            if last_rectangle_frame < last_4_seconds_frames:
                trim_end = min(total_frames, last_rectangle_frame + int(trim_end_seconds * fps))
                print(f"Will trim end: ending at frame {trim_end} ({trim_end_seconds} sec after last kill)")
                should_trim = True
            else:
                print(f"Rectangle present in last 4 seconds. Not trimming end.")
                
        else:
            print(f"No rectangles detected. Moving to unsorted_files.")
            return ('unsorted', test_file)
        
        # Output processing
        if should_trim:
            filename = os.path.basename(test_file)
            base_output_file = os.path.join(output_folder, filename.replace('.mp4', '_trimmed.mp4'))
            
            output_file = base_output_file
            counter = 1
            while os.path.exists(output_file):
                output_file = base_output_file.replace('_trimmed.mp4', f'_trimmed({counter}).mp4')
                counter += 1
            
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
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, trim_start)
            current_frame = trim_start
            
            while current_frame < trim_end:
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
            return ('processed', test_file)
            
        elif rectangle_count > 0 and rectangle_count <= 5 and not should_trim:
            filename = os.path.basename(test_file)
            
            needs_resize = (output_width != video_width or output_height != video_height)
            needs_fps_change = (output_fps_setting != 0 and output_fps_setting != fps)
            
            if needs_resize or needs_fps_change:
                base_output_file = os.path.join(output_folder, filename.replace('.mp4', '_processed.mp4'))
                
                output_file = base_output_file
                counter = 1
                while os.path.exists(output_file):
                    output_file = base_output_file.replace('_processed.mp4', f'_processed({counter}).mp4')
                    counter += 1
                
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
                return ('processed', test_file)
            else:
                base_output_file = os.path.join(output_folder, filename)
                
                output_file = base_output_file
                counter = 1
                while os.path.exists(output_file):
                    output_file = base_output_file.replace('.mp4', f'({counter}).mp4')
                    counter += 1
                
                print(f"No processing needed. Copying to output folder...")
                
                shutil.copy2(test_file, output_file)
                print(f"Video copied to: {output_file}")
                return ('processed', test_file)
        
        return ('processed', test_file)
    
    except KeyboardInterrupt:
        # Allow user to interrupt gracefully
        print(f"\n⚠️  Processing interrupted by user for: {os.path.basename(test_file)}")
        raise  # Re-raise to stop the entire batch
        
    except Exception as e:
        # Catch all other errors
        print(f"\n❌ ERROR processing {os.path.basename(test_file)}")
        print(f"   Error type: {type(e).__name__}")
        print(f"   Error message: {str(e)}")
        print(f"   Moving to unsorted folder...\n")
        return ('error', test_file)


# ========================================
# MAIN EXECUTION
# ========================================

if __name__ == '__main__':
    multiprocessing.freeze_support()
    
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
            'worker_count': worker_count
        }
        
        video_args = [
            (video_files[i], i, num_videos_to_test, config_params) 
            for i in range(num_videos_to_test)
        ]
        
        unsorted_files = []
        error_files = []
        
        if worker_count > 1:
            print(f"Starting multiprocessing with {worker_count} workers...\n")
            with Pool(processes=worker_count) as pool:
                results = pool.map(process_single_video, video_args)
            
            unsorted_files = [file for status, file in results if status == 'unsorted']
            error_files = [file for status, file in results if status == 'error']
        else:
            print("Processing videos sequentially...\n")
            for args in video_args:
                result = process_single_video(args)
                if result[0] == 'unsorted':
                    unsorted_files.append(result[1])
                elif result[0] == 'error':
                    error_files.append(result[1])
    
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
    
    if len(all_unsorted) > 0:
        if not os.path.exists(unsorted_folder):
            os.makedirs(unsorted_folder)
        
        for unsorted_file in all_unsorted:
            if os.path.isfile(unsorted_file):
                filename = os.path.basename(unsorted_file)
                base_destination = os.path.join(unsorted_folder, filename)
                
                destination = base_destination
                counter = 1
                while os.path.exists(destination):
                    destination = base_destination.replace('.mp4', f'({counter}).mp4')
                    counter += 1
                
                shutil.copy2(unsorted_file, destination)
                
                # Indicate if it was an error vs unsorted
                if unsorted_file in error_files:
                    print(f"  Moved ERROR file: {filename}")
                else:
                    print(f"  Moved UNSORTED file: {filename}")
    else:
        print("No files to move to unsorted folder.")