import cv2
import numpy as np
import os
import json
import multiprocessing
from multiprocessing import Pool, cpu_count
import shutil
import platform

# Load configuration from config file - always in same directory as script
script_dir = os.path.dirname(os.path.abspath(__file__))
config_file = os.path.join(script_dir, "settings_config.json")

default_config = {
    "video_folder": r"D:\Videos\Counter-strike 2",
    "output_folder": r"D:/Videos/Counter-strike 2/CS2Clipper/trimmedvideos",
    "unsorted_folder": r"D:/Videos/Counter-strike 2/CS2Clipper/unsortedfiles",
    "trim_start_seconds": 4,
    "trim_end_seconds": 3,
    "output_resolution_width": 1920,
    "output_resolution_height": 1080,
    "output_fps": 0,
    "num_videos_to_process": -1,
    "show_preview": False,
    "multiprocessing_mode": "Auto"
}

# Load config or use defaults
if os.path.exists(config_file):
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
            config = {**default_config, **config}
    except Exception as e:
        print(f"Error loading config: {e}. Using defaults.")
        config = default_config
else:
    print(f"Config file not found at: {config_file}")
    print("Using default settings.")
    config = default_config

# Extract settings from config
video_folder = config["video_folder"]
output_folder = config["output_folder"]
trim_start_seconds = config["trim_start_seconds"]
trim_end_seconds = config["trim_end_seconds"]
output_resolution_width = config.get("output_resolution_width", 1920)
output_resolution_height = config.get("output_resolution_height", 1080)
output_fps = config.get("output_fps", 0)
num_videos_setting = config["num_videos_to_process"]
unsorted_folder = config["unsorted_folder"]
show_preview = config["show_preview"]
multiprocessing_mode = config.get("multiprocessing_mode", "Auto")


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

print(f"Loaded settings:")
print(f"  Config file location: {config_file}")
print(f"  Video folder: {video_folder}")
print(f"  Output folder: {output_folder}")
print(f"  Unsorted folder: {unsorted_folder}")
print(f"  Trim start: {trim_start_seconds} seconds before first kill")
print(f"  Trim end: {trim_end_seconds} seconds after last kill")
print(f"  Output resolution: {output_resolution_width}x{output_resolution_height}")
print(f"  Output FPS: {'Source FPS' if output_fps == 0 else f'{output_fps} FPS'}")
print(f"  Videos to process: {'All' if num_videos_setting == -1 else num_videos_setting}")
print(f"  Show preview: {'Enabled' if show_preview else 'Disabled (Optimized)'}")
print(f"  Multiprocessing mode: {multiprocessing_mode}")
print(f"  Parallel workers: {worker_count}")

# Create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Create unsorted folder if it doesn't exist
if not os.path.exists(unsorted_folder):
    os.makedirs(unsorted_folder)


def process_single_video(args):
    """Process a single video file - designed to be called by multiprocessing"""
    test_file, video_index, num_videos_to_test, config_params = args
    
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
        return ('unsorted', test_file)
    
    frame_count = 0
    rectangle_count = 0
    recent_rectangles = []
    pending_rectangles = []
    confirmed_rectangles_data = []
    DISAPPEAR_THRESHOLD = 60
    CONFIRMATION_FRAMES = 3
    
    first_rectangle_frame = None
    last_rectangle_frame = None
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Define killfeed region as proportions of the source video dimensions
    # Based on typical CS2 killfeed location (top-right corner):
    # - Top edge: ~60 pixels from top on 1080p (60/1080 = 0.0556)
    # - Bottom edge: ~3/11 down from top (0.2727)
    # - Left edge: 4/5 from left (0.8)
    # These ratios scale automatically with any video resolution
    killfeed_top_ratio = 60 / 1080  # ~5.56% from top
    killfeed_bottom_ratio = 3 / 11   # ~27.27% from top
    killfeed_left_ratio = 4 / 5      # 80% from left
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        if frame_count % 30 == 0 or frame_count == 1:
            current_progress = (frame_count / total_frames) * 100
            overall_progress = ((video_index + (frame_count / total_frames)) / num_videos_to_test) * 100
            print(f"\r  Video {video_index + 1}/{num_videos_to_test}: {current_progress:.1f}% | Overall: {overall_progress:.1f}%", end='', flush=True)
        
        height, width = frame.shape[:2]
        killfeed_top = int(height * killfeed_top_ratio)
        killfeed_bottom = int(height * killfeed_bottom_ratio)
        killfeed_left = int(width * killfeed_left_ratio)
        killfeed_region = frame[killfeed_top:killfeed_bottom, killfeed_left:width]
        
        hsv = cv2.cvtColor(killfeed_region, cv2.COLOR_BGR2HSV)
        
        lower_red1 = np.array([0, 80, 80])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([165, 80, 80])
        upper_red2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = mask1 + mask2
        
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        current_rectangles = []

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            if w > 10 and 10 < h < 36:
                roi_mask = red_mask[y:y+h, x:x+w]
                red_pixel_count = cv2.countNonZero(roi_mask)
                
                rect_area = w * h
                fill_ratio = red_pixel_count / rect_area if rect_area > 0 else 0
                
                if rect_area > 3000:
                    cv2.rectangle(killfeed_region, (x, y), (x+w, y+h), (0, 255, 255), 1)
                    
                    if 0.05 < fill_ratio < 0.22:
                        current_rectangles.append((x, y, w, h))
                        cv2.rectangle(killfeed_region, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        for x, y, w, h in current_rectangles:
            is_new = True
            matched_index = None
            
            center_x = x + w // 2
            center_y = y + h // 2
            
            for i, (rx, ry, rw, rh, rframe) in enumerate(recent_rectangles):
                if (frame_count - rframe) <= DISAPPEAR_THRESHOLD:
                    prev_center_x = rx + rw // 2
                    prev_center_y = ry + rh // 2
                    
                    width_ratio = abs(w - rw) / max(w, rw) if max(w, rw) > 0 else 0
                    height_ratio = abs(h - rh) / max(h, rh) if max(h, rh) > 0 else 0
                    size_match = width_ratio < 0.5 and height_ratio < 0.25
                    
                    horizontal_match = abs(center_x - prev_center_x) < 90
                    
                    vertical_diff = center_y - prev_center_y
                    vertical_match = -110 <= vertical_diff <= 10
                    
                    if size_match and horizontal_match and vertical_match:
                        is_new = False
                        matched_index = i
                        break

            if is_new:
                found_pending = False
                for p_idx, (px, py, pw, ph, p_first_frame, p_count) in enumerate(pending_rectangles):
                    p_center_x = px + pw // 2
                    p_center_y = py + ph // 2
                    
                    p_width_ratio = abs(w - pw) / max(w, pw) if max(w, pw) > 0 else 0
                    p_height_ratio = abs(h - ph) / max(h, ph) if max(h, ph) > 0 else 0
                    p_size_match = p_width_ratio < 0.35 and p_height_ratio < 0.2
                    
                    p_horizontal_match = abs(center_x - p_center_x) < 60
                    
                    p_vertical_diff = center_y - p_center_y
                    p_vertical_match = -80 <= p_vertical_diff <= 10
                    
                    if p_size_match and p_horizontal_match and p_vertical_match:
                        pending_rectangles[p_idx] = (x, y, w, h, frame_count, p_count + 1)
                        
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
                recent_rectangles[matched_index] = (x, y, w, h, frame_count)
        
        pending_rectangles = [(x, y, w, h, f, c) for (x, y, w, h, f, c) in pending_rectangles
                              if frame_count - f <= 60]
        recent_rectangles = [(x, y, w, h, f) for (x, y, w, h, f) in recent_rectangles 
                             if frame_count - f <= DISAPPEAR_THRESHOLD]

        cv2.putText(killfeed_region, f"Total: {rectangle_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

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
            position_match = abs(pos1[0] - pos2[0]) < 5 and abs(pos1[1] - pos2[1]) < 5
            
            if area_diff_ratio < 0.05 and position_match:
                matches += 1
                print(f"  Rectangle {i+1} matches {i+half+1}: area {rect1_area} ~= {rect2_area}, pos {pos1} ~= {pos2}")
        
        if matches >= half * 0.8:
            is_repeating = True
            print(f"\nVIDEO REPEAT DETECTED: {matches}/{half} rectangles match!")
    
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
    
    if should_trim:
        filename = os.path.basename(test_file)
        base_output_file = os.path.join(output_folder, filename.replace('.mp4', '_trimmed.mp4'))
        
        output_file = base_output_file
        counter = 1
        while os.path.exists(output_file):
            output_file = base_output_file.replace('_trimmed.mp4', f'_trimmed({counter}).mp4')
            counter += 1
        
        print(f"\nStarting video trimming from frame {trim_start} to {trim_end}...")
        
        # Determine output settings
        needs_resize = (output_width != video_width or output_height != video_height)
        final_fps = fps if output_fps_setting == 0 else output_fps_setting
        
        if needs_resize:
            print(f"Output will be resized from {video_width}x{video_height} to {output_width}x{output_height}")
        if output_fps_setting != 0 and output_fps_setting != fps:
            print(f"Output FPS will be changed from {fps} to {final_fps}")
        
        cap = cv2.VideoCapture(test_file)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # Use output resolution settings
        out = cv2.VideoWriter(output_file, fourcc, final_fps, (output_width, output_height))
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, trim_start)
        current_frame = trim_start
        
        while current_frame < trim_end:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize frame if output resolution is different
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
        
        # Check if we need to process the video (resize or change FPS)
        needs_resize = (output_width != video_width or output_height != video_height)
        needs_fps_change = (output_fps_setting != 0 and output_fps_setting != fps)
        
        if needs_resize or needs_fps_change:
            # Need to process the video
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
            # No processing needed - just copy
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
        
        if worker_count > 1:
            print(f"Starting multiprocessing with {worker_count} workers...\n")
            with Pool(processes=worker_count) as pool:
                results = pool.map(process_single_video, video_args)
            
            unsorted_files = [file for status, file in results if status == 'unsorted']
        else:
            print("Processing videos sequentially...\n")
            for args in video_args:
                result = process_single_video(args)
                if result[0] == 'unsorted':
                    unsorted_files.append(result[1])
    
    print(f"\nUnsorted files: {len(unsorted_files)}")
    
    if len(unsorted_files) > 0:
        if not os.path.exists(unsorted_folder):
            os.makedirs(unsorted_folder)
        
        for unsorted_file in unsorted_files:
            if os.path.isfile(unsorted_file):
                filename = os.path.basename(unsorted_file)
                base_destination = os.path.join(unsorted_folder, filename)
                
                destination = base_destination
                counter = 1
                while os.path.exists(destination):
                    destination = base_destination.replace('.mp4', f'({counter}).mp4')
                    counter += 1
                
                shutil.copy2(unsorted_file, destination)
                print(f"Copied to unsorted: {destination}")
    else:
        print("No unsorted files to copy.")