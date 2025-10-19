import cv2
import numpy as np
import os
import json

# Load configuration from config file - always in same directory as script
script_dir = os.path.dirname(os.path.abspath(__file__))
config_file = os.path.join(script_dir, "cs2clipper_config.json")

default_config = {
    "video_folder": r"D:\Videos\Counter-strike 2",
    "output_folder": r"C:\Users\dgkbr\OneDrive\Documents\Desktop\Misc\Projects\CS2Clipper\trimmedvideos",
    "trim_start_seconds": 4,
    "trim_end_seconds": 3,
    "resolution_width": 1920,
    "resolution_height": 1080,
    "num_videos_to_process": -1,
    "unsorted_folder": r"C:\Users\dgkbr\OneDrive\Documents\Desktop\Misc\Projects\CS2Clipper\unsortedfiles",
    "show_preview": False
}

# Load config or use defaults
if os.path.exists(config_file):
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
            # Merge with defaults in case new settings were added
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
resolution_width = config["resolution_width"]
resolution_height = config["resolution_height"]
num_videos_setting = config["num_videos_to_process"]
unsorted_folder = config["unsorted_folder"]
show_preview = config["show_preview"]

print(f"Loaded settings:")
print(f"  Config file location: {config_file}")
print(f"  Video folder: {video_folder}")
print(f"  Output folder: {output_folder}")
print(f"  Unsorted folder: {unsorted_folder}")
print(f"  Trim start: {trim_start_seconds} seconds before first kill")
print(f"  Trim end: {trim_end_seconds} seconds after last kill")
print(f"  Target resolution: {resolution_width}x{resolution_height}")
print(f"  Videos to process: {'All' if num_videos_setting == -1 else num_videos_setting}")
print(f"  Show preview: {'Enabled' if show_preview else 'Disabled (Optimized)'}")

video_files = []
unsorted_files = []

# Create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Create unsorted folder if it doesn't exist
if not os.path.exists(unsorted_folder):
    os.makedirs(unsorted_folder)

# Get all video files (only .mp4 files, skip folders and other file types)
for filename in os.listdir(video_folder):
    filepath = os.path.join(video_folder, filename)
    # Only process files (not folders) that end with .mp4
    if os.path.isfile(filepath) and filename.lower().endswith('.mp4'):
        video_files.append(filepath)

# Determine number of videos to process
num_videos_to_test = len(video_files) if num_videos_setting == -1 else min(num_videos_setting, len(video_files))

print(f"\nFound {len(video_files)} video files. Processing {num_videos_to_test} videos.\n")

for video_index in range(num_videos_to_test):
    test_file = video_files[video_index]
    print(f"\n{'='*60}")
    print(f"Processing Video {video_index + 1}/{num_videos_to_test}: {test_file}")
    print(f"{'='*60}\n")
    
    cap = cv2.VideoCapture(test_file)

    if not cap.isOpened():
        print(f"Error opening {test_file}")
        unsorted_files.append(test_file)
    else:
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
        
        # Calculate killfeed region proportionally based on actual video resolution
        killfeed_top_ratio = 60 / video_height
        killfeed_bottom_ratio = (3 / 11)
        killfeed_left_ratio = 4 / 5
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Progress indicator - update every 30 frames to reduce console spam
            if frame_count % 30 == 0 or frame_count == 1:
                current_progress = (frame_count / total_frames) * 100
                overall_progress = ((video_index + (frame_count / total_frames)) / num_videos_to_test) * 100
                print(f"\r  Video {video_index + 1}/{num_videos_to_test}: {current_progress:.1f}% | Overall: {overall_progress:.1f}%", end='', flush=True)
            
            # Calculate killfeed region using proportions from actual video
            height, width = frame.shape[:2]
            killfeed_top = int(height * killfeed_top_ratio)
            killfeed_bottom = int(height * killfeed_bottom_ratio)
            killfeed_left = int(width * killfeed_left_ratio)
            killfeed_region = frame[killfeed_top:killfeed_bottom, killfeed_left:width]
            
            # Convert to HSV for better red detection
            hsv = cv2.cvtColor(killfeed_region, cv2.COLOR_BGR2HSV)
            
            # Define red color range
            lower_red1 = np.array([0, 80, 80])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([165, 80, 80])
            upper_red2 = np.array([180, 255, 255])
            
            # Create masks for red
            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            red_mask = mask1 + mask2
            
            # Find contours
            contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            current_rectangles = []

            # Check each contour for rectangle-like shapes
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
            
            # Check for new rectangles
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
                        # else:
                        #     if i == 0 and is_new:
                        #         if not size_match:
                        #             print(f"    No match: size diff w={abs(w-rw)}, h={abs(h-rh)}, width_ratio={width_ratio:.2f}, height_ratio={height_ratio:.2f}")
                        #         if not horizontal_match:
                        #             print(f"    No match: horizontal diff={abs(center_x - prev_center_x)}")
                        #         if not vertical_match:
                        #             print(f"    No match: vertical_diff={vertical_diff} (out of range)")

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

            # Only show preview if enabled
            if show_preview:
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
        if show_preview:
            cv2.destroyAllWindows()
        
        # Clear the progress line and print final stats
        print("\r" + " " * 80)  # Clear the progress line
        print(f"\rFinished processing!")
        print(f"Total frames processed: {frame_count}")
        print(f"Total unique red rectangles detected: {rectangle_count}")
        
        # Check for video repeats
        is_repeating = False
        repeat_end_frame = None
        
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
                last_first_half_frame = first_half_clips[-1]['frame']
                repeat_end_frame = min(total_frames, last_first_half_frame + int(trim_end_seconds * fps))
                print(f"\nVIDEO REPEAT DETECTED: {matches}/{half} rectangles match!")
                print(f"Will trim video to end at frame {repeat_end_frame} ({trim_end_seconds} seconds after first half ends)")
        
        # DETERMINE TRIMMING LOGIC
        should_trim = False
        trim_start = 0
        trim_end = total_frames
        
        if is_repeating:
            print(f"Processing repeating clip with {half} unique kills...")
            
            second_half_first_frame = second_half_clips[0]['frame']
            second_half_last_frame = second_half_clips[-1]['frame']
            
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
            unsorted_files.append(test_file)
            
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
            unsorted_files.append(test_file)
            
        # PERFORM TRIMMING
        if should_trim:
            filename = os.path.basename(test_file)
            base_output_file = os.path.join(output_folder, filename.replace('.mp4', '_trimmed.mp4'))
            
            # Check for existing file and add (1), (2), etc. if needed
            output_file = base_output_file
            counter = 1
            while os.path.exists(output_file):
                output_file = base_output_file.replace('_trimmed.mp4', f'_trimmed({counter}).mp4')
                counter += 1
            
            print(f"\nStarting video trimming from frame {trim_start} to {trim_end}...")
            
            cap = cv2.VideoCapture(test_file)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_file, fourcc, fps, (video_width, video_height))
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, trim_start)
            current_frame = trim_start
            
            while current_frame < trim_end:
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)
                current_frame += 1
            
            cap.release()
            out.release()
            print(f"Trimmed video saved to: {output_file}")
        elif rectangle_count > 0 and rectangle_count <= 5 and not should_trim:
            filename = os.path.basename(test_file)
            base_output_file = os.path.join(output_folder, filename)
            
            # Check for existing file and add (1), (2), etc. if needed
            output_file = base_output_file
            counter = 1
            while os.path.exists(output_file):
                output_file = base_output_file.replace('.mp4', f'({counter}).mp4')
                counter += 1
            
            print(f"No trimming needed. Copying to output folder...")
            
            import shutil
            shutil.copy2(test_file, output_file)
            print(f"Video copied to: {output_file}")

if num_videos_to_test == 0:
    print("No video files found!")

print(f"\nUnsorted files: {len(unsorted_files)}")

# Copy unsorted files to unsorted folder
if len(unsorted_files) > 0:
    if not os.path.exists(unsorted_folder):
        os.makedirs(unsorted_folder)
    
    import shutil
    for unsorted_file in unsorted_files:
        # Only copy if it's actually a file (not a folder)
        if os.path.isfile(unsorted_file):
            filename = os.path.basename(unsorted_file)
            destination = os.path.join(unsorted_folder, filename)
            shutil.copy2(unsorted_file, destination)
            print(f"Copied to unsorted: {destination}")
else:
    print("No unsorted files to copy.")