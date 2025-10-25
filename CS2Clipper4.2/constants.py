"""
CS2 Clipper - Constants and Configuration
Centralized constants to avoid magic numbers and duplication
"""

import os

# ========================================
# FILE PATHS
# ========================================

# Get script directory for relative paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Control file for pause/resume functionality
CONTROL_FILE_PATH = os.path.join(SCRIPT_DIR, "process_control.txt")

# Configuration file
CONFIG_FILE_NAME = "settings_config.json"

# Icon folder for weapon detection
WEAPON_ICONS_FOLDER = os.path.join(SCRIPT_DIR, 'CS2 Icons', 'Weapons')

# ========================================
# PROCESSING THRESHOLDS
# ========================================

# File cleanup thresholds
INCOMPLETE_FILE_MAX_AGE_SECONDS = 300  # 5 minutes
INCOMPLETE_FILE_MAX_SIZE_BYTES = 100000  # 100KB

# Video processing
MAX_KILLS_FOR_AUTO_PROCESSING = 5
MIN_KILLS_FOR_PROCESSING = 1

# Frame checking intervals
PAUSE_CHECK_INTERVAL_FRAMES = 60  # Check every 60 frames

# ========================================
# HARDWARE DETECTION
# ========================================

# RAM thresholds for multiprocessing
RAM_THRESHOLD_MINIMUM_GB = 6.0  # Below this: sequential only
RAM_THRESHOLD_MODERATE_GB = 10.0  # Below this: 2 workers max
RAM_DEFAULT_FALLBACK_GB = 8.0  # Assume this if detection fails

# Worker count limits
MAX_PARALLEL_WORKERS = 4
MIN_PARALLEL_WORKERS = 2

# ========================================
# VIDEO DETECTION THRESHOLDS (1080p reference)
# ========================================

# Rectangle detection (calibrated for 1920x1080)
REFERENCE_WIDTH = 1920
REFERENCE_HEIGHT = 1080

# Rectangle size constraints
RECT_MIN_WIDTH = 150
RECT_MIN_HEIGHT = 24
RECT_MAX_HEIGHT = 36
RECT_MIN_AREA = 3500

# Rectangle validation (NEW)
RECT_MIN_WIDTH_THRESHOLD = 100  # Minimum width for valid kill indicator
RECT_MIN_WHITE_RATIO = 0.03  # At least 3% white pixels (weapon icon)

# Rectangle tracking
RECT_DISAPPEAR_THRESHOLD_FRAMES = 30
RECT_CONFIRMATION_FRAMES = 4

# Position matching
HORIZONTAL_MATCH_THRESHOLD = 90
VERTICAL_MATCH_MAX = 110
VERTICAL_MATCH_MIN = 10

PENDING_HORIZONTAL_THRESHOLD = 60
PENDING_VERTICAL_MAX = 80
PENDING_VERTICAL_MIN = 10
PENDING_CLEANUP_FRAMES = 60

# Repeat detection
POSITION_MATCH_THRESHOLD = 5

# Killfeed region (scaled from 1080p)
KILLFEED_TOP = 60
KILLFEED_BOTTOM = 295
KILLFEED_LEFT = 1536

# ========================================
# ICON DETECTION
# ========================================

# Confidence thresholds
ICON_CERTAINTY_THRESHOLD = 0.70  # 70% match minimum
ICON_MINIMUM_THRESHOLD = 0.70

# Multi-scale parameters
ICON_SCALE_MIN = 0.2
ICON_SCALE_MAX = 0.6
ICON_SCALE_STEP = 0.01

# Minimum scaled icon size
ICON_MIN_SCALED_SIZE = 10  # pixels

# ========================================
# DISCORD COMPRESSION
# ========================================

# Target file size
DISCORD_MAX_SIZE_MB = 8.0
DISCORD_MAX_SIZE_BYTES = int(DISCORD_MAX_SIZE_MB * 1024 * 1024)

# Compression settings table
# (max_seconds, video_bitrate, audio_bitrate, resolution, fps)
DISCORD_COMPRESSION_TABLE = [
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

# Resolution dimensions
RESOLUTION_DIMENSIONS = {
    "720p": (1280, 720),
    "540p": (960, 540),
    "360p": (640, 360),
    "240p": (426, 240)
}

# ========================================
# DEFAULT SETTINGS
# ========================================

DEFAULT_SETTINGS = {
    # Folder settings
    "video_folder": r"D:\Videos\Counter-strike 2",
    "output_folder": r"D:\Videos\Counter-strike 2\CS2Clipper\trimmedvideos",
    "unsorted_folder": r"D:\Videos\Counter-strike 2\CS2Clipper\unsortedfiles",
    
    # Trim settings
    "trim_start_seconds": 4,
    "trim_end_seconds": 3,
    
    # Output video settings
    "output_resolution_width": 1920,
    "output_resolution_height": 1080,
    "output_fps": 0,
    
    # Processing settings
    "num_videos_to_process": -1,
    "show_preview": False,
    "multiprocessing_mode": "Auto",
    "compress_for_discord": False,
    "save_versions_mode": "Trimmed and Compressed"
}

# ========================================
# VALIDATION LIMITS
# ========================================

# Resolution constraints
MIN_OUTPUT_WIDTH = 320
MIN_OUTPUT_HEIGHT = 240
MAX_OUTPUT_WIDTH = 7680  # 8K
MAX_OUTPUT_HEIGHT = 4320  # 8K

MIN_ASPECT_RATIO = 0.5
MAX_ASPECT_RATIO = 3.0

# FPS constraints
MIN_FPS = 0  # 0 = use source
MAX_FPS = 240

# ========================================
# ICON PRIORITY ORDER
# ========================================

WEAPON_PRIORITY = [
    'ak47', 'awp', 'm4a4', 'm4a1s', 'scout',
    'galil', 'famas', 'aug', 'sg556', 'scar20', 'g3sg1',
    'mac10', 'mp9', 'mp7', 'ump45', 'p90', 'ppbizon',
    'usp', 'glock', 'deagle', 'tec9', 'dualies',
    'p2000', 'p250', 'fiveseven', 'cz75', 'r8revolver', 'zeus',
    'mag7', 'xm1014', 'nova', 'sawedoff', 'negev', 'm249'
]
# ========================================
# WEAPON FIXED SCALES
# ========================================

# Fixed scales for weapon icons (measured from actual detections)
# Weapons not listed here will use multi-scale detection
WEAPON_FIXED_SCALES = {
    'mac10': 0.261,
    'mp9': 0.430,
    'ak47': 0.527,
    'awp': 0.618,
    'm4a4': 0.454,
    'glock': 0.258,
    'p2000': 0.188,
    'fiveseven': 0.227,
    'p250': 0.219,
    'deagle': 0.297,
    'usp': 0.391,
    'g3sg1': 0.530,
    'scout': 0.579,
    'sg556': 0.522,
    'mp7': 0.288,
    'p90': 0.399,
    'famas': 0.459,
    'galil': 0.524,
    'negev': 0.480,
    'm249': 0.547,
    'dualies': 0.400,
    'r8revolver': 0.315,
    'tec9': 0.305,
    'sawedoff': 0.500,
    'xm1014': 0.572,
    'nova': 0.540,
    'mag7': 0.391,
    'm4a1s': 0.563,
    'scar20': 0.571,
    'aug': 0.445,
    'ump45': 0.485,
    'ppbizon': 0.525,
    'cz75': 0.275
}