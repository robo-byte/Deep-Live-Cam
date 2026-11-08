# --- START OF FILE globals.py ---

import os
from typing import List, Dict, Any

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKFLOW_DIR = os.path.join(ROOT_DIR, "workflow")

# Canonical media extensions, defined once so the file dialogs and
# has_image_extension never drift. GIF is intentionally excluded: OpenCV's
# cv2.imread/imwrite (the only image I/O this app uses) cannot decode or
# encode GIF on 4.10 or 4.11, so offering it would silently fail. WEBP works
# via the libwebp bundled with opencv-python.
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
VIDEO_EXTENSIONS = (".mp4", ".mkv")

# Face Mapping Data
source_target_map: List[Dict[str, Any]] = [] # Stores detailed map for image/video processing
simple_map: Dict[str, Any] = {}             # Stores simplified map (embeddings/faces) for live/simple mode

# Paths
source_path: str | None = None
target_path: str | None = None
output_path: str | None = None

# Processing Options
frame_processors: List[str] = []
keep_fps: bool = True
keep_audio: bool = True
keep_frames: bool = False
many_faces: bool = False         # Process all detected faces with default source (single-face default for GTX 1050 Ti)
map_faces: bool = False          # Use source_target_map or simple_map for specific swaps
poisson_blend: bool = False      # Enable Poisson Blending for smoother face swaps
color_correction: bool = False   # Enable color correction (implementation specific)
nsfw_filter: bool = False

# Video Output Options
video_encoder: str | None = None
video_quality: int | None = None # Typically a CRF value or bitrate

# Live Mode Options
live_mirror: bool = False
live_resizable: bool = True
camera_input_combobox: Any | None = None # Placeholder for UI element if needed
webcam_preview_running: bool = False
show_fps: bool = True

# Virtual Camera & GTX 1050 Ti Performance Profile Options
virtual_cam: bool = True         # Stream live output to DirectShow 'Face Swap Cam'
profile: str = "balanced"        # 'fast', 'balanced', or 'quality'
frame_skip: int = 2              # Number of pose-warped intermediate frames per neural swap (0 = every frame neural swap)
capture_fps: int = 30            # Webcam capture FPS limit (30 FPS for low latency on GTX 1050 Ti)
auto_live: bool = False          # Automatically launch Live webcam + Virtual Cam on startup
camera_index: int | None = None  # Explicit camera index override

# Mode Selector: 'face_swap', 'clothes_tryon', or 'both'
tryon_mode: str = "face_swap"
garment_path: str | None = None

# System Configuration
max_memory: int | None = None        # Memory limit in GB? (Needs clarification)
execution_providers: List[str] = []  # e.g., ['CUDAExecutionProvider', 'CPUExecutionProvider']
execution_threads: int | None = None # Number of threads for CPU execution
headless: bool | None = None         # Run without UI?
log_level: str = "error"             # Logging level (e.g., 'debug', 'info', 'warning', 'error')

# Face Processor UI Toggles (GFPGAN / enhancers OFF by default for 4GB VRAM)
fp_ui: Dict[str, bool] = {"face_enhancer": False, "face_enhancer_gpen256": False, "face_enhancer_gpen512": False}

# Face Swapper Specific Options
face_swapper_enabled: bool = True # General toggle for the swapper processor
opacity: float = 1.0              # Blend factor for the swapped face (0.0-1.0)
sharpness: float = 0.0            # Sharpness enhancement for swapped face (0.0-1.0+)

# Mouth Mask Options (OFF by default for GTX 1050 Ti speed)
mouth_mask: bool = False           # Enable mouth area masking/pasting
show_mouth_mask_box: bool = False  # Visualize the mouth mask area (for debugging)
mask_feather_ratio: int = 12       # Denominator for feathering calculation (higher = smaller feather)
mask_down_size: float = 0.1        # Expansion factor for lower lip mask (relative)
mask_size: float = 1.0             # Expansion factor for upper lip mask (relative)
mouth_mask_size: float = 0.0       # Mouth mask size (0-100; 0=off, 100=mouth to chin)

# --- START: Added for Frame Interpolation ---
enable_interpolation: bool = True # Toggle temporal smoothing
interpolation_weight: float = 0  # Blend weight for current frame (0.0-1.0). Lower=smoother.
# --- END: Added for Frame Interpolation ---

# Face detection resolution (160, 320, 480, or 640).
# Lower = faster detection, fewer FLOPs, less accurate at distance.
# Changes require face analyser re-init (handled in UI by clearing FACE_ANALYSER).
# DEFAULT_DET_SIZE is 320 for GTX 1050 Ti Balanced profile.
DEFAULT_DET_SIZE: int = 320
det_size: int = DEFAULT_DET_SIZE

# Webcam capture resolution requested via cv2.CAP_PROP_FRAME_WIDTH/HEIGHT.
# Default 640x480: native on virtually every webcam, fastest reliable mode.
capture_resolution: tuple = (640, 480)

PROFILES: Dict[str, Dict[str, Any]] = {
    "fast": {
        "capture_resolution": (480, 360),
        "capture_fps": 30,
        "det_size": 160,
        "frame_skip": 3,
        "many_faces": False,
        "mouth_mask": False,
        "poisson_blend": False,
    },
    "balanced": {
        "capture_resolution": (640, 480),
        "capture_fps": 30,
        "det_size": 320,
        "frame_skip": 2,
        "many_faces": False,
        "mouth_mask": False,
        "poisson_blend": False,
    },
    "quality": {
        "capture_resolution": (640, 480),
        "capture_fps": 30,
        "det_size": 480,
        "frame_skip": 0,
        "many_faces": False,
        "mouth_mask": False,
        "poisson_blend": False,
    },
}


def apply_profile(profile_name: str) -> None:
    """Apply one of the GTX 1050 Ti preset profiles ('fast', 'balanced', 'quality')."""
    global profile, capture_resolution, capture_fps, det_size, frame_skip, many_faces, mouth_mask, poisson_blend, fp_ui
    p = profile_name.lower().strip()
    if p not in PROFILES:
        return
    cfg = PROFILES[p]
    profile = p
    capture_resolution = cfg["capture_resolution"]
    capture_fps = cfg["capture_fps"]
    det_size = cfg["det_size"]
    frame_skip = cfg["frame_skip"]
    many_faces = cfg["many_faces"]
    mouth_mask = cfg["mouth_mask"]
    poisson_blend = cfg["poisson_blend"]
    fp_ui["face_enhancer"] = False
    fp_ui["face_enhancer_gpen256"] = False
    fp_ui["face_enhancer_gpen512"] = False


# --- END OF FILE globals.py ---

import threading
dml_lock = threading.Lock()
