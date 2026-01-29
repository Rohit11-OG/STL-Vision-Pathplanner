"""
Configuration file for STL Object Detection System
All adjustable parameters are defined here.

Settings can be overridden via settings.yaml file.
"""

import os
from pathlib import Path
import yaml

# =====================================================
# SETTINGS FILE LOADER
# =====================================================

def load_settings(settings_path: str = None) -> dict:
    """
    Load user settings from YAML file.
    
    Args:
        settings_path: Path to settings.yaml
        
    Returns:
        Dictionary of settings (empty if file not found)
    """
    if settings_path is None:
        settings_path = Path(__file__).parent / "settings.yaml"
    
    try:
        with open(settings_path, 'r') as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}
    except Exception as e:
        print(f"Warning: Could not load settings: {e}")
        return {}

def reload_settings():
    """Reload settings from file and update globals."""
    global CONFIDENCE_THRESHOLD, IOU_THRESHOLD, MAX_DETECTIONS
    global DEFAULT_PATH_STRATEGY, NUM_PATH_POINTS, DEFAULT_PATH_VELOCITY
    global SURFACE_OFFSET, APPROACH_DISTANCE, MIN_CLEARANCE
    global COLLISION_CHECK_ENABLED, SHOW_PATH_OVERLAY, SHOW_FPS
    
    settings = load_settings()
    
    # Detection settings
    det = settings.get('detection', {})
    CONFIDENCE_THRESHOLD = det.get('confidence', 0.65)
    IOU_THRESHOLD = det.get('iou_threshold', 0.45)
    MAX_DETECTIONS = det.get('max_detections', 10)
    
    # Path planning settings
    path = settings.get('path_planning', {})
    DEFAULT_PATH_STRATEGY = path.get('strategy', 'contour')
    NUM_PATH_POINTS = path.get('num_waypoints', 20)
    DEFAULT_PATH_VELOCITY = path.get('velocity', 0.1)
    SURFACE_OFFSET = path.get('surface_offset', 0.02)
    APPROACH_DISTANCE = path.get('approach_distance', 0.05)
    
    # Collision settings
    coll = settings.get('collision', {})
    COLLISION_CHECK_ENABLED = coll.get('enabled', True)
    MIN_CLEARANCE = coll.get('min_clearance', 0.03)
    
    # Visualization settings
    vis = settings.get('visualization', {})
    SHOW_PATH_OVERLAY = vis.get('show_path', True)
    SHOW_FPS = vis.get('show_fps', True)

# Load settings on import
_USER_SETTINGS = load_settings()

# =====================================================
# PATH CONFIGURATIONS
# =====================================================

# Base directory
BASE_DIR = Path(__file__).parent.absolute()

# STL file path - UPDATE THIS TO YOUR STL FILE
STL_FILE = BASE_DIR / "your_object.stl"

# Dataset directories
DATASET_DIR = BASE_DIR / "dataset"
IMAGES_DIR = DATASET_DIR / "images"
LABELS_DIR = DATASET_DIR / "labels"
TRAIN_IMAGES_DIR = DATASET_DIR / "train" / "images"
TRAIN_LABELS_DIR = DATASET_DIR / "train" / "labels"
VAL_IMAGES_DIR = DATASET_DIR / "val" / "images"
VAL_LABELS_DIR = DATASET_DIR / "val" / "labels"

# Model output directory
MODEL_DIR = BASE_DIR / "runs" / "detect"
BEST_MODEL_PATH = MODEL_DIR / "stl_object" / "weights" / "best.pt"

# =====================================================
# RENDERING CONFIGURATIONS
# =====================================================

# Image size for rendering
RENDER_WIDTH = 640
RENDER_HEIGHT = 480

# Number of synthetic images to generate
NUM_SYNTHETIC_IMAGES = 1000

# Train/Val split ratio
TRAIN_VAL_SPLIT = 0.8  # 80% train, 20% val

# Camera distance range (from object center)
CAMERA_DISTANCE_MIN = 1.5
CAMERA_DISTANCE_MAX = 4.0

# Object scale variations
OBJECT_SCALE_MIN = 0.7
OBJECT_SCALE_MAX = 1.3

# Lighting configurations
AMBIENT_LIGHT_INTENSITY = 0.3
DIRECT_LIGHT_INTENSITY_MIN = 2.0
DIRECT_LIGHT_INTENSITY_MAX = 5.0

# Background types: 'solid', 'gradient', 'noise', 'texture'
BACKGROUND_TYPES = ['solid', 'gradient', 'noise']

# =====================================================
# AUGMENTATION CONFIGURATIONS
# =====================================================

# Enable/disable augmentations
ENABLE_BLUR = True
ENABLE_NOISE = True
ENABLE_BRIGHTNESS = True
ENABLE_CONTRAST = True

# Augmentation probabilities
BLUR_PROBABILITY = 0.3
NOISE_PROBABILITY = 0.3
BRIGHTNESS_PROBABILITY = 0.4
CONTRAST_PROBABILITY = 0.4

# Augmentation ranges
BLUR_KERNEL_SIZE = (3, 7)
NOISE_INTENSITY = (0, 25)
BRIGHTNESS_RANGE = (-30, 30)
CONTRAST_RANGE = (0.8, 1.2)

# =====================================================
# TRAINING CONFIGURATIONS
# =====================================================

# YOLO model to use: 'yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x'
YOLO_MODEL = "yolov8n.pt"  # nano for faster training, use 'yolov8s.pt' or 'yolov8m.pt' for better accuracy

# Training parameters
EPOCHS = 100
BATCH_SIZE = 16
IMAGE_SIZE = 640
PATIENCE = 20  # Early stopping patience

# Device: 'cpu', 'cuda', '0', '1', etc.
DEVICE = "0"  # Use GPU 0 if available, change to 'cpu' for CPU training

# Learning rate
LEARNING_RATE = 0.01

# Workers for data loading
NUM_WORKERS = 4

# =====================================================
# DETECTION CONFIGURATIONS
# =====================================================

# Detection confidence threshold
CONFIDENCE_THRESHOLD = 0.65

# IOU threshold for NMS
IOU_THRESHOLD = 0.45

# Maximum detections per image
MAX_DETECTIONS = 10

# =====================================================
# REALSENSE CAMERA CONFIGURATIONS
# =====================================================

# Camera ID (0 for default webcam, or specific RealSense serial)
CAMERA_ID = 0

# RealSense configurations
REALSENSE_WIDTH = 640
REALSENSE_HEIGHT = 480
REALSENSE_FPS = 30

# Enable depth stream
ENABLE_DEPTH = True

# Depth range (in meters)
DEPTH_MIN = 0.3
DEPTH_MAX = 10.0

# =====================================================
# OUTPUT CONFIGURATIONS
# =====================================================

# Save detection results
SAVE_DETECTIONS = True
DETECTION_OUTPUT_DIR = BASE_DIR / "detections"

# Show FPS on detection window
SHOW_FPS = True

# Show depth information
SHOW_DEPTH = True

# Proximity warning threshold (in meters) - object turns red when closer
PROXIMITY_THRESHOLD = 0.4  # 400mm

# =====================================================
# CLASS CONFIGURATIONS
# =====================================================

# Class name for the object (will be used in training)
CLASS_NAME = "custom_object"

# Number of classes
NUM_CLASSES = 1

# =====================================================
# TOOL PATH CONFIGURATIONS
# =====================================================

# Output directory for generated paths
PATH_OUTPUT_DIR = BASE_DIR / "paths"

# Default path strategy: 'contour', 'approach', 'grid', 'surface', 'spiral', 'zigzag'
DEFAULT_PATH_STRATEGY = "contour"

# Available strategies for UI/CLI
PATH_STRATEGIES = ['contour', 'approach', 'grid', 'surface', 'spiral', 'zigzag']

# Number of waypoints for paths
NUM_PATH_POINTS = 20

# Approach distance for pick operations (meters)
APPROACH_DISTANCE = 0.05

# Safety offset from object surface (meters)
SURFACE_OFFSET = 0.02

# Frame ID for ROS integration
PATH_FRAME_ID = "camera_link"

# =====================================================
# COLLISION AVOIDANCE CONFIGURATIONS
# =====================================================

# Enable collision checking
COLLISION_CHECK_ENABLED = True

# Minimum clearance from obstacles (meters)
MIN_CLEARANCE = 0.03

# Safety margin multiplier
SAFETY_MARGIN_MULTIPLIER = 1.5

# =====================================================
# PATH VISUALIZATION CONFIGURATIONS
# =====================================================

# Show path overlay on camera feed
SHOW_PATH_OVERLAY = True

# Path visualization colors (BGR format)
PATH_WAYPOINT_COLOR = (0, 255, 0)      # Green
PATH_APPROACH_COLOR = (0, 255, 255)    # Yellow
PATH_GRASP_COLOR = (0, 0, 255)         # Red
PATH_RETREAT_COLOR = (255, 255, 0)     # Cyan
PATH_LINE_COLOR = (255, 128, 0)        # Orange

# =====================================================
# TRAJECTORY CONFIGURATIONS
# =====================================================

# Default velocity for waypoints (m/s)
DEFAULT_PATH_VELOCITY = 0.1

# Maximum velocity limit (m/s)
MAX_VELOCITY = 0.5

# Maximum acceleration (m/s²)
MAX_ACCELERATION = 0.2

# Compute timing automatically based on distance and velocity
AUTO_COMPUTE_TIMING = True


def create_directories():
    """Create all necessary directories"""
    dirs = [
        DATASET_DIR, IMAGES_DIR, LABELS_DIR,
        TRAIN_IMAGES_DIR, TRAIN_LABELS_DIR,
        VAL_IMAGES_DIR, VAL_LABELS_DIR,
        MODEL_DIR, DETECTION_OUTPUT_DIR,
        PATH_OUTPUT_DIR
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def get_dataset_yaml_content():
    """Generate dataset.yaml content for YOLO training"""
    return f"""
path: {DATASET_DIR}
train: train/images
val: val/images

nc: {NUM_CLASSES}
names: ['{CLASS_NAME}']
"""


if __name__ == "__main__":
    create_directories()
    print("Configuration loaded successfully!")
    print(f"Base directory: {BASE_DIR}")
    print(f"STL file path: {STL_FILE}")
    print(f"Dataset directory: {DATASET_DIR}")
