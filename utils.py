"""
Utility Functions
Helper functions for image processing, visualization, and logging.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import logging
from datetime import datetime


def setup_logging(log_file: str = None, level: int = logging.INFO) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        log_file: Path to log file (optional)
        level: Logging level
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger("stl_detector")
    logger.setLevel(level)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger


# =====================================================
# IMAGE UTILITIES
# =====================================================

def resize_with_aspect_ratio(image: np.ndarray, 
                             width: int = None, 
                             height: int = None) -> np.ndarray:
    """
    Resize image while maintaining aspect ratio.
    
    Args:
        image: Input image
        width: Target width (optional)
        height: Target height (optional)
        
    Returns:
        Resized image
    """
    h, w = image.shape[:2]
    
    if width is None and height is None:
        return image
    
    if width is None:
        ratio = height / h
        width = int(w * ratio)
    elif height is None:
        ratio = width / w
        height = int(h * ratio)
    
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def apply_clahe(image: np.ndarray, 
                clip_limit: float = 2.0, 
                tile_size: int = 8) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
    
    Args:
        image: Input image (BGR)
        clip_limit: Contrast limit
        tile_size: Tile grid size
        
    Returns:
        Enhanced image
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    l = clahe.apply(l)
    
    # Merge and convert back
    lab = cv2.merge([l, a, b])
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    return result


def add_salt_pepper_noise(image: np.ndarray, 
                          noise_ratio: float = 0.02) -> np.ndarray:
    """
    Add salt and pepper noise to image.
    
    Args:
        image: Input image
        noise_ratio: Ratio of noisy pixels
        
    Returns:
        Noisy image
    """
    noisy = image.copy()
    h, w = image.shape[:2]
    
    num_salt = int(noise_ratio * h * w)
    num_pepper = int(noise_ratio * h * w)
    
    # Salt
    coords = [np.random.randint(0, i, num_salt) for i in image.shape[:2]]
    noisy[coords[0], coords[1]] = 255
    
    # Pepper
    coords = [np.random.randint(0, i, num_pepper) for i in image.shape[:2]]
    noisy[coords[0], coords[1]] = 0
    
    return noisy


def random_crop(image: np.ndarray, 
                crop_size: Tuple[int, int]) -> np.ndarray:
    """
    Random crop from image.
    
    Args:
        image: Input image
        crop_size: (width, height) of crop
        
    Returns:
        Cropped image
    """
    h, w = image.shape[:2]
    crop_w, crop_h = crop_size
    
    if crop_w >= w or crop_h >= h:
        return image
    
    x = np.random.randint(0, w - crop_w)
    y = np.random.randint(0, h - crop_h)
    
    return image[y:y+crop_h, x:x+crop_w]


# =====================================================
# BOUNDING BOX UTILITIES
# =====================================================

def yolo_to_xyxy(bbox: Tuple[float, float, float, float], 
                 img_width: int, img_height: int) -> Tuple[int, int, int, int]:
    """
    Convert YOLO format bbox to xyxy format.
    
    Args:
        bbox: (x_center, y_center, width, height) normalized
        img_width: Image width
        img_height: Image height
        
    Returns:
        (x1, y1, x2, y2) in pixels
    """
    x_center, y_center, w, h = bbox
    
    x1 = int((x_center - w/2) * img_width)
    y1 = int((y_center - h/2) * img_height)
    x2 = int((x_center + w/2) * img_width)
    y2 = int((y_center + h/2) * img_height)
    
    return x1, y1, x2, y2


def xyxy_to_yolo(bbox: Tuple[int, int, int, int], 
                 img_width: int, img_height: int) -> Tuple[float, float, float, float]:
    """
    Convert xyxy format bbox to YOLO format.
    
    Args:
        bbox: (x1, y1, x2, y2) in pixels
        img_width: Image width
        img_height: Image height
        
    Returns:
        (x_center, y_center, width, height) normalized
    """
    x1, y1, x2, y2 = bbox
    
    x_center = (x1 + x2) / 2 / img_width
    y_center = (y1 + y2) / 2 / img_height
    w = (x2 - x1) / img_width
    h = (y2 - y1) / img_height
    
    return x_center, y_center, w, h


def calculate_iou(box1: Tuple[int, int, int, int], 
                  box2: Tuple[int, int, int, int]) -> float:
    """
    Calculate Intersection over Union (IoU) between two boxes.
    
    Args:
        box1: (x1, y1, x2, y2)
        box2: (x1, y1, x2, y2)
        
    Returns:
        IoU value
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0


# =====================================================
# VISUALIZATION UTILITIES
# =====================================================

def draw_bbox(image: np.ndarray, 
              bbox: Tuple[int, int, int, int],
              label: str = None,
              color: Tuple[int, int, int] = (0, 255, 0),
              thickness: int = 2) -> np.ndarray:
    """
    Draw bounding box with label on image.
    
    Args:
        image: Input image
        bbox: (x1, y1, x2, y2)
        label: Label text
        color: Box color (BGR)
        thickness: Line thickness
        
    Returns:
        Image with drawn box
    """
    result = image.copy()
    x1, y1, x2, y2 = bbox
    
    # Draw box
    cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)
    
    # Draw label
    if label:
        (label_w, label_h), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(result, (x1, y1 - label_h - 5), (x1 + label_w + 5, y1), color, -1)
        cv2.putText(result, label, (x1 + 2, y1 - 3),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return result


def create_image_grid(images: List[np.ndarray], 
                      cols: int = 4,
                      cell_size: Tuple[int, int] = (200, 200)) -> np.ndarray:
    """
    Create a grid of images.
    
    Args:
        images: List of images
        cols: Number of columns
        cell_size: (width, height) of each cell
        
    Returns:
        Grid image
    """
    if not images:
        return np.zeros((cell_size[1], cell_size[0], 3), dtype=np.uint8)
    
    n = len(images)
    rows = (n + cols - 1) // cols
    
    # Resize all images
    resized = [cv2.resize(img, cell_size) for img in images]
    
    # Pad to fill grid
    while len(resized) < rows * cols:
        resized.append(np.zeros((cell_size[1], cell_size[0], 3), dtype=np.uint8))
    
    # Create grid
    grid_rows = []
    for i in range(rows):
        row_images = resized[i*cols:(i+1)*cols]
        grid_rows.append(np.hstack(row_images))
    
    return np.vstack(grid_rows)


def visualize_depth(depth_image: np.ndarray, 
                    min_depth: float = 0.3,
                    max_depth: float = 5.0) -> np.ndarray:
    """
    Convert depth image to colorized visualization.
    
    Args:
        depth_image: Depth image (in mm typically)
        min_depth: Minimum depth (meters)
        max_depth: Maximum depth (meters)
        
    Returns:
        Colorized depth image
    """
    # Normalize depth
    depth_normalized = np.clip(depth_image / 1000.0, min_depth, max_depth)
    depth_normalized = ((depth_normalized - min_depth) / (max_depth - min_depth) * 255).astype(np.uint8)
    
    # Apply colormap
    depth_colorized = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
    
    return depth_colorized


# =====================================================
# FILE UTILITIES
# =====================================================

def get_image_files(directory: str, 
                    extensions: List[str] = None) -> List[Path]:
    """
    Get all image files in directory.
    
    Args:
        directory: Directory path
        extensions: List of file extensions
        
    Returns:
        List of image file paths
    """
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    
    directory = Path(directory)
    files = []
    
    for ext in extensions:
        files.extend(directory.glob(f'*{ext}'))
        files.extend(directory.glob(f'*{ext.upper()}'))
    
    return sorted(files)


def ensure_dir(path: str) -> Path:
    """
    Ensure directory exists.
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_timestamp() -> str:
    """
    Get current timestamp string.
    
    Returns:
        Timestamp string
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# =====================================================
# VALIDATION UTILITIES
# =====================================================

def validate_label_file(label_path: str, num_classes: int = 1) -> bool:
    """
    Validate YOLO label file format.
    
    Args:
        label_path: Path to label file
        num_classes: Number of classes
        
    Returns:
        True if valid
    """
    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                return False
            
            class_id = int(parts[0])
            if class_id < 0 or class_id >= num_classes:
                return False
            
            values = [float(p) for p in parts[1:]]
            if not all(0 <= v <= 1 for v in values):
                return False
        
        return True
        
    except Exception:
        return False


if __name__ == "__main__":
    # Test utilities
    print("Testing utilities...")
    
    # Test image operations
    test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    resized = resize_with_aspect_ratio(test_img, width=320)
    print(f"Resized: {test_img.shape} -> {resized.shape}")
    
    enhanced = apply_clahe(test_img)
    print(f"CLAHE applied: {enhanced.shape}")
    
    noisy = add_salt_pepper_noise(test_img)
    print(f"Noise added: {noisy.shape}")
    
    # Test bbox operations
    yolo_bbox = (0.5, 0.5, 0.3, 0.4)
    xyxy_bbox = yolo_to_xyxy(yolo_bbox, 640, 480)
    print(f"YOLO to XYXY: {yolo_bbox} -> {xyxy_bbox}")
    
    back_to_yolo = xyxy_to_yolo(xyxy_bbox, 640, 480)
    print(f"XYXY to YOLO: {xyxy_bbox} -> {back_to_yolo}")
    
    print("All tests passed!")
