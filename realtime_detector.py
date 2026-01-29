"""
Real-time Object Detection Module
Uses RealSense D435i camera with trained YOLO model.

Industrial-Ready Version with:
- Graceful shutdown handling
- Camera auto-reconnect
- Centralized logging
"""

import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
import time
import signal
import sys

try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False

import config
from logger import get_logger
from tool_path_planner import ToolPathPlanner


class RealSenseCamera:
    """
    RealSense D435i camera interface.
    """
    
    def __init__(self, width: int = None, height: int = None, fps: int = None):
        """
        Initialize RealSense camera.
        
        Args:
            width: Frame width
            height: Frame height
            fps: Frames per second
        """
        self.width = width or config.REALSENSE_WIDTH
        self.height = height or config.REALSENSE_HEIGHT
        self.fps = fps or config.REALSENSE_FPS
        
        self.pipeline = None
        self.align = None
        self.depth_scale = None
        self.is_connected = False
        
    def connect(self) -> bool:
        """
        Connect to RealSense camera.
        
        Returns:
            True if connection successful
        """
        if not REALSENSE_AVAILABLE:
            print("RealSense SDK not available")
            return False
        
        try:
            # Create pipeline
            self.pipeline = rs.pipeline()
            rs_config = rs.config()
            
            # Configure streams
            rs_config.enable_stream(
                rs.stream.color, 
                self.width, self.height, 
                rs.format.bgr8, self.fps
            )
            
            if config.ENABLE_DEPTH:
                rs_config.enable_stream(
                    rs.stream.depth,
                    self.width, self.height,
                    rs.format.z16, self.fps
                )
            
            # Start pipeline
            profile = self.pipeline.start(rs_config)
            
            # Get depth scale
            if config.ENABLE_DEPTH:
                depth_sensor = profile.get_device().first_depth_sensor()
                self.depth_scale = depth_sensor.get_depth_scale()
                
                # Create align object
                self.align = rs.align(rs.stream.color)
            
            self.is_connected = True
            print(f"RealSense camera connected: {self.width}x{self.height}@{self.fps}fps")
            
            return True
            
        except Exception as e:
            print(f"Failed to connect RealSense camera: {e}")
            return False
    
    def read(self):
        """
        Read frame from camera.
        
        Returns:
            Tuple of (color_frame, depth_frame) or (color_frame, None)
        """
        if not self.is_connected:
            return None, None
        
        try:
            # Wait for frames with timeout
            frames = self.pipeline.wait_for_frames(timeout_ms=5000)
            
            # Align depth to color
            if self.align:
                frames = self.align.process(frames)
            
            # Get color frame
            color_frame = frames.get_color_frame()
            if not color_frame:
                return None, None
            
            color_image = np.asanyarray(color_frame.get_data())
            
            # Get depth frame
            depth_image = None
            if config.ENABLE_DEPTH:
                depth_frame = frames.get_depth_frame()
                if depth_frame:
                    depth_image = np.asanyarray(depth_frame.get_data())
            
            return color_image, depth_image
            
        except Exception as e:
            print(f"Error reading frame: {e}")
            return None, None
    
    def get_distance(self, depth_image: np.ndarray, x: int, y: int, 
                     region_size: int = 5) -> float:
        """
        Get distance at a point in the depth image.
        
        Args:
            depth_image: Depth image array
            x: X coordinate
            y: Y coordinate
            region_size: Size of region to average
            
        Returns:
            Distance in meters
        """
        if depth_image is None:
            return 0.0
        
        h, w = depth_image.shape
        
        # Calculate region bounds
        x1 = max(0, x - region_size // 2)
        x2 = min(w, x + region_size // 2 + 1)
        y1 = max(0, y - region_size // 2)
        y2 = min(h, y + region_size // 2 + 1)
        
        # Get region and filter zeros
        region = depth_image[y1:y2, x1:x2]
        valid_depths = region[region > 0]
        
        if len(valid_depths) == 0:
            return 0.0
        
        # Calculate median distance
        depth_value = np.median(valid_depths)
        distance = depth_value * self.depth_scale
        
        return distance
    
    def get_3d_coordinates(self, depth_image: np.ndarray, pixel_x: int, pixel_y: int,
                           region_size: int = 5):
        """
        Get 3D world coordinates (X, Y, Z) from pixel position.
        
        Args:
            depth_image: Depth image array
            pixel_x: X coordinate in image
            pixel_y: Y coordinate in image
            region_size: Size of region to average for depth
            
        Returns:
            Tuple of (X, Y, Z) in meters, or (0, 0, 0) if invalid
        """
        if depth_image is None or not self.is_connected:
            return 0.0, 0.0, 0.0
        
        # Get depth value
        z = self.get_distance(depth_image, pixel_x, pixel_y, region_size)
        
        if z <= 0:
            return 0.0, 0.0, 0.0
        
        # RealSense D435i camera intrinsics (approximate for 640x480)
        # These are typical values - for precise values, get from camera profile
        fx = 615.0  # Focal length x
        fy = 615.0  # Focal length y
        cx = 320.0  # Principal point x (image center)
        cy = 240.0  # Principal point y (image center)
        
        # Try to get actual intrinsics from camera
        try:
            profile = self.pipeline.get_active_profile()
            depth_profile = profile.get_stream(rs.stream.depth)
            intrinsics = depth_profile.as_video_stream_profile().get_intrinsics()
            fx = intrinsics.fx
            fy = intrinsics.fy
            cx = intrinsics.ppx
            cy = intrinsics.ppy
        except:
            pass  # Use default values
        
        # Convert pixel coordinates to 3D world coordinates
        # X = (pixel_x - cx) * Z / fx
        # Y = (pixel_y - cy) * Z / fy
        x = (pixel_x - cx) * z / fx
        y = (pixel_y - cy) * z / fy
        
        return x, y, z
    
    def get_orientation(self, depth_image: np.ndarray, bbox: tuple,
                        min_depth: float = 0.1, max_depth: float = 3.0):
        """
        Calculate object orientation using PCA on depth point cloud.
        
        Args:
            depth_image: Depth image array
            bbox: Bounding box (x1, y1, x2, y2)
            min_depth: Minimum valid depth in meters
            max_depth: Maximum valid depth in meters
            
        Returns:
            Tuple of (roll, pitch, yaw) in degrees, or (0, 0, 0) if invalid
        """
        if depth_image is None or not self.is_connected:
            return 0.0, 0.0, 0.0
        
        x1, y1, x2, y2 = map(int, bbox)
        h, w = depth_image.shape[:2]
        
        # Clamp bbox to image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return 0.0, 0.0, 0.0
        
        # Get camera intrinsics
        fx, fy, cx, cy = 615.0, 615.0, 320.0, 240.0
        try:
            profile = self.pipeline.get_active_profile()
            depth_profile = profile.get_stream(rs.stream.depth)
            intrinsics = depth_profile.as_video_stream_profile().get_intrinsics()
            fx, fy = intrinsics.fx, intrinsics.fy
            cx, cy = intrinsics.ppx, intrinsics.ppy
        except:
            pass
        
        # Extract depth region
        depth_roi = depth_image[y1:y2, x1:x2]
        
        # Convert to meters
        depth_meters = depth_roi.astype(np.float32) * self.depth_scale
        
        # Create point cloud from valid depth pixels
        points = []
        step = 2  # Sample every 2nd pixel for speed
        
        for py in range(0, depth_roi.shape[0], step):
            for px in range(0, depth_roi.shape[1], step):
                z = depth_meters[py, px]
                if min_depth < z < max_depth:
                    # Convert to 3D
                    img_x = x1 + px
                    img_y = y1 + py
                    x_3d = (img_x - cx) * z / fx
                    y_3d = (img_y - cy) * z / fy
                    points.append([x_3d, y_3d, z])
        
        if len(points) < 10:
            return 0.0, 0.0, 0.0
        
        points = np.array(points)
        
        # Center the points
        centroid = np.mean(points, axis=0)
        centered = points - centroid
        
        # PCA to find principal axes
        try:
            cov_matrix = np.cov(centered.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            
            # Sort by eigenvalue (largest = main axis)
            idx = np.argsort(eigenvalues)[::-1]
            eigenvectors = eigenvectors[:, idx]
            
            # Extract orientation angles from eigenvectors
            # Main axis is the first eigenvector
            main_axis = eigenvectors[:, 0]
            secondary_axis = eigenvectors[:, 1]
            
            # Calculate Euler angles
            # Yaw: rotation around Y axis (horizontal rotation)
            yaw = np.degrees(np.arctan2(main_axis[0], main_axis[2]))
            
            # Pitch: rotation around X axis (up/down tilt)
            pitch = np.degrees(np.arctan2(-main_axis[1], 
                              np.sqrt(main_axis[0]**2 + main_axis[2]**2)))
            
            # Roll: rotation around Z axis
            roll = np.degrees(np.arctan2(secondary_axis[1], secondary_axis[0]))
            
            return roll, pitch, yaw
            
        except Exception:
            return 0.0, 0.0, 0.0
    
    def disconnect(self):
        """Disconnect camera"""
        if self.pipeline:
            self.pipeline.stop()
            self.is_connected = False
            print("RealSense camera disconnected")


class OpenCVCamera:
    """
    Fallback OpenCV camera interface.
    """
    
    def __init__(self, camera_id: int = 0, width: int = None, height: int = None):
        """
        Initialize OpenCV camera.
        
        Args:
            camera_id: Camera device ID
            width: Frame width
            height: Frame height
        """
        self.camera_id = camera_id
        self.width = width or config.REALSENSE_WIDTH
        self.height = height or config.REALSENSE_HEIGHT
        self.cap = None
        self.is_connected = False
        
    def connect(self) -> bool:
        """Connect to camera"""
        self.cap = cv2.VideoCapture(self.camera_id)
        
        if not self.cap.isOpened():
            print(f"Failed to open camera {self.camera_id}")
            return False
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        self.is_connected = True
        print(f"OpenCV camera connected: {self.camera_id}")
        
        return True
    
    def read(self):
        """Read frame"""
        if not self.is_connected:
            return None, None
        
        ret, frame = self.cap.read()
        if not ret:
            return None, None
        
        return frame, None  # No depth with regular camera
    
    def get_distance(self, depth_image, x, y, region_size=5):
        """Not available for regular camera"""
        return 0.0
    
    def disconnect(self):
        """Disconnect camera"""
        if self.cap:
            self.cap.release()
            self.is_connected = False


class RealtimeDetector:
    """
    Real-time object detection using trained model.
    """
    
    def __init__(self, model_path: str = None, use_realsense: bool = True):
        """
        Initialize detector.
        
        Args:
            model_path: Path to trained YOLO model
            use_realsense: Try to use RealSense camera
        """
        self.model_path = model_path or str(config.BEST_MODEL_PATH)
        self.use_realsense = use_realsense and REALSENSE_AVAILABLE
        
        self.model = None
        self.camera = None
        self.is_running = False
        
        # Stats
        self.frame_count = 0
        self.fps = 0
        self.last_fps_time = time.time()
        
        # Store last detection for tool path generation
        self.last_detection = None
        self.path_planner = None
        self.current_strategy = None
        
    def load_model(self) -> bool:
        """
        Load YOLO model.
        
        Returns:
            True if model loaded successfully
        """
        if not Path(self.model_path).exists():
            print(f"Model not found: {self.model_path}")
            return False
        
        print(f"Loading model: {self.model_path}")
        self.model = YOLO(self.model_path)
        print("Model loaded successfully")
        
        return True
    
    def connect_camera(self, camera_id: int = None) -> bool:
        """
        Connect to camera.
        
        Args:
            camera_id: Camera device ID
            
        Returns:
            True if connected successfully
        """
        camera_id = camera_id if camera_id is not None else config.CAMERA_ID
        
        if self.use_realsense:
            self.camera = RealSenseCamera()
            if self.camera.connect():
                return True
            print("Falling back to OpenCV camera")
        
        self.camera = OpenCVCamera(camera_id)
        return self.camera.connect()
    
    def _update_fps(self):
        """Update FPS calculation"""
        self.frame_count += 1
        current_time = time.time()
        elapsed = current_time - self.last_fps_time
        
        if elapsed >= 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.last_fps_time = current_time
    
    def _draw_detections(self, frame: np.ndarray, results, 
                         depth_image: np.ndarray = None,
                         lock_best: bool = True) -> np.ndarray:
        """
        Draw detection results on frame.
        
        Args:
            frame: Input frame
            results: YOLO detection results
            depth_image: Depth image (optional)
            lock_best: If True, only show the best (highest confidence) detection
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        if results and len(results) > 0:
            boxes = results[0].boxes
            
            if len(boxes) == 0:
                return annotated
            
            # If lock_best is True, only show the highest confidence detection
            if lock_best and len(boxes) > 0:
                # Find the best detection (highest confidence)
                confidences = [float(box.conf[0]) for box in boxes]
                best_idx = confidences.index(max(confidences))
                boxes_to_draw = [boxes[best_idx]]
            else:
                boxes_to_draw = boxes
            
            for box in boxes_to_draw:
                # Get box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                
                # Get class name
                class_name = results[0].names.get(class_id, f"class_{class_id}")
                
                # Calculate center for depth
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                # Get distance and 3D coordinates if depth available
                distance = 0.0
                coord_x, coord_y, coord_z = 0.0, 0.0, 0.0
                if depth_image is not None and config.SHOW_DEPTH:
                    distance = self.camera.get_distance(depth_image, center_x, center_y)
                    # Get 3D coordinates if camera supports it
                    if hasattr(self.camera, 'get_3d_coordinates'):
                        coord_x, coord_y, coord_z = self.camera.get_3d_coordinates(
                            depth_image, center_x, center_y
                        )
                
                # Check for proximity warning
                is_too_close = distance > 0 and distance <= config.PROXIMITY_THRESHOLD
                
                # Choose colors based on proximity
                if is_too_close:
                    color = (0, 0, 255)  # Red for danger
                    bracket_color = (0, 0, 255)  # Red
                    label_bg_color = (0, 0, 255)  # Red background
                else:
                    color = (0, 255, 0)  # Green
                    bracket_color = (0, 255, 255)  # Yellow
                    label_bg_color = (0, 255, 255)  # Yellow background
                
                # Draw locked bounding box with thicker lines
                thickness = 4 if is_too_close else (3 if lock_best else 2)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
                
                # Draw corner brackets for "locked" effect
                if lock_best:
                    bracket_len = 20
                    # Top-left
                    cv2.line(annotated, (x1, y1), (x1 + bracket_len, y1), bracket_color, 3)
                    cv2.line(annotated, (x1, y1), (x1, y1 + bracket_len), bracket_color, 3)
                    # Top-right
                    cv2.line(annotated, (x2, y1), (x2 - bracket_len, y1), bracket_color, 3)
                    cv2.line(annotated, (x2, y1), (x2, y1 + bracket_len), bracket_color, 3)
                    # Bottom-left
                    cv2.line(annotated, (x1, y2), (x1 + bracket_len, y2), bracket_color, 3)
                    cv2.line(annotated, (x1, y2), (x1, y2 - bracket_len), bracket_color, 3)
                    # Bottom-right
                    cv2.line(annotated, (x2, y2), (x2 - bracket_len, y2), bracket_color, 3)
                    cv2.line(annotated, (x2, y2), (x2, y2 - bracket_len), bracket_color, 3)
                
                # Prepare label
                if is_too_close:
                    label = f"WARNING! {distance*1000:.0f}mm"
                else:
                    label = f"LOCKED: {confidence:.2f}"
                    if distance > 0:
                        label += f" | {distance:.2f}m"
                
                # Draw label background
                (label_w, label_h), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                )
                cv2.rectangle(
                    annotated, 
                    (x1, y1 - label_h - 10), 
                    (x1 + label_w + 10, y1),
                    label_bg_color, -1
                )
                
                # Draw label text
                text_color = (255, 255, 255) if is_too_close else (0, 0, 0)
                cv2.putText(
                    annotated, label,
                    (x1 + 5, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2
                )
                
                # Draw 3D coordinates below the bounding box
                if coord_z > 0:
                    # Convert to millimeters for display
                    coord_label = f"X:{coord_x*1000:.0f} Y:{coord_y*1000:.0f} Z:{coord_z*1000:.0f}mm"
                    (coord_w, coord_h), _ = cv2.getTextSize(
                        coord_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                    )
                    # Background for coordinates
                    cv2.rectangle(
                        annotated,
                        (x1, y2),
                        (x1 + coord_w + 10, y2 + coord_h + 10),
                        (50, 50, 50), -1  # Dark gray background
                    )
                    # Coordinates text
                    cv2.putText(
                        annotated, coord_label,
                        (x1 + 5, y2 + coord_h + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
                    )
                
                # Calculate and display orientation (PCA-based)
                if depth_image is not None and hasattr(self.camera, 'get_orientation'):
                    roll, pitch, yaw = self.camera.get_orientation(
                        depth_image, (x1, y1, x2, y2)
                    )
                    
                    if abs(roll) > 0.1 or abs(pitch) > 0.1 or abs(yaw) > 0.1:
                        # Draw orientation label
                        orient_y = y2 + (coord_h + 15 if coord_z > 0 else 5)
                        orient_label = f"R:{roll:.0f} P:{pitch:.0f} Y:{yaw:.0f}"
                        
                        (orient_w, orient_h), _ = cv2.getTextSize(
                            orient_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                        )
                        
                        # Background for orientation
                        cv2.rectangle(
                            annotated,
                            (x1, orient_y),
                            (x1 + orient_w + 10, orient_y + orient_h + 10),
                            (80, 80, 80), -1
                        )
                        
                        # Orientation text
                        cv2.putText(
                            annotated, orient_label,
                            (x1 + 5, orient_y + orient_h + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 128, 0), 2
                        )
                        
                        # Draw visual orientation axes at center
                        axis_len = 30
                        
                        # Calculate axis directions from orientation angles
                        yaw_rad = np.radians(yaw)
                        pitch_rad = np.radians(pitch)
                        
                        # X axis (red) - affected by yaw
                        ax_x = int(center_x + axis_len * np.cos(yaw_rad))
                        ax_y = int(center_y - axis_len * np.sin(pitch_rad) * 0.5)
                        cv2.arrowedLine(annotated, (center_x, center_y), (ax_x, center_y), 
                                       (0, 0, 255), 2, tipLength=0.3)
                        
                        # Y axis (green) - affected by pitch
                        ay_x = center_x
                        ay_y = int(center_y - axis_len * np.cos(pitch_rad))
                        cv2.arrowedLine(annotated, (center_x, center_y), (center_x, ay_y),
                                       (0, 255, 0), 2, tipLength=0.3)
                        
                        # Z axis (blue) - depth direction
                        az_x = int(center_x + axis_len * np.sin(yaw_rad) * 0.7)
                        az_y = int(center_y + axis_len * 0.7)
                        cv2.arrowedLine(annotated, (center_x, center_y), (az_x, az_y),
                                       (255, 0, 0), 2, tipLength=0.3)
                
                # Draw center crosshair
                crosshair_color = (0, 0, 255) if is_too_close else (0, 0, 255)
                crosshair_size = 15
                cv2.line(annotated, (center_x - crosshair_size, center_y), 
                        (center_x + crosshair_size, center_y), crosshair_color, 2)
                cv2.line(annotated, (center_x, center_y - crosshair_size), 
                        (center_x, center_y + crosshair_size), crosshair_color, 2)
                cv2.circle(annotated, (center_x, center_y), 5, crosshair_color, -1)
                
                # Draw big warning overlay if too close
                if is_too_close:
                    h, w = annotated.shape[:2]
                    # Flash warning text at top
                    warning_text = "!! TOO CLOSE !!"
                    (tw, th), _ = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
                    text_x = (w - tw) // 2
                    # Red background bar
                    cv2.rectangle(annotated, (0, 50), (w, 100), (0, 0, 200), -1)
                    cv2.putText(annotated, warning_text, (text_x, 90),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        
        return annotated
    
    def _draw_info(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw info overlay on frame.
        
        Args:
            frame: Input frame
            
        Returns:
            Frame with info overlay
        """
        annotated = frame.copy()
        h, w = frame.shape[:2]
        
        # FPS
        if config.SHOW_FPS:
            fps_text = f"FPS: {self.fps:.1f}"
            cv2.putText(
                annotated, fps_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
        
        # Confidence level (top right)
        conf_text = f"Conf: {getattr(self, 'confidence', 0.5):.2f}"
        cv2.putText(
            annotated, conf_text,
            (w - 150, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
        )
        
        # Strategy indicator if path planner exists
        if hasattr(self, 'current_strategy') and self.current_strategy:
            strat_text = f"Strategy: {self.current_strategy}"
            cv2.putText(
                annotated, strat_text,
                (w - 200, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2
            )
        
        # Controls hint (two lines)
        controls1 = "p=path | 1-6=strategy | v=viz | r=reload"
        controls2 = "+/-=conf | s=save | q=quit"
        cv2.putText(
            annotated, controls1,
            (10, h - 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1
        )
        cv2.putText(
            annotated, controls2,
            (10, h - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1
        )
        
        return annotated
    
    def _draw_path_overlay(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw tool path overlay on frame.
        
        Args:
            frame: Input frame
            
        Returns:
            Frame with path overlay
        """
        if not hasattr(self, 'path_planner') or self.path_planner is None:
            return frame
        
        if not self.path_planner.path:
            return frame
        
        if not getattr(config, 'SHOW_PATH_OVERLAY', True):
            return frame
        
        annotated = frame.copy()
        
        # Camera intrinsics for projection
        fx, fy, cx, cy = 615.0, 615.0, 320.0, 240.0
        
        # Convert 3D waypoints to 2D pixels
        pixels = []
        for point in self.path_planner.path:
            if point.z > 0:
                px = int(point.x * fx / point.z + cx)
                py = int(point.y * fy / point.z + cy)
                pixels.append((px, py, point.point_type, point.id))
        
        if not pixels:
            return annotated
        
        # Define colors
        colors = {
            'waypoint': getattr(config, 'PATH_WAYPOINT_COLOR', (0, 255, 0)),
            'approach': getattr(config, 'PATH_APPROACH_COLOR', (0, 255, 255)),
            'grasp': getattr(config, 'PATH_GRASP_COLOR', (0, 0, 255)),
            'retreat': getattr(config, 'PATH_RETREAT_COLOR', (255, 255, 0))
        }
        line_color = getattr(config, 'PATH_LINE_COLOR', (255, 128, 0))
        
        # Draw lines connecting waypoints
        for i in range(len(pixels) - 1):
            pt1 = (pixels[i][0], pixels[i][1])
            pt2 = (pixels[i+1][0], pixels[i+1][1])
            cv2.line(annotated, pt1, pt2, line_color, 2)
        
        # Draw waypoints
        for px, py, ptype, pid in pixels:
            color = colors.get(ptype, (0, 255, 0))
            cv2.circle(annotated, (px, py), 6, color, -1)
            cv2.circle(annotated, (px, py), 8, (255, 255, 255), 1)
            
            # Draw waypoint number
            cv2.putText(
                annotated, str(pid),
                (px + 10, py - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1
            )
        
        # Draw path info box
        h, w = annotated.shape[:2]
        info_text = f"Path: {len(pixels)} pts"
        cv2.rectangle(annotated, (w - 120, h - 60), (w - 5, h - 35), (50, 50, 50), -1)
        cv2.putText(
            annotated, info_text,
            (w - 115, h - 42),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
        )
        
        return annotated

    
    def _save_frame(self, frame: np.ndarray):
        """Save current frame"""
        config.create_directories()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = config.DETECTION_OUTPUT_DIR / f"detection_{timestamp}.jpg"
        cv2.imwrite(str(filename), frame)
        print(f"Frame saved: {filename}")
    
    def run(self, camera_id: int = None, confidence: float = None):
        """
        Run real-time detection with industrial-grade stability.
        
        Args:
            camera_id: Camera device ID
            confidence: Detection confidence threshold
        """
        self.confidence = confidence or config.CONFIDENCE_THRESHOLD
        self.logger = get_logger()
        self._shutdown_requested = False
        
        # Setup signal handlers for graceful shutdown
        def signal_handler(signum, frame):
            signal_name = signal.Signals(signum).name
            self.logger.warning(f"Received signal {signal_name}, initiating graceful shutdown...")
            self._shutdown_requested = True
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Load model
        if not self.load_model():
            self.logger.error("Failed to load model. Please train model first.")
            return
        
        # Connect camera with retry
        max_retries = 3
        for attempt in range(max_retries):
            if self.connect_camera(camera_id):
                break
            self.logger.warning(f"Camera connection attempt {attempt + 1}/{max_retries} failed")
            if attempt < max_retries - 1:
                time.sleep(2)
        else:
            self.logger.error("Failed to connect camera after all retries")
            return
        
        camera_type = "RealSense" if isinstance(self.camera, RealSenseCamera) else "OpenCV"
        self.logger.log_startup(self.model_path, camera_type)
        
        self.logger.info(f"Confidence threshold: {self.confidence}")
        self.logger.info("Controls: q=quit, s=save, +/-=adjust confidence")
        
        self.is_running = True
        consecutive_failures = 0
        max_consecutive_failures = 30  # ~1 second at 30fps
        
        try:
            while self.is_running and not self._shutdown_requested:
                # Read frame
                color_frame, depth_frame = self.camera.read()
                
                if color_frame is None:
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        self.logger.warning("Camera connection lost, attempting reconnect...")
                        self.camera.disconnect()
                        time.sleep(1)
                        if self.connect_camera(camera_id):
                            self.logger.info("Camera reconnected successfully")
                            consecutive_failures = 0
                        else:
                            self.logger.error("Failed to reconnect camera")
                            break
                    continue
                
                consecutive_failures = 0
                
                # Run detection with current confidence
                results = self.model(
                    color_frame,
                    conf=self.confidence,
                    iou=config.IOU_THRESHOLD,
                    max_det=config.MAX_DETECTIONS,
                    verbose=False
                )
                
                # Draw results
                annotated = self._draw_detections(color_frame, results, depth_frame)
                annotated = self._draw_info(annotated)
                annotated = self._draw_path_overlay(annotated)
                
                # Update FPS
                self._update_fps()
                
                # Show frame
                cv2.imshow("STL Object Detection", annotated)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.logger.info("Quit requested by user")
                    break
                elif key == ord('s'):
                    self._save_frame(annotated)
                    self.logger.info("Frame saved")
                elif key == ord('+') or key == ord('='):
                    self.confidence = min(0.95, self.confidence + 0.05)
                    self.logger.info(f"Confidence increased to {self.confidence:.2f}")
                elif key == ord('-') or key == ord('_'):
                    self.confidence = max(0.05, self.confidence - 0.05)
                    self.logger.info(f"Confidence decreased to {self.confidence:.2f}")
                elif key == ord('p'):
                    # Generate tool path from current detection
                    if results and len(results) > 0 and len(results[0].boxes) > 0:
                        output_file = self.generate_tool_path(
                            results, color_frame, depth_frame
                        )
                        if output_file:
                            self.logger.info(f"Tool path saved: {output_file}")
                    else:
                        self.logger.warning("No detection available for path generation")
                elif key == ord('v'):
                    # Toggle path visualization
                    config.SHOW_PATH_OVERLAY = not getattr(config, 'SHOW_PATH_OVERLAY', True)
                    status = "ON" if config.SHOW_PATH_OVERLAY else "OFF"
                    print(f"Path visualization: {status}")
                elif key in [ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6')]:
                    # Quick strategy selection
                    strategies = ['contour', 'approach', 'grid', 'surface', 'spiral', 'zigzag']
                    idx = key - ord('1')
                    self.current_strategy = strategies[idx]
                    config.DEFAULT_PATH_STRATEGY = strategies[idx]
                    print(f"Strategy: {strategies[idx]}")
                elif key == ord('r'):
                    # Reload settings from file
                    config.reload_settings()
                    print("Settings reloaded from settings.yaml")
                    
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            self.is_running = False
            self.camera.disconnect()
            cv2.destroyAllWindows()
            self.logger.log_shutdown("Normal" if not self._shutdown_requested else "Signal received")
    
    def detect_image(self, image_path: str, 
                     confidence: float = None,
                     save_output: bool = True) -> np.ndarray:
        """
        Detect objects in a single image.
        
        Args:
            image_path: Path to input image
            confidence: Detection confidence threshold
            save_output: Save annotated image
            
        Returns:
            Annotated image
        """
        confidence = confidence or config.CONFIDENCE_THRESHOLD
        
        # Load model
        if self.model is None:
            if not self.load_model():
                print("Failed to load model")
                return None
        
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to read image: {image_path}")
            return None
        
        # Run detection
        results = self.model(
            image,
            conf=confidence,
            iou=config.IOU_THRESHOLD,
            max_det=config.MAX_DETECTIONS,
            verbose=False
        )
        
        # Draw results
        annotated = self._draw_detections(image, results)
        
        # Save output
        if save_output:
            config.create_directories()
            output_path = config.DETECTION_OUTPUT_DIR / f"detected_{Path(image_path).name}"
            cv2.imwrite(str(output_path), annotated)
            print(f"Detection saved: {output_path}")
        
        return annotated
    
    def generate_tool_path(self, results, color_frame: np.ndarray, 
                           depth_frame: np.ndarray = None,
                           strategy: str = None,
                           output_path: str = None) -> str:
        """
        Generate tool path from current detection results.
        
        Args:
            results: YOLO detection results
            color_frame: Current color frame
            depth_frame: Current depth frame (optional)
            strategy: Path strategy ('contour', 'approach', 'grid', 'surface', 'spiral', 'zigzag')
            output_path: Custom output path for YAML file
            
        Returns:
            Path to generated YAML file, or None if failed
        """
        if not results or len(results) == 0 or len(results[0].boxes) == 0:
            return None
        
        strategy = strategy or config.DEFAULT_PATH_STRATEGY
        self.current_strategy = strategy
        
        # Get the best detection
        boxes = results[0].boxes
        confidences = [float(box.conf[0]) for box in boxes]
        best_idx = confidences.index(max(confidences))
        box = boxes[best_idx]
        
        # Extract detection data
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = float(box.conf[0])
        class_id = int(box.cls[0])
        class_name = results[0].names.get(class_id, f"class_{class_id}")
        
        # Calculate center
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        # Get 3D coordinates if depth available
        if depth_frame is not None and hasattr(self.camera, 'get_3d_coordinates'):
            coord_x, coord_y, coord_z = self.camera.get_3d_coordinates(
                depth_frame, center_x, center_y
            )
        else:
            # Fallback to approximate depth
            coord_x, coord_y, coord_z = 0.0, 0.0, 0.5
        
        # Get orientation if available
        if depth_frame is not None and hasattr(self.camera, 'get_orientation'):
            roll, pitch, yaw = self.camera.get_orientation(
                depth_frame, (x1, y1, x2, y2)
            )
        else:
            roll, pitch, yaw = 0.0, 0.0, 0.0
        
        # Build object data for planner
        object_data = {
            'center_3d': (coord_x, coord_y, coord_z),
            'orientation': (roll, pitch, yaw),
            'bounding_box': (x1, y1, x2, y2),
            'class_name': class_name,
            'confidence': confidence,
            'depth_image': depth_frame
        }
        
        # Store last detection
        self.last_detection = object_data
        
        # Create path planner
        self.path_planner = ToolPathPlanner(object_data)
        
        # Apply config settings
        self.path_planner.surface_offset = config.SURFACE_OFFSET
        self.path_planner.approach_distance = config.APPROACH_DISTANCE
        self.path_planner.frame_id = config.PATH_FRAME_ID
        
        # Generate path
        path = self.path_planner.generate_path(
            strategy=strategy,
            num_points=config.NUM_PATH_POINTS
        )
        
        # Generate output path if not specified
        if output_path is None:
            config.create_directories()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = str(config.PATH_OUTPUT_DIR / f"tool_path_{class_name}_{timestamp}.yaml")
        
        # Export to YAML
        try:
            output_file = self.path_planner.export_to_yaml(output_path)
            
            # Also overlay path on displayed frame if possible
            print(f"\n{'='*50}")
            print(f"TOOL PATH GENERATED")
            print(f"{'='*50}")
            print(f"Object: {class_name} (confidence: {confidence:.2f})")
            print(f"Position: X={coord_x*1000:.1f}mm, Y={coord_y*1000:.1f}mm, Z={coord_z*1000:.1f}mm")
            print(f"Orientation: Roll={roll:.1f}°, Pitch={pitch:.1f}°, Yaw={yaw:.1f}°")
            print(f"Strategy: {strategy}")
            print(f"Waypoints: {len(path)}")
            print(f"Output: {output_file}")
            print(f"{'='*50}\n")
            
            return output_file
        except Exception as e:
            print(f"Failed to export tool path: {e}")
            return None


def main():
    """Main function for real-time detection"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Real-time object detection")
    parser.add_argument("--model", type=str, default=None, help="Path to model")
    parser.add_argument("--camera", type=int, default=0, help="Camera ID")
    parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--image", type=str, default=None, help="Detect in single image")
    parser.add_argument("--no-realsense", action="store_true", help="Don't use RealSense")
    
    args = parser.parse_args()
    
    detector = RealtimeDetector(
        model_path=args.model,
        use_realsense=not args.no_realsense
    )
    
    if args.image:
        detector.detect_image(args.image, confidence=args.confidence)
    else:
        detector.run(camera_id=args.camera, confidence=args.confidence)


if __name__ == "__main__":
    main()
