#!/usr/bin/env python3
"""
ROS 2 Detection Node for STL Object Detection

This node subscribes to camera images, runs YOLO detection,
and publishes detection results with 3D coordinates.

Topics Published:
    /detections (DetectionArray): All detected objects
    /best_detection (Detection): Highest confidence detection
    /detection_image (sensor_msgs/Image): Annotated image

Topics Subscribed:
    /camera/color/image_raw (sensor_msgs/Image): RGB image
    /camera/aligned_depth_to_color/image_raw (sensor_msgs/Image): Depth image

Author: Generated for Dobot Nova5 painting application
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Point, Vector3, TransformStamped
from std_msgs.msg import Header

from cv_bridge import CvBridge
import cv2
import numpy as np
from pathlib import Path
import sys
import os

# Add parent directory to path for importing existing modules
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # Go up to STL directory
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("WARNING: ultralytics not installed. Install with: pip install ultralytics")

try:
    import tf2_ros
    from tf2_ros import TransformBroadcaster
    TF2_AVAILABLE = True
except ImportError:
    TF2_AVAILABLE = False


class DetectionNode(Node):
    """
    ROS 2 Node for real-time object detection using YOLO.
    
    This node provides object detection capabilities suitable for
    robotic arm applications like painting with Dobot Nova5.
    """
    
    def __init__(self):
        super().__init__('stl_detector_node')
        
        # Declare parameters
        self.declare_parameter('model_path', '')
        self.declare_parameter('confidence_threshold', 0.65)
        self.declare_parameter('iou_threshold', 0.45)
        self.declare_parameter('publish_tf', True)
        self.declare_parameter('camera_frame', 'camera_color_optical_frame')
        self.declare_parameter('detection_frame_prefix', 'detected_object_')
        self.declare_parameter('show_all_detections', False)
        self.declare_parameter('max_detections', 10)
        # CRITICAL: Only detect these specific classes - ignore everything else
        self.declare_parameter('allowed_classes', ['bottle', 'mouse'])
        
        # Get parameters
        model_path = self.get_parameter('model_path').value
        self.confidence_threshold = self.get_parameter('confidence_threshold').value
        self.iou_threshold = self.get_parameter('iou_threshold').value
        self.publish_tf = self.get_parameter('publish_tf').value
        self.camera_frame = self.get_parameter('camera_frame').value
        self.detection_frame_prefix = self.get_parameter('detection_frame_prefix').value
        self.show_all_detections = self.get_parameter('show_all_detections').value
        self.max_detections = self.get_parameter('max_detections').value
        # Get allowed classes - ONLY these will be detected
        self.filter_classes = self.get_parameter('allowed_classes').value
        self.get_logger().info(f"CLASS FILTER ACTIVE: Only detecting {self.filter_classes}")
        
        # Find model if not specified
        if not model_path:
            # Try default locations - ONLY custom trained models
            default_paths = [
                PROJECT_ROOT / 'runs' / 'detect' / 'stl_object' / 'weights' / 'best.pt',
                PROJECT_ROOT / 'best.pt',
                Path.home() / 'stl_detector' / 'best.pt',
            ]
            for path in default_paths:
                if path.exists():
                    model_path = str(path)
                    break
        
        # CRITICAL: Validate model to prevent "detects all objects" issue
        if not model_path or not Path(model_path).exists():
            self.get_logger().error(f"Model not found! Searched: {default_paths}")
            self.get_logger().error("Please specify model_path parameter or train a model first.")
            self.get_logger().error("DO NOT use pretrained yolov8n.pt - it detects 80 COCO classes!")
            self.model = None
            self.allowed_classes = []
        else:
            self.get_logger().info(f"Loading YOLO model from: {model_path}")
            if YOLO_AVAILABLE:
                self.model = YOLO(model_path)
                
                # CRITICAL VALIDATION: Check number of classes
                # Pretrained COCO model has 80 classes, custom should have fewer
                num_classes = len(self.model.names)
                self.allowed_classes = list(self.model.names.values())
                
                self.get_logger().info(f"Model classes ({num_classes}): {self.allowed_classes}")
                
                # SAFETY CHECK: Warn if model has too many classes (likely pretrained)
                if num_classes > 10:
                    self.get_logger().warn("=" * 60)
                    self.get_logger().warn("WARNING: Model has more than 10 classes!")
                    self.get_logger().warn(f"Classes: {self.allowed_classes[:10]}...")
                    self.get_logger().warn("This might be a PRETRAINED model (COCO has 80 classes)")
                    self.get_logger().warn("If you see 'person', 'car', 'bottle' etc., use your TRAINED model!")
                    self.get_logger().warn("Trained model location: runs/detect/stl_object/weights/best.pt")
                    self.get_logger().warn("=" * 60)
                
                # HARD BLOCK: Refuse to run with obvious pretrained model
                coco_classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 
                               'bus', 'train', 'truck', 'boat', 'traffic light']
                is_coco = sum(1 for c in coco_classes if c in self.allowed_classes) >= 5
                
                if is_coco:
                    self.get_logger().error("=" * 60)
                    self.get_logger().error("BLOCKED: This appears to be a PRETRAINED COCO model!")
                    self.get_logger().error("This model will detect ALL 80 object types.")
                    self.get_logger().error("You MUST use your custom-trained model instead:")
                    self.get_logger().error(f"  {PROJECT_ROOT}/runs/detect/stl_object/weights/best.pt")
                    self.get_logger().error("=" * 60)
                    self.model = None  # Block the model
                else:
                    self.get_logger().info(f"✓ Model validated: {num_classes} custom classes")
                    self.get_logger().info("Model loaded successfully")
            else:
                self.model = None
                self.allowed_classes = []
                self.get_logger().error("YOLO not available - install ultralytics")
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # Camera intrinsics (will be updated from CameraInfo)
        self.fx = 615.0
        self.fy = 615.0
        self.cx = 320.0
        self.cy = 240.0
        self.depth_scale = 0.001  # Default for RealSense (1mm)
        self.camera_info_received = False
        
        # Image buffers
        self.color_image = None
        self.depth_image = None
        self.last_detection_time = self.get_clock().now()
        
        # QoS for sensor data
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Callback group for concurrent processing
        self.callback_group = ReentrantCallbackGroup()
        
        # Subscribers
        self.color_sub = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.color_callback,
            sensor_qos,
            callback_group=self.callback_group
        )
        
        self.depth_sub = self.create_subscription(
            Image,
            '/camera/aligned_depth_to_color/image_raw',
            self.depth_callback,
            sensor_qos,
            callback_group=self.callback_group
        )
        
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/camera/color/camera_info',
            self.camera_info_callback,
            sensor_qos,
            callback_group=self.callback_group
        )
        
        # Publishers - using dynamic imports for custom messages
        # Since custom messages may not be built yet, we use a flexible approach
        self.setup_publishers()
        
        # TF broadcaster
        if self.publish_tf and TF2_AVAILABLE:
            self.tf_broadcaster = TransformBroadcaster(self)
        else:
            self.tf_broadcaster = None
        
        # Detection timer (run at 30Hz)
        self.detection_timer = self.create_timer(
            1.0 / 30.0,  # 30 Hz
            self.detection_callback,
            callback_group=self.callback_group
        )
        
        self.get_logger().info("STL Detector Node initialized")
        self.get_logger().info(f"  Confidence threshold: {self.confidence_threshold}")
        self.get_logger().info(f"  Camera frame: {self.camera_frame}")
        self.get_logger().info(f"  Publish TF: {self.publish_tf}")
    
    def setup_publishers(self):
        """Setup publishers with fallback for when custom messages aren't built."""
        # Always available publishers
        self.image_pub = self.create_publisher(
            Image,
            '/detection_image',
            10
        )
        
        # Try to import custom messages
        try:
            from stl_detector_ros2.msg import Detection, DetectionArray
            self.Detection = Detection
            self.DetectionArray = DetectionArray
            self.custom_msgs_available = True
            
            self.detections_pub = self.create_publisher(
                DetectionArray,
                '/detections',
                10
            )
            
            self.best_detection_pub = self.create_publisher(
                Detection,
                '/best_detection',
                10
            )
            
            self.get_logger().info("Custom messages available - publishing to /detections")
            
        except ImportError:
            self.custom_msgs_available = False
            self.get_logger().warn(
                "Custom messages not built yet. Build package first:\n"
                "  cd ~/ros2_ws && colcon build --packages-select stl_detector_ros2\n"
                "Using JSON string topic as fallback."
            )
            # Fallback to string messages with JSON
            from std_msgs.msg import String
            self.String = String
            
            self.detections_pub = self.create_publisher(
                String,
                '/detections_json',
                10
            )
            
            self.best_detection_pub = self.create_publisher(
                String,
                '/best_detection_json',
                10
            )
    
    def color_callback(self, msg: Image):
        """Handle incoming color image."""
        try:
            self.color_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f"Error converting color image: {e}")
    
    def depth_callback(self, msg: Image):
        """Handle incoming depth image."""
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, 'passthrough')
        except Exception as e:
            self.get_logger().error(f"Error converting depth image: {e}")
    
    def camera_info_callback(self, msg: CameraInfo):
        """Handle camera info for intrinsics."""
        if not self.camera_info_received:
            self.fx = msg.k[0]  # K[0,0]
            self.fy = msg.k[4]  # K[1,1]
            self.cx = msg.k[2]  # K[0,2]
            self.cy = msg.k[5]  # K[1,2]
            self.camera_info_received = True
            self.get_logger().info(
                f"Camera intrinsics received: fx={self.fx:.1f}, fy={self.fy:.1f}, "
                f"cx={self.cx:.1f}, cy={self.cy:.1f}"
            )
    
    def get_3d_position(self, pixel_x: int, pixel_y: int, depth_image: np.ndarray) -> tuple:
        """
        Convert pixel coordinates to 3D position in camera frame.
        
        Args:
            pixel_x: X coordinate in image
            pixel_y: Y coordinate in image
            depth_image: Depth image array
            
        Returns:
            Tuple of (x, y, z) in meters
        """
        if depth_image is None:
            return 0.0, 0.0, 0.0
        
        h, w = depth_image.shape[:2]
        
        # Get depth value (average over small region for stability)
        region_size = 5
        x1 = max(0, pixel_x - region_size // 2)
        x2 = min(w, pixel_x + region_size // 2 + 1)
        y1 = max(0, pixel_y - region_size // 2)
        y2 = min(h, pixel_y + region_size // 2 + 1)
        
        region = depth_image[y1:y2, x1:x2]
        valid_depths = region[region > 0]
        
        if len(valid_depths) == 0:
            return 0.0, 0.0, 0.0
        
        # Use median for robustness
        depth_value = np.median(valid_depths)
        z = depth_value * self.depth_scale
        
        if z <= 0:
            return 0.0, 0.0, 0.0
        
        # Convert to 3D (camera optical frame: X-right, Y-down, Z-forward)
        x = (pixel_x - self.cx) * z / self.fx
        y = (pixel_y - self.cy) * z / self.fy
        
        return x, y, z
    
    def get_orientation(self, depth_image: np.ndarray, bbox: tuple) -> tuple:
        """
        Estimate object orientation using PCA on depth point cloud.
        
        Args:
            depth_image: Depth image array
            bbox: Bounding box (x1, y1, x2, y2)
            
        Returns:
            Tuple of (roll, pitch, yaw) in degrees
        """
        if depth_image is None:
            return 0.0, 0.0, 0.0
        
        x1, y1, x2, y2 = map(int, bbox)
        h, w = depth_image.shape[:2]
        
        # Clamp to image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return 0.0, 0.0, 0.0
        
        # Extract depth region
        depth_roi = depth_image[y1:y2, x1:x2]
        depth_meters = depth_roi.astype(np.float32) * self.depth_scale
        
        # Build point cloud
        points = []
        step = 2
        
        for py in range(0, depth_roi.shape[0], step):
            for px in range(0, depth_roi.shape[1], step):
                z = depth_meters[py, px]
                if 0.1 < z < 3.0:
                    img_x = x1 + px
                    img_y = y1 + py
                    x_3d = (img_x - self.cx) * z / self.fx
                    y_3d = (img_y - self.cy) * z / self.fy
                    points.append([x_3d, y_3d, z])
        
        if len(points) < 10:
            return 0.0, 0.0, 0.0
        
        points = np.array(points)
        
        try:
            # PCA for orientation
            centroid = np.mean(points, axis=0)
            centered = points - centroid
            cov_matrix = np.cov(centered.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            
            idx = np.argsort(eigenvalues)[::-1]
            eigenvectors = eigenvectors[:, idx]
            
            main_axis = eigenvectors[:, 0]
            secondary_axis = eigenvectors[:, 1]
            
            yaw = np.degrees(np.arctan2(main_axis[0], main_axis[2]))
            pitch = np.degrees(np.arctan2(-main_axis[1], 
                              np.sqrt(main_axis[0]**2 + main_axis[2]**2)))
            roll = np.degrees(np.arctan2(secondary_axis[1], secondary_axis[0]))
            
            return roll, pitch, yaw
            
        except Exception:
            return 0.0, 0.0, 0.0
    
    def detection_callback(self):
        """Main detection loop - runs at timer frequency."""
        if self.color_image is None or self.model is None:
            return
        
        # Run YOLO inference
        results = self.model(
            self.color_image,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            verbose=False
        )
        
        if not results or len(results) == 0:
            return
        
        boxes = results[0].boxes
        if len(boxes) == 0:
            return
        
        # Process detections
        detections = []
        annotated_frame = self.color_image.copy()
        
        for i, box in enumerate(boxes):
            if i >= self.max_detections:
                break
            
            # Extract detection info
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = results[0].names.get(class_id, f"class_{class_id}")
            
            # CRITICAL: Skip any class not in allowed list
            if self.filter_classes and class_name not in self.filter_classes:
                # Silently skip - this prevents detecting unwanted classes
                continue
            
            # Calculate center
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # Get 3D position
            pos_x, pos_y, pos_z = self.get_3d_position(
                center_x, center_y, self.depth_image
            )
            
            # Get orientation
            roll, pitch, yaw = self.get_orientation(
                self.depth_image, (x1, y1, x2, y2)
            )
            
            detection = {
                'class_name': class_name,
                'class_id': class_id,
                'confidence': confidence,
                'bbox': [x1, y1, x2, y2],
                'center_x': center_x,
                'center_y': center_y,
                'position': {'x': pos_x, 'y': pos_y, 'z': pos_z},
                'orientation': {'roll': roll, 'pitch': pitch, 'yaw': yaw},
                'distance': pos_z
            }
            detections.append(detection)
            
            # Draw on annotated frame
            color = (0, 255, 0) if pos_z > 0.4 else (0, 0, 255)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            label = f"{class_name}: {confidence:.2f}"
            if pos_z > 0:
                label += f" | {pos_z:.2f}m"
            
            cv2.putText(annotated_frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw center crosshair
            cv2.drawMarker(annotated_frame, (center_x, center_y),
                          (0, 0, 255), cv2.MARKER_CROSS, 10, 2)
            
            # Publish TF for this detection
            if self.publish_tf and self.tf_broadcaster and pos_z > 0:
                self.publish_detection_tf(detection, i)
        
        # Publish results
        self.publish_detections(detections)
        
        # Publish annotated image
        try:
            img_msg = self.bridge.cv2_to_imgmsg(annotated_frame, 'bgr8')
            img_msg.header.stamp = self.get_clock().now().to_msg()
            img_msg.header.frame_id = self.camera_frame
            self.image_pub.publish(img_msg)
        except Exception as e:
            self.get_logger().error(f"Error publishing image: {e}")
    
    def publish_detections(self, detections: list):
        """Publish detection results."""
        now = self.get_clock().now().to_msg()
        
        if self.custom_msgs_available:
            # Publish using custom messages
            det_array = self.DetectionArray()
            det_array.header.stamp = now
            det_array.header.frame_id = self.camera_frame
            det_array.count = len(detections)
            
            for det in detections:
                msg = self.Detection()
                msg.header.stamp = now
                msg.header.frame_id = self.camera_frame
                msg.class_name = det['class_name']
                msg.class_id = det['class_id']
                msg.confidence = det['confidence']
                msg.bbox = det['bbox']
                msg.center_x = det['center_x']
                msg.center_y = det['center_y']
                msg.position = Point(
                    x=det['position']['x'],
                    y=det['position']['y'],
                    z=det['position']['z']
                )
                msg.orientation = Vector3(
                    x=det['orientation']['roll'],
                    y=det['orientation']['pitch'],
                    z=det['orientation']['yaw']
                )
                msg.distance = det['distance']
                det_array.detections.append(msg)
            
            self.detections_pub.publish(det_array)
            
            # Publish best detection
            if detections:
                best = max(detections, key=lambda d: d['confidence'])
                best_msg = self.Detection()
                best_msg.header.stamp = now
                best_msg.header.frame_id = self.camera_frame
                best_msg.class_name = best['class_name']
                best_msg.class_id = best['class_id']
                best_msg.confidence = best['confidence']
                best_msg.bbox = best['bbox']
                best_msg.center_x = best['center_x']
                best_msg.center_y = best['center_y']
                best_msg.position = Point(
                    x=best['position']['x'],
                    y=best['position']['y'],
                    z=best['position']['z']
                )
                best_msg.orientation = Vector3(
                    x=best['orientation']['roll'],
                    y=best['orientation']['pitch'],
                    z=best['orientation']['yaw']
                )
                best_msg.distance = best['distance']
                self.best_detection_pub.publish(best_msg)
        
        else:
            # Fallback to JSON strings
            import json
            
            det_json = json.dumps({
                'stamp': {'sec': now.sec, 'nanosec': now.nanosec},
                'frame_id': self.camera_frame,
                'count': len(detections),
                'detections': detections
            })
            
            msg = self.String()
            msg.data = det_json
            self.detections_pub.publish(msg)
            
            if detections:
                best = max(detections, key=lambda d: d['confidence'])
                best_msg = self.String()
                best_msg.data = json.dumps(best)
                self.best_detection_pub.publish(best_msg)
    
    def publish_detection_tf(self, detection: dict, index: int):
        """Publish TF transform for detected object."""
        if not self.tf_broadcaster:
            return
        
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.camera_frame
        t.child_frame_id = f"{self.detection_frame_prefix}{index}"
        
        # Position from detection
        t.transform.translation.x = detection['position']['x']
        t.transform.translation.y = detection['position']['y']
        t.transform.translation.z = detection['position']['z']
        
        # Convert RPY to quaternion (simplified - assumes small angles)
        roll = np.radians(detection['orientation']['roll'])
        pitch = np.radians(detection['orientation']['pitch'])
        yaw = np.radians(detection['orientation']['yaw'])
        
        # Quaternion from Euler angles (ZYX convention)
        cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
        cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
        cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
        
        t.transform.rotation.w = cr * cp * cy + sr * sp * sy
        t.transform.rotation.x = sr * cp * cy - cr * sp * sy
        t.transform.rotation.y = cr * sp * cy + sr * cp * sy
        t.transform.rotation.z = cr * cp * sy - sr * sp * cy
        
        self.tf_broadcaster.sendTransform(t)


def main(args=None):
    """Main entry point."""
    rclpy.init(args=args)
    
    node = DetectionNode()
    
    # Use multi-threaded executor for concurrent callbacks
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
