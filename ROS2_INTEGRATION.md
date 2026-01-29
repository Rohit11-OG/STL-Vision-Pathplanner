# ROS 2 Humble Integration Guide

This guide explains how to integrate the STL Object Detection system with ROS 2 Humble for use with the Dobot Nova5 robotic arm.

## Quick Start

### 1. Prerequisites

```bash
# ROS 2 Humble
sudo apt install ros-humble-desktop

# RealSense ROS 2 wrapper
sudo apt install ros-humble-realsense2-camera

# Python dependencies
pip install ultralytics opencv-python numpy
```

### 2. Build the Package

```bash
# Create workspace (if not exists)
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src

# Link or copy the package
ln -s /home/rohit/Desktop/STL/stl_detector_ros2 .

# Build
cd ~/ros2_ws
colcon build --packages-select stl_detector_ros2
source install/setup.bash
```

### 3. Launch Detection

```bash
# Terminal 1: Start RealSense camera
ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true

# Terminal 2: Start detection node
ros2 launch stl_detector_ros2 detection.launch.py
```

---

## Published Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/detections` | `stl_detector_ros2/DetectionArray` | All detected objects |
| `/best_detection` | `stl_detector_ros2/Detection` | Highest confidence detection |
| `/detection_image` | `sensor_msgs/Image` | Annotated image with bounding boxes |

## Detection Message Format

```python
# stl_detector_ros2/Detection
Header header
string class_name          # e.g., "bottle", "mouse"
int32 class_id
float32 confidence         # 0.0 - 1.0
int32[4] bbox              # [x1, y1, x2, y2] in pixels
int32 center_x, center_y   # pixel center
Point position             # 3D position (x, y, z) in meters
Vector3 orientation        # (roll, pitch, yaw) in degrees
float32 distance           # distance from camera in meters
```

---

## Integration with Dobot Nova5

### Example: Subscribe to Detections

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from stl_detector_ros2.msg import Detection

class DobotPaintingNode(Node):
    def __init__(self):
        super().__init__('dobot_painting_node')
        
        self.subscription = self.create_subscription(
            Detection,
            '/best_detection',
            self.detection_callback,
            10
        )
    
    def detection_callback(self, msg):
        # Get object position in camera frame
        x = msg.position.x  # meters
        y = msg.position.y  # meters  
        z = msg.position.z  # meters (distance)
        
        self.get_logger().info(
            f"Detected {msg.class_name} at ({x:.3f}, {y:.3f}, {z:.3f})m"
        )
        
        # TODO: Transform to robot base frame
        # TODO: Send to Dobot Nova5 for painting

def main():
    rclpy.init()
    node = DobotPaintingNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### TF2 Coordinate Transformation

The detection node publishes TF transforms for each detected object. To transform coordinates to the robot base frame:

```python
import tf2_ros
from tf2_geometry_msgs import do_transform_point
from geometry_msgs.msg import PointStamped

# Create TF buffer and listener
tf_buffer = tf2_ros.Buffer()
tf_listener = tf2_ros.TransformListener(tf_buffer, self)

# Get transform from camera to robot base
transform = tf_buffer.lookup_transform(
    'base_link',           # Target frame (robot base)
    'camera_color_optical_frame',  # Source frame (camera)
    rclpy.time.Time()
)

# Transform detection point
point_camera = PointStamped()
point_camera.header.frame_id = 'camera_color_optical_frame'
point_camera.point.x = detection.position.x
point_camera.point.y = detection.position.y
point_camera.point.z = detection.position.z

point_robot = do_transform_point(point_camera, transform)
# point_robot.point now contains coordinates in robot base frame
```

---

## Configuration

Edit `config/detection_params.yaml`:

```yaml
stl_detector_node:
  ros__parameters:
    confidence_threshold: 0.65  # Adjust for your needs
    publish_tf: true            # Enable for robot integration
    camera_frame: "camera_color_optical_frame"
```

---

## Dobot Nova5 Setup

1. Install Dobot ROS 2 driver from: https://github.com/Dobot-Arm/TCP-IP-ROS-6AXis
2. Configure camera-to-robot transform (eye-on-hand calibration)
3. Subscribe to `/best_detection` topic
4. Transform coordinates and send to robot

## Troubleshooting

**No detections published:**
- Check camera is running: `ros2 topic echo /camera/color/image_raw`
- Verify model path in parameters
- Lower confidence threshold

**Wrong 3D coordinates:**
- Ensure depth alignment is enabled
- Check camera intrinsics are received
- Verify TF tree: `ros2 run tf2_tools view_frames`
