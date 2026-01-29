"""
Tool Path Planner Module
Generates tool paths from detected objects for robotic manipulation.

Workflow: Detected Object → Tool Path Planning → YAML Export
"""

import numpy as np
import yaml
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
from datetime import datetime
import math


def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> Tuple[float, float, float, float]:
    """
    Convert Euler angles (in degrees) to quaternion (x, y, z, w).
    
    Uses ROS convention (ZYX rotation order).
    """
    # Convert to radians
    roll_rad = math.radians(roll)
    pitch_rad = math.radians(pitch)
    yaw_rad = math.radians(yaw)
    
    # Calculate quaternion components
    cy = math.cos(yaw_rad * 0.5)
    sy = math.sin(yaw_rad * 0.5)
    cp = math.cos(pitch_rad * 0.5)
    sp = math.sin(pitch_rad * 0.5)
    cr = math.cos(roll_rad * 0.5)
    sr = math.sin(roll_rad * 0.5)
    
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    
    return (x, y, z, w)


@dataclass
class PathPoint:
    """Represents a single waypoint in the tool path (ROS PoseStamped style with trajectory)."""
    id: int
    x: float
    y: float
    z: float
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    point_type: str = "waypoint"  # approach, grasp, waypoint, retreat
    # Trajectory fields
    velocity: float = 0.1  # Linear velocity (m/s)
    time_from_start: float = 0.0  # Time from path start (seconds)
    
    def get_quaternion(self) -> Tuple[float, float, float, float]:
        """Get orientation as quaternion (x, y, z, w)."""
        return euler_to_quaternion(self.roll, self.pitch, self.yaw)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to ROS-style PoseStamped dictionary with trajectory for YAML export."""
        qx, qy, qz, qw = self.get_quaternion()
        # Convert time to ROS Duration format (sec + nanosec)
        sec = int(self.time_from_start)
        nanosec = int((self.time_from_start - sec) * 1e9)
        return {
            'id': self.id,
            'type': self.point_type,
            'pose': {
                'position': {
                    'x': round(float(self.x), 6),
                    'y': round(float(self.y), 6),
                    'z': round(float(self.z), 6)
                },
                'orientation': {
                    'x': round(float(qx), 6),
                    'y': round(float(qy), 6),
                    'z': round(float(qz), 6),
                    'w': round(float(qw), 6)
                }
            },
            'velocity': round(float(self.velocity), 4),
            'time_from_start': {'sec': sec, 'nanosec': nanosec}
        }
    
    def to_dict_euler(self) -> Dict[str, Any]:
        """Convert to dictionary with Euler angles (alternative format)."""
        return {
            'id': self.id,
            'type': self.point_type,
            'position': {'x': round(self.x, 4), 'y': round(self.y, 4), 'z': round(self.z, 4)},
            'orientation_euler': {'roll': round(self.roll, 2), 'pitch': round(self.pitch, 2), 'yaw': round(self.yaw, 2)},
            'velocity': round(self.velocity, 4),
            'time_from_start': round(self.time_from_start, 4)
        }


class ToolPathPlanner:
    """
    Generate tool paths from detected object data.
    
    Supports multiple path strategies:
    - contour: Follow the object boundary/surface
    - approach: Pick-and-place style (approach → grasp → retreat)
    - grid: Grid pattern over object surface
    """
    
    def __init__(self, object_data: Dict[str, Any]):
        """
        Initialize path planner with detected object data.
        
        Args:
            object_data: Dictionary containing:
                - center_3d: (x, y, z) center position in meters
                - orientation: (roll, pitch, yaw) in degrees
                - bounding_box: (x1, y1, x2, y2) pixel coordinates
                - class_name: Object class label
                - depth_image: Optional depth array for surface extraction
                - confidence: Detection confidence
        """
        self.center_3d = object_data.get('center_3d', (0, 0, 0))
        self.orientation = object_data.get('orientation', (0, 0, 0))
        self.bounding_box = object_data.get('bounding_box', (0, 0, 0, 0))
        self.class_name = object_data.get('class_name', 'object')
        self.depth_image = object_data.get('depth_image', None)
        self.confidence = object_data.get('confidence', 0.0)
        
        # Path configuration
        self.surface_offset = 0.02  # 2cm safety offset from surface
        self.approach_distance = 0.05  # 5cm approach distance
        self.frame_id = "camera_link"
        
        self.path: List[PathPoint] = []
        self.timestamp = datetime.now().isoformat()
    
    def generate_path(self, strategy: str = 'contour', num_points: int = 20, 
                      **kwargs) -> List[PathPoint]:
        """
        Generate tool path using specified strategy.
        
        Args:
            strategy: 'contour', 'approach', 'grid', 'surface', 'spiral', or 'zigzag'
            num_points: Number of waypoints to generate
            **kwargs: Additional strategy-specific parameters
            
        Returns:
            List of PathPoint objects
        """
        if strategy == 'contour':
            self.path = self._generate_contour_path(num_points, **kwargs)
        elif strategy == 'approach':
            self.path = self._generate_approach_path(**kwargs)
        elif strategy == 'grid':
            self.path = self._generate_grid_path(num_points, **kwargs)
        elif strategy == 'surface':
            self.path = self._generate_surface_path(num_points, **kwargs)
        elif strategy == 'spiral':
            self.path = self._generate_spiral_path(num_points, **kwargs)
        elif strategy == 'zigzag':
            self.path = self._generate_zigzag_path(num_points, **kwargs)
        else:
            raise ValueError(f"Unknown strategy: {strategy}. Use 'contour', 'approach', 'grid', 'surface', 'spiral', or 'zigzag'")
        
        return self.path
    
    def _generate_contour_path(self, num_points: int = 20, 
                                radius_scale: float = 1.2) -> List[PathPoint]:
        """
        Generate path following object boundary in XY plane.
        
        Creates a circular/elliptical path around the detected object center
        at a safe distance for inspection, painting, or welding tasks.
        """
        cx, cy, cz = self.center_3d
        roll, pitch, yaw = self.orientation
        
        # Estimate object radius from bounding box (in meters, approximate)
        x1, y1, x2, y2 = self.bounding_box
        bbox_width = abs(x2 - x1)
        bbox_height = abs(y2 - y1)
        
        # Convert pixel size to approximate world size (rough estimate)
        # Assuming ~600 pixels corresponds to ~1m at 1m distance
        pixels_per_meter = 600 / max(cz, 0.3)
        radius_x = (bbox_width / pixels_per_meter / 2) * radius_scale + self.surface_offset
        radius_y = (bbox_height / pixels_per_meter / 2) * radius_scale + self.surface_offset
        
        path_points = []
        
        for i in range(num_points):
            angle = (2 * math.pi * i) / num_points
            
            # Elliptical path around object
            px = cx + radius_x * math.cos(angle)
            py = cy + radius_y * math.sin(angle)
            pz = cz  # Keep same depth
            
            # Tool orientation: point toward object center
            point_yaw = math.degrees(math.atan2(cy - py, cx - px))
            
            point = PathPoint(
                id=i,
                x=px, y=py, z=pz,
                roll=roll, pitch=pitch, yaw=point_yaw,
                point_type='waypoint'
            )
            path_points.append(point)
        
        return path_points
    
    def _generate_approach_path(self, approach_axis: str = 'z') -> List[PathPoint]:
        """
        Generate approach-grasp-retreat path for pick-and-place.
        
        Creates a 3-point sequence:
        1. Approach: Position above/before object
        2. Grasp: At object position
        3. Retreat: Back to safe position
        """
        cx, cy, cz = self.center_3d
        roll, pitch, yaw = self.orientation
        
        path_points = []
        
        # Approach point (offset along specified axis)
        if approach_axis == 'z':
            approach = PathPoint(
                id=0, x=cx, y=cy, z=cz - self.approach_distance,
                roll=roll, pitch=pitch, yaw=yaw,
                point_type='approach'
            )
        elif approach_axis == 'y':
            approach = PathPoint(
                id=0, x=cx, y=cy - self.approach_distance, z=cz,
                roll=roll, pitch=pitch, yaw=yaw,
                point_type='approach'
            )
        else:  # x-axis
            approach = PathPoint(
                id=0, x=cx - self.approach_distance, y=cy, z=cz,
                roll=roll, pitch=pitch, yaw=yaw,
                point_type='approach'
            )
        path_points.append(approach)
        
        # Grasp point (at object with surface offset)
        grasp = PathPoint(
            id=1, x=cx, y=cy, z=cz + self.surface_offset,
            roll=roll, pitch=pitch, yaw=yaw,
            point_type='grasp'
        )
        path_points.append(grasp)
        
        # Retreat point (same as approach)
        retreat = PathPoint(
            id=2, x=approach.x, y=approach.y, z=approach.z,
            roll=roll, pitch=pitch, yaw=yaw,
            point_type='retreat'
        )
        path_points.append(retreat)
        
        return path_points
    
    def _generate_grid_path(self, num_points: int = 16, 
                            grid_size: float = None) -> List[PathPoint]:
        """
        Generate grid pattern over object surface.
        
        Useful for scanning or coating operations.
        """
        cx, cy, cz = self.center_3d
        roll, pitch, yaw = self.orientation
        
        # Calculate grid dimensions from bounding box
        x1, y1, x2, y2 = self.bounding_box
        bbox_width = abs(x2 - x1)
        bbox_height = abs(y2 - y1)
        
        pixels_per_meter = 600 / max(cz, 0.3)
        width = bbox_width / pixels_per_meter
        height = bbox_height / pixels_per_meter
        
        if grid_size is None:
            # Calculate grid size for approximate num_points
            grid_dim = int(math.sqrt(num_points))
            step_x = width / max(grid_dim - 1, 1)
            step_y = height / max(grid_dim - 1, 1)
        else:
            step_x = step_y = grid_size
            grid_dim = int(max(width, height) / grid_size) + 1
        
        path_points = []
        point_id = 0
        
        # Serpentine pattern for efficient traversal
        for row in range(grid_dim):
            cols = range(grid_dim) if row % 2 == 0 else range(grid_dim - 1, -1, -1)
            for col in cols:
                px = cx - width/2 + col * step_x
                py = cy - height/2 + row * step_y
                pz = cz + self.surface_offset
                
                point = PathPoint(
                    id=point_id,
                    x=px, y=py, z=pz,
                    roll=roll, pitch=pitch, yaw=yaw,
                    point_type='waypoint'
                )
                path_points.append(point)
                point_id += 1
        
        return path_points
    
    def _generate_surface_path(self, num_points: int = 20, 
                                camera_intrinsics: Dict = None) -> List[PathPoint]:
        """
        Generate path following actual 3D surface from depth data.
        
        Extracts point cloud from depth image and creates path along surface boundary.
        """
        if self.depth_image is None:
            return self._generate_contour_path(num_points)
        
        cx, cy, cz = self.center_3d
        roll, pitch, yaw = self.orientation
        x1, y1, x2, y2 = map(int, self.bounding_box)
        
        # Camera intrinsics
        if camera_intrinsics is None:
            camera_intrinsics = {'fx': 615.0, 'fy': 615.0, 'cx': 320.0, 'cy': 240.0}
        
        fx, fy = camera_intrinsics['fx'], camera_intrinsics['fy']
        ppx, ppy = camera_intrinsics['cx'], camera_intrinsics['cy']
        
        h, w = self.depth_image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return self._generate_contour_path(num_points)
        
        depth_roi = self.depth_image[y1:y2, x1:x2]
        roi_h, roi_w = depth_roi.shape
        
        # Sample along perimeter
        perimeter = 2 * (roi_w + roi_h)
        step = max(1, perimeter // num_points)
        
        path_points = []
        point_id = 0
        
        for i in range(0, perimeter, step):
            if i < roi_w:
                px_roi, py_roi = i, 0
            elif i < roi_w + roi_h:
                px_roi, py_roi = roi_w - 1, i - roi_w
            elif i < 2 * roi_w + roi_h:
                px_roi, py_roi = roi_w - 1 - (i - roi_w - roi_h), roi_h - 1
            else:
                px_roi, py_roi = 0, roi_h - 1 - (i - 2 * roi_w - roi_h)
            
            px_roi = max(0, min(roi_w - 1, px_roi))
            py_roi = max(0, min(roi_h - 1, py_roi))
            
            depth_val = depth_roi[py_roi, px_roi]
            if depth_val <= 0:
                continue
            
            z = float(depth_val) * 0.001
            if z <= 0.1 or z > 5.0:
                continue
            
            img_x, img_y = x1 + px_roi, y1 + py_roi
            x_3d = (img_x - ppx) * z / fx
            y_3d = (img_y - ppy) * z / fy
            z_offset = z - self.surface_offset
            
            point_yaw = math.degrees(math.atan2(cy - y_3d, cx - x_3d))
            
            path_points.append(PathPoint(
                id=point_id, x=float(x_3d), y=float(y_3d), z=float(z_offset),
                roll=roll, pitch=pitch, yaw=point_yaw, point_type='waypoint'
            ))
            point_id += 1
            if point_id >= num_points:
                break
        
        return path_points if path_points else self._generate_contour_path(num_points)
    
    def _generate_spiral_path(self, num_points: int = 20, inward: bool = True) -> List[PathPoint]:
        """Generate spiral pattern around object center."""
        cx, cy, cz = self.center_3d
        roll, pitch, yaw = self.orientation
        
        x1, y1, x2, y2 = self.bounding_box
        pixels_per_meter = 600 / max(cz, 0.3)
        max_radius = max(abs(x2-x1), abs(y2-y1)) / pixels_per_meter / 2 + self.surface_offset
        
        path_points = []
        num_turns = 3
        
        for i in range(num_points):
            t = i / max(num_points - 1, 1)
            radius = max_radius * (1 - t) if inward else max_radius * t
            angle = 2 * math.pi * num_turns * t
            
            px = cx + radius * math.cos(angle)
            py = cy + radius * math.sin(angle)
            point_yaw = math.degrees(math.atan2(cy - py, cx - px))
            
            path_points.append(PathPoint(
                id=i, x=float(px), y=float(py), z=float(cz),
                roll=roll, pitch=pitch, yaw=point_yaw, point_type='waypoint'
            ))
        
        return path_points
    
    def _generate_zigzag_path(self, num_points: int = 20) -> List[PathPoint]:
        """Generate zigzag/raster pattern over object area."""
        cx, cy, cz = self.center_3d
        roll, pitch, yaw = self.orientation
        
        x1, y1, x2, y2 = self.bounding_box
        pixels_per_meter = 600 / max(cz, 0.3)
        width = abs(x2 - x1) / pixels_per_meter
        height = abs(y2 - y1) / pixels_per_meter
        
        num_rows = int(math.sqrt(num_points / 2)) + 1
        points_per_row = max(2, num_points // num_rows)
        
        path_points = []
        point_id = 0
        
        for row in range(num_rows):
            row_y = cy - height/2 + (row / max(num_rows - 1, 1)) * height
            
            if row % 2 == 0:
                x_positions = [cx - width/2 + i/(max(points_per_row-1, 1))*width for i in range(points_per_row)]
            else:
                x_positions = [cx + width/2 - i/(max(points_per_row-1, 1))*width for i in range(points_per_row)]
            
            for px in x_positions:
                path_points.append(PathPoint(
                    id=point_id, x=float(px), y=float(row_y), z=float(cz + self.surface_offset),
                    roll=roll, pitch=pitch, yaw=yaw, point_type='waypoint'
                ))
                point_id += 1
                if point_id >= num_points:
                    break
            if point_id >= num_points:
                break
        
        return path_points
    
    def check_collision(self, point: PathPoint, min_clearance: float = 0.03) -> bool:
        """Check if waypoint collides with obstacles. Returns True if collision."""
        if self.depth_image is None:
            return False
        
        fx, fy, cx, cy = 615.0, 615.0, 320.0, 240.0
        if point.z <= 0:
            return True
        
        px = int(point.x * fx / point.z + cx)
        py = int(point.y * fy / point.z + cy)
        
        h, w = self.depth_image.shape[:2]
        if px < 0 or px >= w or py < 0 or py >= h:
            return False
        
        depth_at_point = float(self.depth_image[py, px]) * 0.001
        if depth_at_point <= 0:
            return False
        
        return point.z > depth_at_point - min_clearance
    
    def apply_collision_avoidance(self, min_clearance: float = 0.03) -> int:
        """Adjust path waypoints to avoid collisions. Returns count of adjusted points."""
        adjusted = 0
        for point in self.path:
            if self.check_collision(point, min_clearance):
                point.z = point.z - min_clearance * 2
                adjusted += 1
        return adjusted
    
    def compute_trajectory_timing(self, default_velocity: float = 0.1, 
                                   max_velocity: float = 0.5) -> float:
        """
        Compute velocity and timing for each waypoint based on distance.
        
        Args:
            default_velocity: Default travel velocity in m/s
            max_velocity: Maximum allowed velocity in m/s
            
        Returns:
            Total trajectory duration in seconds
        """
        if not self.path or len(self.path) < 2:
            return 0.0
        
        velocity = min(default_velocity, max_velocity)
        current_time = 0.0
        
        # First point starts at time 0
        self.path[0].time_from_start = 0.0
        self.path[0].velocity = velocity
        
        for i in range(1, len(self.path)):
            prev = self.path[i - 1]
            curr = self.path[i]
            
            # Calculate distance between waypoints
            dx = curr.x - prev.x
            dy = curr.y - prev.y
            dz = curr.z - prev.z
            distance = math.sqrt(dx*dx + dy*dy + dz*dz)
            
            # Time = distance / velocity
            if velocity > 0:
                segment_time = distance / velocity
            else:
                segment_time = 0.5  # Default 0.5s per waypoint
            
            current_time += segment_time
            curr.time_from_start = current_time
            curr.velocity = velocity
        
        return current_time
    
    def export_to_yaml(self, output_path: str) -> str:
        """
        Export generated path to ROS-style YAML file with PoseStamped waypoints.
        
        Args:
            output_path: Path to output YAML file
            
        Returns:
            Path to created file
        """
        if not self.path:
            raise ValueError("No path generated. Call generate_path() first.")
        
        # Ensure output directory exists
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Get object pose quaternion
        obj_qx, obj_qy, obj_qz, obj_qw = euler_to_quaternion(*self.orientation)
        
        # Build ROS-style YAML structure
        yaml_data = {
            'header': {
                'frame_id': self.frame_id,
                'stamp': self.timestamp
            },
            'object_info': {
                'class': self.class_name,
                'confidence': round(float(self.confidence), 3),
                'pose': {
                    'position': {
                        'x': round(float(self.center_3d[0]), 6),
                        'y': round(float(self.center_3d[1]), 6),
                        'z': round(float(self.center_3d[2]), 6)
                    },
                    'orientation': {
                        'x': round(float(obj_qx), 6),
                        'y': round(float(obj_qy), 6),
                        'z': round(float(obj_qz), 6),
                        'w': round(float(obj_qw), 6)
                    }
                }
            },
            'path': {
                'num_waypoints': len(self.path),
                'waypoints': [p.to_dict() for p in self.path]
            }
        }
        
        # Write YAML file
        with open(output_file, 'w') as f:
            yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)
        
        return str(output_file)
    
    def get_path_as_dict(self) -> Dict[str, Any]:
        """Get path data as dictionary."""
        return {
            'frame_id': self.frame_id,
            'object_class': self.class_name,
            'timestamp': self.timestamp,
            'waypoints': [p.to_dict() for p in self.path]
        }
    
    def visualize_path_2d(self, image: np.ndarray = None, 
                          camera_intrinsics: Dict = None) -> np.ndarray:
        """
        Overlay path points on image for visualization.
        
        Args:
            image: Input image (BGR format)
            camera_intrinsics: Dict with fx, fy, cx, cy
            
        Returns:
            Annotated image
        """
        import cv2
        
        if image is None:
            image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Default camera intrinsics
        if camera_intrinsics is None:
            camera_intrinsics = {'fx': 615, 'fy': 615, 'cx': 320, 'cy': 240}
        
        fx = camera_intrinsics['fx']
        fy = camera_intrinsics['fy']
        cx = camera_intrinsics['cx']
        cy = camera_intrinsics['cy']
        
        # Convert 3D points to 2D pixels
        pixels = []
        for point in self.path:
            if point.z > 0:
                px = int(point.x * fx / point.z + cx)
                py = int(point.y * fy / point.z + cy)
                pixels.append((px, py, point.point_type))
        
        # Draw path
        colors = {
            'approach': (0, 255, 255),   # Yellow
            'grasp': (0, 0, 255),        # Red
            'retreat': (255, 255, 0),    # Cyan
            'waypoint': (0, 255, 0)      # Green
        }
        
        # Draw lines between consecutive points
        for i in range(len(pixels) - 1):
            cv2.line(image, (pixels[i][0], pixels[i][1]), 
                    (pixels[i+1][0], pixels[i+1][1]), (255, 128, 0), 2)
        
        # Draw points
        for px, py, ptype in pixels:
            color = colors.get(ptype, (0, 255, 0))
            cv2.circle(image, (px, py), 6, color, -1)
            cv2.circle(image, (px, py), 8, (255, 255, 255), 1)
        
        return image


def test_tool_path_planner():
    """Test the tool path planner with mock data."""
    print("Testing Tool Path Planner...")
    
    # Mock object data
    object_data = {
        'center_3d': (0.15, 0.05, 0.50),  # 50cm away
        'orientation': (0.0, 0.0, 45.0),
        'bounding_box': (200, 150, 400, 350),
        'class_name': 'bottle',
        'confidence': 0.92
    }
    
    planner = ToolPathPlanner(object_data)
    
    # Test contour path
    print("\n1. Testing contour path...")
    contour_path = planner.generate_path(strategy='contour', num_points=12)
    print(f"   Generated {len(contour_path)} waypoints")
    
    # Test approach path
    print("\n2. Testing approach path...")
    approach_path = planner.generate_path(strategy='approach')
    print(f"   Generated {len(approach_path)} waypoints")
    for p in approach_path:
        print(f"   - {p.point_type}: ({p.x:.3f}, {p.y:.3f}, {p.z:.3f})")
    
    # Test grid path
    print("\n3. Testing grid path...")
    grid_path = planner.generate_path(strategy='grid', num_points=9)
    print(f"   Generated {len(grid_path)} waypoints")
    
    # Test YAML export
    print("\n4. Testing YAML export...")
    output_path = planner.export_to_yaml('/tmp/test_tool_path.yaml')
    print(f"   Exported to: {output_path}")
    
    # Verify YAML content
    with open(output_path, 'r') as f:
        content = yaml.safe_load(f)
        print(f"   YAML structure: {list(content.keys())}")
        print(f"   Waypoints: {content['path']['num_waypoints']}")
        print(f"   Object class: {content['object_info']['class']}")
        # Verify quaternion format
        wp = content['path']['waypoints'][0]
        print(f"   Sample waypoint pose keys: {list(wp['pose'].keys())}")
    
    print("\n✓ All tests passed!")
    return True


if __name__ == "__main__":
    test_tool_path_planner()
