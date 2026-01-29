#!/usr/bin/env python3
"""
ROS2 Path Publisher Node

Publishes generated tool paths to ROS2 topics for robot control.
Works standalone - gracefully handles missing ROS2 installation.

Usage:
    ros2 run stl_detection path_publisher
    OR
    python ros2_path_publisher.py
"""

import os
import sys
import yaml
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List

# Try to import ROS2 - graceful fallback if not available
ROS2_AVAILABLE = False
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
    from nav_msgs.msg import Path as NavPath
    from std_srvs.srv import Trigger
    from std_msgs.msg import Header
    ROS2_AVAILABLE = True
except ImportError:
    print("ROS2 not available - running in standalone mode")
    Node = object  # Placeholder


class PathPublisher:
    """
    Publishes tool paths to ROS2 topics.
    
    Can run standalone without ROS2 for testing.
    """
    
    def __init__(self, paths_dir: str = None):
        """
        Initialize path publisher.
        
        Args:
            paths_dir: Directory containing YAML path files
        """
        if paths_dir is None:
            paths_dir = Path(__file__).parent / "paths"
        self.paths_dir = Path(paths_dir)
        self.current_path = None
        self.last_file_time = {}
        
    def load_path_from_yaml(self, yaml_path: str) -> Optional[Dict]:
        """Load path from YAML file."""
        try:
            with open(yaml_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading {yaml_path}: {e}")
            return None
    
    def get_latest_path_file(self) -> Optional[Path]:
        """Get the most recently modified path file."""
        if not self.paths_dir.exists():
            return None
        
        yaml_files = list(self.paths_dir.glob("*.yaml"))
        if not yaml_files:
            return None
        
        return max(yaml_files, key=lambda p: p.stat().st_mtime)
    
    def watch_for_new_paths(self, callback=None):
        """Watch for new path files and trigger callback."""
        while True:
            latest = self.get_latest_path_file()
            if latest:
                mtime = latest.stat().st_mtime
                if str(latest) not in self.last_file_time or \
                   self.last_file_time[str(latest)] < mtime:
                    self.last_file_time[str(latest)] = mtime
                    path_data = self.load_path_from_yaml(str(latest))
                    if path_data and callback:
                        callback(path_data, str(latest))
            time.sleep(1.0)
    
    def convert_to_ros_path(self, path_data: Dict) -> Dict:
        """Convert internal path format to ROS Path message structure."""
        header = path_data.get('header', {})
        waypoints = path_data.get('path', {}).get('waypoints', [])
        
        poses = []
        for wp in waypoints:
            pose_data = wp.get('pose', {})
            pos = pose_data.get('position', {})
            ori = pose_data.get('orientation', {})
            
            poses.append({
                'header': {
                    'frame_id': header.get('frame_id', 'camera_link'),
                    'stamp': wp.get('time_from_start', {'sec': 0, 'nanosec': 0})
                },
                'pose': {
                    'position': pos,
                    'orientation': ori
                }
            })
        
        return {
            'header': header,
            'poses': poses
        }


if ROS2_AVAILABLE:
    class PathPublisherNode(Node):
        """ROS2 Node for publishing tool paths."""
        
        def __init__(self):
            super().__init__('path_publisher')
            
            # Declare parameters
            self.declare_parameter('paths_dir', '')
            self.declare_parameter('frame_id', 'camera_link')
            self.declare_parameter('publish_rate', 1.0)
            
            paths_dir = self.get_parameter('paths_dir').get_parameter_value().string_value
            if not paths_dir:
                paths_dir = str(Path(__file__).parent / "paths")
            
            self.path_publisher = PathPublisher(paths_dir)
            self.frame_id = self.get_parameter('frame_id').get_parameter_value().string_value
            
            # QoS for reliable delivery
            qos = QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
                depth=10
            )
            
            # Publishers
            self.path_pub = self.create_publisher(NavPath, '/tool_path', qos)
            
            # Services
            self.trigger_srv = self.create_service(
                Trigger, 
                '/publish_latest_path', 
                self.publish_latest_callback
            )
            
            # Timer for periodic publishing
            rate = self.get_parameter('publish_rate').get_parameter_value().double_value
            if rate > 0:
                self.timer = self.create_timer(1.0 / rate, self.timer_callback)
            
            self.last_published = None
            self.get_logger().info(f'Path Publisher started. Watching: {paths_dir}')
        
        def publish_latest_callback(self, request, response):
            """Service callback to publish latest path."""
            latest = self.path_publisher.get_latest_path_file()
            if latest:
                success = self.publish_path_file(str(latest))
                response.success = success
                response.message = f"Published: {latest.name}" if success else "Failed to publish"
            else:
                response.success = False
                response.message = "No path files found"
            return response
        
        def timer_callback(self):
            """Periodically check for new paths."""
            latest = self.path_publisher.get_latest_path_file()
            if latest and str(latest) != self.last_published:
                self.publish_path_file(str(latest))
        
        def publish_path_file(self, yaml_path: str) -> bool:
            """Load and publish a path from YAML file."""
            path_data = self.path_publisher.load_path_from_yaml(yaml_path)
            if not path_data:
                return False
            
            try:
                nav_path = self.convert_to_nav_path(path_data)
                self.path_pub.publish(nav_path)
                self.last_published = yaml_path
                self.get_logger().info(f'Published path: {Path(yaml_path).name}')
                return True
            except Exception as e:
                self.get_logger().error(f'Error publishing: {e}')
                return False
        
        def convert_to_nav_path(self, path_data: Dict) -> NavPath:
            """Convert YAML path data to nav_msgs/Path message."""
            nav_path = NavPath()
            
            # Header
            header = path_data.get('header', {})
            nav_path.header.frame_id = header.get('frame_id', self.frame_id)
            nav_path.header.stamp = self.get_clock().now().to_msg()
            
            # Poses
            waypoints = path_data.get('path', {}).get('waypoints', [])
            for wp in waypoints:
                pose_stamped = PoseStamped()
                pose_stamped.header.frame_id = nav_path.header.frame_id
                
                # Time from start
                tfs = wp.get('time_from_start', {})
                if isinstance(tfs, dict):
                    sec = tfs.get('sec', 0)
                    nanosec = tfs.get('nanosec', 0)
                else:
                    sec = int(tfs)
                    nanosec = int((tfs - sec) * 1e9)
                pose_stamped.header.stamp.sec = sec
                pose_stamped.header.stamp.nanosec = nanosec
                
                # Position
                pose = wp.get('pose', {})
                pos = pose.get('position', {})
                pose_stamped.pose.position.x = float(pos.get('x', 0))
                pose_stamped.pose.position.y = float(pos.get('y', 0))
                pose_stamped.pose.position.z = float(pos.get('z', 0))
                
                # Orientation
                ori = pose.get('orientation', {})
                pose_stamped.pose.orientation.x = float(ori.get('x', 0))
                pose_stamped.pose.orientation.y = float(ori.get('y', 0))
                pose_stamped.pose.orientation.z = float(ori.get('z', 0))
                pose_stamped.pose.orientation.w = float(ori.get('w', 1))
                
                nav_path.poses.append(pose_stamped)
            
            return nav_path


def run_standalone_test():
    """Run standalone test without ROS2."""
    print("\n" + "="*60)
    print("ROS2 PATH PUBLISHER - STANDALONE MODE")
    print("="*60)
    
    publisher = PathPublisher()
    
    latest = publisher.get_latest_path_file()
    if latest:
        print(f"\nLatest path file: {latest.name}")
        path_data = publisher.load_path_from_yaml(str(latest))
        if path_data:
            ros_path = publisher.convert_to_ros_path(path_data)
            print(f"Converted to ROS Path with {len(ros_path['poses'])} poses")
            print(f"Frame ID: {ros_path['header'].get('frame_id', 'unknown')}")
            
            # Print first pose
            if ros_path['poses']:
                first = ros_path['poses'][0]
                pos = first['pose']['position']
                print(f"First pose: ({pos['x']:.3f}, {pos['y']:.3f}, {pos['z']:.3f})")
    else:
        print("\nNo path files found in paths/ directory")
    
    print("\n" + "="*60)


def main():
    """Main entry point."""
    if ROS2_AVAILABLE:
        rclpy.init()
        node = PathPublisherNode()
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
        finally:
            node.destroy_node()
            rclpy.shutdown()
    else:
        run_standalone_test()


if __name__ == '__main__':
    main()
