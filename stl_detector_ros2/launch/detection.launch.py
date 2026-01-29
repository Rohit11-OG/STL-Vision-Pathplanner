#!/usr/bin/env python3
"""
Launch file for STL Object Detection with RealSense camera.

This launch file starts:
1. RealSense camera node (optional - if camera is not already running)
2. STL Detection node

Usage:
    ros2 launch stl_detector_ros2 detection.launch.py
    ros2 launch stl_detector_ros2 detection.launch.py use_realsense:=true
    ros2 launch stl_detector_ros2 detection.launch.py model_path:=/path/to/model.pt
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.launch_description_sources import PythonLaunchDescriptionSource

import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Get package directories
    pkg_share = get_package_share_directory('stl_detector_ros2')
    
    # Declare launch arguments
    use_realsense_arg = DeclareLaunchArgument(
        'use_realsense',
        default_value='false',
        description='Launch RealSense camera node (set to true if camera is not already running)'
    )
    
    model_path_arg = DeclareLaunchArgument(
        'model_path',
        # IMPORTANT: Use your TRAINED model, NOT yolov8n.pt!
        # Pretrained yolov8n.pt has 80 COCO classes and will detect EVERYTHING
        default_value='/home/rohit/Desktop/STL/runs/detect/stl_object/weights/best.pt',
        description='Path to trained YOLO model (.pt file). MUST be your custom-trained model!'
    )
    
    confidence_arg = DeclareLaunchArgument(
        'confidence_threshold',
        default_value='0.65',
        description='Detection confidence threshold (0.0-1.0)'
    )
    
    publish_tf_arg = DeclareLaunchArgument(
        'publish_tf',
        default_value='true',
        description='Publish TF transforms for detected objects'
    )
    
    camera_frame_arg = DeclareLaunchArgument(
        'camera_frame',
        default_value='camera_color_optical_frame',
        description='Camera optical frame name for TF'
    )
    
    # RealSense camera node (optional)
    # This requires realsense2_camera package to be installed
    realsense_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('realsense2_camera'),
                'launch',
                'rs_launch.py'
            ])
        ]),
        launch_arguments={
            'align_depth.enable': 'true',
            'pointcloud.enable': 'false',
            'enable_sync': 'true',
        }.items(),
        condition=IfCondition(LaunchConfiguration('use_realsense'))
    )
    
    # STL Detection node
    detection_node = Node(
        package='stl_detector_ros2',
        executable='detection_node.py',
        name='stl_detector_node',
        output='screen',
        parameters=[{
            'model_path': LaunchConfiguration('model_path'),
            'confidence_threshold': LaunchConfiguration('confidence_threshold'),
            'publish_tf': LaunchConfiguration('publish_tf'),
            'camera_frame': LaunchConfiguration('camera_frame'),
            'iou_threshold': 0.45,
            'max_detections': 10,
            'show_all_detections': False,
            'detection_frame_prefix': 'detected_object_',
        }],
        remappings=[
            # Remap topics if needed for your specific camera setup
            # ('/camera/color/image_raw', '/your_camera/color/image_raw'),
            # ('/camera/aligned_depth_to_color/image_raw', '/your_camera/depth/image_rect_raw'),
        ]
    )
    
    return LaunchDescription([
        # Launch arguments
        use_realsense_arg,
        model_path_arg,
        confidence_arg,
        publish_tf_arg,
        camera_frame_arg,
        
        # Nodes
        realsense_node,
        detection_node,
    ])
