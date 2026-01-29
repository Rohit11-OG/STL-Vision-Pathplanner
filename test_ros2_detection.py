#!/usr/bin/env python3
"""
Test script to validate the ROS 2 detection node logic WITHOUT ROS 2.
This tests:
1. Model loading and validation
2. Class filter (only bottle, mouse)
3. Real-time detection with RealSense camera
"""

import sys
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
from ultralytics import YOLO

# Configuration - same as ROS 2 node
MODEL_PATH = PROJECT_ROOT / 'runs' / 'detect' / 'stl_object' / 'weights' / 'best.pt'
CONFIDENCE_THRESHOLD = 0.65
ALLOWED_CLASSES = ['bottle', 'mouse']  # Only detect these!

def validate_model(model_path: Path):
    """Validate model is custom-trained, not pretrained COCO."""
    print("=" * 60)
    print("MODEL VALIDATION")
    print("=" * 60)
    
    if not model_path.exists():
        print(f"❌ ERROR: Model not found at {model_path}")
        return None
    
    print(f"Loading model from: {model_path}")
    model = YOLO(str(model_path))
    
    num_classes = len(model.names)
    class_names = list(model.names.values())
    
    print(f"✓ Model loaded successfully")
    print(f"  Number of classes: {num_classes}")
    print(f"  Class names: {class_names}")
    
    # Check for COCO classes
    coco_classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 
                   'bus', 'train', 'truck', 'boat', 'traffic light']
    is_coco = sum(1 for c in coco_classes if c in class_names) >= 5
    
    if is_coco:
        print("\n❌ BLOCKED: This appears to be a PRETRAINED COCO model!")
        print("   This would detect ALL 80 object types.")
        print("   Use your custom-trained model instead!")
        return None
    
    if num_classes > 10:
        print(f"\n⚠️  WARNING: Model has {num_classes} classes (might be pretrained)")
    
    print(f"\n✓ Model validated: {num_classes} custom classes")
    print("=" * 60)
    return model


def run_detection_test(model):
    """Run detection with class filtering."""
    print("\nCLASS FILTER ACTIVE: Only detecting", ALLOWED_CLASSES)
    print("Press 'q' to quit\n")
    
    # Try RealSense first
    try:
        import pyrealsense2 as rs
        
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        
        profile = pipeline.start(config)
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        align = rs.align(rs.stream.color)
        
        print("✓ RealSense camera connected")
        use_realsense = True
        
    except Exception as e:
        print(f"RealSense not available: {e}")
        print("Trying OpenCV camera...")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ No camera available")
            return
        use_realsense = False
        print("✓ OpenCV camera connected")
    
    try:
        while True:
            # Get frame
            if use_realsense:
                frames = pipeline.wait_for_frames()
                frames = align.process(frames)
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                
                if not color_frame:
                    continue
                    
                frame = np.asanyarray(color_frame.get_data())
                depth = np.asanyarray(depth_frame.get_data()) if depth_frame else None
            else:
                ret, frame = cap.read()
                if not ret:
                    continue
                depth = None
            
            # Run detection
            results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
            
            annotated = frame.copy()
            filtered_count = 0
            total_count = 0
            
            if results and len(results) > 0:
                boxes = results[0].boxes
                
                for box in boxes:
                    class_id = int(box.cls[0])
                    class_name = results[0].names.get(class_id, f"class_{class_id}")
                    confidence = float(box.conf[0])
                    total_count += 1
                    
                    # APPLY CLASS FILTER - same as ROS 2 node
                    if class_name not in ALLOWED_CLASSES:
                        print(f"  [FILTERED OUT] {class_name} ({confidence:.2f})")
                        continue
                    
                    filtered_count += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Draw
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Get distance if depth available
                    dist_text = ""
                    if depth is not None:
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                        d = depth[cy, cx] * depth_scale if use_realsense else 0
                        if d > 0:
                            dist_text = f" | {d:.2f}m"
                    
                    label = f"{class_name}: {confidence:.2f}{dist_text}"
                    cv2.putText(annotated, label, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Status bar
            status = f"Detected: {filtered_count}/{total_count} | Filter: {ALLOWED_CLASSES} | Press 'q' to quit"
            cv2.putText(annotated, status, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            cv2.imshow("ROS 2 Detection Test", annotated)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        try:
            if use_realsense:
                pipeline.stop()
            else:
                cap.release()
        except:
            pass
        cv2.destroyAllWindows()
        print("\nTest completed.")


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("ROS 2 DETECTION NODE TEST")
    print("=" * 60 + "\n")
    
    # Validate model
    model = validate_model(MODEL_PATH)
    
    if model is None:
        print("\n❌ Model validation failed. Cannot proceed.")
        sys.exit(1)
    
    # Run detection test
    run_detection_test(model)
