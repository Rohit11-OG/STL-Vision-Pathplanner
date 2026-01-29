"""
Real Image Labeler and Trainer
Uses real captured images to train a better detector.
"""

import cv2
import numpy as np
from pathlib import Path
import os
import shutil

import config


class RealImageLabeler:
    """
    Interactive tool to label real images for training.
    """
    
    def __init__(self, image_dir: str = None, output_dir: str = None):
        self.image_dir = Path(image_dir) if image_dir else config.DETECTION_OUTPUT_DIR
        self.output_dir = Path(output_dir) if output_dir else config.DATASET_DIR / "real"
        self.current_image = None
        self.current_path = None
        self.drawing = False
        self.ix, self.iy = -1, -1
        self.boxes = []
        self.current_box = None
        
        # Create output directories
        os.makedirs(self.output_dir / "images", exist_ok=True)
        os.makedirs(self.output_dir / "labels", exist_ok=True)
    
    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for drawing bounding boxes"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix, self.iy = x, y
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.current_box = (self.ix, self.iy, x, y)
                
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            if abs(x - self.ix) > 10 and abs(y - self.iy) > 10:
                x1, y1 = min(self.ix, x), min(self.iy, y)
                x2, y2 = max(self.ix, x), max(self.iy, y)
                self.boxes.append((x1, y1, x2, y2))
            self.current_box = None
    
    def label_images(self):
        """
        Interactively label images with bounding boxes.
        
        Controls:
            - Draw box: Click and drag
            - Undo: Press 'u'
            - Save and next: Press 'n' or SPACE
            - Skip: Press 's'
            - Quit: Press 'q'
        """
        image_files = list(self.image_dir.glob("*.jpg")) + list(self.image_dir.glob("*.png"))
        
        if not image_files:
            print(f"No images found in {self.image_dir}")
            return
        
        print(f"Found {len(image_files)} images to label")
        print("\nControls:")
        print("  Draw box: Click and drag")
        print("  Undo last box: Press 'u'")
        print("  Save and next: Press 'n' or SPACE")
        print("  Skip image: Press 's'")
        print("  Quit: Press 'q'")
        
        cv2.namedWindow('Label Image')
        cv2.setMouseCallback('Label Image', self._mouse_callback)
        
        labeled_count = 0
        
        for img_path in image_files:
            self.current_path = img_path
            self.current_image = cv2.imread(str(img_path))
            self.boxes = []
            
            if self.current_image is None:
                continue
            
            h, w = self.current_image.shape[:2]
            
            while True:
                # Draw current state
                display = self.current_image.copy()
                
                # Draw completed boxes
                for box in self.boxes:
                    cv2.rectangle(display, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                
                # Draw current box being drawn
                if self.current_box:
                    cv2.rectangle(display, 
                                 (self.current_box[0], self.current_box[1]),
                                 (self.current_box[2], self.current_box[3]),
                                 (0, 255, 255), 2)
                
                # Show instructions
                cv2.putText(display, f"Boxes: {len(self.boxes)} | n=save s=skip u=undo q=quit",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(display, str(img_path.name),
                           (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                cv2.imshow('Label Image', display)
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    cv2.destroyAllWindows()
                    print(f"\nLabeled {labeled_count} images")
                    return
                    
                elif key == ord('u'):
                    if self.boxes:
                        self.boxes.pop()
                        
                elif key == ord('s'):
                    print(f"Skipped: {img_path.name}")
                    break
                    
                elif key == ord('n') or key == ord(' '):
                    if self.boxes:
                        # Save image and labels
                        self._save_labeled_image(img_path, self.boxes, w, h)
                        labeled_count += 1
                        print(f"Saved: {img_path.name} with {len(self.boxes)} boxes")
                    else:
                        print(f"No boxes drawn, skipping: {img_path.name}")
                    break
        
        cv2.destroyAllWindows()
        print(f"\nLabeling complete! Labeled {labeled_count} images")
        print(f"Saved to: {self.output_dir}")
    
    def _save_labeled_image(self, img_path: Path, boxes: list, width: int, height: int):
        """Save image and YOLO format labels"""
        # Copy image
        dest_img = self.output_dir / "images" / img_path.name
        shutil.copy(img_path, dest_img)
        
        # Save labels
        label_name = img_path.stem + ".txt"
        label_path = self.output_dir / "labels" / label_name
        
        with open(label_path, 'w') as f:
            for box in boxes:
                x1, y1, x2, y2 = box
                x_center = (x1 + x2) / 2 / width
                y_center = (y1 + y2) / 2 / height
                w = (x2 - x1) / width
                h = (y2 - y1) / height
                f.write(f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")


def auto_label_with_prompts(image_dir: str = None):
    """
    Simple script to help capture and label more training data.
    """
    import pyrealsense2 as rs
    
    save_dir = Path(image_dir) if image_dir else config.DETECTION_OUTPUT_DIR / "capture"
    os.makedirs(save_dir, exist_ok=True)
    
    # Try RealSense first, fallback to webcam
    try:
        pipeline = rs.pipeline()
        rs_config = rs.config()
        rs_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        pipeline.start(rs_config)
        use_realsense = True
        print("Using RealSense camera")
    except:
        cap = cv2.VideoCapture(0)
        use_realsense = False
        print("Using webcam")
    
    print("\nCapture training images:")
    print("  - Position the bottle in different poses/backgrounds")
    print("  - Press SPACE to capture")
    print("  - Press 'q' to quit")
    
    count = len(list(save_dir.glob("*.jpg")))
    
    try:
        while True:
            if use_realsense:
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                frame = np.asanyarray(color_frame.get_data())
            else:
                ret, frame = cap.read()
                if not ret:
                    continue
            
            display = frame.copy()
            cv2.putText(display, f"Captured: {count} | SPACE=capture q=quit",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("Capture Training Data", display)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord(' '):
                filename = save_dir / f"train_{count:04d}.jpg"
                cv2.imwrite(str(filename), frame)
                print(f"Saved: {filename}")
                count += 1
                
    finally:
        if use_realsense:
            pipeline.stop()
        else:
            cap.release()
        cv2.destroyAllWindows()
    
    print(f"\nCaptured {count} images in {save_dir}")
    print("Now run labeling with: python real_image_labeler.py label")


def combine_datasets():
    """Combine synthetic and real datasets for training"""
    real_dir = config.DATASET_DIR / "real"
    
    if not real_dir.exists():
        print("No real data found. Capture and label real images first.")
        return
    
    # Count real images
    real_images = list((real_dir / "images").glob("*.jpg"))
    print(f"Found {len(real_images)} real labeled images")
    
    # Copy real images to training set (adding to existing synthetic)
    train_images_dir = config.TRAIN_IMAGES_DIR
    train_labels_dir = config.TRAIN_LABELS_DIR
    
    # Find next index
    existing = list(train_images_dir.glob("*.jpg"))
    start_idx = len(existing)
    
    for i, img_path in enumerate(real_images):
        new_idx = start_idx + i
        
        # Copy image
        dest_img = train_images_dir / f"img_{new_idx:05d}.jpg"
        shutil.copy(img_path, dest_img)
        
        # Copy label
        label_name = img_path.stem + ".txt"
        src_label = real_dir / "labels" / label_name
        dest_label = train_labels_dir / f"img_{new_idx:05d}.txt"
        
        if src_label.exists():
            shutil.copy(src_label, dest_label)
    
    print(f"Added {len(real_images)} real images to training set")
    print(f"Total training images: {len(list(train_images_dir.glob('*.jpg')))}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python real_image_labeler.py capture  - Capture new images")
        print("  python real_image_labeler.py label    - Label captured images")
        print("  python real_image_labeler.py combine  - Add real data to training set")
    else:
        command = sys.argv[1]
        
        if command == "capture":
            auto_label_with_prompts()
        elif command == "label":
            labeler = RealImageLabeler()
            labeler.label_images()
        elif command == "combine":
            combine_datasets()
        else:
            print(f"Unknown command: {command}")
