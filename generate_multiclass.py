"""
Multi-class data generation script.
Generates synthetic data for both bottle and mouse with separate class labels.
"""

import sys
sys.path.insert(0, '/home/rohit/Desktop/STL')

from data_generator import SyntheticDataGenerator
import config
from pathlib import Path
import shutil
import os

# Objects to train
OBJECTS = [
    {"stl": "bottle.stl", "class_id": 0, "class_name": "bottle", "num_images": 250},
    {"stl": "mouse.stl", "class_id": 1, "class_name": "mouse", "num_images": 250},
]

def generate_multi_class_data():
    base_dir = Path("/home/rohit/Desktop/STL")
    train_images = base_dir / "dataset/train/images"
    train_labels = base_dir / "dataset/train/labels"
    val_images = base_dir / "dataset/val/images"
    val_labels = base_dir / "dataset/val/labels"
    
    # Ensure directories exist
    for d in [train_images, train_labels, val_images, val_labels]:
        d.mkdir(parents=True, exist_ok=True)
    
    img_counter = 0
    val_counter = 0
    
    for obj in OBJECTS:
        print(f"\n=== Generating data for {obj['class_name']} ===")
        stl_path = base_dir / obj['stl']
        
        if not stl_path.exists():
            print(f"STL file not found: {stl_path}")
            continue
        
        # Generate to temporary directory
        temp_dir = base_dir / f"temp_{obj['class_name']}"
        temp_dir.mkdir(exist_ok=True)
        
        generator = SyntheticDataGenerator(
            stl_path=str(stl_path),
            output_dir=str(temp_dir)
        )
        
        count = generator.generate(
            num_images=obj['num_images'],
            show_progress=True
        )
        
        print(f"Generated {count} images for {obj['class_name']}")
        
        # Move and relabel train images
        temp_train_imgs = temp_dir / "train/images"
        temp_train_lbls = temp_dir / "train/labels"
        
        if temp_train_imgs.exists():
            for img_path in sorted(temp_train_imgs.glob("*.jpg")):
                # Copy image with new name
                new_img_name = f"img_{img_counter:05d}.jpg"
                shutil.copy(img_path, train_images / new_img_name)
                
                # Update label with correct class ID
                label_path = temp_train_lbls / (img_path.stem + ".txt")
                if label_path.exists():
                    with open(label_path, 'r') as f:
                        lines = f.readlines()
                    
                    new_label = train_labels / f"img_{img_counter:05d}.txt"
                    with open(new_label, 'w') as f:
                        for line in lines:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                # Replace class ID with correct one
                                parts[0] = str(obj['class_id'])
                                f.write(' '.join(parts) + '\n')
                
                img_counter += 1
        
        # Move and relabel val images
        temp_val_imgs = temp_dir / "val/images"
        temp_val_lbls = temp_dir / "val/labels"
        
        if temp_val_imgs.exists():
            for img_path in sorted(temp_val_imgs.glob("*.jpg")):
                # Copy image with new name
                new_img_name = f"val_{val_counter:05d}.jpg"
                shutil.copy(img_path, val_images / new_img_name)
                
                # Update label with correct class ID
                label_path = temp_val_lbls / (img_path.stem + ".txt")
                if label_path.exists():
                    with open(label_path, 'r') as f:
                        lines = f.readlines()
                    
                    new_label = val_labels / f"val_{val_counter:05d}.txt"
                    with open(new_label, 'w') as f:
                        for line in lines:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                parts[0] = str(obj['class_id'])
                                f.write(' '.join(parts) + '\n')
                
                val_counter += 1
        
        # Cleanup temp directory
        shutil.rmtree(temp_dir)
    
    # Create dataset.yaml for 2 classes
    yaml_content = f"""path: {base_dir / 'dataset'}
train: train/images
val: val/images

nc: 2
names: ['bottle', 'mouse']
"""
    
    with open(base_dir / "dataset/dataset.yaml", 'w') as f:
        f.write(yaml_content)
    
    print(f"\n=== Multi-class dataset created ===")
    print(f"Train images: {img_counter}")
    print(f"Val images: {val_counter}")
    print(f"Classes: bottle (0), mouse (1)")

if __name__ == "__main__":
    generate_multi_class_data()
