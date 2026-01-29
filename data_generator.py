"""
Synthetic Data Generator Module
Generates training images by rendering STL meshes from multiple viewpoints.
"""

import os
import cv2
import numpy as np
import trimesh
import pyrender
from pathlib import Path
from typing import Tuple, List, Optional
from tqdm import tqdm
import random
import shutil

from stl_processor import STLProcessor
import config


class SyntheticDataGenerator:
    """
    Generate synthetic training data by rendering STL meshes.
    Creates YOLO-format dataset with images and labels.
    """
    
    def __init__(self, stl_path: str, output_dir: str = None):
        """
        Initialize the data generator.
        
        Args:
            stl_path: Path to STL file
            output_dir: Output directory for dataset
        """
        self.stl_path = stl_path
        self.output_dir = Path(output_dir) if output_dir else config.DATASET_DIR
        self.processor = STLProcessor(stl_path)
        self.renderer = None
        
        # Ensure output directories exist
        config.create_directories()
        
    def _create_renderer(self, width: int = None, height: int = None):
        """Create offscreen renderer"""
        w = width or config.RENDER_WIDTH
        h = height or config.RENDER_HEIGHT
        
        # Set environment variable for headless rendering
        os.environ['PYOPENGL_PLATFORM'] = 'egl'
        
        try:
            self.renderer = pyrender.OffscreenRenderer(w, h)
        except Exception as e:
            print(f"EGL renderer failed, trying osmesa: {e}")
            os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
            self.renderer = pyrender.OffscreenRenderer(w, h)
        
        return self.renderer
    
    def _create_camera_pose(self, distance: float, 
                            theta: float, phi: float) -> np.ndarray:
        """
        Create camera pose matrix looking at origin.
        
        Args:
            distance: Distance from origin
            theta: Azimuthal angle (0 to 2*pi)
            phi: Polar angle (0 to pi)
            
        Returns:
            4x4 camera pose matrix
        """
        # Spherical to Cartesian
        x = distance * np.sin(phi) * np.cos(theta)
        y = distance * np.sin(phi) * np.sin(theta)
        z = distance * np.cos(phi)
        
        # Camera position
        camera_pos = np.array([x, y, z])
        
        # Look-at matrix (looking at origin)
        forward = -camera_pos / np.linalg.norm(camera_pos)
        up = np.array([0, 0, 1])
        
        # Handle gimbal lock when looking straight up/down
        if np.abs(np.dot(forward, up)) > 0.99:
            up = np.array([0, 1, 0])
        
        right = np.cross(up, forward)
        right = right / (np.linalg.norm(right) + 1e-10)
        up = np.cross(forward, right)
        
        # Create pose matrix
        pose = np.eye(4)
        pose[:3, 0] = right
        pose[:3, 1] = up
        pose[:3, 2] = -forward  # OpenGL convention
        pose[:3, 3] = camera_pos
        
        return pose
    
    def _generate_background(self, width: int, height: int, 
                             bg_type: str = None) -> np.ndarray:
        """
        Generate random background image.
        
        Args:
            width: Image width
            height: Image height
            bg_type: Type of background ('solid', 'gradient', 'noise')
            
        Returns:
            Background image (RGB)
        """
        if bg_type is None:
            bg_type = random.choice(config.BACKGROUND_TYPES)
        
        if bg_type == 'solid':
            # Random solid color
            color = np.random.randint(30, 230, 3)
            bg = np.full((height, width, 3), color, dtype=np.uint8)
            
        elif bg_type == 'gradient':
            # Random gradient
            color1 = np.random.randint(30, 200, 3)
            color2 = np.random.randint(30, 200, 3)
            
            # Vertical or horizontal gradient
            if random.random() > 0.5:
                gradient = np.linspace(0, 1, height).reshape(-1, 1, 1)
            else:
                gradient = np.linspace(0, 1, width).reshape(1, -1, 1)
            
            bg = (color1 * (1 - gradient) + color2 * gradient).astype(np.uint8)
            if bg.shape[0] == 1:
                bg = np.tile(bg, (height, 1, 1))
            elif bg.shape[1] == 1:
                bg = np.tile(bg, (1, width, 1))
                
        elif bg_type == 'noise':
            # Random noise with color tint
            base_color = np.random.randint(50, 180, 3)
            noise = np.random.randint(-50, 50, (height, width, 3))
            bg = np.clip(base_color + noise, 0, 255).astype(np.uint8)
            
        else:
            # Default to solid gray
            bg = np.full((height, width, 3), 128, dtype=np.uint8)
        
        return bg
    
    def _apply_augmentations(self, image: np.ndarray) -> np.ndarray:
        """
        Apply random augmentations to image.
        
        Args:
            image: Input image (BGR)
            
        Returns:
            Augmented image
        """
        img = image.copy()
        
        # Gaussian blur
        if config.ENABLE_BLUR and random.random() < config.BLUR_PROBABILITY:
            kernel_size = random.choice(range(config.BLUR_KERNEL_SIZE[0], 
                                               config.BLUR_KERNEL_SIZE[1] + 1, 2))
            img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
        
        # Add noise
        if config.ENABLE_NOISE and random.random() < config.NOISE_PROBABILITY:
            noise_intensity = random.randint(*config.NOISE_INTENSITY)
            noise = np.random.normal(0, noise_intensity, img.shape).astype(np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Brightness adjustment
        if config.ENABLE_BRIGHTNESS and random.random() < config.BRIGHTNESS_PROBABILITY:
            brightness = random.randint(*config.BRIGHTNESS_RANGE)
            img = np.clip(img.astype(np.int16) + brightness, 0, 255).astype(np.uint8)
        
        # Contrast adjustment
        if config.ENABLE_CONTRAST and random.random() < config.CONTRAST_PROBABILITY:
            contrast = random.uniform(*config.CONTRAST_RANGE)
            mean = np.mean(img)
            img = np.clip((img - mean) * contrast + mean, 0, 255).astype(np.uint8)
        
        return img
    
    def _extract_bounding_box(self, mask: np.ndarray, 
                              width: int, height: int) -> Optional[Tuple[float, float, float, float]]:
        """
        Extract YOLO format bounding box from depth mask.
        
        Args:
            mask: Binary mask where object is present
            width: Image width
            height: Image height
            
        Returns:
            Tuple of (x_center, y_center, width, height) normalized to [0,1]
            or None if object not visible
        """
        if not np.any(mask):
            return None
        
        # Find bounding box coordinates
        coords = np.argwhere(mask)
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        # Add small padding
        padding = 2
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(width - 1, x_max + padding)
        y_max = min(height - 1, y_max + padding)
        
        # Convert to YOLO format (normalized center x, y, width, height)
        x_center = (x_min + x_max) / 2 / width
        y_center = (y_min + y_max) / 2 / height
        bbox_width = (x_max - x_min) / width
        bbox_height = (y_max - y_min) / height
        
        # Filter out very small objects
        if bbox_width < 0.01 or bbox_height < 0.01:
            return None
        
        return (x_center, y_center, bbox_width, bbox_height)
    
    def generate(self, num_images: int = None, 
                 show_progress: bool = True) -> str:
        """
        Generate synthetic training dataset.
        
        Args:
            num_images: Number of images to generate
            show_progress: Show progress bar
            
        Returns:
            Path to dataset directory
        """
        num_images = num_images or config.NUM_SYNTHETIC_IMAGES
        
        print(f"Generating {num_images} synthetic images...")
        
        # Load and normalize mesh
        self.processor.load()
        self.processor.normalize()
        
        # Create renderer
        width, height = config.RENDER_WIDTH, config.RENDER_HEIGHT
        self._create_renderer(width, height)
        
        # Temporary storage for all images
        all_images = []
        all_labels = []
        
        # Generate images
        iterator = tqdm(range(num_images)) if show_progress else range(num_images)
        
        for i in iterator:
            try:
                # Create mesh variation (without color changes that cause issues)
                mesh = self.processor.create_rotated_variation()
                scale = random.uniform(config.OBJECT_SCALE_MIN, config.OBJECT_SCALE_MAX)
                mesh.apply_scale(scale)
                
                # Apply a random single color using vertex colors (more compatible)
                color = np.random.randint(80, 255, 3).tolist() + [255]
                vertex_colors = np.tile(color, (len(mesh.vertices), 1)).astype(np.uint8)
                mesh.visual.vertex_colors = vertex_colors
                
                # Create pyrender scene
                scene = pyrender.Scene(
                    ambient_light=[config.AMBIENT_LIGHT_INTENSITY] * 3,
                    bg_color=[0, 0, 0, 0]  # Transparent background
                )
                
                # Add mesh to scene with smooth=False to avoid face color issues
                mesh_pyrender = pyrender.Mesh.from_trimesh(mesh, smooth=False)
                scene.add(mesh_pyrender)
                
                # Random camera position
                distance = random.uniform(config.CAMERA_DISTANCE_MIN, 
                                         config.CAMERA_DISTANCE_MAX)
                theta = random.uniform(0, 2 * np.pi)
                phi = random.uniform(0.1, np.pi - 0.1)  # Avoid poles
                
                camera_pose = self._create_camera_pose(distance, theta, phi)
                
                # Add camera
                camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
                scene.add(camera, pose=camera_pose)
                
                # Add lights (random positions and intensities)
                num_lights = random.randint(1, 3)
                for _ in range(num_lights):
                    light_intensity = random.uniform(
                        config.DIRECT_LIGHT_INTENSITY_MIN,
                        config.DIRECT_LIGHT_INTENSITY_MAX
                    )
                    light_color = [
                        random.uniform(0.8, 1.0),
                        random.uniform(0.8, 1.0),
                        random.uniform(0.8, 1.0)
                    ]
                    light = pyrender.DirectionalLight(
                        color=light_color,
                        intensity=light_intensity
                    )
                    
                    # Random light direction
                    light_theta = random.uniform(0, 2 * np.pi)
                    light_phi = random.uniform(0.2, np.pi / 2)
                    light_pose = self._create_camera_pose(1.0, light_theta, light_phi)
                    scene.add(light, pose=light_pose)
                
                # Render
                color, depth = self.renderer.render(scene)
                
                # Create mask from depth
                mask = (depth > 0).astype(np.uint8)
                
                # Extract bounding box
                bbox = self._extract_bounding_box(mask, width, height)
                
                if bbox is None:
                    continue  # Skip if object not visible
                
                # Generate background
                background = self._generate_background(width, height)
                
                # Composite object onto background
                mask_3d = mask[:, :, np.newaxis]
                final_image = color * mask_3d + background * (1 - mask_3d)
                final_image = final_image.astype(np.uint8)
                
                # Convert RGB to BGR for OpenCV
                final_image = cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR)
                
                # Apply augmentations
                final_image = self._apply_augmentations(final_image)
                
                # Store image and label
                all_images.append(final_image)
                all_labels.append(bbox)
                
            except Exception as e:
                if show_progress:
                    print(f"\nWarning: Error generating image {i}: {e}")
                continue
        
        # Cleanup renderer
        self.renderer.delete()
        
        print(f"Successfully generated {len(all_images)} images")
        
        # Split into train/val
        self._save_dataset(all_images, all_labels)
        
        # Create dataset.yaml
        yaml_path = self.output_dir / "dataset.yaml"
        with open(yaml_path, 'w') as f:
            f.write(config.get_dataset_yaml_content())
        
        print(f"Dataset saved to: {self.output_dir}")
        print(f"Dataset config: {yaml_path}")
        
        return str(self.output_dir)
    
    def _save_dataset(self, images: List[np.ndarray], 
                      labels: List[Tuple[float, float, float, float]]):
        """
        Save images and labels with train/val split.
        
        Args:
            images: List of images
            labels: List of bounding boxes
        """
        # Shuffle data
        combined = list(zip(images, labels))
        random.shuffle(combined)
        images, labels = zip(*combined)
        
        # Split
        split_idx = int(len(images) * config.TRAIN_VAL_SPLIT)
        
        train_images = images[:split_idx]
        train_labels = labels[:split_idx]
        val_images = images[split_idx:]
        val_labels = labels[split_idx:]
        
        print(f"Train set: {len(train_images)} images")
        print(f"Val set: {len(val_images)} images")
        
        # Save train set
        for i, (img, bbox) in enumerate(zip(train_images, train_labels)):
            img_path = config.TRAIN_IMAGES_DIR / f"img_{i:05d}.jpg"
            label_path = config.TRAIN_LABELS_DIR / f"img_{i:05d}.txt"
            
            cv2.imwrite(str(img_path), img)
            with open(label_path, 'w') as f:
                f.write(f"0 {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")
        
        # Save val set
        for i, (img, bbox) in enumerate(zip(val_images, val_labels)):
            img_path = config.VAL_IMAGES_DIR / f"img_{i:05d}.jpg"
            label_path = config.VAL_LABELS_DIR / f"img_{i:05d}.txt"
            
            cv2.imwrite(str(img_path), img)
            with open(label_path, 'w') as f:
                f.write(f"0 {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")
    
    def preview_sample(self, num_samples: int = 5) -> List[np.ndarray]:
        """
        Generate and display sample images without saving.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            List of sample images
        """
        print(f"Generating {num_samples} preview samples...")
        
        # Load and normalize mesh
        if self.processor.normalized_mesh is None:
            self.processor.load()
            self.processor.normalize()
        
        # Create renderer
        width, height = config.RENDER_WIDTH, config.RENDER_HEIGHT
        self._create_renderer(width, height)
        
        samples = []
        
        for i in range(num_samples):
            try:
                # Create mesh variation (without color changes that cause issues)
                mesh = self.processor.create_rotated_variation()
                scale = random.uniform(config.OBJECT_SCALE_MIN, config.OBJECT_SCALE_MAX)
                mesh.apply_scale(scale)
                
                # Apply random vertex colors
                color = np.random.randint(80, 255, 3).tolist() + [255]
                vertex_colors = np.tile(color, (len(mesh.vertices), 1)).astype(np.uint8)
                mesh.visual.vertex_colors = vertex_colors
                
                # Create scene
                scene = pyrender.Scene(
                    ambient_light=[config.AMBIENT_LIGHT_INTENSITY] * 3
                )
                
                mesh_pyrender = pyrender.Mesh.from_trimesh(mesh, smooth=False)
                scene.add(mesh_pyrender)
                
                # Camera
                distance = random.uniform(config.CAMERA_DISTANCE_MIN, 
                                         config.CAMERA_DISTANCE_MAX)
                theta = random.uniform(0, 2 * np.pi)
                phi = random.uniform(0.3, np.pi - 0.3)
                
                camera_pose = self._create_camera_pose(distance, theta, phi)
                camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
                scene.add(camera, pose=camera_pose)
                
                # Light
                light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
                scene.add(light, pose=camera_pose)
                
                # Render
                color, depth = self.renderer.render(scene)
                
                # Background
                mask = (depth > 0).astype(np.uint8)
                background = self._generate_background(width, height)
                
                mask_3d = mask[:, :, np.newaxis]
                final_image = color * mask_3d + background * (1 - mask_3d)
                final_image = final_image.astype(np.uint8)
                
                # Draw bounding box
                bbox = self._extract_bounding_box(mask, width, height)
                if bbox:
                    x_center, y_center, w, h = bbox
                    x1 = int((x_center - w/2) * width)
                    y1 = int((y_center - h/2) * height)
                    x2 = int((x_center + w/2) * width)
                    y2 = int((y_center + h/2) * height)
                    
                    final_bgr = cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR)
                    cv2.rectangle(final_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    samples.append(final_bgr)
                
            except Exception as e:
                print(f"Error generating sample {i}: {e}")
                continue
        
        self.renderer.delete()
        
        # Display samples
        if samples:
            combined = np.hstack(samples[:5])
            cv2.imshow("Sample Previews", combined)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return samples


def main():
    """Main function for testing"""
    import sys
    
    if len(sys.argv) > 1:
        stl_path = sys.argv[1]
    else:
        # Create test cube if no STL provided
        print("No STL file provided, creating test cube...")
        test_mesh = trimesh.creation.box(extents=[1, 1, 1])
        stl_path = str(config.BASE_DIR / "test_cube.stl")
        test_mesh.export(stl_path)
        print(f"Test cube saved to: {stl_path}")
    
    # Generate dataset
    generator = SyntheticDataGenerator(stl_path)
    
    # Generate preview first
    # generator.preview_sample(5)
    
    # Generate full dataset
    generator.generate(num_images=100)  # Small number for testing


if __name__ == "__main__":
    main()
