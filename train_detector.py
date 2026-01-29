"""
YOLO Training Module
Trains YOLOv8 model on synthetic dataset.
"""

import os
from pathlib import Path
from ultralytics import YOLO
import torch

import config


class YOLOTrainer:
    """
    Train YOLOv8 model on synthetic STL-rendered dataset.
    """
    
    def __init__(self, dataset_path: str = None, model_type: str = None):
        """
        Initialize trainer.
        
        Args:
            dataset_path: Path to dataset directory
            model_type: YOLO model type (e.g., 'yolov8n.pt', 'yolov8s.pt')
        """
        self.dataset_path = Path(dataset_path) if dataset_path else config.DATASET_DIR
        self.model_type = model_type or config.YOLO_MODEL
        self.model = None
        self.trained_model_path = None
        
    def check_gpu(self) -> bool:
        """
        Check if GPU is available.
        
        Returns:
            True if CUDA is available
        """
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_count = torch.cuda.device_count()
            print(f"GPU Available: {gpu_name}")
            print(f"GPU Count: {gpu_count}")
        else:
            print("No GPU available, will use CPU")
        return cuda_available
    
    def validate_dataset(self) -> bool:
        """
        Validate that dataset is properly structured.
        
        Returns:
            True if dataset is valid
        """
        yaml_path = self.dataset_path / "dataset.yaml"
        train_images = self.dataset_path / "train" / "images"
        train_labels = self.dataset_path / "train" / "labels"
        val_images = self.dataset_path / "val" / "images"
        val_labels = self.dataset_path / "val" / "labels"
        
        checks = [
            (yaml_path.exists(), f"dataset.yaml exists at {yaml_path}"),
            (train_images.exists(), f"Train images directory exists"),
            (train_labels.exists(), f"Train labels directory exists"),
            (val_images.exists(), f"Val images directory exists"),
            (val_labels.exists(), f"Val labels directory exists"),
        ]
        
        # Count files
        train_img_count = len(list(train_images.glob("*.jpg"))) if train_images.exists() else 0
        train_lbl_count = len(list(train_labels.glob("*.txt"))) if train_labels.exists() else 0
        val_img_count = len(list(val_images.glob("*.jpg"))) if val_images.exists() else 0
        val_lbl_count = len(list(val_labels.glob("*.txt"))) if val_labels.exists() else 0
        
        print("\n=== Dataset Validation ===")
        all_valid = True
        for is_valid, message in checks:
            status = "✓" if is_valid else "✗"
            print(f"  {status} {message}")
            all_valid = all_valid and is_valid
        
        print(f"\n  Train images: {train_img_count}")
        print(f"  Train labels: {train_lbl_count}")
        print(f"  Val images: {val_img_count}")
        print(f"  Val labels: {val_lbl_count}")
        
        if train_img_count == 0:
            print("\n  ✗ No training images found!")
            all_valid = False
        
        if train_img_count != train_lbl_count:
            print("\n  ⚠ Mismatch between train images and labels count")
        
        return all_valid
    
    def train(self, 
              epochs: int = None,
              batch_size: int = None,
              image_size: int = None,
              device: str = None,
              patience: int = None,
              resume: bool = False) -> str:
        """
        Train the YOLO model.
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size
            image_size: Input image size
            device: Device to train on ('cpu', '0', 'cuda')
            patience: Early stopping patience
            resume: Resume from last checkpoint
            
        Returns:
            Path to best trained model
        """
        # Validate dataset
        if not self.validate_dataset():
            raise RuntimeError("Dataset validation failed. Please generate dataset first.")
        
        # Load model
        print(f"\nLoading {self.model_type}...")
        self.model = YOLO(self.model_type)
        
        # Determine device
        if device is None:
            device = config.DEVICE
        if device != 'cpu' and not torch.cuda.is_available():
            print("GPU not available, falling back to CPU")
            device = 'cpu'
        
        # Training parameters
        epochs = epochs or config.EPOCHS
        batch_size = batch_size or config.BATCH_SIZE
        image_size = image_size or config.IMAGE_SIZE
        patience = patience or config.PATIENCE
        
        print(f"\n=== Training Configuration ===")
        print(f"  Model: {self.model_type}")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Image size: {image_size}")
        print(f"  Device: {device}")
        print(f"  Early stopping patience: {patience}")
        
        # Train
        print("\n=== Starting Training ===")
        
        yaml_path = self.dataset_path / "dataset.yaml"
        
        results = self.model.train(
            data=str(yaml_path),
            epochs=epochs,
            imgsz=image_size,
            batch=batch_size,
            device=device,
            patience=patience,
            project=str(config.MODEL_DIR),
            name="stl_object",
            exist_ok=True,
            verbose=True,
            plots=True,
            save=True,
            resume=resume,
            lr0=config.LEARNING_RATE,
            workers=config.NUM_WORKERS,
        )
        
        # Get best model path
        self.trained_model_path = config.MODEL_DIR / "stl_object" / "weights" / "best.pt"
        
        print(f"\n=== Training Complete ===")
        print(f"Best model saved to: {self.trained_model_path}")
        
        return str(self.trained_model_path)
    
    def evaluate(self, model_path: str = None) -> dict:
        """
        Evaluate trained model on validation set.
        
        Args:
            model_path: Path to model weights
            
        Returns:
            Evaluation metrics
        """
        model_path = model_path or self.trained_model_path
        
        if model_path is None or not Path(model_path).exists():
            raise RuntimeError("No trained model found. Train model first.")
        
        print(f"\nEvaluating model: {model_path}")
        
        model = YOLO(model_path)
        yaml_path = self.dataset_path / "dataset.yaml"
        
        results = model.val(
            data=str(yaml_path),
            split='val',
            verbose=True
        )
        
        # Extract metrics
        metrics = {
            'mAP50': float(results.box.map50),
            'mAP50-95': float(results.box.map),
            'precision': float(results.box.mp),
            'recall': float(results.box.mr),
        }
        
        print("\n=== Evaluation Results ===")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
        
        return metrics
    
    def export(self, model_path: str = None, format: str = 'onnx') -> str:
        """
        Export model to different formats.
        
        Args:
            model_path: Path to model weights
            format: Export format ('onnx', 'torchscript', 'tflite', etc.)
            
        Returns:
            Path to exported model
        """
        model_path = model_path or self.trained_model_path
        
        if model_path is None or not Path(model_path).exists():
            raise RuntimeError("No trained model found. Train model first.")
        
        print(f"\nExporting model to {format}...")
        
        model = YOLO(model_path)
        export_path = model.export(format=format)
        
        print(f"Model exported to: {export_path}")
        
        return str(export_path)


def main():
    """Main function for training"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train YOLO detector")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=None, help="Batch size")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu, 0, cuda)")
    parser.add_argument("--resume", action="store_true", help="Resume training")
    parser.add_argument("--evaluate", action="store_true", help="Only evaluate")
    
    args = parser.parse_args()
    
    trainer = YOLOTrainer()
    trainer.check_gpu()
    
    if args.evaluate:
        trainer.evaluate()
    else:
        trainer.train(
            epochs=args.epochs,
            batch_size=args.batch,
            device=args.device,
            resume=args.resume
        )
        trainer.evaluate()


if __name__ == "__main__":
    main()
