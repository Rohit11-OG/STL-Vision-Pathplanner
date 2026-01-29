"""
STL Object Detection System
Main entry point with CLI interface.

Usage:
    python main.py generate --stl your_object.stl --num-images 1000
    python main.py train --epochs 100
    python main.py detect --camera 0
    python main.py detect --image test.jpg
    python main.py full --stl your_object.stl
"""

import argparse
import sys
from pathlib import Path

import config
from stl_processor import STLProcessor
from data_generator import SyntheticDataGenerator
from train_detector import YOLOTrainer
from realtime_detector import RealtimeDetector


def generate_data(args):
    """Generate synthetic training data"""
    print("\n" + "="*60)
    print("STEP 1: GENERATING SYNTHETIC TRAINING DATA")
    print("="*60)
    
    stl_path = args.stl or str(config.STL_FILE)
    
    if not Path(stl_path).exists():
        print(f"\nError: STL file not found: {stl_path}")
        print("Please provide a valid STL file with --stl option")
        return False
    
    generator = SyntheticDataGenerator(stl_path)
    
    # Preview samples first
    if args.preview:
        generator.preview_sample(5)
    
    # Generate full dataset
    generator.generate(num_images=args.num_images)
    
    print("\nSynthetic data generation complete!")
    return True


def train_model(args):
    """Train YOLO detector"""
    print("\n" + "="*60)
    print("STEP 2: TRAINING YOLO DETECTOR")
    print("="*60)
    
    trainer = YOLOTrainer()
    
    # Check GPU
    trainer.check_gpu()
    
    # Train
    model_path = trainer.train(
        epochs=args.epochs,
        batch_size=args.batch,
        device=args.device,
        resume=args.resume
    )
    
    # Evaluate
    trainer.evaluate(model_path)
    
    print("\nTraining complete!")
    return True


def run_detection(args):
    """Run real-time detection or detect in image"""
    print("\n" + "="*60)
    print("STEP 3: RUNNING OBJECT DETECTION")
    print("="*60)
    
    detector = RealtimeDetector(
        model_path=args.model,
        use_realsense=not args.no_realsense
    )
    
    if args.image:
        # Single image detection
        detector.detect_image(
            args.image,
            confidence=args.confidence,
            save_output=True
        )
    else:
        # Real-time detection
        detector.run(
            camera_id=args.camera,
            confidence=args.confidence
        )
    
    return True


def run_path_generation(args):
    """Run detection with tool path generation"""
    print("\n" + "="*60)
    print("TOOL PATH GENERATION MODE")
    print("="*60)
    print(f"\nStrategy: {args.strategy}")
    print(f"Waypoints: {args.num_points}")
    print("\nControls:")
    print("  p     - Generate tool path from current detection")
    print("  1-6   - Quick strategy (1=contour 2=approach 3=grid")
    print("                         4=surface 5=spiral 6=zigzag)")
    print("  v     - Toggle path visualization")
    print("  r     - Reload settings.yaml")
    print("  s     - Save frame")
    print("  +/-   - Adjust confidence threshold")
    print("  q     - Quit")
    print("\n" + "="*60)
    
    detector = RealtimeDetector(
        model_path=args.model,
        use_realsense=not args.no_realsense
    )
    
    # Set strategy for path generation
    detector.current_strategy = args.strategy
    config.DEFAULT_PATH_STRATEGY = args.strategy
    config.NUM_PATH_POINTS = args.num_points
    
    # Run detection (path generation available via 'p' key)
    detector.run(
        camera_id=args.camera,
        confidence=args.confidence
    )
    
    return True


def full_pipeline(args):
    """Run full pipeline: generate -> train -> detect"""
    print("\n" + "="*60)
    print("FULL PIPELINE: STL TO REAL-TIME DETECTION")
    print("="*60)
    
    # Step 1: Generate data
    stl_path = args.stl or str(config.STL_FILE)
    
    if not Path(stl_path).exists():
        print(f"\nError: STL file not found: {stl_path}")
        print("Please provide a valid STL file with --stl option")
        return False
    
    print("\n[1/3] Generating synthetic training data...")
    generator = SyntheticDataGenerator(stl_path)
    generator.generate(num_images=args.num_images)
    
    # Step 2: Train
    print("\n[2/3] Training detector...")
    trainer = YOLOTrainer()
    trainer.check_gpu()
    trainer.train(
        epochs=args.epochs,
        batch_size=args.batch,
        device=args.device
    )
    trainer.evaluate()
    
    # Step 3: Detect
    print("\n[3/3] Starting real-time detection...")
    detector = RealtimeDetector(use_realsense=not args.no_realsense)
    detector.run(
        camera_id=args.camera,
        confidence=args.confidence
    )
    
    return True


def show_info(args):
    """Show system information"""
    import torch
    
    print("\n" + "="*60)
    print("STL OBJECT DETECTION SYSTEM - INFO")
    print("="*60)
    
    print(f"\nConfiguration:")
    print(f"  Base directory: {config.BASE_DIR}")
    print(f"  STL file: {config.STL_FILE}")
    print(f"  Dataset directory: {config.DATASET_DIR}")
    print(f"  Model directory: {config.MODEL_DIR}")
    
    print(f"\nTraining settings:")
    print(f"  YOLO model: {config.YOLO_MODEL}")
    print(f"  Epochs: {config.EPOCHS}")
    print(f"  Batch size: {config.BATCH_SIZE}")
    print(f"  Device: {config.DEVICE}")
    
    print(f"\nGPU Status:")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU count: {torch.cuda.device_count()}")
    
    # Check for existing model
    if config.BEST_MODEL_PATH.exists():
        print(f"\nTrained model found: {config.BEST_MODEL_PATH}")
    else:
        print(f"\nNo trained model found")
    
    # Check for dataset
    train_images = config.TRAIN_IMAGES_DIR
    if train_images.exists():
        num_train = len(list(train_images.glob("*.jpg")))
        print(f"Training images: {num_train}")
    else:
        print("No training dataset found")


def main():
    parser = argparse.ArgumentParser(
        description="STL Object Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate synthetic training data
  python main.py generate --stl my_object.stl --num-images 1000

  # Train the detector
  python main.py train --epochs 100 --device 0

  # Run real-time detection with RealSense
  python main.py detect --camera 0

  # Run full pipeline
  python main.py full --stl my_object.stl --num-images 500 --epochs 50
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate synthetic training data")
    gen_parser.add_argument("--stl", type=str, help="Path to STL file")
    gen_parser.add_argument("--num-images", type=int, default=1000, help="Number of images")
    gen_parser.add_argument("--preview", action="store_true", help="Preview samples first")
    gen_parser.set_defaults(func=generate_data)
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train YOLO detector")
    train_parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    train_parser.add_argument("--batch", type=int, default=None, help="Batch size")
    train_parser.add_argument("--device", type=str, default=None, help="Device (cpu, 0, cuda)")
    train_parser.add_argument("--resume", action="store_true", help="Resume training")
    train_parser.set_defaults(func=train_model)
    
    # Detect command
    detect_parser = subparsers.add_parser("detect", help="Run detection")
    detect_parser.add_argument("--model", type=str, default=None, help="Path to model")
    detect_parser.add_argument("--camera", type=int, default=0, help="Camera ID")
    detect_parser.add_argument("--image", type=str, default=None, help="Image path for single detection")
    detect_parser.add_argument("--confidence", type=float, default=config.CONFIDENCE_THRESHOLD, help="Confidence threshold")
    detect_parser.add_argument("--no-realsense", action="store_true", help="Don't use RealSense")
    detect_parser.set_defaults(func=run_detection)
    
    # Full pipeline command
    full_parser = subparsers.add_parser("full", help="Run full pipeline")
    full_parser.add_argument("--stl", type=str, help="Path to STL file")
    full_parser.add_argument("--num-images", type=int, default=500, help="Number of images")
    full_parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    full_parser.add_argument("--batch", type=int, default=16, help="Batch size")
    full_parser.add_argument("--device", type=str, default=None, help="Device")
    full_parser.add_argument("--camera", type=int, default=0, help="Camera ID")
    full_parser.add_argument("--confidence", type=float, default=config.CONFIDENCE_THRESHOLD, help="Confidence threshold")
    full_parser.add_argument("--no-realsense", action="store_true", help="Don't use RealSense")
    full_parser.set_defaults(func=full_pipeline)
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Show system info")
    info_parser.set_defaults(func=show_info)
    
    # Path generation command
    path_parser = subparsers.add_parser("path", help="Run detection with tool path generation")
    path_parser.add_argument("--model", type=str, default=None, help="Path to model")
    path_parser.add_argument("--camera", type=int, default=0, help="Camera ID")
    path_parser.add_argument("--confidence", type=float, default=config.CONFIDENCE_THRESHOLD, help="Confidence threshold")
    path_parser.add_argument("--strategy", type=str, default=config.DEFAULT_PATH_STRATEGY, 
                            choices=config.PATH_STRATEGIES, help="Path strategy")
    path_parser.add_argument("--num-points", type=int, default=config.NUM_PATH_POINTS, help="Number of waypoints")
    path_parser.add_argument("--no-realsense", action="store_true", help="Don't use RealSense")
    path_parser.set_defaults(func=run_path_generation)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(0)
    
    # Create directories
    config.create_directories()
    
    # Run command
    try:
        success = args.func(args)
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()
