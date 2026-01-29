"""
Centralized Logging Module for Industrial Deployment
Provides structured logging with file rotation and console output.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler

import config


class DetectorLogger:
    """
    Industrial-grade logging for the STL object detection system.
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if DetectorLogger._initialized:
            return
        
        self.logger = logging.getLogger("STLDetector")
        self.logger.setLevel(logging.DEBUG)
        
        # Create logs directory
        self.log_dir = config.BASE_DIR / "logs"
        self.log_dir.mkdir(exist_ok=True)
        
        # Console handler (INFO and above)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_format)
        
        # File handler with rotation (DEBUG and above)
        log_file = self.log_dir / "detector.log"
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB per file
            backupCount=5,  # Keep 5 backup files
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        
        # Detection log (for tracking detections)
        detection_file = self.log_dir / "detections.log"
        self.detection_handler = RotatingFileHandler(
            detection_file,
            maxBytes=50*1024*1024,  # 50MB per file
            backupCount=10,
            encoding='utf-8'
        )
        self.detection_handler.setLevel(logging.INFO)
        detection_format = logging.Formatter(
            '%(asctime)s,%(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.detection_handler.setFormatter(detection_format)
        
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        
        DetectorLogger._initialized = True
        self.logger.info("Logging system initialized")
        self.logger.info(f"Log directory: {self.log_dir}")
    
    def debug(self, msg, *args, **kwargs):
        self.logger.debug(msg, *args, **kwargs)
    
    def info(self, msg, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)
    
    def warning(self, msg, *args, **kwargs):
        self.logger.warning(msg, *args, **kwargs)
    
    def error(self, msg, *args, **kwargs):
        self.logger.error(msg, *args, **kwargs)
    
    def critical(self, msg, *args, **kwargs):
        self.logger.critical(msg, *args, **kwargs)
    
    def log_detection(self, confidence: float, distance: float, 
                      x: float, y: float, z: float):
        """Log a detection event with coordinates."""
        msg = f"{confidence:.3f},{distance:.3f},{x:.3f},{y:.3f},{z:.3f}"
        record = logging.LogRecord(
            name="detection",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg=msg,
            args=(),
            exc_info=None
        )
        self.detection_handler.emit(record)
    
    def log_startup(self, model_path: str, camera_type: str):
        """Log system startup."""
        self.info("="*60)
        self.info("STL OBJECT DETECTION SYSTEM STARTING")
        self.info("="*60)
        self.info(f"Model: {model_path}")
        self.info(f"Camera: {camera_type}")
        self.info(f"Timestamp: {datetime.now().isoformat()}")
    
    def log_shutdown(self, reason: str = "Normal"):
        """Log system shutdown."""
        self.info("="*60)
        self.info(f"SYSTEM SHUTDOWN: {reason}")
        self.info("="*60)


# Global logger instance
def get_logger() -> DetectorLogger:
    """Get the global logger instance."""
    return DetectorLogger()
