"""
Setup script for STL Object Detection System
Industrial-ready package for deployment
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text() if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    requirements = [
        line.strip() 
        for line in requirements_path.read_text().splitlines() 
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="stl-object-detector",
    version="1.0.0",
    description="Industrial-grade STL object detection system with RealSense camera support",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="STL Detection Team",
    python_requires=">=3.8",
    
    py_modules=[
        "config",
        "data_generator",
        "logger",
        "main",
        "real_image_labeler",
        "realtime_detector",
        "stl_processor",
        "train_detector",
        "utils",
    ],
    
    install_requires=requirements,
    
    entry_points={
        "console_scripts": [
            "stl-detect=main:main",
        ],
    },
    
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Manufacturing",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    
    package_data={
        "": ["*.yaml", "*.yml", "*.stl"],
    },
    
    include_package_data=True,
)
