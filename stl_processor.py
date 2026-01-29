"""
STL File Processor Module
Handles loading, normalizing, and processing STL/3D mesh files.
"""

import numpy as np
import trimesh
from pathlib import Path
from typing import Tuple, Optional, List
import copy


class STLProcessor:
    """
    Process STL files for synthetic data generation.
    Handles mesh loading, normalization, and variations.
    """
    
    def __init__(self, stl_path: str):
        """
        Initialize STL processor.
        
        Args:
            stl_path: Path to the STL file
        """
        self.stl_path = Path(stl_path)
        self.original_mesh = None
        self.normalized_mesh = None
        self.mesh_info = {}
        
    def load(self) -> trimesh.Trimesh:
        """
        Load STL file and store mesh data.
        
        Returns:
            Loaded trimesh object
        """
        if not self.stl_path.exists():
            raise FileNotFoundError(f"STL file not found: {self.stl_path}")
        
        print(f"Loading STL file: {self.stl_path}")
        
        # Load mesh
        self.original_mesh = trimesh.load(str(self.stl_path))
        
        # Handle scene vs mesh
        if isinstance(self.original_mesh, trimesh.Scene):
            # Combine all meshes in scene
            meshes = []
            for name, geometry in self.original_mesh.geometry.items():
                if isinstance(geometry, trimesh.Trimesh):
                    meshes.append(geometry)
            if meshes:
                self.original_mesh = trimesh.util.concatenate(meshes)
            else:
                raise ValueError("No valid meshes found in STL file")
        
        # Store mesh info
        self.mesh_info = {
            'vertices': len(self.original_mesh.vertices),
            'faces': len(self.original_mesh.faces),
            'bounds': self.original_mesh.bounds.tolist(),
            'extents': self.original_mesh.extents.tolist(),
            'center': self.original_mesh.centroid.tolist(),
            'volume': float(self.original_mesh.volume) if self.original_mesh.is_watertight else None,
            'is_watertight': self.original_mesh.is_watertight
        }
        
        print(f"Mesh loaded successfully:")
        print(f"  - Vertices: {self.mesh_info['vertices']}")
        print(f"  - Faces: {self.mesh_info['faces']}")
        print(f"  - Extents: {self.mesh_info['extents']}")
        print(f"  - Watertight: {self.mesh_info['is_watertight']}")
        
        return self.original_mesh
    
    def normalize(self, target_size: float = 1.0) -> trimesh.Trimesh:
        """
        Normalize mesh to be centered at origin with specified size.
        
        Args:
            target_size: Target size (max extent) for the normalized mesh
            
        Returns:
            Normalized trimesh object
        """
        if self.original_mesh is None:
            self.load()
        
        # Create a copy for normalization
        self.normalized_mesh = copy.deepcopy(self.original_mesh)
        
        # Center at origin
        self.normalized_mesh.apply_translation(-self.normalized_mesh.centroid)
        
        # Scale to target size
        current_max_extent = np.max(self.normalized_mesh.extents)
        if current_max_extent > 0:
            scale_factor = target_size / current_max_extent
            self.normalized_mesh.apply_scale(scale_factor)
        
        print(f"Mesh normalized: centered at origin, max extent = {target_size}")
        
        return self.normalized_mesh
    
    def get_mesh(self, normalized: bool = True) -> trimesh.Trimesh:
        """
        Get the mesh object.
        
        Args:
            normalized: If True, return normalized mesh
            
        Returns:
            Trimesh object
        """
        if normalized:
            if self.normalized_mesh is None:
                self.normalize()
            return self.normalized_mesh
        else:
            if self.original_mesh is None:
                self.load()
            return self.original_mesh
    
    def create_color_variation(self, color: Tuple[int, int, int, int] = None) -> trimesh.Trimesh:
        """
        Create a copy of the mesh with a specific color.
        
        Args:
            color: RGBA color tuple (0-255 for each channel)
            
        Returns:
            Colored mesh copy
        """
        mesh = copy.deepcopy(self.get_mesh())
        
        if color is None:
            # Random color
            color = tuple(np.random.randint(50, 255, 3).tolist()) + (255,)
        
        # Convert to face colors visual to avoid smooth mesh issues
        try:
            # Create a new ColorVisuals with face colors
            face_colors = np.tile(color, (len(mesh.faces), 1)).astype(np.uint8)
            mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, face_colors=face_colors)
        except Exception:
            # Fallback: apply vertex colors instead
            vertex_colors = np.tile(color, (len(mesh.vertices), 1)).astype(np.uint8)
            mesh.visual.vertex_colors = vertex_colors
        
        return mesh
    
    def create_rotated_variation(self, 
                                  rotation_angles: Tuple[float, float, float] = None) -> trimesh.Trimesh:
        """
        Create a rotated copy of the mesh.
        
        Args:
            rotation_angles: Rotation angles (x, y, z) in radians. If None, random rotation.
            
        Returns:
            Rotated mesh copy
        """
        mesh = copy.deepcopy(self.get_mesh())
        
        if rotation_angles is None:
            # Random rotation
            rotation_angles = np.random.uniform(0, 2 * np.pi, 3)
        
        # Create rotation matrix
        rx = trimesh.transformations.rotation_matrix(rotation_angles[0], [1, 0, 0])
        ry = trimesh.transformations.rotation_matrix(rotation_angles[1], [0, 1, 0])
        rz = trimesh.transformations.rotation_matrix(rotation_angles[2], [0, 0, 1])
        
        # Apply rotations
        mesh.apply_transform(rx)
        mesh.apply_transform(ry)
        mesh.apply_transform(rz)
        
        return mesh
    
    def create_scaled_variation(self, scale_factor: float = None) -> trimesh.Trimesh:
        """
        Create a scaled copy of the mesh.
        
        Args:
            scale_factor: Scale factor. If None, random scale between 0.7 and 1.3.
            
        Returns:
            Scaled mesh copy
        """
        mesh = copy.deepcopy(self.get_mesh())
        
        if scale_factor is None:
            scale_factor = np.random.uniform(0.7, 1.3)
        
        mesh.apply_scale(scale_factor)
        
        return mesh
    
    def create_variation(self,
                        color: Tuple[int, int, int, int] = None,
                        rotation: Tuple[float, float, float] = None,
                        scale: float = None,
                        random_all: bool = False) -> trimesh.Trimesh:
        """
        Create a variation of the mesh with optional transformations.
        
        Args:
            color: RGBA color tuple
            rotation: Rotation angles (x, y, z) in radians
            scale: Scale factor
            random_all: If True, randomize all parameters
            
        Returns:
            Transformed mesh copy
        """
        mesh = copy.deepcopy(self.get_mesh())
        
        # Apply scale
        if scale is not None or random_all:
            s = scale if scale is not None else np.random.uniform(0.7, 1.3)
            mesh.apply_scale(s)
        
        # Apply rotation
        if rotation is not None or random_all:
            angles = rotation if rotation is not None else np.random.uniform(0, 2 * np.pi, 3)
            rx = trimesh.transformations.rotation_matrix(angles[0], [1, 0, 0])
            ry = trimesh.transformations.rotation_matrix(angles[1], [0, 1, 0])
            rz = trimesh.transformations.rotation_matrix(angles[2], [0, 0, 1])
            mesh.apply_transform(rx)
            mesh.apply_transform(ry)
            mesh.apply_transform(rz)
        
        # Apply color
        if color is not None or random_all:
            c = color if color is not None else tuple(np.random.randint(50, 255, 3).tolist()) + (255,)
            try:
                # Create a new ColorVisuals with face colors
                face_colors = np.tile(c, (len(mesh.faces), 1)).astype(np.uint8)
                mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, face_colors=face_colors)
            except Exception:
                # Fallback: apply vertex colors instead
                vertex_colors = np.tile(c, (len(mesh.vertices), 1)).astype(np.uint8)
                mesh.visual.vertex_colors = vertex_colors
        
        return mesh
    
    def get_info(self) -> dict:
        """
        Get mesh information.
        
        Returns:
            Dictionary with mesh properties
        """
        if not self.mesh_info:
            self.load()
        return self.mesh_info
    
    def visualize(self):
        """
        Open an interactive visualization of the mesh.
        """
        mesh = self.get_mesh()
        mesh.show()


def test_stl_processor():
    """Test function for STL processor"""
    import sys
    
    if len(sys.argv) > 1:
        stl_path = sys.argv[1]
    else:
        print("Usage: python stl_processor.py <path_to_stl_file>")
        print("No STL file provided, creating a test cube...")
        
        # Create a test cube
        test_mesh = trimesh.creation.box(extents=[1, 1, 1])
        test_path = Path(__file__).parent / "test_cube.stl"
        test_mesh.export(str(test_path))
        stl_path = str(test_path)
        print(f"Test cube saved to: {test_path}")
    
    # Test the processor
    processor = STLProcessor(stl_path)
    processor.load()
    processor.normalize()
    
    print("\nMesh Info:")
    for key, value in processor.get_info().items():
        print(f"  {key}: {value}")
    
    # Create variations
    print("\nCreating variations...")
    colored = processor.create_color_variation((255, 0, 0, 255))
    rotated = processor.create_rotated_variation()
    scaled = processor.create_scaled_variation(1.5)
    random_var = processor.create_variation(random_all=True)
    
    print("Variations created successfully!")


if __name__ == "__main__":
    test_stl_processor()
