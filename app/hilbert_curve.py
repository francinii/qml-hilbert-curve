from typing import Tuple, Union, List
import numpy as np
from hilbertcurve.hilbertcurve import HilbertCurve  # type: ignore
import matplotlib.pyplot as plt
from PIL import Image 
import nibabel as nib  # type: ignore
from scipy.ndimage import zoom


class HilbertTransformer:
    """Handles Hilbert curve transformations for 2D and 3D data."""
    
    def __init__(self, dimensions: int):
        if dimensions not in [2, 3]:
            raise ValueError("Dimensions must be 2 or 3")
        self.dimensions = dimensions
    
    def _validate_power_of_2(self, size: int) -> int:
        """Validates and returns the order for a power of 2 size."""
        order = int(np.log2(size))
        if 2**order != size:
            raise ValueError(f"Size {size} must be a power of 2")
        return order
    
    def _validate_cubic_shape(self, data: np.ndarray) -> None:
        """Validates that data has cubic shape."""
        if not all(dim == data.shape[0] for dim in data.shape):
            raise ValueError("Data must have cubic shape")

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transforms data to 1D vector using Hilbert curve."""
        if len(data.shape) != self.dimensions:
            raise ValueError(f"Expected {self.dimensions}D data, got {len(data.shape)}D")
        
        self._validate_cubic_shape(data)
        order = self._validate_power_of_2(data.shape[0])
        
        hilbert = HilbertCurve(order, self.dimensions)
        num_points = 2**(self.dimensions * order)
        
        coords = [hilbert.point_from_distance(i) for i in range(num_points)]
        
        if self.dimensions == 2:
            return np.array([data[x, y] for x, y in coords])
        else:
            return np.array([data[x, y, z] for x, y, z in coords])


class ImageProcessor:
    """Handles image processing and resizing operations."""
    
    @staticmethod
    def load_image(file_path: str, target_size: int = 128) -> np.ndarray:
        """Loads and resizes an image to target size."""
        image = Image.open(file_path).convert("L")
        image = image.resize((target_size, target_size))
        return np.array(image)
    
    @staticmethod
    def resize_to_power_of_2(data: np.ndarray, target_size: int) -> np.ndarray:
        """Resizes data to target size using nearest neighbor interpolation."""
        if data.shape[0] == target_size:
            return data
        
        scale_factor = target_size / data.shape[0]
        return zoom(data, scale_factor, order=0)
    
    @staticmethod
    def make_cubic(data: np.ndarray) -> np.ndarray:
        """Crops data to cubic shape using minimum dimension."""
        min_dim = min(data.shape)
        return data[:min_dim, :min_dim, :min_dim]


class NiftiProcessor:
    """Handles NIfTI file loading and preprocessing."""
    
    @staticmethod
    def load_volume(file_path: str) -> Tuple[np.ndarray, Tuple[int, ...]]:
        """Loads NIfTI file and returns volume data with original shape."""
        nifti_img = nib.load(file_path)  # type: ignore
        volume = nifti_img.get_fdata()  # type: ignore
        return volume, volume.shape
    
    @staticmethod
    def validate_3d_volume(volume: np.ndarray) -> None:
        """Validates that volume is 3D."""
        if len(volume.shape) != 3:
            raise ValueError(f"Expected 3D volume, got shape {volume.shape}")


class HilbertCurveProcessor:
    """Main class for processing data with Hilbert curve transformations."""
    
    def __init__(self):
        self.image_transformer = HilbertTransformer(2)
        self.volume_transformer = HilbertTransformer(3)
        self.image_processor = ImageProcessor()
        self.nifti_processor = NiftiProcessor()
    
    def process_image(self, file_path: str, target_size: int = 128) -> np.ndarray:
        """Processes 2D image with Hilbert curve transformation."""
        image = self.image_processor.load_image(file_path, target_size)
        return self.image_transformer.transform(image)
    
    def process_nifti(
        self, 
        file_path: str, 
        resize: bool = True, 
        target_size: int = 64, 
        channel: int = 0
    ) -> Tuple[np.ndarray, Tuple[int, ...], Tuple[int, ...]]:
        """Processes 3D NIfTI volume with Hilbert curve transformation."""
        volume, original_shape = self.nifti_processor.load_volume(file_path)
        # If 4D, select the specified channel
        if len(volume.shape) == 4:
            volume = volume[..., channel]
        self.nifti_processor.validate_3d_volume(volume)
        
        if resize:
            volume = self.image_processor.make_cubic(volume)
            volume = self.image_processor.resize_to_power_of_2(volume, target_size)
        
        vector = self.volume_transformer.transform(volume)
        return vector, original_shape, volume.shape


def main():
    """Example usage of the HilbertCurveProcessor."""
    processor = HilbertCurveProcessor()
    
    # Process 2D image
    image_vector = processor.process_image("MNIST_32.png")
    print(f"Image vector length: {len(image_vector)}")
    print(f"Image vector sample: {image_vector[600:760]}")
    
    # Process 3D NIfTI (uncomment when you have NIfTI files)
    vector, original_shape, resized_shape = processor.process_nifti("data/imagesTr/BRATS_001.nii.gz", channel=0)
    print(f"NIfTI vector length: {len(vector)}")


if __name__ == "__main__":
    main()
