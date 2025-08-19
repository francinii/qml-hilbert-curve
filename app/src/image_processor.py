import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
from PIL import Image

class ImageProcessor:
    """Handles 2D image loading and preprocessing."""

    def __init__(self):
        """Initializes the ImageProcessor."""
        pass

    #def load_and_resize(self, file_path: str, target_size: int = 128) -> np.ndarray:
    #    """
    #    Loads a 2D image, converts it to grayscale, and resizes it.

    #    Args:
    #        file_path (str): The path to the image file.
    #        target_size (int): The target width and height for the image.

    #    Returns:
    #        np.ndarray: The processed image as a NumPy array.
    #    """
    #    image = Image.open(file_path).convert("L")  # Convert to grayscale
    #    image = image.resize((target_size, target_size), Image.Resampling.LANCZOS)
    #    return np.array(image)

    def load_and_resize(self, image_array: np.ndarray, target_size: int = 128) -> np.ndarray:
        """
        Creates a PIL image from a NumPy array, converts it to grayscale, and resizes it.
        """
        # Create an image directly from the NumPy array
        image = Image.fromarray(image_array).convert("L") # ✅ CORRECT
        image = image.resize((target_size, target_size), Image.Resampling.LANCZOS)
        return np.array(image)


class NiftiProcessor:
    """Handles NIfTI file loading and 3D volume preprocessing."""

    def __init__(self):
        """Initializes the NiftiProcessor."""
        pass

    def load_volume(self, file_path: str) -> tuple[np.ndarray, tuple[int, ...]]:
        """
        Loads a NIfTI file and returns its data volume and original shape.

        Args:
            file_path (str): The path to the NIfTI file.

        Returns:
            A tuple containing:
                - np.ndarray: The NIfTI data volume.
                - tuple[int, ...]: The original shape of the volume.
        """
        nifti_img = nib.load(file_path)
        volume = nifti_img.get_fdata()
        return volume, volume.shape

    def _make_cubic(self, data: np.ndarray) -> np.ndarray:
        """
        Crops a 3D array to a cubic shape using its smallest dimension.

        Args:
            data (np.ndarray): The input 3D NumPy array.

        Returns:
            np.ndarray: The cropped, cubic NumPy array.
        """
        min_dim = min(data.shape)
        return data[:min_dim, :min_dim, :min_dim]

    def _resize_to_power_of_2(
        self, data: np.ndarray, target_size: int
    ) -> np.ndarray:
        """
        Resizes a 3D array to a target cubic size using nearest-neighbor interpolation.

        Args:
            data (np.ndarray): The input 3D NumPy array, assumed to be cubic.
            target_size (int): The target size for each dimension.

        Returns:
            np.ndarray: The resized 3D NumPy array.
        """
        if data.shape[0] == target_size:
            return data
        scale_factor = target_size / data.shape[0]
        return zoom(data, scale_factor, order=0)  # order=0 is nearest neighbor

    def preprocess(self, volume: np.ndarray, target_size: int = 64) -> np.ndarray:
        """
        Preprocesses a 3D volume for Hilbert transformation.
        This involves cropping to a cube and resizing to a power-of-2 dimension.

        Args:
            volume (np.ndarray): The input 3D NIfTI volume.
            target_size (int): The final cubic dimension size. Must be a power of 2.

        Returns:
            np.ndarray: The processed 3D volume.
        """
        processed_volume = self._make_cubic(volume)
        processed_volume = self._resize_to_power_of_2(processed_volume, target_size)
        return processed_volume

