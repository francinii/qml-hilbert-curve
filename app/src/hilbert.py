from __future__ import annotations
import numpy as np
from hilbertcurve.hilbertcurve import HilbertCurve
from image_processor import ImageProcessor

class HilbertTransformer:
    """
    Handles the mapping of 2D or 3D data to a 1D vector using a Hilbert curve.

    The Hilbert space-filling curve is used to preserve data locality, meaning
    points that are close in the multi-dimensional space are likely to be
    close in the resulting 1D vector.
    """

    def __init__(self, dimensions: int):
        """
        Initializes the HilbertTransformer.

        Args:
            dimensions (int): The number of dimensions of the data (must be 2 or 3).

        Raises:
            ValueError: If dimensions are not 2 or 3.
        """
        if dimensions not in [2, 3]:
            raise ValueError("Dimensions must be 2 or 3.")
        self.dimensions = dimensions

    def _validate_power_of_2(self, size: int) -> int:
        """
        Validates that the input size is a power of 2 and returns its order.
        The order 'p' is such that size = 2^p.

        Args:
            size (int): The size of one dimension of the data.

        Returns:
            int: The order of the Hilbert curve.

        Raises:
            ValueError: If the size is not a perfect power of 2.
        """
        if size <= 0 or (size & (size - 1)) != 0:
            raise ValueError(f"Size '{size}' must be a power of 2.")
        return int(np.log2(size))

    def _validate_cubic_shape(self, data: np.ndarray) -> None:
        """
        Validates that the input data array has a cubic shape (all dimensions are equal).

        Args:
            data (np.ndarray): The input data array.

        Raises:
            ValueError: If the data does not have a cubic shape.
        """
        if not all(dim == data.shape[0] for dim in data.shape):
            raise ValueError("Data must have a cubic shape (e.g., 64x64x64).")

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Transforms a multi-dimensional array into a 1D vector using the Hilbert curve.

        The input data must have a cubic shape with side lengths that are a power of 2.

        Args:
            data (np.ndarray): The 2D or 3D input data array.

        Returns:
            np.ndarray: A 1D vector containing the data mapped along the Hilbert curve.

        Raises:
            ValueError: If the data's dimensions do not match the transformer's dimensions.
        """
        if data.ndim != self.dimensions:
            raise ValueError(
                f"Expected {self.dimensions}D data, but got {data.ndim}D data."
            )

        self._validate_cubic_shape(data)
        order = self._validate_power_of_2(data.shape[0])

        hilbert_curve = HilbertCurve(order, self.dimensions)
        num_points = data.size

        # Generate coordinates along the Hilbert curve
        coords = [hilbert_curve.point_from_distance(i) for i in range(num_points)]

        # Map the data from the multi-dimensional array to the 1D vector
        if self.dimensions == 2:
            return np.array([data[x, y] for x, y in coords])
        else:  # self.dimensions == 3
            return np.array([data[x, y, z] for x, y, z in coords])


class HilbertCurveProcessor:
    """
    Main class for orchestrating data processing with Hilbert curve transformations.
    """

    def __init__(self):
        """Initializes all necessary processors and transformers."""
        self.image_transformer = HilbertTransformer(2) #two dimensions
        self.image_processor = ImageProcessor()


    def process_image(self, file_path: str, target_size: int = 128) -> np.ndarray:
        """
        Loads, preprocesses, and transforms a 2D image into a 1D vector.

        Args:
            file_path (str): The path to the image file.
            target_size (int): The size to which the image will be resized.
                               Must be a power of 2.

        Returns:
            np.ndarray: The resulting 1D data vector.
        """
        image_data = self.image_processor.load_and_resize(file_path, target_size)
        return self.image_transformer.transform(image_data)


    def process_image_batch(self, file_paths: List[str], target_size: int = 128) -> np.ndarray:
        """
        Processes a batch of images from a list of file paths.

        Itera sobre cada ruta de archivo, aplica la transformación de Hilbert
        y devuelve todos los vectores resultantes apilados en un único array.

        Args:
            file_paths (List[str]): Una lista de rutas a los archivos de imagen.
            target_size (int): El tamaño al que cada imagen será redimensionada.
                               Debe ser una potencia de 2.

        Returns:
            np.ndarray: Un array 2D de NumPy donde cada fila es el vector 1D
                        procesado de la imagen correspondiente.
        """
        # Usamos una "list comprehension" para procesar cada imagen de forma concisa
        processed_vectors = [self.process_image(path, target_size) for path in file_paths]
        
        # Convertimos la lista de vectores en un único array de NumPy
        return np.array(processed_vectors)   


def main():
    """
    Demonstrates the usage of the HilbertCurveProcessor with example files.
    """
    processor = HilbertCurveProcessor()

    # --- Process a 2D image ---
    # The image must be in the same directory or a valid path must be provided.
    # The target size must be a power of 2 (e.g., 32, 64, 128).
    try:
        print("--- Processing 2D Image ---")
        image_vector = processor.process_image("../data/dataset_binary/Training/tumor/glioma_tumor_train_0004.jpg", target_size=512)
        print(f"Original image shape: (512, 512)")
        print(f"Transformed image vector length: {len(image_vector)}")
        print(f"Image vector sample (first 10 values): {image_vector[25000:26000].round(2)}")
        print("-" * 20)

    except FileNotFoundError:
        print("Skipping 2D image processing: '' not found.")
    except Exception as e:
        print(f"An error occurred during image processing: {e}")
