import os
from typing import Tuple, List
import numpy as np
from PIL import Image
import random

def prepare_data_multiclass(
    data_dir: str = "data/dataset_v2/Training/",
    image_size: int = 128,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Carga, redimensiona y etiqueta imágenes de las 4 clases para clasificación multiclase.
    Realiza downsampling para balancear las clases (todas tendrán el mismo número de imágenes que la clase minoritaria).
    Guarda un log de las imágenes seleccionadas en results/graphics/downsampling_log.txt.
    Devuelve (X, y) como arrays de numpy.
    """
    import random
    import os
    random.seed(seed)
    class_map = {
        "glioma_tumor": 0,
        "meningioma_tumor": 1,
        "pituitary_tumor": 2,
        "no_tumor": 3
    }
    files_by_class = {}
    for class_name in class_map:
        class_dir = os.path.join(data_dir, class_name)
        files = [os.path.join(class_dir, f) for f in os.listdir(class_dir) if f.lower().endswith('.jpg')]
        files_by_class[class_name] = files
    # Downsampling: encontrar la clase minoritaria
    min_count = min(len(files) for files in files_by_class.values())
    selected_files_log = {}
    for class_name in files_by_class:
        files = files_by_class[class_name]
        random.shuffle(files)
        selected = []
        step = 10
        block = 2
        for i in range(0, len(files), step):
            selected.extend(files[i:i+block])
        selected = selected[:min_count]  # Por si se pasa del límite
        files_by_class[class_name] = selected
        selected_files_log[class_name] = selected
    # Guardar log
    os.makedirs('results/logs', exist_ok=True)
    with open('results/logs/downsampling_log.txt', 'w') as f:
        for class_name, files in selected_files_log.items():
            f.write(f"{class_name} ({len(files)}):\n")
            for file in files:
                f.write(f"    {file}\n")
            f.write("\n")
    X, y = [], []
    for class_name, label in class_map.items():
        for f in files_by_class[class_name]:
            img = Image.open(f).convert('L').resize((image_size, image_size))
            X.append(np.array(img))
            y.append(label)
    X = np.stack(X)
    y = np.array(y)
    return X, y 