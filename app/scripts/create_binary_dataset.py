import os
import shutil
import random

def create_binary_dataset(
    base_multiclass_path="app3/data/dataset_multiclase",
    base_binary_path="app3/data/dataset_binary",
    target_tumor_types=["glioma_tumor", "pituitary_tumor", "meningioma_tumor"],
    no_tumor_class="no_tumor",
    splits=["Training", "Testing", "Validation"],
    default_total_tumor_images=2000 # Used if no_tumor class is empty or not found
):
    """
    Creates a binary classification dataset from a multi-class dataset.
    Combines specified tumor types into a single 'tumor' class and copies 'no_tumor' class.

    Args:
        base_multiclass_path (str): Base path to the multi-class dataset.
        base_binary_path (str): Base path for the new binary dataset.
        target_tumor_types (list): List of tumor class names to combine into the 'tumor' class.
        no_tumor_class (str): Name of the non-tumor class.
        splits (list): List of dataset splits (e.g., "Training", "Testing", "Validation").
        default_total_tumor_images (int): Default total images for the 'tumor' class
                                         if the 'no_tumor' class count cannot be determined.
    """

    for split in splits:
        multiclass_split_path = os.path.join(base_multiclass_path, split)
        binary_split_path = os.path.join(base_binary_path, split)

        # Create destination split directory if it doesn't exist
        os.makedirs(binary_split_path, exist_ok=True)

        # --- Handle the 'no_tumor' class ---
        no_tumor_source_path = os.path.join(multiclass_split_path, no_tumor_class)
        no_tumor_dest_path = os.path.join(binary_split_path, no_tumor_class)
        no_tumor_images = []

        if os.path.exists(no_tumor_source_path) and os.path.isdir(no_tumor_source_path):
            no_tumor_images = [f for f in os.listdir(no_tumor_source_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
            print(f"Found {len(no_tumor_images)} '{no_tumor_class}' images in {no_tumor_source_path}")

            # Ensure destination for no_tumor exists and copy images
            os.makedirs(no_tumor_dest_path, exist_ok=True)
            for img_name in no_tumor_images:
                shutil.copy(os.path.join(no_tumor_source_path, img_name), os.path.join(no_tumor_dest_path, img_name))
            print(f"Copied {len(no_tumor_images)} '{no_tumor_class}' images to {no_tumor_dest_path}")
        else:
            print(f"Warning: '{no_tumor_class}' directory not found at {no_tumor_source_path}")

        # --- Handle the combined 'tumor' class ---
        tumor_dest_path = os.path.join(binary_split_path, "tumor")
        os.makedirs(tumor_dest_path, exist_ok=True)
        print(f"Created/Ensured directory: {tumor_dest_path}")

        # Determine the target number of images for the combined 'tumor' class
        # Prioritize matching 'no_tumor' count for balance
        total_tumor_images_to_copy = len(no_tumor_images)
        if total_tumor_images_to_copy == 0:
            total_tumor_images_to_copy = default_total_tumor_images
            print(f"Using default total of {default_total_tumor_images} images for 'tumor' class due to missing/empty '{no_tumor_class}'.")
        else:
            print(f"Targeting {total_tumor_images_to_copy} images for 'tumor' class (matching '{no_tumor_class}' count).")

        images_per_tumor_type = total_tumor_images_to_copy // len(target_tumor_types)
        print(f"Will attempt to copy approximately {images_per_tumor_type} images from each tumor type.")

        combined_tumor_count = 0
        for tumor_type in target_tumor_types:
            tumor_source_path = os.path.join(multiclass_split_path, tumor_type)
            if os.path.exists(tumor_source_path) and os.path.isdir(tumor_source_path):
                all_tumor_images = [f for f in os.listdir(tumor_source_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
                print(f"Found {len(all_tumor_images)} '{tumor_type}' images in {tumor_source_path}")

                # Sample images to ensure equitable distribution
                num_to_copy = min(images_per_tumor_type, len(all_tumor_images))
                selected_images = random.sample(all_tumor_images, num_to_copy)

                for img_name in selected_images:
                    shutil.copy(os.path.join(tumor_source_path, img_name), os.path.join(tumor_dest_path, img_name))
                combined_tumor_count += num_to_copy
                print(f"Copied {num_to_copy} '{tumor_type}' images to {tumor_dest_path}")
            else:
                print(f"Warning: '{tumor_type}' directory not found at {tumor_source_path}")

        print(f"Finished processing split '{split}'. Total 'tumor' images copied: {combined_tumor_count}\n")

    print("Binary dataset creation complete!")

if __name__ == "__main__":
    # Ensure you set these paths correctly based on your project structure.
    # This assumes 'app3' is in the current working directory where you run the script.
    create_binary_dataset(
        base_multiclass_path="app3/data/dataset_multiclase",
        base_binary_path="app3/data/dataset_binary"
    )
