import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
from tifffile import imread
# python batch_slicer.py

# --- Import the functions from our other script ---
# Make sure virtual_slicer.py is in the same directory.
try:
    from virtual_slicer import create_cube, slice_matrix, visualize_volumes
except ImportError:
    print("Error: Could not import from 'virtual_slicer.py'.")
    print("Please make sure 'virtual_slicer.py' is in the same directory as this script.")
    exit()

# ====================================================================
# Configuration
# ====================================================================

# Set to True to save the output slices of each volume as 2D PNG images.
SAVE_OUTPUT_IMAGES = True

# Set to True to show the interactive 3D visualization for each sliced volume.
# WARNING: This will pause the script after each iteration until you close the window.
VISUALIZE_INTERACTIVELY = False

# The base directory where all output folders will be created.
OUTPUT_DIR = Path("./slicer_output")

# The list of normal vectors to iterate over. These are some common examples.
# Feel free to add or remove vectors. They will be normalized automatically.
NORMAL_VECTORS_TO_PROCESS = [
    [1, 0, 0],  # Sagittal slice
    [0, 1, 0],  # Coronal slice
    [0, 0, 1],  # Axial slice (original orientation)
    [1, 1, 0],  # 45-degree slice in XY plane
    [1, 0, 1],  # 45-degree slice in XZ plane
    [0, 1, 1],  # 45-degree slice in YZ plane

    [0.5, 0.5, 0],  # 45-degree slice in XY plane
    [0.5, 0, 0.5],  # 45-degree slice in XZ plane
    [0, 0.5, 0.5],  # 45-degree slice in YZ plane

    [1, 1, 1],  # Oblique slice through the main diagonal
]

# ====================================================================
# Helper Function for Saving
# ====================================================================

def save_volume_as_images(volume, base_dir, normal_vector):
    """
    Saves a 3D numpy array as a sequence of 2D grayscale images.
    
    Args:
        volume (np.ndarray): The 3D volume to save.
        base_dir (Path): The root output directory.
        normal_vector (list): The slicing normal, used for naming the subdirectory.
    """
    # Create a descriptive subdirectory name from the normal vector
    vec_name = f"sliced_on_{normal_vector[0]}_{normal_vector[1]}_{normal_vector[2]}"
    slice_output_dir = base_dir / vec_name
    
    # Create the directory if it doesn't exist
    slice_output_dir.mkdir(parents=True, exist_ok=True)
    
    num_slices = volume.shape[2]
    print(f"  Saving {num_slices} slices to '{slice_output_dir}'...")
    
    # Check if volume data needs normalization to 0-255 for saving
    is_uint8 = volume.dtype == np.uint8
    
    for i in range(num_slices):
        slice_2d = volume[:, :, i]
        
        # Prepare the slice for saving as a grayscale image
        if not is_uint8:
            # Normalize to 0-255 if the data is not already uint8
            if np.max(slice_2d) > 0:
                slice_2d = (slice_2d / np.max(slice_2d) * 255).astype(np.uint8)
            else:
                slice_2d = slice_2d.astype(np.uint8)
        
        # Define the output filename with zero-padding for correct sorting
        filename = f"slice_{i:04d}.png"
        filepath = slice_output_dir / filename
        
        # Save the 2D slice as a grayscale PNG image
        plt.imsave(filepath, slice_2d, cmap='gray')
        
    print(f"  Successfully saved volume.")


# ====================================================================
# Main Execution Block
# ====================================================================

if __name__ == "__main__":
    
    # --- 1. Create the single base volume for all operations ---
    print("Creating the original 3D volume (sphere)...")
    #original_volume = create_cube(64, 64, 64)  # Create a cube of size 64x64x64

    original_volume = imread('/data/jhahn/data/brain_lightsheet/mask_stack.tif')
    print("original_volume",original_volume.shape)
    print(np.max(np.max(original_volume[100,:,:],axis=1)))
    print(np.min(np.min(original_volume[100,:,:],axis=1)))
    # --- 2. Create the main output directory ---
    if SAVE_OUTPUT_IMAGES:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        print(f"Output will be saved in: '{OUTPUT_DIR.resolve()}'")

    # --- 3. Loop through each normal vector and process the volume ---
    print("\nStarting batch processing...")
    
    for vector in NORMAL_VECTORS_TO_PROCESS:
        print("-" * 50)
        print(f"Processing normal vector: {vector}")
        
        # Perform the virtual slicing
        transformed_volume = slice_matrix(original_volume, normal=vector)
        
        print(f"  Original shape: {original_volume.shape} -> Transformed shape: {transformed_volume.shape}")
        
        if transformed_volume.size == 0:
            print("  Skipping this vector as the resulting volume is empty.")
            continue
            
        # Interactively visualize the result if the flag is set
        if VISUALIZE_INTERACTIVELY:
            print("  Displaying interactive 3D visualization...")
            visualize_volumes(
                original_volume,
                transformed_volume,
                original_title="Original Sphere",
                transformed_title=f"Resliced on Normal {vector}"
            )
        
        # Save the resulting 3D matrix as a series of 2D images if the flag is set
        if SAVE_OUTPUT_IMAGES:
            save_volume_as_images(transformed_volume, OUTPUT_DIR, vector)

    print("-" * 50)
    print("\nBatch processing complete.")