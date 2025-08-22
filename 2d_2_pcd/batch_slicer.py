import numpy as np
import os
from PIL import Image

from pathlib import Path
import matplotlib.pyplot as plt
from tifffile import imread
# python batch_slicer.py
import png_to_obj
import multiprocessing
from tqdm import tqdm # 1. tqdm 라이브러리를 임포트합니다.

# --- Import the functions from our other script ---
# Make sure virtual_slicer.py is in the same directory.
try:
    from virtual_slicer import create_cube, slice_matrix, visualize_volumes, show_slice_montage
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
OUTPUT_DIR = Path("/data/jhahn/data/brain_lightsheet/slices")

# The list of normal vectors to iterate over. These are some common examples.
# Feel free to add or remove vectors. They will be normalized automatically.
NORMAL_VECTORS_TO_PROCESS = [
    [1, 0, 0],  # Sagittal slice
    [0, 1, 0],  # Coronal slice
    [0, 0, 1],  # Axial slice (original orientation)
    [1, 1, 0],  # 45-degree slice in XY plane
    [1, 0, 1],  # 45-degree slice in XZ plane
    [0, 1, 1],  # 45-degree slice in YZ plane
    [1, 1, 1],  # Oblique slice through the main diagonal
]

# ====================================================================
# Helper Function for Saving
# ====================================================================

def save_volume_as_images(volume, slice_output_dir):
    """
    Saves a 3D numpy array as a sequence of 2D grayscale images.
    
    Args:
        volume (np.ndarray): The 3D volume to save.
        base_dir (Path): The root output directory.
        normal_vector (list): The slicing normal, used for naming the subdirectory.
    """
    # Create a descriptive subdirectory name from the normal vector

    
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

original_volume = None
def initializer_func():
    global original_volume
    if original_volume is None:
        # --- 1. Create the single base volume for all operations ---
        print("Creating the original 3D volume (sphere)...")
        #original_volume = create_cube(64, 64, 64)  # Create a cube of size 64x64x64

        #original_volume = imread('/data/jhahn/data/brain_lightsheet/stack.tif')
        data = np.load('/data/jhahn/data/brain_lightsheet/mask_volume.npz')
        original_volume = data['my_3d_array']

# ====================================================================
# Helper Functions for Mesh and Volume Creation
# ====================================================================
def save_numpy_array_as_tiffs(numpy_array, output_folder, filename_prefix=""):
    """
    Saves a 3D NumPy array as sequential TIFF images in a specified folder.

    Args:
        numpy_array (np.ndarray): The input 3D NumPy array.
                                  Expected shape: (num_images, height, width) for grayscale
                                  or (num_images, height, width, channels) for color.
        output_folder (str): The path to the folder where images will be saved.
        filename_prefix (str, optional): A prefix for the filenames.
                                         Images will be named like 'prefix_000.tif', 'prefix_001.tif', etc.
                                         Defaults to "image".
    """
    if not isinstance(numpy_array, np.ndarray):
        print("Error: Input is not a NumPy array.")
        return

    # Create the output folder if it doesn't exist
    try:
        os.makedirs(output_folder, exist_ok=True)
        print(f"Ensured output folder exists: {output_folder}")
    except OSError as e:
        print(f"Error creating folder {output_folder}: {e}")
        return

    # Determine if it's a grayscale or RGB stack
    if numpy_array.ndim == 3:
        # Grayscale stack (num_images, height, width)
        image_stack = np.transpose(numpy_array, (2, 0, 1))
        image_mode = None # PIL will infer 'L' for grayscale
        print("Detected grayscale image stack.")
    elif numpy_array.ndim == 4 and numpy_array.shape[-1] in [3, 4]:
        # RGB or RGBA stack (num_images, height, width, channels)
        image_mode = 'RGB' if numpy_array.shape[-1] == 3 else 'RGBA'
        print(f"Detected {image_mode} image stack.")
    else:
        print(f"Error: Unsupported NumPy array dimensions. Expected 3D (grayscale) or 4D (RGB/RGBA). Got {numpy_array.ndim}D array with shape {numpy_array.shape}.")
        return

    # Loop through each 2D slice (image) in the 3D or 4D array
    for i, image_slice in enumerate(image_stack):
        try:
            # Convert the NumPy array slice to a PIL Image object
            if image_mode: # For RGB/RGBA
                pil_image = Image.fromarray(image_slice, mode=image_mode)
            else: # For grayscale
                pil_image = Image.fromarray(image_slice)

            # Define the output filename with sequential numbering and the specified prefix
            filename = os.path.join(output_folder, f"{filename_prefix}{i:03d}.tif")

            # Save the image as a TIFF file
            pil_image.save(filename)
            #print(f"Saved {filename}")

        except Exception as e:
            print(f"Error saving image {i}: {e}")
# ====================================================================
# Main Execution Block
# ====================================================================
def _task(OUTPUT_DIR,vector):
    global original_volume
    # Perform the virtual slicing

    vec_name = f"sliced_on_{vector[0]}_{vector[1]}_{vector[2]}"
    
    slice_output_dir = OUTPUT_DIR / vec_name
    if os.path.exists(slice_output_dir):
        print("EXISTS:",slice_output_dir)
        return
    print(vec_name)

    slice_gap = 1
    transformed_volume = slice_matrix(original_volume, normal=vector, debug=False, slice_gap=slice_gap)
    
    #print(f"  Original shape: {original_volume.shape} -> Transformed shape: {transformed_volume.shape}")
    
    if transformed_volume.size == 0:
        print("  Skipping this vector as the resulting volume is empty.")
        return
        

    #save_volume_as_images(transformed_volume, slice_output_dir)
    save_numpy_array_as_tiffs(transformed_volume, slice_output_dir)
    #unique_values, counts = np.unique(transformed_volume.flatten(), return_counts=True)
    #for i in range(len(unique_values)):
    #    print(i, " ** ", f"{unique_values[i]} : {counts[i]} ")

    #show_slice_montage(transformed_volume,title =f"sliced on ({vector[0]}, {vector[1]}, {vector[2]})",output_filename=str(OUTPUT_DIR )+"/"+ f"{vec_name}.png")
    #png_to_obj.convert(slice_output_dir, vector, str(OUTPUT_DIR )+"/"+ f"{vec_name}.obj", slice_gap)
    #print("  png_to_obj done")
if __name__ == "__main__":
    

    #print("original_volume",original_volume.shape)
    #print(np.max(np.max(original_volume[100,:,:],axis=1)))
    #print(np.min(np.min(original_volume[100,:,:],axis=1)))
    # --- 2. Create the main output directory ---
    #if SAVE_OUTPUT_IMAGES:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output will be saved in: '{OUTPUT_DIR.resolve()}'")

    # --- 3. Loop through each normal vector and process the volume ---
    #print("\nStarting batch processing...")
    
    #png_to_obj.convert('/data/jhahn/data/brain_lightsheet/0408', [1,0,0], '/data/jhahn/data/brain_lightsheet/0408.obj')
    #if True:
    #    exit(1)
    tasks_to_run = []
    for vector in NORMAL_VECTORS_TO_PROCESS:
    #for vector in [[1.0, 1.0, 0.0]]:
        gap = 1
        tasks_to_run.append((OUTPUT_DIR, vector))

    #print(f'Total number of jobs: {len(tasks_to_run)}')

    #plane_normal_vector_list = plane_normal_vector_list[:3]
    #gap_list = gap_list[:3]
    #output_dir_list = output_dir_list[:3]

    print(f'the number of jobs:{len(tasks_to_run)}')
    with multiprocessing.Pool(
        initializer=initializer_func,
       # processes=1
    ) as pool:# Use a pool of 4 processes
        #pool.starmap(slice_tiff, zip(output_dir_list, gap_list, plane_normal_vector_list))
        pool.starmap(_task, tqdm(tasks_to_run, total=len(tasks_to_run), desc="Slicing Images"))

    

    '''
    for vector in NORMAL_VECTORS_TO_PROCESS:
    #for vector in [[0.5, 0.5, 0]]:
        print("-" * 50)
        print(f"Processing normal vector: {vector}")
   '''     



    print("-" * 50)
    print("\nBatch processing complete.")


    