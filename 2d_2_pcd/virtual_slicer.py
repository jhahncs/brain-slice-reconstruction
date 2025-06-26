import numpy as np
import pyvista as pv
from scipy.ndimage import map_coordinates
import matplotlib.pyplot as plt

# ====================================================================
# Helper Functions for Mesh and Volume Creation
# ====================================================================

# conda create -n slicer python=3.9 numpy scipy scikit-image matplotlib pyvista -c conda-forge
# conda activate slicer   
# pip install pyqt5
# python virtual_slicer.py
def create_cube(size_x, size_y, size_z):
    """
    Creates a 3D numpy array with shape (2*size_x, 2*size_y, 2*size_z),
    containing a solid cube of size (size_x, size_y, size_z) centered in the array.
    """
    # Create a zero matrix of double the cube size
    arr = np.zeros((2*size_x, 2*size_y, 2*size_z), dtype=np.uint8)
    # Calculate start and end indices for centering the cube
    start_x = size_x // 2
    start_y = size_y // 2
    start_z = size_z // 2
    end_x = start_x + size_x
    end_y = start_y + size_y
    end_z = start_z + size_z
    # Place the cube in the center
    arr[start_x:end_x, start_y:end_y, start_z:end_z] = 255
    return arr

def create_sphere(size, radius, value=255):
    """Creates a 3D numpy array representing a sphere."""
    coords = np.ogrid[:size, :size, :size]
    center = size / 2
    distance = np.sqrt((coords[0] - center)**2 + (coords[1] - center)**2 + (coords[2] - center)**2)
    sphere = np.zeros((size, size, size), dtype=np.uint8)
    sphere[distance <= radius] = value
    return sphere

def remove_zero_slices(matrix):
    """Removes slices from a 3D matrix that contain only zeros."""
    non_zero_slices = [i for i in range(matrix.shape[2]) if np.any(matrix[:, :, i])]
    return matrix[:, :, non_zero_slices]

# ====================================================================
# Core Slicing Function
# ====================================================================

def slice_matrix(V, normal, tolerance=0.5):
    """
    Reslices a 3D volume V along a plane defined by the normal vector.
    This function is a Python equivalent of MATLAB's obliqueslice.
    """
    # Normalize the normal vector
    normal = np.array(normal) / np.linalg.norm(normal)

    # Volume dimensions and center
    original_shape = np.array(V.shape)
    array_center = original_shape / 2

    # Create an orthonormal basis for the slicing plane
    u = np.array([-normal[1], normal[0], 0])
    if np.linalg.norm(u) < 1e-6:
        u = np.array([0, -normal[2], normal[1]])
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)

    # Determine the size of the new slice matrix
    max_dim = int(np.ceil(np.linalg.norm(original_shape)))
    slice_size = (max_dim, max_dim)
    
    # Create grid points for the new slice
    x_slice, y_slice = np.meshgrid(np.arange(-max_dim/2, max_dim/2),
                                   np.arange(-max_dim/2, max_dim/2))

    sliced_matrix_list = []
    
    # Iterate through slices along the normal vector
    max_travel = int(np.ceil(np.linalg.norm(array_center))) * 2
    for s in range(max_travel):
        # Calculate the center point for the current slice
        pt = array_center + normal * (s - max_travel / 2)
        
        # Check if the plane is reasonably within the volume bounds
        if not (0 <= pt[0] < original_shape[0] and 
                0 <= pt[1] < original_shape[1] and 
                0 <= pt[2] < original_shape[2]):
            if len(sliced_matrix_list) > 0 : # Stop if we have collected slices and moved out
                 break
            else: # continue until we are in the volume
                 continue


        # Convert 2D slice grid points to 3D coordinates in the original volume
        points_3d = pt[:, np.newaxis, np.newaxis] + \
                    u[:, np.newaxis, np.newaxis] * x_slice + \
                    v[:, np.newaxis, np.newaxis] * y_slice

        # Sample the volume at these 3D coordinates
        # Note: map_coordinates expects coordinates in (z, y, x) order for a (depth, height, width) array
        coords_for_map = [points_3d[2], points_3d[1], points_3d[0]]
        oblique_slice = map_coordinates(V, coords_for_map, order=0, mode='constant', cval=0.0)
        
        # If the slice is not empty, add it to our list
        if np.any(oblique_slice):
            sliced_matrix_list.append(oblique_slice)

    if not sliced_matrix_list:
        print("Warning: No data was captured in the slices. The volume might be empty or the normal is pointing away.")
        return np.zeros((10, 10, 10)) # Return a small empty matrix

    # Stack the collected slices into a new 3D matrix
    sliced_matrix = np.stack(sliced_matrix_list, axis=-1)
    
    return remove_zero_slices(sliced_matrix)


# ====================================================================
# Visualization Functions
# ====================================================================

def visualize_volumes(original_volume, transformed_volume, original_title="Original Volume", transformed_title="Transformed Volume"):
    """
    Visualizes the original and transformed volumes side-by-side using PyVista.
    """
    # Convert numpy arrays to PyVista UniformGrid objects
    grid_orig = pv.wrap(original_volume)
    grid_transformed = pv.wrap(transformed_volume)

    # Set up the plotter
    plotter = pv.Plotter(shape=(1, 2), window_size=[1600, 800])

    # Plot original volume
    plotter.subplot(0, 0)
    plotter.add_text(original_title, font_size=15)
    plotter.add_volume(grid_orig, cmap="viridis", shade=True)
    plotter.add_axes()

    # Plot transformed volume
    plotter.subplot(0, 1)
    plotter.add_text(transformed_title, font_size=15)
    plotter.add_volume(grid_transformed, cmap="magma", shade=True)
    plotter.add_axes()

    # Link cameras so they move together
    plotter.link_views()
    
    print("Displaying original and transformed volumes. Close the window to continue.")
    plotter.show()


def show_slice_montage(volume, title="Slice Montage"):
    """Displays a montage of the slices in the volume."""
    num_slices = volume.shape[2]
    cols = int(np.ceil(np.sqrt(num_slices)))
    rows = int(np.ceil(num_slices / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
    axes = axes.flatten()
    
    for i in range(num_slices):
        axes[i].imshow(volume[:, :, i], cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(f'Slice {i}')
        
    for i in range(num_slices, len(axes)):
        axes[i].axis('off') # Hide unused subplots
        
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()


# ====================================================================
# Main Execution Block
# ====================================================================

if __name__ == "__main__":
    
    # --- 1. Define Parameters ---
    # Define the normal vector for the slicing plane. [0,0,1] is a standard axial slice.
    # Try [1,1,1] or [1,0,1] for an oblique slice.
    slicing_normal = [1, 0, 5] 
    
    # --- 2. Create Initial Data ---
    print("Creating initial 3D volumes (cube and sphere)...")
    # Create a cube volume
    mask_volume = create_cube(64, 64, 64)
    # mask_volume = create_sphere(64, radius=20)
    
    # For demonstration, we'll use the cube as our primary volume
    # This makes the slicing effect more visually apparent
    original_volume = mask_volume
    
    # --- 3. Perform Slicing / Rotation ---
    print(f"Slicing volume with normal vector: {slicing_normal}...")
    
    # Check if a transformation is needed
    if np.array_equal(slicing_normal, [0, 0, 1]) or np.array_equal(slicing_normal, [0, 0, -1]):
        print("Normal vector is aligned with Z-axis. No transformation needed.")
        transformed_volume = original_volume
    else:
        # Perform the virtual slicing
        transformed_volume = slice_matrix(original_volume, normal=slicing_normal)

    print("Slicing complete.")
    print(f"Original volume shape: {original_volume.shape}")
    print(f"Transformed volume shape: {transformed_volume.shape}")

    # --- 4. Visualize Results ---
    print("Generating visualizations...")
    
    # Show slice montages for a 2D view
    show_slice_montage(original_volume, title="Original Volume Slices")
    show_slice_montage(transformed_volume, title=f"Sliced Volume (Normal: {slicing_normal})")
    
    # Show interactive 3D renderings
    visualize_volumes(
        original_volume, 
        transformed_volume,
        original_title="Original Sphere",
        transformed_title=f"Sphere Resliced with Normal {slicing_normal}"
    )
    
    print("Script finished.")