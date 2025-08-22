
from PIL import Image
slice_spacing_mm = 1.0
pixel_spacing_mm = 0.5
# A threshold to decide if a pixel is part of the object (0-255).
BRIGHTNESS_THRESHOLD = 50
import os
# A step to downsample the points. 1 means every point, 5 means every 5th point.
DOWNSAMPLE_STEP = 100
import glob
import numpy as np

def save_points_to_obj(points, filename):
    """Saves a list of 3D points to an OBJ file."""
    with open(filename, 'w') as f:
        f.write("# OBJ file generated from a point cloud\n")
        f.write(f"# {len(points)} vertices\n")
        for p in points:
            f.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")


def create_plane_basis(normal):
    """
    Creates an orthonormal basis (u, v, n) for a plane defined by its normal.
    """
    # Normalize the normal vector
    n = normal / np.linalg.norm(normal)
    
    # Create a non-parallel vector. If normal is close to z-axis, use y-axis.
    if np.allclose(n, [0, 0, 1]) or np.allclose(n, [0, 0, -1]):
        temp_vec = np.array([0, 1, 0])
    else:
        temp_vec = np.array([0, 0, 1])
        
    # Create the first basis vector using the cross product, then normalize
    u = np.cross(n, temp_vec)
    u /= np.linalg.norm(u)
    
    # Create the second basis vector, which is already orthogonal and normalized
    v = np.cross(n, u)
    
    return u, v

def convert(data_dir, normal_vector, output_obj_path, slice_gap):
    
            
    # --- 2. GENERATE POINT CLOUD ---
    image_files = sorted(glob.glob(os.path.join(data_dir, '*.png')))
    if not image_files:
        raise FileNotFoundError(f"No PNG files found in '{data_dir}'")
    
    # Get the in-plane basis vectors from the normal
    u_vec, v_vec = create_plane_basis(normal_vector)
    n_vec = normal_vector / np.linalg.norm(normal_vector)

    all_points = []
    print("Processing slices to generate point cloud...")
    for i, file_path in enumerate(image_files):
        # Calculate the origin of this slice's plane in 3D space
        plane_origin = i * (slice_spacing_mm * slice_gap) * n_vec
        
        # Open the image and convert to a numpy array
        img_array = np.array(Image.open(file_path).convert('L'))
        
        # Find the (row, col) coordinates of all pixels above the threshold
        rows, cols = np.nonzero(img_array > BRIGHTNESS_THRESHOLD)
        
        # Downsample the points for performance
        rows = rows[::DOWNSAMPLE_STEP]
        cols = cols[::DOWNSAMPLE_STEP]
        
        # Get image center to calculate offsets correctly
        center_row, center_col = img_array.shape[0] / 2, img_array.shape[1] / 2
        
        # Calculate the 3D coordinate for each pixel and add to the list
        for r, c in zip(rows, cols):
            # Calculate displacement from the center of the image plane
            u_disp = (c - center_col) * pixel_spacing_mm
            v_disp = (r - center_row) * pixel_spacing_mm # Often rows are inverted in 3D graphics
            
            # Calculate final 3D point and append
            point = plane_origin + u_disp * u_vec + v_disp * v_vec
            all_points.append(point)

    print(f"Generated a point cloud with {len(all_points)} points.")

    # --- 3. SAVE POINT CLOUD TO OBJ FILE --- 💾
    print(f"Saving point cloud to '{output_obj_path}'...")
    save_points_to_obj(all_points, output_obj_path)

    print("Done.")
import matplotlib.pyplot as plt

def show_slice_montage(volume, title="Slices", output_filename=""):
    """Displays a montage of the slices in the volume."""
    num_slices = volume.shape[2]
    cols = int(np.ceil(np.sqrt(num_slices)))
    rows = int(np.ceil(num_slices / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 12) )#, gridspec_kw={'wspace': 0.2, 'hspace': 0.2})
    axes = axes.flatten()
    
    for i in range(num_slices):
        axes[i].imshow(volume[:, :, i], cmap='gray')
        axes[i].axis('off')
        #axes[i].set_title(f'Slice {i}')
        
    for i in range(num_slices, len(axes)):
        axes[i].axis('off') # Hide unused subplots
        
    fig.suptitle(f'{num_slices} slices', fontsize=16)
    plt.tight_layout()
    #plt.show()
    plt.savefig(output_filename) 

