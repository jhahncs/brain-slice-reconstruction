import tifffile
import os
import argparse
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.spatial.transform import Rotation
import numpy as np
from itertools import product
import multiprocessing
import traceback
import sys
def get_slice_from_plane(volume, plane_normal, slice_index, debug=False):
    """
    Extracts a 2D slice from a 3D volume defined by a plane normal.

    Args:
        volume (np.ndarray): The 3D image volume (D, H, W).
        plane_normal (np.ndarray): The normal vector of the slicing plane.
        slice_index (int): The position of the slice along the slicing axis.

    Returns:
        np.ndarray: The extracted 2D slice.
    """
    # Normalize the plane normal vector
    plane_normal = plane_normal / np.linalg.norm(plane_normal)

    # --- Calculate Rotation Matrix ---
    # The initial slicing direction is along the first axis (depth).
    initial_normal = np.array([1.0, 0.0, 0.0])
    
    # Calculate the rotation required to align the initial normal with the desired plane normal.
    rotation, _ = Rotation.align_vectors(plane_normal, initial_normal)
    transform = rotation.as_matrix()

    # Get volume dimensions and center
    depth, height, width = volume.shape
    center_d, center_h, center_w = (depth - 1) / 2, (height - 1) / 2, (width - 1) / 2

    # --- Create a grid of coordinates for the output slice ---
    # The output slice will have the same dimensions (height, width) as the original slices.
    # This ensures consistent output dimensions, though it may result in black areas
    # for angled slices where the plane extends beyond the original volume.
    x_coords = np.arange(width) - center_w
    y_coords = np.arange(height) - center_h
    x_grid, y_grid = np.meshgrid(x_coords, y_coords)

    # The slice_index determines the position of the plane along its normal.
    z_grid = np.full_like(x_grid, slice_index - center_d)
    if debug: 
        print('x_grid',x_grid.shape)
        print('z_grid',z_grid.shape)
        print('z_grid',z_grid[:5,:5])
        print('x_grid',x_grid[:5,:5])
        print('slice_index',slice_index)
        print('center_d',center_d)
    # Prepare coordinates for transformation. The order is (depth, height, width)
    # to match the volume's axes for map_coordinates.
    coords = np.vstack((z_grid.ravel(), y_grid.ravel(), x_grid.ravel()))
    if debug: 
        print('coords',coords.shape)
        print('z_grid',z_grid.ravel().shape)
        print('y_grid',y_grid.ravel().shape)
        print('x_grid',x_grid.ravel().shape)

    # Apply the inverse rotation to the grid coordinates
    inv_transform = np.linalg.inv(transform)
    rotated_coords = inv_transform @ coords
    if debug: print('inv_transform',inv_transform)
    # Shift coordinates to be relative to the volume's center.
    rotated_coords[0, :] += center_d
    rotated_coords[1, :] += center_h
    rotated_coords[2, :] += center_w
    if debug: 
        print('z_grid',rotated_coords[0, :5])
        print('x_grid',rotated_coords[1, :5])
    # Interpolate the values at the rotated coordinates using linear interpolation.
    sliced_image_flat = map_coordinates(volume, rotated_coords, order=1, mode='constant', cval=0.0)
    
    return sliced_image_flat.reshape(x_grid.shape)

shared_images = None
shared_num_images = None


def slice_tiff(  base_output_dir, gap, plane_normal):
    """
    Slices a TIFF file with specified gaps and a cutting plane, saving the 2D images
    into a uniquely named folder based on the parameters.

    Args:
        input_file (str): Path to the input TIFF file.
        base_output_dir (str): Path to the base directory to save the output folder.
        gaps (list[int]): A list of integers for the gaps between slices.
        plane_normal (np.ndarray): The normal vector of the slicing plane.
    """

    global shared_images
    global shared_num_images

    try:
        # --- Construct the output folder name from parameters ---
        base_filename = 'mask'#os.path.splitext(os.path.basename(input_file))[0]

        normal_str = f"{plane_normal[0]:.1f}_{plane_normal[1]:.1f}_{plane_normal[2]:.1f}".replace('-', 'm')
        #gaps_str = f"gaps_{'_'.join(map(str, gaps))}"
        gaps_str = f"gap_{gap}"
        
        # Combine parts to create a unique folder name for this run
        run_folder_name = f"{base_filename}_{normal_str}_{gaps_str}"
        
        # Create the full path for the output directory for this specific run
        output_dir = os.path.join(base_output_dir, run_folder_name)

        # Create the output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        #print(f"Output will be saved in: {output_dir}")


        current_index = 0
        gap_cycle_index = 0
        slice_count = 0
        
        is_orthogonal = np.allclose(plane_normal, [1, 0, 0])

        while current_index < shared_num_images:
            # Get the slice
            if is_orthogonal:
                # Fast path for standard, non-rotated slices
                image_slice = shared_images[current_index]
            else:
                # Slower path for angled slices requiring interpolation
                #print(f"Generating angled slice at index {current_index}...")
                image_slice = get_slice_from_plane(shared_images, plane_normal, current_index)

            # Construct the output filename
            output_filename = f"{current_index:04d}.tif"
            output_path = os.path.join(output_dir, output_filename)

            # Save the 2D image slice
            tifffile.imwrite(output_path, image_slice.astype(shared_images.dtype))
            #print(f"Saved {output_path}")
            slice_count += 1

            # Determine the next index based on the variable gap
            #gap = gaps[gap_cycle_index % len(gaps)]
            current_index += gap + 1
            gap_cycle_index += 1

        print(f"\nSlicing complete. {slice_count} images on {output_dir}")


    except Exception as e:

        # --- 특정 정보만 추출하고 싶을 경우 ---
        print("\n--- 상세 정보만 추출하기 ---")
        exc_type, exc_value, exc_traceback = sys.exc_info()
        
        # 가장 최근의 콜 스택 정보 (에러가 직접 발생한 위치)
        latest_call = traceback.extract_tb(exc_traceback)[-1]
        
        filename = latest_call.filename
        line_number = latest_call.lineno
        function_name = latest_call.name
        line_content = latest_call.line
        
        print(f"파일: '{filename}'")
        print(f"라인: {line_number}")
        print(f"함수: {function_name}()")
        print(f"코드: '{line_content}'")
        
        print(f"An error occurred: {e}")

def initializer_func(input_file):
    """각 작업자 프로세스가 시작될 때 한 번만 실행되는 초기화 함수"""
    global shared_images
    global shared_num_images
    if shared_images is None:
        # Read the entire TIFF stack into a NumPy array
        with tifffile.TiffFile(input_file) as tif:
            shared_images = tif.asarray()
            shared_num_images = shared_images.shape[0]
            #print(f"Found {shared_num_images} images in the TIFF file.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Slice a multi-page TIFF file with variable gaps and a specified cut plane.",
        formatter_class=argparse.RawTextHelpFormatter)
        
    parser.add_argument("input_file", help="Path to the input TIFF file.")
    parser.add_argument("output_dir", help="Base directory to save the parameter-named output folder.")
    args = parser.parse_args()

    # 각 차원에서 0.0부터 1.0까지 0.1 간격으로 값을 생성합니다.
    values = np.arange(0.0, 1.1, 0.1)

    # 3개의 차원에 대한 데카르트 곱을 사용하여 3차원 벡터를 생성합니다.
    vectors = list(product(values, repeat=3))

    # 부동 소수점 오류를 방지하기 위해 각 값을 소수점 첫째 자리까지 반올림합니다.
    vectors = [tuple(round(val, 1) for val in vec) for vec in vectors][1:]

    plane_normal_vector_list = []
    gap_list = []
    output_dir_list = []
    for v in vectors:       
        #for g in range(1,11): 
        plane_normal_vector_list.append(v)    
        gap_list.append(1)
        output_dir_list.append(args.output_dir)


    #plane_normal_vector_list = plane_normal_vector_list[:3]
    #gap_list = gap_list[:3]
    #output_dir_list = output_dir_list[:3]

    print(f'the number of jobs:{len(output_dir_list)}')
    with multiprocessing.Pool(
        initializer=initializer_func,
        initargs=(args.input_file,) # initargs는 튜플 형태로 전달해야 함
    ) as pool:# Use a pool of 4 processes
        pool.starmap(slice_tiff, zip(output_dir_list, gap_list, plane_normal_vector_list))


    #slice_tiff(args.input_file, args.output_dir, gap_list, plane_normal_vector)


