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


import glob

def visualize_tiff_grid(folder_path, rows, cols, ext):
    """
    지정된 폴더에서 TIFF 파일을 찾아 rows x cols 그리드 형태로 시각화합니다.
    """

    MAX_IMAGES = rows * cols
    # 폴더 내의 모든 .tif 또는 .tiff 파일을 찾습니다. (대소문자 구분 없이)
    #search_pattern = os.path.join(folder_path, '*.[tT][iI][fF]*')
    search_pattern = os.path.join(folder_path, '*.'+ext)
    tiff_files = glob.glob(search_pattern)
    
    #tiff_files.sort(key=lambda path: int(os.path.splitext(os.path.basename(path))[0]))
    tiff_files.sort(key=lambda path: int( os.path.basename(path).split(".")[0] ))
    if not tiff_files:
        print(f"오류: 지정된 경로 '{folder_path}'에서 TIFF 파일(*.tif, *.tiff)을 찾을 수 없습니다.")
        # 파일이 없으면 더 이상 진행하지 않고 종료
        return

    # 시각화할 이미지 개수를 최대 크기로 제한합니다.
    images_to_display = tiff_files[:MAX_IMAGES]
    num_images = len(images_to_display)
    
    print(f"총 {len(tiff_files)}개의 파일 중, {num_images}개의 파일을 {rows}x{cols} 그리드에 시각화합니다.")

    # Matplotlib 서브플롯 설정
    # figsize는 그리드 크기에 맞춰 조정할 수 있습니다.
    fig, axes = plt.subplots(rows, cols, figsize=(20, 20))
    # axes가 1차원 배열일 경우 2차원 배열로 reshape
    if rows * cols == 1:
        axes = np.array([[axes]])
    elif rows == 1 or cols == 1:
        axes = axes.reshape(rows, cols)
    
    # 0부터 MAX_IMAGES - 1까지의 인덱스를 순회합니다.
    for i in range(MAX_IMAGES):
        row = i // cols
        col = i % cols
        ax = axes[row, col]

        # 이미지 파일이 남아 있다면 시각화
        if i < num_images:
            file_path = images_to_display[i]
            
            try:
                # PIL을 사용하여 TIFF 파일 열기
                img = Image.open(file_path)
                
                # 이미지를 서브플롯에 표시
                # TIFF 파일은 단일 채널(흑백), 3채널(RGB) 등 다양할 수 있습니다.
                # 'gray' colormap은 단일 채널 이미지에 적합합니다.
                if img.mode == 'L' or img.mode == 'I': # 단일 채널 (8-bit 또는 16-bit 흑백)
                     ax.imshow(img, cmap='gray')
                else: # RGB 또는 다른 모드
                     ax.imshow(img)
                
                # 파일 이름을 제목으로 표시 (선택 사항)
                ax.set_title(os.path.basename(file_path), fontsize=8)

            except Exception as e:
                ax.text(0.1, 0.5, f"Error: {e}", color='red', transform=ax.transAxes)
                print(f"파일을 로드하는 중 오류 발생: {file_path}. 오류: {e}")
        
        # 축(Axis) 정보 제거 (깔끔한 시각화를 위해)
        ax.axis('off')
        
    # 전체 플롯의 레이아웃을 조정하여 겹치는 부분을 방지합니다.
    plt.tight_layout()
    # 그래프를 화면에 표시
    plt.show()

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
        filename = f"{i:04d}.png"
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

import cv2
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
    bx=10000
    by=10000
    bw=0
    bh= 0
    # Loop through each 2D slice (image) in the 3D or 4D array
    for i, image_slice in enumerate(image_stack):
        try:
            # Convert the NumPy array slice to a PIL Image object
            if image_mode: # For RGB/RGBA
                pil_image = Image.fromarray(image_slice, mode=image_mode)
            else: # For grayscale
                pil_image = Image.fromarray(image_slice)


            img_np = np.array(pil_image)
            output_img_np = img_np.copy()

            # 3. 전처리: 그레이스케일 및 이진화
            #img_gray = cv2.cvtColor(output_img_np, cv2.COLOR_RGB2GRAY)
            # 객체가 흰색, 배경이 검은색이 되도록 임계값 처리
            ret, thresh = cv2.threshold(output_img_np, 127, 255, cv2.THRESH_BINARY) 

            # 4. 윤곽선 찾기
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


            # 4. 각 윤곽선에 Bounding Box 적용
            for cnt in contours:
                # 윤곽선 영역이 너무 작으면 무시 (노이즈 제거)
                if cv2.contourArea(cnt) > 500:
                    
                    # cv2.boundingRect()를 사용하여 Bounding Box 좌표 얻기
                    # x, y: 좌측 상단 모서리 좌표
                    # w, h: 너비와 높이
                    x, y, w, h = cv2.boundingRect(cnt)
                    #print(x, y, w, h)
                    if x < bx:
                        bx = x
                    if y < by:
                        by = y
                    if bw < w:
                        bw = w
                    if bh < h:
                        bh = h
                        


            # Define the output filename with sequential numbering and the specified prefix
            filename = os.path.join(output_folder, f"{filename_prefix}{i:03d}.tif")

            # Save the image as a TIFF file
            pil_image.save(filename)
            #print(f"Saved {filename}")

        except Exception as e:
            print(f"Error saving image {i}: {e}")
    return len(image_stack), (bx,by,bw,bh)
# ====================================================================
# Main Execution Block
# ====================================================================
from PIL import Image

def _task(OUTPUT_DIR,vector, V = None, slice_gap=1):
    if V is None:
        global original_volume
        V = original_volume
    # Perform the virtual slicing

    vec_name = f"sliced_on_{vector[0]}_{vector[1]}_{vector[2]}"
    
    slice_output_dir = OUTPUT_DIR / vec_name
    #if os.path.exists(slice_output_dir):
    #    print("EXISTS:",slice_output_dir)
    #    return
    print('vec_name',vec_name)

    
    transformed_volume = slice_matrix(V, normal=vector, debug=False, slice_gap=slice_gap)
    
    print(f"  Original shape: {V.shape} -> Transformed shape: {transformed_volume.shape}")
    
    if transformed_volume.size == 0:
        print("  Skipping this vector as the resulting volume is empty.")
        return
        

    #save_volume_as_images(transformed_volume, slice_output_dir)
    num_of_slices,(bx,by,bw,bh) = save_numpy_array_as_tiffs(transformed_volume, slice_output_dir)
    #unique_values, counts = np.unique(transformed_volume.flatten(), return_counts=True)
    #for i in range(len(unique_values)):
    #    print(i, " ** ", f"{unique_values[i]} : {counts[i]} ")

    #show_slice_montage(transformed_volume,title =f"sliced on ({vector[0]}, {vector[1]}, {vector[2]})",output_filename=str(OUTPUT_DIR )+"/"+ f"{vec_name}.png")
    #png_to_obj.convert(slice_output_dir, vector, str(OUTPUT_DIR )+"/"+ f"{vec_name}.obj", slice_gap)
    #print("  png_to_obj done")


    

    for _idx in range(num_of_slices-1):
        input_filepath = str(slice_output_dir) + f"/{str(_idx).zfill(3)}.tif"
        #output_filepath_o = input_filepath[:-3]+"png"
        img = Image.open(input_filepath)
        
        cropped_img = img.crop( (bx - 20,by - 20,bx+bw + 20,by+bh + 20) )
        #cropped_img.save(output_filepath_c, format="PNG")

        #output_filepath_r = input_filepath[:-3]+"_r.png"
        img_rescaled = cropped_img.resize((500, 500), Image.LANCZOS)
        img_rescaled.save(input_filepath, format="TIFF")

        #img_rescaled.save(output_filepath_o, format="PNG")
        

    return num_of_slices, (bx,by,bw,bh)
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


    