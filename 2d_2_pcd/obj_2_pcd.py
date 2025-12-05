import warnings
# 반드시 numpy나 torch를 import하기 '전'에 작성해야 합니다.
warnings.filterwarnings("ignore", message=".*smallest subnormal.*")
import torch.multiprocessing as mp

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import multiprocessing
from vedo import dataurl, printc, Plotter, Points, Mesh, Text2D
import torch
import shutil
import slice_util
import argparse
import trimesh
import time
from tqdm import tqdm # 1. tqdm 라이브러리를 임포트합니다.
import open3d as o3d

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

def pcd_2_mesh(pdc_filename, mesh_filename):
    mesh = Mesh(pdc_filename)
    pts0 = Points(mesh, r=3)#.add_gaussian_noise(1)
    pts1 = pts0.clone().smooth_mls_2d(f=0.8)
    pts1.subsample(0.005)
    reco = pts1.reconstruct_surface(dims=10, radius=0.2)
    reco.write(mesh_filename)

            
def padding_to_image(image, padding_size=100):
    empty_row = np.array([[0,0,0]*padding_size]*image.shape[0]).reshape(image.shape[0],padding_size,3)

    _image = np.concatenate([empty_row, image, empty_row], axis=1)

    empty_col = np.array([[0,0,0]*padding_size]*_image.shape[1]).reshape(padding_size,_image.shape[1],3)
    _image = np.concatenate([empty_col, _image, empty_col], axis=0)
    

    return _image.astype(np.float32)


import torch
import numpy as np

def interpolate_points_between_borders(points1, points2, num_interpolated_points, device=None):
    """
    GPU를 사용하여 두 점 집합 사이를 보간합니다.
    (Open3D KDTree 및 for loop 제거 -> PyTorch 행렬 연산)
    """
    
    # 1. 데이터를 GPU Tensor로 변환
    # 입력이 이미 텐서라면 그대로 사용, numpy라면 변환
    if not torch.is_tensor(points1):
        p1_tensor = torch.tensor(points1, dtype=torch.float32, device=device)
    else:
        p1_tensor = points1.to(device)
        
    if not torch.is_tensor(points2):
        p2_tensor = torch.tensor(points2, dtype=torch.float32, device=device)
    else:
        p2_tensor = points2.to(device)

    # 2. Nearest Neighbor Search (KDTree 대체)
    # 모든 p1 점들에 대해 p2에서 가장 가까운 점 찾기
    # cdist: 두 텐서 간의 거리를 계산 (N x M 행렬 생성)
    # 주의: 점의 개수가 수십만 개 이상일 경우 메모리 부족이 발생할 수 있으므로 배치 처리가 필요할 수 있음
    dists = torch.cdist(p1_tensor, p2_tensor) 
    
    # 각 p1에 대해 거리가 가장 짧은 p2의 인덱스(argmin)를 찾음
    min_indices = torch.argmin(dists, dim=1)
    
    # 가장 가까운 p2 점들을 추출 (p1과 순서가 매칭됨)
    closest_p2 = p2_tensor[min_indices] # Shape: [N, 3]

    # 3. 벡터화된 보간 (Vectorized Interpolation)
    # 기존 이중 for loop 제거
    
    # t 값 생성: 0부터 1까지 (num_interpolated_points + 1) 단계
    steps = num_interpolated_points + 1
    t = torch.linspace(0, 1, steps, device=device) # Shape: [steps]
    
    # 브로드캐스팅을 위한 차원 확장
    # P1: [N, 1, 3]
    # P2: [N, 1, 3]
    # t : [1, steps, 1]
    p1_expanded = p1_tensor.unsqueeze(1)
    p2_expanded = closest_p2.unsqueeze(1)
    t_expanded = t.view(1, -1, 1)
    
    # 보간 공식: (1 - t) * P1 + t * P2
    # 결과 Shape: [N, steps, 3]
    interpolated = (1 - t_expanded) * p1_expanded + t_expanded * p2_expanded
    
    # 4. 결과 형상 변경 (Flatten)
    # [N * steps, 3] 형태로 변경하여 모든 점을 리스트업
    final_points = interpolated.reshape(-1, 3)
    
    # 필요시 CPU Numpy로 변환하여 반환 (후속 작업에 따라 Tensor 반환 권장)
    return final_points

# --- 사용 예시 ---
# N = 10000 # 점 개수
# points_A = np.random.rand(N, 3)
# points_B = np.random.rand(N, 3)

# result = interpolate_points_between_borders_gpu(points_A, points_B, 5)
# print(f"생성된 총 점의 개수: {result.shape[0]}")

def interpolate_points_between_borders_cpu(points1, points2, num_interpolated_points):
    """
    두 슬라이스의 경계선 사이를 보간하여 새로운 점을 생성합니다.
    """
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(points1)
    
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(points2)
    
    #kdtree1 = o3d.geometry.KDTreeFlann(pcd1)
    kdtree2 = o3d.geometry.KDTreeFlann(pcd2)
    
    interpolated_points = []
    
    # 첫 번째 슬라이스에서 점을 순회하며 두 번째 슬라이스의 가장 가까운 점을 찾고 보간합니다.
    for p1 in points1:
        [k, idx, _] = kdtree2.search_knn_vector_3d(p1, 1)
        if k > 0:
            p2 = points2[idx[0]]
            for i in range(0, num_interpolated_points+1 ):
                t = i / (num_interpolated_points )
                new_point = (1 - t) * p1 + t * p2
                #print(new_point)
                interpolated_points.append(new_point)
                
    return np.array(interpolated_points)


def tiff_2_xyz_cpu(tiff_filename_full, y_min, y_max = 0, 
               max_num_of_points = 100, is_curvature = True, estimated_num_of_missing_slices = 10,
               device = None):

    image_top = cv2.imread(tiff_filename_full, cv2.IMREAD_COLOR)
    
    #new_width, new_height = 150, 150
    #image_top = cv2.resize(image_top, (new_width, new_height))

    gray_image_top = cv2.cvtColor(image_top, cv2.COLOR_BGR2GRAY)
    canny_image_top = cv2.Canny(gray_image_top,100,200)
    
    points_top = []


    if is_curvature:
        for i, x in np.ndenumerate(canny_image_top):
            if x > 0 :
                points_top.append([(canny_image_top.shape[1]-i[1])/(canny_image_top.shape[1]), y_min, (canny_image_top.shape[0]-i[0])/(canny_image_top.shape[0])])

    else:
        '''
        y_random_numbers = [random.uniform(y_min, y_max) for _ in range(50*int((estimated_num_of_missing_slices+1)))]

        for i, x in np.ndenumerate(canny_image_top):
            if x > 0 :
                points_top.append([(canny_image_top.shape[1]-i[1])/(canny_image_top.shape[1]), y_min, (canny_image_top.shape[0]-i[0])/(canny_image_top.shape[0])])
                points_top.append([(canny_image_top.shape[1]-i[1])/(canny_image_top.shape[1]), y_max, (canny_image_top.shape[0]-i[0])/(canny_image_top.shape[0])])
                for y in y_random_numbers:
                    points_top.append([(canny_image_top.shape[1]-i[1])/(canny_image_top.shape[1]), y, (canny_image_top.shape[0]-i[0])/(canny_image_top.shape[0])])
        
        '''

        # -----------------------------------------------------

        ## 1. 초기 데이터 준비 및 텐서 변환

        # canny_image_top을 GPU 텐서로 변환
        canny_tensor = torch.from_numpy(canny_image_top).to(device, dtype=torch.float32)
        H, W = canny_tensor.shape

        # 1-1. y_random_numbers 계산 및 GPU 텐서 변환
        num_random = 50 * int((estimated_num_of_missing_slices + 1))
        y_random_numbers = torch.tensor([random.uniform(y_min, y_max) for _ in range(num_random)], 
                                        dtype=torch.float32, device=device)

        ## 2. 병렬화를 위한 마스킹 및 인덱싱

        # 2-1. 에지(양수)가 있는 픽셀의 위치(인덱스) 찾기
        # (H, W) 크기의 텐서에서 값이 0보다 큰(> 0) 모든 위치의 (row, col) 인덱스를 반환
        mask = canny_tensor > 0
        edge_indices = torch.nonzero(mask, as_tuple=False)  # (N, 2) 크기, N은 에지 픽셀 수

        # 에지 픽셀의 개수
        N = edge_indices.shape[0]

        # 2-2. X 및 Z 좌표 계산 (정규화)
        # i[0]은 row(Z) 인덱스, i[1]은 col(X) 인덱스입니다.
        # X 좌표: (W - col) / W
        x_coords = (W - edge_indices[:, 1].float()) / W
        # Z 좌표: (H - row) / H
        z_coords = (H - edge_indices[:, 0].float()) / H
        
        # 2-3. X, Z 좌표를 N x 1 텐서로 결합 (N, 1)
        xz_coords = torch.stack([x_coords, z_coords], dim=1) # (N, 2)
        
        # 2-4. Y_min, Y_max, Y_random 데이터 생성
        
        # A. (X, Y_min, Z) 점들 생성: N x 3
        y_min_tensor = torch.full((N, 1), y_min, device=device) # (N, 1)
        points_ymin = torch.cat((xz_coords[:, 0:1], y_min_tensor, xz_coords[:, 1:2]), dim=1) # (N, 3)

        # B. (X, Y_max, Z) 점들 생성: N x 3
        y_max_tensor = torch.full((N, 1), y_max, device=device) # (N, 1)
        points_ymax = torch.cat((xz_coords[:, 0:1], y_max_tensor, xz_coords[:, 1:2]), dim=1) # (N, 3)
        
        # C. (X, Y_random, Z) 점들 생성: N * num_random x 3
        # N개의 (X, Z) 쌍을 num_random번 반복 (N * num_random, 2)
        # torch.repeat_interleave는 PyTorch에서 가장 빠른 방법 중 하나입니다.
        repeated_xz = xz_coords.repeat_interleave(num_random, dim=0) # (N * num_random, 2)
        
        # num_random개의 y_random_numbers를 N번 반복 (N * num_random, 1)
        repeated_y_random = y_random_numbers.repeat(N).unsqueeze(1) # (N * num_random, 1)
        
        # 최종 랜덤 포인트 결합
        # X와 Z 사이에 Y_random을 삽입합니다.
        points_yrandom = torch.cat((repeated_xz[:, 0:1], repeated_y_random, repeated_xz[:, 1:2]), dim=1) # (N * num_random, 3)

        # 2-5. 모든 점들을 하나로 결합
        # (2*N + N*num_random, 3) 크기의 최종 텐서
        points_top_tensor = torch.cat((points_ymin, points_ymax, points_yrandom), dim=0)
        points_top = points_top_tensor.cpu()
        # 3. 결과 확인 (선택 사항)
        
        

    points_top = np.array(points_top)
    #print('points_top',points_top.shape)
    if points_top.shape[0] > max_num_of_points:
        points_top_sampled_idx = np.random.choice(points_top.shape[0], size=max_num_of_points, replace=False)
        points_top = points_top[points_top_sampled_idx]



    return points_top, image_top.shape
def tiff_2_xyz(tiff_filename_full, y_min,
               max_num_of_points = 100, 
               device = None):


    # 2. 이미지 로드 및 Canny Edge Detection (CPU 수행)
    # OpenCV의 기본 연산은 CPU에서 수행되지만 C++ 기반이라 충분히 빠릅니다.
    image_top = cv2.imread(tiff_filename_full, cv2.IMREAD_COLOR)
    gray_image_top = cv2.cvtColor(image_top, cv2.COLOR_BGR2GRAY)
    canny_image_top = cv2.Canny(gray_image_top, 100, 200)

    # 3. 데이터를 GPU 텐서로 변환
    # canny_image_top을 GPU로 이동 (dtype은 float 계산을 위해 미리 변환하지 않고 인덱싱 후 변환)
    canny_tensor = torch.from_numpy(canny_image_top).to(device)
    H, W = canny_tensor.shape

    # --------------------------------------------------------------------------
    # [핵심 변경] 느린 Python 루프 -> PyTorch 벡터 연산으로 대체
    # --------------------------------------------------------------------------

    # 4. 값이 0보다 큰(에지인) 픽셀의 인덱스 추출 (병렬 처리)
    # torch.nonzero: 조건을 만족하는 모든 인덱스를 (N, 2) 텐서로 반환 [[row, col], ...]
    indices = torch.nonzero(canny_tensor > 0, as_tuple=False)

    
    points_top = []

    # 에지가 하나도 없는 경우 예외 처리

    # 5. 좌표 계산 (Vectorized Operation)
    # indices[:, 0] -> Row (y축 인덱스, i[0])
    # indices[:, 1] -> Col (x축 인덱스, i[1])
    
    # 원본 수식: (width - col) / width
    # PyTorch는 나눗셈 시 자동으로 float로 형변환됩니다.
    x_coords = (W - indices[:, 1]) / W
    
    # 원본 수식: (height - row) / height
    z_coords = (H - indices[:, 0]) / H
    
    # y_min 값 채우기 (모든 점에 대해 동일)
    y_coords = torch.full_like(x_coords, y_min)
    
    # [x, y_min, z] 형태로 스택 (N, 3)
    points_top = torch.stack([x_coords, y_coords, z_coords], dim=1)



    '''
    points_top = np.array(points_top)
    #print('points_top',points_top.shape)
    if points_top.shape[0] > max_num_of_points:
        points_top_sampled_idx = np.random.choice(points_top.shape[0], size=max_num_of_points, replace=False)
        points_top = points_top[points_top_sampled_idx]

    '''
    # 6. 랜덤 샘플링 (GPU 가속)
    num_points = points_top.shape[0]
    #print(num_points)
    if num_points > max_num_of_points:
        # torch.randperm: 0부터 n-1까지의 숫자를 랜덤하게 섞은 순열 생성 (GPU에서 수행)
        perm = torch.randperm(num_points, device=device)
        
        # 앞쪽에서 max_num_of_points 개수만큼 인덱스 가져오기
        idx = perm[:max_num_of_points]
        
        # 인덱싱으로 데이터 추출
        points_top = points_top[idx]
    return points_top, image_top.shape


def tiff_list_2_pcd(_cur_y, tiff_filename_list, 
                         output_dir, tickness = 0.05, is_curvature= True, file_ext='glb', 
                         max_num_of_points_for_a_tiff_file = 100,
                         max_num_of_points_for_a_pcd  = 5000,
                         device=None, DEBUG=False):

    #warnings.filterwarnings("ignore", message=".*smallest subnormal.*", category=UserWarning)

    
    slice_filename_arr = tiff_filename_list[0].split("/")
    slice_filename_itself = slice_filename_arr[len(slice_filename_arr)-1].split(".")[0]
    pcd_filename = f'{output_dir}/{slice_filename_itself}.{file_ext}'

    os.makedirs(output_dir, exist_ok = True)

    if DEBUG: print(pcd_filename)


    xyz_list = []
    xyz_list = torch.tensor(xyz_list, dtype=torch.float32, device=device)
    #print(tiff_filename_list)
    tickness = tickness
    num_interpolated = int(tickness*100)
    #num_interpolated = 3
    _pre_points = None
    #_cur_y = image_relative_index * (tickness ) 
    tiff_filename_first = tiff_filename_list[0]

    tickess_for_a_tiff_file = ( tickness / len(tiff_filename_list))
    for _i, tiff_filename in enumerate(tiff_filename_list):
        #start_time = time.time()
        
        #print(tiff_filename)
        #print(_cur_y)

        if is_curvature:
            points_this, (h, w, c ) = tiff_2_xyz(tiff_filename, _cur_y, max_num_of_points=max_num_of_points_for_a_tiff_file, device = device)
        else:
            points_this, (h, w, c ) = tiff_2_xyz(tiff_filename_first, _cur_y,max_num_of_points=max_num_of_points_for_a_tiff_file,  device = device)

        #print(points_this.shape)
        #if DEBUG: print('points',f'{np.min(points_this, axis=0)},{np.max(points_this, axis=0)}')
        #end_time = time.time()
        #elapsed_time = end_time - start_time

        #print(f"{elapsed_time:.4f} {tiff_filename.split('/')[-1]}")
        if _pre_points is not None:
            #print(f'{_i}, {_cur_y:.3f} / {np.min(_pre_points, axis=0)[1]:.3f}, {np.max(_pre_points, axis=0)[1]:.3f} / {np.min(points_this, axis=0)[1]:.3f}, {np.max(points_this, axis=0)[1]:.3f}')
            #start_time = time.time()

            #if is_curvature:
            _xyz_list = interpolate_points_between_borders(_pre_points, points_this, num_interpolated, device = device)
            

                #_xyz_list, (h, w, c ) = tiff_2_xyz(tiff_filename_first, _cur_y + tickess_for_a_tiff_file, 
                 #                                device = device)

            #if DEBUG: print("inter",f'{np.min(_xyz_list, axis=0)},{np.max(_xyz_list, axis=0)}')
            #xyz_list.extend(_xyz_list)
            xyz_list = torch.cat([xyz_list, _xyz_list], dim=0)
            #end_time = time.time()
            #elapsed_time = end_time - start_time
            #print(f"{elapsed_time:.4f} {tiff_filename.split('/')[-1]}_inter")
        _cur_y += tickess_for_a_tiff_file
        _pre_points = points_this
   
    #points_bottom, (h, w, c ) = tiff_2_xyz(tiff_filename_full_bottom, y_max)

    # --- 사용 예시 ---



    # --- 랜덤 샘플링 ---

    # 4. 점의 개수가 기준보다 많을 경우 샘플링
    num_points = xyz_list.shape[0]
    #print(num_points)
    if num_points > max_num_of_points_for_a_pcd:
        # GPU 상에서 랜덤 순열(Permutation) 생성
        perm = torch.randperm(num_points, device=device)
        
        # 앞에서부터 필요한 개수만큼 인덱스 자르기
        sampled_indices = perm[:max_num_of_points_for_a_pcd]
        
        # 인덱싱을 통해 데이터 추출 (Slicing)
        xyz_list = xyz_list[sampled_indices]

    xyz_list[:, 0] = 1.0 - xyz_list[:, 0]
    xyz_list[:, 2] = 1.0 - xyz_list[:, 2]

    # 3. Z 좌표 스케일링 (xyz[2] *= (h/w))
    # 모든 행의 2번(z) 컬럼에 대해 곱셈 수행
    xyz_list[:, 2] *= (h / w)

    '''
    for xyz in xyz_list:
        xyz[0] = 1.0 - xyz[0] 
        xyz[2] = 1.0 - xyz[2] 

    for xyz in xyz_list:
        xyz[2] *= (h/w)
    xyz_list = np.array(xyz_list)
    #print('_xyz_list',xyz_list.shape)
    if xyz_list.shape[0] > max_num_of_points_for_a_pcd:
        xyz_list_sampled_idx = np.random.choice(xyz_list.shape[0], size=max_num_of_points_for_a_pcd, replace=False)
        xyz_list = xyz_list[xyz_list_sampled_idx]
    #print(f'{tickness},{np.min(xyz_list, axis=0)[1]:.3f},{np.max(xyz_list, axis=0)[1]:.3f}')
    #if DEBUG: print("inter",f'{np.min(_xyz_list, axis=0)},{np.max(_xyz_list, axis=0)}')
    '''
    xyz_list = xyz_list.detach().cpu().numpy()

    if file_ext == 'glb':
        point_cloud = trimesh.PointCloud(vertices=xyz_list)
        point_cloud.export(file_obj=pcd_filename)
    elif file_ext =='obj':        
        #with open(pcd_filename,'w') as f:
        #    for xyz in xyz_list:
        #        f.write(f'v {xyz[0]} {xyz[1]} {xyz[2]}\n')
        np.savetxt(pcd_filename, xyz_list, fmt='v %f %f %f')#, delimiter='\n')

    return pcd_filename

def tiff_2_pcd(num_of_missing_slices, offset_y, tiff_filename_full,tiff_filename_full2=None, output_dir=None, 
               tickness=0.001, overwrite=True, no_gap_between_slices = True):


    slice_filename_arr = tiff_filename_full.split("/")
    slice_filename_itself = slice_filename_arr[len(slice_filename_arr)-1].split(".")[0]
    #obj_filename = f'{slice_filename_itself}.obj'
    obj_filename = f'{slice_filename_itself}.glb'
    pcd_filename = f'{output_dir}/{obj_filename}'
    if not overwrite and os.path.exists(pcd_filename):
        return pcd_filename

    os.makedirs(output_dir, exist_ok = True)
    
    #row_index = int(index/10)
    #col_index = int(index%10)
    image = cv2.imread(tiff_filename_full, cv2.IMREAD_COLOR)
    #image = padding_to_image(image)
    #image = cv2.copyMakeBorder(image, 100, 100, 100, 100,cv2.BORDER_CONSTANT,value=[0,0,0])
    h, w, c = image.shape
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    canny_image = cv2.Canny(gray_image,100,200)
    
    #plt.figure(figsize=(8, 5))
    #plt.imshow(canny_image)
    #plt.title('hough_image')
    #plt.xticks([]), plt.yticks([])
    
    _image = gray_image.copy()
    contours, _ = cv2.findContours(_image, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)
    hull_area_list = []
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        hull = cv2.convexHull(contours[i])
        hull_area_list.append((contours[i], hull, area))

    hull_area_list.sort(key = lambda x :x[2], reverse=True)
    outer_points = hull_area_list[0][0].squeeze(axis=1)
    
    y_min = offset_y * tickness

    #num_of_poinst_tickness = 10
    if no_gap_between_slices:
        tickness = tickness *  (num_of_missing_slices+1)
        #num_of_poinst_tickness = 10 *  int( (num_of_missing_slices+1) )
    
    #print('y_min',y_min)
    y_max = y_min + tickness
    y_random_numbers = [random.uniform(y_min, y_max) for _ in range(5 *int((num_of_missing_slices+1)))]

    xyz_list = []
    for i, x in np.ndenumerate(canny_image):
        if x > 0 :
            xyz_list.append([(canny_image.shape[1]-i[1])/(canny_image.shape[1]), y_min, (canny_image.shape[0]-i[0])/(canny_image.shape[0])])
            xyz_list.append([(canny_image.shape[1]-i[1])/(canny_image.shape[1]), y_max, (canny_image.shape[0]-i[0])/(canny_image.shape[0])])
            for y in y_random_numbers:
                xyz_list.append([(canny_image.shape[1]-i[1])/(canny_image.shape[1]), y, (canny_image.shape[0]-i[0])/(canny_image.shape[0])])

    '''
    min_value = y_min
    max_value = tickness + y_min
    random_numbers = [random.uniform(min_value, max_value) for _ in range(50 *int((num_of_missing_slices+1)))]
    
    for i in range(outer_points.shape[0]):
        #min_value = (tickness/num_of_poinst_tickness)*1 + y_min
        #max_value = (tickness/num_of_poinst_tickness)*num_of_poinst_tickness + y_min


        # 5개의 랜덤한 소수점 숫자 생성
        
        for y in random_numbers:
            xyz_list.append([(gray_image.shape[1]-outer_points[i][0])/(gray_image.shape[1]), y, (gray_image.shape[0]-outer_points[i][1])/(gray_image.shape[0])])
    '''
    for xyz in xyz_list:
        xyz[0] = 1.0 - xyz[0] 
        #xyz[1] *= 0.1
        xyz[2] = 1.0 - xyz[2] 

    for xyz in xyz_list:
        xyz[2] *= (h/w)

        
    dx = -0.1
    dz = -0.3
    #for xyz in xyz_list:
    #    xyz[0] += dx
    #    xyz[1] += dz


    #print(f'{tiff_filename_full}_{np.min(xyz_list, axis=0)[1]}_{np.max(xyz_list, axis=0)[1]}_{y_min}_{y_max}')
    '''
    with open(pcd_filename,'w') as f:
        for xyz in xyz_list:
            f.write(f'v {xyz[0]} {xyz[1]} {xyz[2]}\n')
    '''


    point_cloud = trimesh.PointCloud(vertices=np.array(xyz_list))

    #print(f"Exporting to binary GLB format at '{glb_path}'...")
    # 'export' handles the conversion to a self-contained binary file
    point_cloud.export(file_obj=pcd_filename)
    #os.remove(pcd_filename)
    #mesh_filename = f'{slice_filename_itself}.ply'
    #print(f'{output_dir}/{mesh_filename}')
    #pcd_2_mesh(f'{output_dir}/{obj_filename}',f'{output_dir}/{mesh_filename}')
    return pcd_filename




def tiff_2_obj_parallel_test_mode(tiff_dir_root, slice_angle, tickness:float, 
                        num_of_missing_slices = 5, obj_dir_root = "" , no_gap_between_slices = True,
                from_index = 0, to_index = 0, max_num_of_slices = 19, is_curvature = True):
    obj_dir_list = []
    tasks_to_run = []
    
    tiff_dir = f'{tiff_dir_root}'
    image_filename_list = []
    for f in os.listdir(tiff_dir):
        if os.path.isdir(f):
            continue
        image_filename_list.append(tiff_dir+"/"+f)
    image_filename_list.sort(key = lambda x: int(x.split("/")[-1].split(".")[-2]))
    #print('image_filename_list',image_filename_list)
    _num_of_slices = 0
    for _i in range(0, len(image_filename_list)):
        
        _num_of_slices += 1

        if max_num_of_slices <= _num_of_slices:
            break 
        if _i+1 < len(image_filename_list):
            print(f'{_i} / top:{image_filename_list[_i].split("/")[-1]}, bottom:{image_filename_list[_i+1].split("/")[-1]}')
            image_filename_list_sub = [image_filename_list[_i], image_filename_list[_i+1]]
            tasks_to_run.append((num_of_missing_slices, _i , 
                                image_filename_list_sub, obj_dir_root+"/test/fractured_0", 
                            tickness, no_gap_between_slices))

        
        obj_dir_list.append("test/fractured_0")

    print(f'_tiff_2_obj: the number of jobs:{len(tasks_to_run)}')
    with multiprocessing.Pool() as pool: # Use a pool of 4 processes
        if is_curvature:        
            pool.starmap(tiff_list_2_pcd, tqdm(tasks_to_run, total=len(tasks_to_run), desc="tiff_2_pcd_curvature"))
        else:
            pool.starmap(tiff_2_pcd,  tqdm(tasks_to_run, total=len(tasks_to_run), desc="tiff_2_pcd"))

    return obj_dir_list
import warnings
def tiff_lilst_2_brain_obj(image_filename_list_sub,  tickness:float, 
                        num_of_missing_slices_list = 5, obj_slicing_dir = ""  
                        , num_of_slices = 19, is_curvature = True, file_ext = 'obj'):
    
    
    
    # 정규표현식으로 해당 메시지를 포함하는 모든 UserWarning 무시
    #warnings.filterwarnings("ignore", message=".*smallest subnormal.*", category=UserWarning)

    obj_dir_list = []
    
    #if os.path.exists(obj_slicing_dir):
    #    print("EXISTS: ",obj_slicing_dir.split("/")[-2])
    #    return obj_dir_list
    
    tasks_to_run = []
    start_idx = 0
    _iter_idx = 0
    #print('len(image_filename_list_sub)',len(image_filename_list_sub))
    _cur_y = 0
    while _iter_idx < num_of_slices :
        
        end_idx = start_idx + num_of_missing_slices_list[_iter_idx]
        #print('image_filename_list_sub',start_idx, end_idx)
        _image_filename_list_sub = image_filename_list_sub[start_idx:end_idx]
        
        if len(_image_filename_list_sub) <= 0:
            break


        #print(_image_filename_list_sub)
        #tasks_to_run.append((_iter_idx, _image_filename_list_sub, obj_slicing_dir, tickness, is_curvature, file_ext,
        #                         100,5000,device,False))
        adjusted_tickness = tickness * num_of_missing_slices_list[_iter_idx]
        obj_file_name = tiff_list_2_pcd(_cur_y, _image_filename_list_sub, 
                                        obj_slicing_dir, adjusted_tickness, is_curvature, file_ext,
                        max_num_of_points_for_a_tiff_file = 100,
                         max_num_of_points_for_a_pcd  = 5000,
                         device=device, DEBUG=False)
        _cur_y += (tickness * num_of_missing_slices_list[_iter_idx])
        #print(obj_file_name)
        obj_dir_list.append(obj_file_name)
        _iter_idx += 1
        start_idx = end_idx

    #with multiprocessing.Pool( ) as pool: # Use a pool of 4 processes
    #    pool.starmap(tiff_list_2_pcd, tqdm(tasks_to_run, total=len(tasks_to_run), desc="tiff_list_2_pcd"))

    return obj_dir_list

def distribute_obj_files(data_ids, tickness, num_of_missing_slices:int, obj_dir_root, from_index, to_index, max_num_of_slices, no_gap_between_slices):
    for data_id in data_ids:
        for start_index in range(0, num_of_missing_slices):
            obj_dir = f'{obj_dir_root}/{data_id}_{tickness:.4f}_{no_gap_between_slices}/fractured_0'

            obj_filename_list = []
            for f in os.listdir(obj_dir):
                obj_filename_list.append(obj_dir+"/"+f)
            obj_filename_list.sort(key = lambda x: int(x.split("/")[-1].split(".")[-2]))

            #print(obj_filename_list)
            #print(from_index,start_index,to_index,num_of_slices)
            #print(from_index + start_index, to_index, int((to_index - from_index) / num_of_slices) + 1)
            #obj_filename_list_sub = obj_filename_list[from_index + start_index : to_index : int((to_index - from_index) / spacing) + 1]
            obj_filename_list_sub = obj_filename_list[from_index + start_index : to_index : num_of_missing_slices]
            if len(obj_filename_list_sub) > max_num_of_slices:
                obj_filename_list_sub = obj_filename_list_sub[:max_num_of_slices]
            #print(obj_filename_list_sub)
            start_data_id = obj_filename_list_sub[0].split("/")[-1].split(".")[-2]
            #end_data_id = obj_filename_list_sub[-1].split("/")[-1].split(".")[-2]
            obj_slicing_dir = f'{obj_dir_root}/{data_id}_{tickness:.4f}_{no_gap_between_slices}_{num_of_missing_slices}_{start_data_id}_{to_index}/fractured_0'
            if os.path.exists(obj_slicing_dir):
                print("distribute_obj_files EXISTS:",obj_slicing_dir)
                continue
            os.makedirs(obj_slicing_dir, exist_ok = True)
            obj_file_list = []
            for orginal_filename in obj_filename_list_sub:
                filename = orginal_filename.split("/")[-1]
                target_filename = f'{obj_slicing_dir}/{filename}'
                shutil.copyfile(orginal_filename,target_filename )       


 
def load_obj(file_path):
    vertices = []
    faces = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('v '):
                vertex = [float(x) for x in line.strip().split(' ')[1:]]
                vertices.append(vertex)
            elif line.startswith('f '):
                face = [int(x.split('/')[0]) for x in line.strip().split(' ')[1:]]
                faces.append(face)
    return np.array(vertices)


def obj_augmentation(data_name, data_count, tickness, spacing,obj_dir_root, data_id,  from_index, start_index, to_index):

    obj_slicing_dir = f'{obj_dir_root}/{data_id}_{tickness:.4f}_{spacing:.3f}_{start_index}_{to_index}/fractured_0'
    obj_filename_list = []
    for f in os.listdir(obj_slicing_dir):
        obj_filename_list.append(obj_slicing_dir+"/"+f)
        

    #for aug_name, aug_func in [('random_rotation',random_rotation),('random_translation',random_translation),('random_scale',random_scale),('all',all)]:

    for data_group_id in range(data_count):
            
        angle = random.random()
        aug_name = 'rotate'
        obj_dir = f'{obj_dir_root}/{data_id}_{tickness:.4f}_{spacing:.3f}_{start_index}_{to_index}_{aug_name}_{data_name}/fractured_{data_group_id}'
        os.makedirs(obj_dir, exist_ok = True)

        for obj_filename in obj_filename_list:
            filename = obj_filename.split("/")[-1]
            target_filename = f'{obj_dir}/{filename}'
            #vertices = pytorch3d.io.load_obj(obj_filename, device='cpu')[0]
            vertices = load_obj(obj_filename)
            augmented_image, _  = slice_util.random_rotation(vertices, angle)
            
            with open(target_filename,'w') as f:
                for xyz in augmented_image:
                    f.write(f'v {xyz[0]} {xyz[1]} {xyz[2]}\n')

                        
from scipy.stats import truncnorm


def generate_bounded_gaussian_int(mean, variance, n, max_val, min_val=0):
    """
    mean: 평균
    variance: 분산
    n: 추출할 개수
    max_val: 상한값 (이 값보다 작은 정수 추출)
    min_val: 하한값 (기본값 0, 이 값보다 크거나 같은 정수 추출)
    """
    std_dev = np.sqrt(variance)
    
    # [핵심 로직: 반올림 구간 설정]
    # 정수 k가 되기 위한 실수 범위: k - 0.5 <= x < k + 0.5
    # 따라서 min_val(포함) ~ max_val(미포함) 범위를 만들려면:
    
    # 하한: min_val이 되어야 하므로 (min_val - 0.5) 부터 시작
    lower_bound_continuous = min_val - 0.5
    
    # 상한: max_val보다 작아야(즉, max_val-1) 하므로 (max_val - 0.5) 에서 끝
    upper_bound_continuous = max_val - 0.5
    
    # 표준화 (Z-score 변환)
    a = (lower_bound_continuous - mean) / std_dev
    b = (upper_bound_continuous - mean) / std_dev
    
    # 1. 절단된 정규분포에서 실수 추출
    samples_float = truncnorm.rvs(a, b, loc=mean, scale=std_dev, size=n)
    
    # 2. 반올림 후 정수 변환
    samples_int = np.round(samples_float).astype(int)
    
    return samples_int


def pad_list_to_20(input_list, none_value = ''):
    """
    주어진 리스트의 길이가 20보다 작으면 -1로 채워서 20 크기의 리스트를 반환합니다.
    """
    target_size = 20
    
    current_size = len(input_list)
    
    if current_size < target_size:
        # 1. 부족한 크기 계산
        padding_needed = target_size - current_size
        
        # 2. -1로 채워진 패딩 리스트 생성
        padding_list = [none_value] * padding_needed
        
        # 3. 입력 리스트와 패딩 리스트 결합
        result_list = input_list + padding_list
        
        return result_list
    else:
        # 리스트 크기가 20 이상이면, (요구사항에 따라) 앞 20개만 반환합니다.
        # 만약 크기 조절 없이 원본을 그대로 반환하려면 return input_list 로 변경합니다.
        return input_list[:target_size]

def gen_num_of_missing_slices(mode, from_index, to_index, _num_of_slices):
    
    _num_of_missing_slices_list = []

    #_max_num_of_missing_slices = int((to_index - from_index)/ 1)
    _num_of_missing_slices = int((to_index - from_index)/ _num_of_slices)
    _num_of_missing_slices = _num_of_missing_slices - 2
    
    if mode == 'Uniform':
        
        adjusted_to_index = from_index
        for _i in range(_num_of_slices):
            _num_of_missing_slices_list.append(_num_of_missing_slices)
            adjusted_to_index += _num_of_missing_slices
            adjusted_to_index += 2                            
        adjusted_to_index -= 1

    elif mode == 'Gaussian':

        mu = _num_of_missing_slices
        var = 100
        n_samples = _num_of_slices
        limit_m = int((to_index - from_index)/ 1)  # 평균인 10보다 작은 수만 추출
        _tries = 0
        while _tries < 10:
            _num_of_missing_slices_list = generate_bounded_gaussian_int(mu, var, n_samples, limit_m, 0)
            adjusted_to_index = from_index
            for _i in range(_num_of_slices):
                adjusted_to_index += _num_of_missing_slices_list[_i]
                adjusted_to_index += 2                            
            adjusted_to_index -= 1
            _tries += 1
            if (adjusted_to_index <= to_index) and (adjusted_to_index >= to_index - mu ):
                break


        if _tries == 10:
            return None, None

    return _num_of_missing_slices_list, adjusted_to_index


import sys


import pandas as pd
from SlicedVolumeDataset import SlicedVolumeDataset

#DEBUG_MODE = True

from torch.utils.data import DataLoader


def create_data_loader(dataset_annotation_file_name):
 

    _dataset = SlicedVolumeDataset(annotation_file=dataset_annotation_file_name)

    BATCH_SIZE = 1
    NUM_WORKERS = 4  # 데이터를 로드할 프로세스 수 (일반적으로 2, 4, 8 등)
    # -----------------------------------------------



    data_loader = DataLoader(
        _dataset,
        batch_size=BATCH_SIZE,
        shuffle=False, 
        num_workers=NUM_WORKERS,
        # 멀티 레이블에서는 Drop-Last(마지막 남은 배치 버리기)를 True로 설정하는 경우가 많습니다.
        drop_last=True 
    )

    return data_loader
def create_dataset(dataset_annotation_file_name, tiff_dir_root, obj_dir_root ):

        
    os.makedirs(obj_dir_root, exist_ok=True)

    slice_angle_list = os.listdir(tiff_dir_root)
    slice_angle_list = [w for w in slice_angle_list if os.path.isdir(tiff_dir_root+"/"+w)]
    #if DEBUG_MODE: data_ids = data_ids[:1]


    tickness_of_a_slice_list = [0.003]
    is_curvature_list = [False,True]
    num_of_missing_slices_dist_list = ['Uniform','Gaussian']
    #num_of_missing_slices_dist_list = ['Uniform']
    #num_of_missing_slices_list = sorted(list(range(0, 6, 1))) #[0, 1, 2, 3, 4, 5] # 10, 15, 20, 15, 30, 35, 40, 45, 50]
    #num_of_missing_slices_list = [50]
    #from_index_list = [100]
    #if DEBUG_MODE: from_index_list = [100]

    #from_to_index_list = [(100,)]    
    #print('slice_angle: ',slice_angle_list)
    #print('is_curvature: ',is_curvature_list)
    #print('num_of_missing_slices_dist_list: ',num_of_missing_slices_dist_list)
    #print('from_index: ',from_index_list)
    #print('tickness: ',tickness_of_a_slice_list)



    slice_angle_to_range_map = {}
    slice_angle_to_range_map['sliced_on_1_0_0'] = (90,210)
    slice_angle_to_range_map['sliced_on_1_0_1'] = (170,400)
    slice_angle_to_range_map['sliced_on_0_0_1'] = (260,400)
    slice_angle_to_range_map['sliced_on_1_1_0'] = (80,300)
    slice_angle_to_range_map['sliced_on_0_1_1'] = (120,570)
    slice_angle_to_range_map['sliced_on_0_1_0'] = (100,700)
    slice_angle_to_range_map['sliced_on_1_1_0'] = (70,490)
    slice_angle_to_range_map['sliced_on_1_1_1'] = (430,550)
    
    slice_angle_to_range_map['sliced_on_1_0_0'] = (0,366)
    slice_angle_to_range_map['sliced_on_1_0_1'] = (0,490)
    slice_angle_to_range_map['sliced_on_0_0_1'] = (61,398)
    slice_angle_to_range_map['sliced_on_1_1_0'] = (5,608)
    slice_angle_to_range_map['sliced_on_0_1_1'] = (6,569)
    slice_angle_to_range_map['sliced_on_0_1_0'] = (5,797)
    slice_angle_to_range_map['sliced_on_1_1_0'] = (5,608)
    slice_angle_to_range_map['sliced_on_1_1_1'] = (0,640)


    num_of_slices_list = sorted(list(range(2, 19))) #[100, 150, 200, 250, 300, 0, 50]
    max_num_of_slices = np.max(num_of_slices_list)

    sliced_volumn_dataset_list = []

    for slice_angle in slice_angle_list:
        _from_index = slice_angle_to_range_map[slice_angle][0]
        _to_index = slice_angle_to_range_map[slice_angle][1]

        tiff_dir = f'{tiff_dir_root}/{slice_angle}'
        image_filename_list = []
        for f in os.listdir(tiff_dir):
            if os.path.isdir(f):
                continue
            image_filename_list.append(tiff_dir+"/"+f)
        image_filename_list.sort(key = lambda x: int(x.split("/")[-1].split(".")[-2]))


        from_index_list = sorted(list(range(_from_index, _to_index - max_num_of_slices, 50)))
        to_index_list = sorted(list(range(_to_index, _from_index + max_num_of_slices, -50)))

        for from_index in from_index_list:        
            for to_index in to_index_list:

                if from_index >= to_index :
                    continue
                for _num_of_slices in num_of_slices_list:

                    for num_of_missing_slices_dist in num_of_missing_slices_dist_list:
                       

                        _num_of_missing_slices_list, adjusted_to_index = gen_num_of_missing_slices(num_of_missing_slices_dist, 
                                                                                                   from_index, to_index,_num_of_slices)

                        if adjusted_to_index is None:
                            #print('missing slices not',num_of_missing_slices_dist, from_index, to_index,_num_of_slices)
                            continue
                        #adjusted_to_index = from_index + (num_of_missing_slices + 2) * (_num_of_slices -1) + (num_of_missing_slices+1)

                        image_filename_list_for_one_sub_brain = image_filename_list[from_index  : adjusted_to_index + 1]
                        if len(image_filename_list_for_one_sub_brain) <= 0:
                            #print("Empty",slice_angle, from_index, to_index, adjusted_to_index, _num_of_slices)
                            continue
                        #image_filename_list_for_one_sub_brain = pad_list_to_20(image_filename_list_for_one_sub_brain, '')
                        for is_curvature in is_curvature_list:
                            for tickness in tickness_of_a_slice_list:
                                
                                
                                sliced_volumn_dataset = {}
                                sliced_volumn_dataset['obj_dir_root'] = obj_dir_root
                                sliced_volumn_dataset['image_filename_list_for_one_sub_brain'] = ','.join(image_filename_list_for_one_sub_brain)
                                sliced_volumn_dataset['slice_angle'] = slice_angle
                                sliced_volumn_dataset['tickness'] = tickness
                                sliced_volumn_dataset['from_index'] = from_index
                                sliced_volumn_dataset['to_index'] = adjusted_to_index + 1
                                sliced_volumn_dataset['num_of_missing_slices_dist'] = num_of_missing_slices_dist
                                _num_of_missing_slices_list_str = [str(item) for item in _num_of_missing_slices_list]
                                sliced_volumn_dataset['num_of_missing_slices_list'] = ','.join(_num_of_missing_slices_list_str)
                                sliced_volumn_dataset['is_curvature'] = is_curvature
                                sliced_volumn_dataset['num_of_slices'] = _num_of_slices
                                sliced_volumn_dataset_list.append(sliced_volumn_dataset)
                                #if _num_of_slices <= 4 and from_index == _from_index and adjusted_to_index > _to_index - 50:
                                #    sliced_volumn_dataset_list.append(sliced_volumn_dataset)

                                #obj_2_pcd.tiff_2_obj_parallel(**sliced_volumn_dataset)
                                '''
                                print(sliced_volumn_dataset['image_filename_list_for_one_sub_brain'])
                                if len(sliced_volumn_dataset_list) > 3:
                                    keys_list = [
                                    'obj_dir_root',
                                    'image_filename_list_for_one_sub_brain',
                                    'slice_angle',
                                    'tickness',
                                    'from_index',
                                    'to_index',
                                    'num_of_missing_slices_dist',
                                    'num_of_missing_slices_list',
                                    'is_curvature',
                                    'num_of_slices'
                                    ]
                                    
                                    _df = pd.DataFrame(sliced_volumn_dataset_list)
                                    _df.columns = keys_list
                                    _df.to_csv(dataset_annotation_file_name, index=None)

                                    return _df
                                '''

    #if len(sliced_volumn_dataset_list) >= 100:
    keys_list = [
    'obj_dir_root',
    'image_filename_list_for_one_sub_brain',
    'slice_angle',
    'tickness',
    'from_index',
    'to_index',
    'num_of_missing_slices_dist',
    'num_of_missing_slices_list',
    'is_curvature',
    'num_of_slices'
    ]
    
    _df = pd.DataFrame(sliced_volumn_dataset_list)
    _df.columns = keys_list
    _df.to_csv(dataset_annotation_file_name, index=None)

    return _df


if __name__ == "__main__":




    parser = argparse.ArgumentParser(
        description="convert tiff to obj/glb",
        formatter_class=argparse.RawTextHelpFormatter)
        
    parser.add_argument("--tiff_dir_root",required=True, help="Path to the input TIFF file.")
    parser.add_argument("--obj_dir_root",required=True, help="Base directory to save the parameter-named output folder.")
    args = parser.parse_args()


    dataset_annotation_file_name = args.obj_dir_root+"/data.csv"

    _df  = create_dataset(dataset_annotation_file_name, args.tiff_dir_root, args.obj_dir_root)
    #print(_df.head())
    data_loader = create_data_loader(dataset_annotation_file_name)
    print(f"DataLoader 생성 완료. 총 배치 개수: {len(data_loader)}")

        

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")


    ### 2. DataLoader 순회 (Iteration)
    tasks_to_run = []
    # 일반적으로 훈련 루프(Training Loop)에서 사용됩니다.


    # DataLoader를 순회하며 배치 단위로 데이터(이미지)와 레이블을 가져옵니다.
    for batch_idx, (tiff_images, labels, output_dir) in enumerate(data_loader):
        

        #if batch_idx > 10:
        #    break

        missing_slices_list = [t.item() for t in labels['missing_slices_list']]
        _tiff_images = [t[0] for t in tiff_images]
        _output_dir = output_dir[0]
        if os.path.exists(_output_dir):
            print("EXISTS:"+_output_dir)
            continue
        #print((_tiff_images, labels['tickness'],missing_slices_list, _output_dir, labels['num_of_slices'].item(), labels['is_curvature'].item(),'glb', device))
        tasks_to_run.append((_tiff_images, labels['tickness'].item()
                ,missing_slices_list, _output_dir, labels['num_of_slices'].item(), labels['is_curvature'].item(),'glb'))
        

    mp.set_start_method('spawn', force=True)
    with mp.Pool( ) as pool: # Use a pool of 4 processes
        pool.starmap(tiff_lilst_2_brain_obj, tqdm(tasks_to_run, total=len(tasks_to_run), desc="_tiff_2_pcd_func"))
    print("DONE!")