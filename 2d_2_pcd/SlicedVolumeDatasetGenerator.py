import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple, Any
from pathlib import Path
import cv2
import argparse
import trimesh
import random
import multiprocessing
from tqdm import tqdm # 1. tqdm 라이브러리를 임포트합니다.
import open3d as o3d

# 특징을 구분하는 구분자
DELIMITER = '_'

class SlicedVolumeDatasetGenerator:
    """
    특징 목록을 기반으로 데이터셋 폴더 구조와 더미 이미지 파일을 생성합니다.
    """
    def __init__(self, sliced_volume_dir: str):
        self.sliced_volume_dir = sliced_volume_dir
        os.makedirs(self.sliced_volume_dir, exist_ok=True)
        print(f"Generator initialized. Root: {self.sliced_volume_dir} ")

    def generate_dummy_data(self, features_list: List[Dict[str, Any]], num_images_per_folder: int = 5):
        """
        주어진 특징 리스트에 따라 폴더를 생성하고, 더미 이미지를 채웁니다.
        
        Args:
            features_list (List[Dict]): 생성할 각 클래스의 특징 딕셔너리 리스트.
            num_images_per_folder (int): 각 폴더에 생성할 더미 이미지 개수.
        """
        print("\n--- Generating Dummy Dataset Structure ---")
        
        for features in features_list:
            # 특징 딕셔너리를 폴더 이름 형식으로 변환
            # 예: ['0.707', '0.707', '0.0', '1.5um', 'FULL', '500'] -> "0.707_0.707_0.0_1.5um_FULL_500"
            folder_name = f"{features['VectorX']}{DELIMITER}{features['VectorY']}{DELIMITER}{features['VectorZ']}{DELIMITER}{features['SliceThickness']}{features['ThicknessUnit']}{DELIMITER}{features['Scope']}{DELIMITER}{features['SliceCount']}"
            
            class_dir = self.root_dir / folder_name
            os.makedirs(class_dir, exist_ok=True)
            
            # 더미 이미지 생성
            for i in range(num_images_per_folder):
                # 224x224 RGB 더미 이미지 (랜덤 노이즈)
                dummy_array = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
                dummy_image = Image.fromarray(dummy_array)
                
                img_path = class_dir / f"img_{i+1}.png"
                dummy_image.save(img_path)
                
            print(f"  Created folder: {folder_name} with {num_images_per_folder} images.")

    def get_foldername(self, slice_angle, tickness, no_gap_between_slices, num_of_missing_slices, start_data_id, is_curvature, end_data_id, num_of_slices):
        
        return f'{slice_angle}_{tickness:.4f}_{no_gap_between_slices}_{num_of_missing_slices}_{start_data_id}_{is_curvature}_{end_data_id}_HIP_{num_of_slices}/fractured_0'

    def tiffs_2_volumn(self, image_filename_list_sub, slice_angle, tickness:float, 
                            num_of_missing_slices = 5, obj_dir_root = "" , no_gap_between_slices = True,
                    from_index = 0, to_index = 0, num_of_slices = 19, is_curvature = True):
        obj_dir_list = []
        

        '''
        if len(image_filename_list_sub) > max_num_of_slices:
            image_filename_list_sub = image_filename_list_sub[:max_num_of_slices]
        elif len(image_filename_list_sub) <= 3:
            return obj_dir_list
        '''
        #print('image_filename_list_sub',image_filename_list_sub)
        start_data_id = image_filename_list_sub[0].split("/")[-1].split(".")[-2]
        end_data_id = image_filename_list_sub[-1].split("/")[-1].split(".")[-2]
        obj_slicing_dir = f'{obj_dir_root}/{self.get_foldername(slice_angle, tickness, no_gap_between_slices, num_of_missing_slices, start_data_id, is_curvature, end_data_id, num_of_slices)}
        
        if os.path.exists(obj_slicing_dir):
            print("EXISTS: ",obj_slicing_dir.split("/")[-2])
            return obj_dir_list
        
        
        if is_curvature:        
            _tiff_2_pcd_func = self.tiff_2_pcd_curvature
        else:
            _tiff_2_pcd_func = self.tiff_2_pcd
        tasks_to_run = []
        for _i in range(0, len(image_filename_list_sub)-num_of_missing_slices+2, num_of_missing_slices):
            _image_filename_list_sub = image_filename_list_sub[_i:_i+num_of_missing_slices]
            #print('image_filename_list_sub',_image_filename_list_sub)
            #tasks_to_run.append((num_of_missing_slices, _i, 
            #                    _image_filename_list_sub, 
            #                    obj_slicing_dir, tickness, no_gap_between_slices))
            _tiff_2_pcd_func(num_of_missing_slices, _i, 
                                _image_filename_list_sub, 
                                obj_slicing_dir, tickness, no_gap_between_slices)
        

        #with multiprocessing.Pool( ) as pool: # Use a pool of 4 processes
        #    pool.starmap(_tiff_2_pcd_func, tqdm(tasks_to_run, total=len(tasks_to_run), desc="_tiff_2_pcd_func"))

        return obj_dir_list

    def tiff_2_pcd_curvature(self,num_of_missing_slices, image_relative_index, tiff_filename_list, 
                            output_dir, tickness, no_gap_between_slices = True, DEBUG=True):


        
        slice_filename_arr = tiff_filename_list[0].split("/")
        slice_filename_itself = slice_filename_arr[len(slice_filename_arr)-1].split(".")[0]
        obj_filename = f'{slice_filename_itself}.obj'
        #obj_filename = f'{slice_filename_itself}.glb'
        pcd_filename = f'{output_dir}/{obj_filename}'
        #if not overwrite and os.path.exists(pcd_filename):
        #    return pcd_filename

        os.makedirs(output_dir, exist_ok = True)

        if DEBUG: print(obj_filename)


        if no_gap_between_slices:
            y_min = image_relative_index * (tickness )
            y_max = y_min + (tickness )
        else:
            y_min = image_relative_index * tickness
            y_max = y_min + tickness
        if DEBUG: print(image_relative_index, y_min , y_max)

        num_interpolated = int(tickness*1000)
        #num_interpolated = 3
        _pre_points = None
        _cur_y = y_min
        xyz_list = []
        #print(tiff_filename_list)
        for _i, tiff_filename in enumerate(tiff_filename_list):
            #start_time = time.time()
            
            #print(tiff_filename)
            points_this, (h, w, c ) = self.tiff_2_xyz(tiff_filename, _cur_y)
            print(points_this.shape)
            #if DEBUG: print('points',f'{np.min(points_this, axis=0)},{np.max(points_this, axis=0)}')
            #end_time = time.time()
            #elapsed_time = end_time - start_time

            #print(f"{elapsed_time:.4f} {tiff_filename.split('/')[-1]}")
            


            

            if _pre_points is not None:
                #print(f'{_i}, {_cur_y:.3f} / {np.min(_pre_points, axis=0)[1]:.3f}, {np.max(_pre_points, axis=0)[1]:.3f} / {np.min(points_this, axis=0)[1]:.3f}, {np.max(points_this, axis=0)[1]:.3f}')
                #start_time = time.time()
                _xyz_list = self.interpolate_points_between_borders(_pre_points, points_this, num_interpolated)
                #if DEBUG: print("inter",f'{np.min(_xyz_list, axis=0)},{np.max(_xyz_list, axis=0)}')
                xyz_list.extend(_xyz_list)
                #end_time = time.time()
                #elapsed_time = end_time - start_time
                #print(f"{elapsed_time:.4f} {tiff_filename.split('/')[-1]}_inter")
            _cur_y += tickness
            _pre_points = points_this


        #points_bottom, (h, w, c ) = tiff_2_xyz(tiff_filename_full_bottom, y_max)

        # --- 사용 예시 ---


        
        for xyz in xyz_list:
            xyz[0] = 1.0 - xyz[0] 
            xyz[2] = 1.0 - xyz[2] 

        for xyz in xyz_list:
            xyz[2] *= (h/w)
        xyz_list = np.array(xyz_list)
        #print('_xyz_list',xyz_list.shape)
        if xyz_list.shape[0] > 5000:
            xyz_list_sampled_idx = np.random.choice(xyz_list.shape[0], size=5000, replace=False)
            xyz_list = xyz_list[xyz_list_sampled_idx]
        #print(f'{tickness},{np.min(xyz_list, axis=0)[1]:.3f},{np.max(xyz_list, axis=0)[1]:.3f}')
        #if DEBUG: print("inter",f'{np.min(_xyz_list, axis=0)},{np.max(_xyz_list, axis=0)}')
        
        #point_cloud = trimesh.PointCloud(vertices=xyz_list)
        #point_cloud.export(file_obj=pcd_filename)

        
        with open(pcd_filename,'w') as f:
            for xyz in xyz_list:
                f.write(f'v {xyz[0]} {xyz[1]} {xyz[2]}\n')
        
        return pcd_filename

    def interpolate_points_between_borders(self, points1, points2, num_interpolated_points):
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

    def tiff_2_pcd(self,num_of_missing_slices, offset_y, tiff_filename_full,tiff_filename_full2=None, output_dir=None, 
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


    def tiff_2_xyz(self,tiff_filename_full, y):

        image_top = cv2.imread(tiff_filename_full, cv2.IMREAD_COLOR)
        
        #new_width, new_height = 150, 150
        #image_top = cv2.resize(image_top, (new_width, new_height))

        gray_image_top = cv2.cvtColor(image_top, cv2.COLOR_BGR2GRAY)
        canny_image_top = cv2.Canny(gray_image_top,100,200)
        
        points_top = []
        for i, x in np.ndenumerate(canny_image_top):
            if x > 0 :
                points_top.append([(canny_image_top.shape[1]-i[1])/(canny_image_top.shape[1]), y, (canny_image_top.shape[0]-i[0])/(canny_image_top.shape[0])])
        points_top = np.array(points_top)
        #print('points_top',points_top.shape)
        if points_top.shape[0] > 100:
            points_top_sampled_idx = np.random.choice(points_top.shape[0], size=100, replace=False)
            points_top = points_top[points_top_sampled_idx]

        return points_top, image_top.shape

    def run(self, tiff_dir_root, num_of_slices_list, is_curvature_list, num_of_missing_slices_list, tickness_list, slice_angle_to_range_map):
        tasks_to_run = []

        tiff_sub_folders = os.listdir(tiff_dir_root)
        tiff_sub_folders = [w for w in tiff_sub_folders if os.path.isdir(tiff_dir_root+"/"+w)]

        for slice_angle in tiff_sub_folders:
                _from_index = slice_angle_to_range_map[slice_angle][0]
                to_index = slice_angle_to_range_map[slice_angle][1]

                tiff_dir = f'{tiff_dir_root}/{slice_angle}'
                image_filename_list = []
                for f in os.listdir(tiff_dir):
                    if os.path.isdir(f):
                        continue
                    image_filename_list.append(tiff_dir+"/"+f)
                image_filename_list.sort(key = lambda x: int(x.split("/")[-1].split(".")[-2]))

                for _num_of_slices in num_of_slices_list:
                    num_of_missing_slices = int((to_index - _from_index)/_num_of_slices)
                    #for num_of_missing_slices in num_of_missing_slices_list[:1]:
                    if num_of_missing_slices <= 5:
                        continue
                    from_index_list = sorted(list(range(_from_index, to_index - num_of_missing_slices, 1)))
                    from_index_list = sorted(list(range(_from_index, _from_index+30, 1)))
                    #from_index_list = [_from_index]
                    for is_curvature in is_curvature_list:
                        for no_gap_between_slices in no_gap_between_slices_list:
                            for tickness in tickness_list:
                                for from_index in from_index_list:

                                    image_filename_list_sub = image_filename_list[from_index  : to_index ]
                                    _num_of_slices = int((len(image_filename_list_sub)+1)/(num_of_missing_slices )) 
                                    #print(from_index, to_index, int(( len(image_filename_list_sub)+1)/(num_of_missing_slices )))
                                    if  _num_of_slices <= 19:

                                        #print(from_index, num_of_missing_slices, len(image_filename_list_sub))
                                        tasks_to_run.append(( image_filename_list_sub, slice_angle, tickness, num_of_missing_slices,  
                                                        self.sliced_volume_dir, no_gap_between_slices, 
                                                        from_index, to_index, _num_of_slices, is_curvature))
                                        #tiff_2_obj_parallel(image_filename_list_sub, slice_angle, tickness, num_of_missing_slices,  
                                        #                args.obj_dir_root, no_gap_between_slices, 
                                        #                from_index, to_index, _num_of_slices, is_curvature)
                                        #exit()
        tasks_to_run = tasks_to_run[:1]
        print(f'tiff_2_obj_parallel: the number of jobs:{len(tasks_to_run)}')

        
        with multiprocessing.Pool( ) as pool: # Use a pool of 4 processes
            pool.starmap(self.tiffs_2_volumn, tqdm(tasks_to_run, total=len(tasks_to_run), desc="tiff_2_obj_parallel"))



