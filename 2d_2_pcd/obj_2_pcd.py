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

def tiff_2_pcd(offset_y, tiff_filename_full, output_dir, tickness, overwrite=True):


    slice_filename_arr = tiff_filename_full.split("/")
    slice_filename_itself = slice_filename_arr[len(slice_filename_arr)-1].split(".")[0]
    obj_filename = f'{slice_filename_itself}.obj'
    pcd_filename = f'{output_dir}/{obj_filename}'
    if not overwrite and os.path.exists(pcd_filename):
        return pcd_filename

    os.makedirs(output_dir, exist_ok = True)
    num_of_poinst_tickness = 10
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

    
    y_min = offset_y
    #print('y_min',y_min)
    y_max = y_min + tickness
    xyz_list = []
    for i, x in np.ndenumerate(canny_image):
        if x > 0 :
            xyz_list.append([(canny_image.shape[1]-i[1])/(canny_image.shape[1]), y_min, (canny_image.shape[0]-i[0])/(canny_image.shape[0])])
            xyz_list.append([(canny_image.shape[1]-i[1])/(canny_image.shape[1]), y_max, (canny_image.shape[0]-i[0])/(canny_image.shape[0])])
    
    for i in range(outer_points.shape[0]):
        min_value = (tickness/num_of_poinst_tickness)*1 + y_min
        max_value = (tickness/num_of_poinst_tickness)*num_of_poinst_tickness + y_min
        min_value = y_min
        max_value = tickness + y_min

        # 5개의 랜덤한 소수점 숫자 생성
        random_numbers = [random.uniform(min_value, max_value) for _ in range(100)]
        for y in random_numbers:
            xyz_list.append([(gray_image.shape[1]-outer_points[i][0])/(gray_image.shape[1]), y, (gray_image.shape[0]-outer_points[i][1])/(gray_image.shape[0])])
    
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

    with open(pcd_filename,'w') as f:
        for xyz in xyz_list:
            f.write(f'v {xyz[0]} {xyz[1]} {xyz[2]}\n')

    #mesh_filename = f'{slice_filename_itself}.ply'
    #print(f'{output_dir}/{mesh_filename}')
    #pcd_2_mesh(f'{output_dir}/{obj_filename}',f'{output_dir}/{mesh_filename}')
    return pcd_filename



def main():        
    img_dir = '/data/jhahn/data/tg/0083/Mouse_0083_Lhem_cFos_647_PV_488_Sagittal/Atlas_Registration/Dembamaps'
    img_dir = '/data/jhahn/data/brain_lightsheet'


    #tg_names = ['A0242','A0244','A0245','A0247','A0242_manual','A0244_manual','A0245_manual','A0247_manual']
    tg_names = ['A0242','A0244','A0245','A0247','A0247']
    tg_names = ['0408_from_0088','0408_from_0089','0408_from_0090','0408_from_0091','0408_from_0092','0408_from_0093']

    for prefix in  tg_names:

        _img_dir = f'{img_dir}/{prefix}'
        obj_dir = f'/data/jhahn/data/shape_dataset/data/brain_lightsheet/{prefix}/fractured_0'
        try:
            os.makedirs(obj_dir)
        except:
            pass

        print(_img_dir)
        tickness = 0.02

        index_list = []
        slice_file_list = []
        output_dir = []
        tickness_list = []

        img_files = os.listdir(_img_dir)
        img_files = [f for f in img_files if f.endswith('.tif')]
        #for img_file_name in img_files:
        #    if "r.png" in img_file_name or "g.png" in img_file_name  or  "b.png" in img_file_name :
        #        img_file_name

        for _i, img_file_name in enumerate(img_files):
            slice_file_list.append(f'{_img_dir}/{img_file_name}')
            output_dir.append(obj_dir)
            tickness_list.append(tickness)
            index_list.append(_i)
            #pcd_gen(_i, f'{_img_dir}/{img_file_name}', obj_dir, tickness)
            #if True:
            #    break



        print(f'the number of jobs:{len(img_files)}')
        with multiprocessing.Pool() as pool: # Use a pool of 4 processes
            pool.starmap(pcd_gen, zip(index_list,slice_file_list, output_dir, tickness_list))






def tiff_2_obj_parallel(tiff_dir_root, data_ids, tickness, spacing, obj_dir_root):
    offset_y_list = []
    tiff_filename_list = []
    obj_dir_root_list = []
    tickness_list = []
    obj_dir_list = []

    if data_ids is None:
        tiff_dir = f'{tiff_dir_root}'
        image_filename_list = []
        for f in os.listdir(tiff_dir):
            if os.path.isdir(f):
                continue
            image_filename_list.append(tiff_dir+"/"+f)
        image_filename_list.sort(key = lambda x: int(x.split("/")[-1].split(".")[-2]))

        obj_dir = f'{obj_dir_root}/{tickness:.3f}_{spacing:.3f}/fractured_0'
        os.makedirs(obj_dir, exist_ok = True)
        for _i, orginal_filename in enumerate(image_filename_list):
            offset_y_list.append(_i*spacing)
            tiff_filename_list.append(orginal_filename)
            obj_dir_root_list.append(obj_dir)
            tickness_list.append(tickness)
        obj_dir_list.append(f'{tickness:.3f}_{spacing:.3f}')
    else:
            
        for data_id in data_ids:


            tiff_dir = f'{tiff_dir_root}/{data_id}'
            image_filename_list = []
            for f in os.listdir(tiff_dir):
                if os.path.isdir(f):
                    continue
                image_filename_list.append(tiff_dir+"/"+f)
            image_filename_list.sort(key = lambda x: int(x.split("/")[-1].split(".")[-2]))

            obj_dir = f'{obj_dir_root}/{data_id}_{tickness:.3f}_{spacing:.3f}/fractured_0'
            os.makedirs(obj_dir, exist_ok = True)
            for _i, orginal_filename in enumerate(image_filename_list):
                offset_y_list.append(_i*spacing)
                tiff_filename_list.append(orginal_filename)
                obj_dir_root_list.append(obj_dir)
                tickness_list.append(tickness)
            obj_dir_list.append(f'{data_id}_{tickness:.3f}_{spacing:.3f}/fractured_0')


    print(f'_tiff_2_obj: the number of jobs:{len(tiff_filename_list)}')
    with multiprocessing.Pool() as pool: # Use a pool of 4 processes
        pool.starmap(tiff_2_pcd, zip(offset_y_list,tiff_filename_list, obj_dir_root_list, tickness_list))

    return obj_dir_list

def distribute_obj_files(data_ids, tickness, spacing, obj_dir_root, from_index, to_index, num_of_slices):
    for data_id in data_ids:
        for start_index in range(int((to_index-from_index)/num_of_slices)):
            obj_dir = f'{obj_dir_root}/{data_id}_{tickness:.3f}_{spacing:.3f}/fractured_0'

            obj_filename_list = []
            for f in os.listdir(obj_dir):
                obj_filename_list.append(obj_dir+"/"+f)
            obj_filename_list.sort(key = lambda x: int(x.split("/")[-1].split(".")[-2]))


            obj_filename_list_sub = obj_filename_list[from_index + start_index : to_index : int((to_index - from_index) / num_of_slices) + 1]

            start_data_id = obj_filename_list_sub[0].split("/")[-1].split(".")[-2]
            #end_data_id = obj_filename_list_sub[-1].split("/")[-1].split(".")[-2]
            obj_slicing_dir = f'{obj_dir_root}/{data_id}_{tickness:.3f}_{spacing:.3f}_{start_data_id}_{to_index}/fractured_0'
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

    obj_slicing_dir = f'{obj_dir_root}/{data_id}_{tickness:.3f}_{spacing:.3f}_{start_index}_{to_index}/fractured_0'
    obj_filename_list = []
    for f in os.listdir(obj_slicing_dir):
        obj_filename_list.append(obj_slicing_dir+"/"+f)
        

    #for aug_name, aug_func in [('random_rotation',random_rotation),('random_translation',random_translation),('random_scale',random_scale),('all',all)]:

    for data_group_id in range(data_count):
            
        angle = random.random()
        aug_name = 'rotate'
        obj_dir = f'{obj_dir_root}/{data_id}_{tickness:.3f}_{spacing:.3f}_{start_index}_{to_index}_{aug_name}_{data_name}/fractured_{data_group_id}'
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

                        
            










if __name__ == "__main__":

    data_ids = ['0408']

    tiff_dir_root = '/data/jhahn/data/brain_lightsheet'
    obj_dir_root = '/data/jhahn/data/shape_dataset/data/brain_lightsheet'

    data_ids = os.listdir(tiff_dir_root)
    data_ids = [w for w in data_ids if os.path.isdir(tiff_dir_root+"/"+w) and w.startswith("mask_")]


    from_index=100
    start_index=0
    to_index=200
    num_of_slices=20
    tickness_list_const = [0.001, 0.003, 0.005]
    
    spacing = 0.007
    
    for tickness in tickness_list_const:
        print(tickness)
        tiff_2_obj_parallel(tiff_dir_root, data_ids, tickness,spacing,obj_dir_root)
        distribute_obj_files(data_ids, tickness, spacing, obj_dir_root, from_index, to_index, num_of_slices)


    tickness_list = []
    obj_dir_root_list = []
    data_id_list = []
    from_index_list = []
    start_index_list = []
    to_index_list = []

    data_name_list = []
    data_count_list = []

    for data_id in data_ids:
        for tickness in tickness_list_const:
            obj_dir = f'{obj_dir_root}/{data_id}_{tickness:.3f}_{spacing:.3f}/fractured_0'
            obj_filename_list = []
            for f in os.listdir(obj_dir):
                obj_filename_list.append(obj_dir+"/"+f)
            obj_filename_list.sort(key = lambda x: int(x.split("/")[-1].split(".")[-2]))

            for data_name, data_count in [('train',20), ('val',2), ('test',1) ]:
                    


                for start_index in range(int((to_index-from_index)/num_of_slices)):

                            
                    obj_filename_list_sub = obj_filename_list[from_index + start_index : to_index : int((to_index - from_index) / num_of_slices) + 1]

                    start_data_id = obj_filename_list_sub[0].split("/")[-1].split(".")[-2]

                    data_name_list.append(data_name)
                    data_count_list.append(data_count)
                    
                    tickness_list.append(tickness)
                    obj_dir_root_list.append(obj_dir_root)
                    data_id_list.append(data_id)
                    from_index_list.append(from_index)
                    start_index_list.append(start_data_id)
                    to_index_list.append(to_index)

                    #if True:
                #    break

                    
            #obj_augmentation(obj_dir_root, '0408', '0088', '0089','215','0')
    print(f'obj_augmentation: the number of jobs:{len(to_index_list)}')
    with multiprocessing.Pool() as pool: # Use a pool of 4 processes
        pool.starmap(obj_augmentation, zip(data_name_list,data_count_list,tickness_list, obj_dir_root_list, data_id_list, from_index_list, start_index_list, to_index_list))

