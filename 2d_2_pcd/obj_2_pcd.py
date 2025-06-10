import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import multiprocessing
from vedo import dataurl, printc, Plotter, Points, Mesh, Text2D
import torch

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


def random_rotation(points):
    """Rotate point cloud by random angle around a random axis"""
    axis = np.random.randn(3)
    axis = axis/np.linalg.norm(axis)
    angle = np.random.uniform(0, 2*np.pi)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    rotation_matrix = np.array([
        [cos_a + (1-cos_a)*axis[0]**2, (1-cos_a)*axis[0]*axis[1] - sin_a*axis[2], (1-cos_a)*axis[0]*axis[2] + sin_a*axis[1]],
        [(1-cos_a)*axis[0]*axis[1] + sin_a*axis[2], cos_a + (1-cos_a)*axis[1]**2, (1-cos_a)*axis[1]*axis[2] - sin_a*axis[0]],
        [(1-cos_a)*axis[0]*axis[2] - sin_a*axis[1], (1-cos_a)*axis[1]*axis[2] + sin_a*axis[0], cos_a + (1-cos_a)*axis[2]**2]
    ])
    rotated_points = np.dot(points, rotation_matrix)
    return torch.from_numpy(rotated_points).float()
def random_translation(points, max_translation=0.1):
    """Translate point cloud by random vector"""
    translation_vector = np.random.uniform(-max_translation, max_translation, size=3)
    translated_points = points + translation_vector
    if isinstance(translated_points, torch.Tensor):
        return translated_points.to(torch.float)        
    else:
        return torch.from_numpy(translated_points).float()
        
def random_scale(points, scale_min=0.8, scale_max=1.5):
    """Scale point cloud by random factor"""
    scale_factor = np.random.uniform(scale_min, scale_max)
    
    scaled_points = points * scale_factor
    if isinstance(points, torch.Tensor):
        return scaled_points.to(torch.float)        
    else:
        return torch.from_numpy(scaled_points).float()

def tiff_2_pcd(slice_filename, output_dir, tickness):


    slice_filename_arr = slice_filename.split("/")
    slice_filename_itself = slice_filename_arr[len(slice_filename_arr)-1].split(".")[0]
    obj_filename = f'{slice_filename_itself}.obj'
    pcd_filename = f'{output_dir}/{obj_filename}'
    if os.path.exists(pcd_filename):
        return pcd_filename

    os.makedirs(output_dir, exist_ok = True)
    num_of_poinst_tickness = 10
    #row_index = int(index/10)
    #col_index = int(index%10)
    image = cv2.imread(slice_filename, cv2.IMREAD_COLOR)
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

    _i = 0
    y_min = _i * tickness
    y_max = y_min + tickness
    xyz_list = []
    for i, x in np.ndenumerate(canny_image):
        if x > 0 :
            xyz_list.append([(canny_image.shape[1]-i[1])/(canny_image.shape[1]), y_min, (canny_image.shape[0]-i[0])/(canny_image.shape[0])])
            xyz_list.append([(canny_image.shape[1]-i[1])/(canny_image.shape[1]), y_max, (canny_image.shape[0]-i[0])/(canny_image.shape[0])])
    
    for i in range(outer_points.shape[0]):
        min_value = (tickness/num_of_poinst_tickness)*1 + y_min
        max_value = (tickness/num_of_poinst_tickness)*num_of_poinst_tickness + y_min

        # 5개의 랜덤한 소수점 숫자 생성
        random_numbers = [random.uniform(min_value, max_value) for _ in range(5)]
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






def _tiff_2_obj():

    tiff_filename_list = []
    obj_dir_root_list = []
    tickness_list = []
    for data_id in data_ids:


        tiff_dir = f'{tiff_dir_root}/{data_id}'
        image_filename_list = []
        for f in os.listdir(tiff_dir):
            image_filename_list.append(tiff_dir+"/"+f)
        image_filename_list.sort(key = lambda x: int(x.split("/")[-1].split(".")[-2]))

        obj_dir = f'{obj_dir_root}/{data_id}/fractured_0'
        os.makedirs(obj_dir, exist_ok = True)
        for orginal_filename in image_filename_list:
            tiff_filename_list.append(orginal_filename)
            obj_dir_root_list.append(obj_dir)
            tickness_list.append(tickness)

    print(f'the number of jobs:{len(tiff_filename_list)}')
    with multiprocessing.Pool() as pool: # Use a pool of 4 processes
        pool.starmap(tiff_2_pcd, zip(tiff_filename_list, obj_dir_root_list, tickness_list))

def obj_slicing():
    for data_id in data_ids:
        for start_index in range(int((to_index-from_index)/num_of_slices)):
            obj_dir = f'{obj_dir_root}/{data_id}/fractured_0'

            obj_filename_list = []
            for f in os.listdir(obj_dir):
                obj_filename_list.append(obj_dir+"/"+f)
            obj_filename_list.sort(key = lambda x: int(x.split("/")[-1].split(".")[-2]))


            obj_filename_list_sub = obj_filename_list[from_index + start_index : to_index : int((to_index - from_index) / num_of_slices) + 1]

            start_data_id = obj_filename_list_sub[0].split("/")[-1].split(".")[-2]
            #end_data_id = obj_filename_list_sub[-1].split("/")[-1].split(".")[-2]
            obj_slicing_dir = f'{obj_dir_root}/{data_id}_{start_data_id}_{to_index}/fractured_0'
            os.makedirs(obj_slicing_dir, exist_ok = True)
            print(obj_slicing_dir)
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


def obj_augmentation( obj_dir_root, data_id,  from_index, start_index, to_index, group_index):

    obj_slicing_dir = f'{obj_dir_root}/{data_id}_{start_index}_{to_index}/fractured_0'
    obj_filename_list = []
    for f in os.listdir(obj_slicing_dir):
        obj_filename_list.append(obj_slicing_dir+"/"+f)
        

    for aug_name, aug_func in [('random_rotation',random_rotation),('random_translation',random_translation),('random_scale',random_scale),('all',all)]:
        

        obj_dir = f'{obj_dir_root}/{data_id}_{start_index}_{to_index}_{aug_name}/fractured_{group_index}'
        print(obj_dir)
        os.makedirs(obj_dir, exist_ok = True)

        for obj_filename in obj_filename_list:
            filename = obj_filename.split("/")[-1]
            target_filename = f'{obj_dir}/{filename}'
            #vertices = pytorch3d.io.load_obj(obj_filename, device='cpu')[0]
            vertices = load_obj(obj_filename)
            if aug_name =='all':
                augmented_image = random_rotation(vertices)
                augmented_image = random_translation(augmented_image)
                augmented_image = random_scale(augmented_image)    
            else:
                augmented_image = aug_func(vertices)
            
            with open(target_filename,'w') as f:
                for xyz in augmented_image:
                    f.write(f'v {xyz[0]} {xyz[1]} {xyz[2]}\n')

                    
            










if __name__ == "__main__":

    data_ids = ['0408']
    tiff_dir_root = '/data/jhahn/data/brain_lightsheet'
    obj_dir_root = '/data/jhahn/data/shape_dataset/data/brain_lightsheet'
    from_index=88
    start_index=0
    to_index=215
    num_of_slices=20
    tickness = 0.02

    #_tiff_2_obj()
    #obj_slicing()
    

            
    obj_dir_root_list = []
    data_id_list = []
    from_index_list = []
    start_index_list = []
    to_index_list = []
    group_index_list = []

    for data_id in data_ids:

        obj_dir = f'{obj_dir_root}/{data_id}/fractured_0'
        obj_filename_list = []
        for f in os.listdir(obj_dir):
            obj_filename_list.append(obj_dir+"/"+f)
        obj_filename_list.sort(key = lambda x: int(x.split("/")[-1].split(".")[-2]))


        for group_index in range(10):


            for start_index in range(int((to_index-from_index)/num_of_slices)):

                        
                obj_filename_list_sub = obj_filename_list[from_index + start_index : to_index : int((to_index - from_index) / num_of_slices) + 1]

                start_data_id = obj_filename_list_sub[0].split("/")[-1].split(".")[-2]

                obj_dir_root_list.append(obj_dir_root)
                data_id_list.append(data_id)
                from_index_list.append(from_index)
                start_index_list.append(start_data_id)
                to_index_list.append(to_index)
                group_index_list.append(group_index)

                #if True:
                #    break

                
        #obj_augmentation(obj_dir_root, '0408', '0088', '0089','215','0')
        print(f'the number of jobs:{len(group_index_list)}')
        with multiprocessing.Pool() as pool: # Use a pool of 4 processes
            pool.starmap(obj_augmentation, zip( obj_dir_root_list, data_id_list, from_index_list, start_index_list, to_index_list,group_index_list))

