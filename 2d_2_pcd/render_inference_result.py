import os
import json
import torch
import trimesh
import numpy as np
import slice_util
import open3d as o3d
from tqdm import tqdm 
from PIL import Image
from typing import Callable, List, Optional, Tuple
from pytorch3d.io import load_objs_as_meshes, load_obj
import imageio.v2 as iio
import pytorch3d
from pytorch3d.structures import Meshes
import pytorch3d.utils
from scipy.spatial.transform import Rotation as R

from PIL import Image, ImageDraw, ImageFont
from chamferdist import ChamferDistance
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    PointLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    HardPhongShader,
    TexturesUV,
    TexturesVertex,
    Textures
)
from pytorch3d.io import IO
import glob
import copy
from myutils import *
from tqdm import tqdm

from matplotlib import colormaps

cmap_names = list(colormaps)
#print(cmap_names)

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from pytorch3d.transforms import Transform3d, matrix_to_euler_angles
from pytorch3d.transforms.transform3d import (
    Rotate,
    RotateAxisAngle,
    Scale,
    Transform3d,
    Translate,
    
)
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVOrthographicCameras, 
    PointsRasterizationSettings,
    PointsRenderer,
    PulsarPointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
    NormWeightedCompositor
)
import sys
sys.path.insert(0, "../")

import puzzlefusion_plusplus.denoiser.evaluation.transform as puzzle_transform


import imageio
viridis = plt.get_cmap('tab20b_r')
tab10_r = ListedColormap(viridis(np.arange(20)))
tab10_r.colors[0][:3]
tab10_r
def get_vertices(inference_result_dir, objs_dir, device = None, max_points = 10000):

    
    with open(f'{inference_result_dir}/0/mesh_file_path.txt') as f:
        _mesh_file_dir = f.read()
    mesh_file_dir = objs_dir+"/"+_mesh_file_dir


    obj_file_list = slice_util.obj_files(mesh_file_dir)
    #combined_obj_filename = f'{output_dir}/{data_id}/{"combined"}.obj'

    #slice_util.combine_obj_files(obj_file_list,combined_obj_filename )
    #pcd_2_img(device, combined_obj_filename,f'{output_dir}/{data_id}/{"gt"}.png')

    #print(init_pose.shape)
    
    vertice_list = []
    min_list = []
    max_list = []
    for _i, pcd_file_name in enumerate(tqdm(obj_file_list)):
        if pcd_file_name.endswith('.glb'):

            _glb = trimesh.load(pcd_file_name)
            all_vertices = [geom.vertices for geom in _glb.geometry.values()]
            combined_vertices = np.vstack(all_vertices)
            vertices = torch.tensor(combined_vertices, dtype=torch.float32, device=device)

            #vertices = trimesh.PointCloud(vertices=combined_vertices)
        else:
            vertices = pytorch3d.io.load_obj(pcd_file_name, device=device)[0]
        N, dim = vertices.shape
        #print(pcd_file_name,vertices.shape)
        if N > max_points:
            #print(N)
            random_indices = torch.randperm(vertices.shape[0])[:max_points]
            vertices = vertices[random_indices]

        vertice_list.append(vertices)
        #min_list.append(torch.min(vertices,axis=0)[0].cpu().numpy())
        #max_list.append(torch.max(vertices,axis=0)[0].cpu().numpy())

    #print(np.min(min_list,axis=0),np.max(max_list,axis=0))
    #_min = np.min(min_list,axis=0)
    #_max = np.max(max_list,axis=0)
    
    #xlim = (float(_min[0]),  float(_max[0]))
    #ylim = (float(_min[1]), float(_max[1]))
    #zlim = (float(_min[2]),  float(_max[2]))
    #print(xlim, ylim, zlim)
    #print(type(_max[0]))

    obj_id_list = [o.split("/")[-1].split(".")[0] for o in obj_file_list]
    return vertice_list, obj_id_list

def sum_arrays(total):

    summed_array = copy.deepcopy(total[0])
    if len(total) ==  1:
        return summed_array
    #print('total',len(total))
    for arr in total[1:]:
        #print((summed_array > 0 ).sum())
        zero_mask = np.ones(total[0].shape, dtype=bool)
        zero_mask &= (summed_array == 0)
        #print((zero_mask == 1 ).sum())
        
        summed_array = np.where(zero_mask, arr, summed_array)

    return summed_array
'''
def _transform_pc(translated_points, trans, rotate, device):
    tr = Translate(torch.FloatTensor([trans]), dtype=torch.float32, device=device)
    _mean = torch.mean(translated_points, axis=0)
    tr_c = Translate(-_mean[0],-_mean[1],-_mean[2], dtype=torch.float32, device=device)
    tr_c_r = Translate(_mean[0],_mean[1],_mean[2], dtype=torch.float32, device=device)
    rr = Rotate(slice_util.quaternion_to_matrix(torch.FloatTensor(rotate)), dtype=torch.float32, device=device)    
    return [tr_c, rr, tr_c_r, tr]
'''


def _rotate_whole_part_xyz(pc,quat_gt, inverted = False):
    """
    pc: [P, N, 3]
    """

    if isinstance(pc, list):
        rot_mat = R.from_quat(quat_gt, scalar_first=True).as_matrix().T
        if inverted:
            rot_mat = rot_mat.T
        rot_mat = torch.from_numpy(rot_mat).float()
        new_pc = []
        for _part_idx, _part in enumerate(pc):
            new_pc.append((rot_mat @ _part.T).T)
        return new_pc
    else:
            
        P, N, _ = pc.shape
        pc = pc.reshape(-1, 3)
        rot_mat = R.from_quat(quat_gt, scalar_first=True).as_matrix().T
        if inverted:
            rot_mat = rot_mat.T
        pc = (rot_mat @ pc.T).T
        return pc.reshape(P, N, 3)



def _recenter_ref(pc, ref_idx):
    """
    pc: [P, N, 3]
    """
    P, N, _ = pc.shape
    #ref_idx = np.where(ref_part)[0]
    centroid = np.mean(pc[ref_idx], axis=0)
    pc = pc - centroid
    return pc

def _recenter_centroid(pc, centroid, inverted=False):
    """
    pc: [P, N, 3]
    """
    if isinstance(pc, list):
        new_pc = []
        for _part_idx, _part in enumerate(pc):
            if inverted:
                new_pc.append(-_part - centroid)
            else:
                new_pc.append(_part - centroid)
        return new_pc
    else:

        P, N, _ = pc.shape
        #ref_idx = np.where(ref_part)[0]
        #centroid = np.mean(pc[ref_idx], axis=0)
        if inverted:
            pc = -pc 
        pc = pc - centroid
        return pc


def _recenter_pc( pc, centroid):
    """pc: [N, 3]"""
    #centroid = np.mean(pc, axis=0)
    pc = pc - centroid[None]
    return pc

def _rotate_pc_xyz( pc, quat_gt, inverted = False):
    """
    pc: [N, 3]
    """

    rot_mat = R.from_quat(quat_gt, scalar_first=True).as_matrix()
    if inverted:
        rot_mat = rot_mat.T

    #rot_mat = R.random().as_matrix()
    if isinstance(pc, torch.Tensor):
        rot_mat = torch.from_numpy(rot_mat).float().to(pc.device)
        pc = (rot_mat @ pc.T).T
        return pc
    else:
        pc = (rot_mat @ pc.T).T
        return pc

    
    

def transform_pc(device, vertices, init_pose, gt, trans_rotate, step, part_index):

    if not isinstance(vertices, torch.Tensor):
        translated_points = torch.from_numpy(vertices).to(device).float()
    else:
        translated_points = vertices

    


    transformation_elem = []
    total_temp_t = None
    if init_pose is not None and gt is None and trans_rotate is None:
        #translated_points = vertices.clone().cpu().numpy()

        init_trans_reverse = Translate(torch.FloatTensor([-init_pose[:3]]), dtype=torch.float32, device=device)
        init_rotate_reverse = Rotate(slice_util.quaternion_to_matrix(torch.FloatTensor(slice_util.invert_rotation_quaternion(init_pose[3:]))), dtype=torch.float32, device=device)

        transform_matrix = init_rotate_reverse.get_matrix()
        rotation_matrix = transform_matrix[:, :3, :3]
        euler_angles_degrees = torch.rad2deg(matrix_to_euler_angles(rotation_matrix, convention="XYZ"))
        #print('init_rotate_reverse',euler_angles_degrees)
        #print('init_pose',-init_pose[:3])
        

        init_trans = Translate(torch.FloatTensor([init_pose[:3]]), dtype=torch.float32, device=device)
        init_rotate = Rotate(slice_util.quaternion_to_matrix(torch.FloatTensor(init_pose[3:])), dtype=torch.float32, device=device)
        #print('init_trans',init_pose)
        transform_matrix = init_rotate.get_matrix()
        rotation_matrix = transform_matrix[:, :3, :3]
        euler_angles_degrees = torch.rad2deg(matrix_to_euler_angles(rotation_matrix, convention="XYZ"))
        #print('init_rotate',euler_angles_degrees)
        #print('init_pose',init_pose[:3])
        translated_points = Transform3d(device=device).compose(init_trans).transform_points(translated_points)
        translated_points = Transform3d(device=device).compose(init_rotate).transform_points(translated_points)
        if False:

            init_trans = -init_pose[:3]
            init_rotate = init_pose[3:]
            #init_rotate = slice_util.invert_rotation_quaternion(init_pose[3:])
            
            #print('init_rotate',init_rotate)
            #print(slice_util.quaternion_to_matrix(torch.FloatTensor(init_rotate)))
            _mean = torch.mean(translated_points, axis=0)
            #print('_mean',_mean)
            tr_c = Translate(-_mean[0],-_mean[1],-_mean[2], dtype=torch.float32, device=device)
            rr = Rotate(slice_util.quaternion_to_matrix(torch.FloatTensor(init_rotate)), dtype=torch.float32, device=device)  
            #rr = RotateAxisAngle(angle=22, axis="Y", device=device)
            #print('rr',rr.R)
            tr_c_r = Translate(_mean[0],_mean[1],_mean[2], dtype=torch.float32, device=device)
            tr = Translate( init_trans[0],init_trans[1],init_trans[2], dtype=torch.float32, device=device)

            temp_t = Transform3d(device=device).compose(tr).compose(tr_c_r).compose(rr).compose(tr_c)
            #temp_t = Transform3d(device=device).compose(rr).compose(tr)
            translated_points = temp_t.transform_points(translated_points)#.to(torch.float).to(device)

    elif init_pose is not None and gt is not None and trans_rotate is None:
        init_trans = init_pose[:3]
        init_rotate = init_pose[3:]
        #init_rotate = slice_util.invert_rotation_quaternion(init_pose[3:])
        

        
        _mean = torch.mean(translated_points, axis=0)
        tr_c = Translate(-_mean[0],-_mean[1],-_mean[2], dtype=torch.float32, device=device)
        rr = Rotate(slice_util.quaternion_to_matrix(torch.FloatTensor(init_rotate)), dtype=torch.float32, device=device)  
        tr_c_r = Translate(_mean[0],_mean[1],_mean[2], dtype=torch.float32, device=device)
        tr = Translate(torch.FloatTensor([init_trans]), dtype=torch.float32, device=device)

        #temp_t = Transform3d(device=device).compose(tr_c).compose(rr).compose(tr_c_r).compose(tr)
        temp_t = Transform3d(device=device).compose(tr).compose(tr_c_r).compose(rr).compose(tr_c)
        translated_points = temp_t.transform_points(translated_points)#.to(torch.float).to(device)


        gt_trans = gt[part_index,:3]
        gt_rotate = gt[part_index,3:]
        gt_rotate = slice_util.invert_rotation_quaternion(gt[part_index,3:])


        _mean = torch.mean(translated_points, axis=0)
        tr_c = Translate(-_mean[0],-_mean[1],-_mean[2], dtype=torch.float32, device=device)
        rr = Rotate(slice_util.quaternion_to_matrix(torch.FloatTensor(gt_rotate)), dtype=torch.float32, device=device)  
        tr_c_r = Translate(_mean[0],_mean[1],_mean[2], dtype=torch.float32, device=device)
        tr = Translate(torch.FloatTensor([gt_trans]), dtype=torch.float32, device=device)

        temp_t = Transform3d(device=device).compose(tr).compose(tr_c_r).compose(rr).compose(tr_c)
        translated_points = temp_t.transform_points(translated_points)#.to(torch.float).to(device)

    elif init_pose is not None and gt is not None and trans_rotate is not None:
        

        '''
        trans1 = Vector(-init_pose[:3])
        trans_mat1 = Matrix.Translation(trans1)
        rot_mat1 = Quaternion(init_pose[3:]).inverted().to_matrix().to_4x4()

        trans2 = Vector(-gt_transformation[:3])
        trans_mat2 = Matrix.Translation(trans2)
        rot_mat2 = Quaternion(gt_transformation[3:]).inverted().to_matrix().to_4x4()

        trans3 = Vector(transformation[:3])
        trans_mat3 = Matrix.Translation(trans3)
        rot_mat3 = Quaternion(transformation[3:]).normalized().to_matrix().to_4x4()

        trans4 = Vector(init_pose[:3])
        trans_mat4 = Matrix.Translation(trans4)
        rot_mat4 = Quaternion(init_pose[3:]).to_matrix().to_4x4()

        # rotate -> translate -> translate -> rotate -> rotate -> translate
        final_transformation = rot_mat4 @ trans_mat4 @ trans_mat3 @ rot_mat3 @ rot_mat2 @ trans_mat2  @ trans_mat1 @ rot_mat1

        '''
        #x_trans = Translate(torch.FloatTensor([-1,0,0]), dtype=torch.float32, device=device)











        init_trans_reverse = Translate(torch.FloatTensor([-init_pose[:3]]), dtype=torch.float32, device=device)
        init_rotate_reverse = Rotate(slice_util.quaternion_to_matrix(torch.FloatTensor(slice_util.invert_rotation_quaternion(init_pose[3:]))), dtype=torch.float32, device=device)

        gt_trans_reverse = Translate(torch.FloatTensor([-gt[part_index,:3]]), dtype=torch.float32, device=device)
        gt_rotate_reverse = Rotate(slice_util.quaternion_to_matrix(torch.FloatTensor(slice_util.invert_rotation_quaternion(gt[part_index,3:]))), dtype=torch.float32, device=device)
        gt_trans = Translate(torch.FloatTensor([gt[part_index,:3]]), dtype=torch.float32, device=device)


        pred_trans = Translate(torch.FloatTensor([trans_rotate[step, part_index,:3]]), dtype=torch.float32, device=device)
        pred_trans_reverse = Translate(torch.FloatTensor([-trans_rotate[step, part_index,:3]]), dtype=torch.float32, device=device)
        pred_rotate = Rotate(slice_util.quaternion_to_matrix(torch.FloatTensor(trans_rotate[step, part_index,3:])), dtype=torch.float32, device=device)

        init_trans = Translate(torch.FloatTensor([init_pose[:3]]), dtype=torch.float32, device=device)
        init_rotate = Rotate(slice_util.quaternion_to_matrix(torch.FloatTensor(init_pose[3:])), dtype=torch.float32, device=device)
        


        #_mean = torch.mean(translated_points, axis=0)
        #tr_c = Translate(-_mean[0],-_mean[1],-_mean[2], dtype=torch.float32, device=device)
        #tr_c_r = Translate(_mean[0],_mean[1],_mean[2], dtype=torch.float32, device=device)
        translated_points = Transform3d(device=device).compose(init_rotate_reverse).transform_points(translated_points)
        translated_points = Transform3d(device=device).compose(init_trans).transform_points(translated_points)
        translated_points = Transform3d(device=device).compose(gt_trans).transform_points(translated_points)
        translated_points = Transform3d(device=device).compose(gt_rotate_reverse).transform_points(translated_points)
        translated_points = Transform3d(device=device).compose(pred_rotate).transform_points(translated_points)
        translated_points = Transform3d(device=device).compose(pred_trans_reverse).transform_points(translated_points)
        #translated_points = Transform3d(device=device).compose(init_trans_reverse).transform_points(translated_points)
        #translated_points = Transform3d(device=device).compose(init_rotate).transform_points(translated_points)
  
    
        '''
        translated_points = _rotate_whole_part_xyz(translated_points,init_rotate)
        translated_points = _recenter_ref(translated_points,init_trans, 0)

        translated_points = _recenter_pc(translated_points, gt[part_index,:3])
        translated_points = _rotate_pc_xyz(translated_points, gt[part_index,3:])


        _scale = np.max(np.abs(translated_points), axis=(1,2), keepdims=True)
        #_scale = torch.max(torch.abs(translated_points), dim=1, keepdim=True).values
        _scale[_scale == 0] = 1
        #translated_points = translated_points / _scale
        '''

        #translated_points = puzzle_transform.transform_pc(trans_rotate[step, part_index,:3], trans_rotate[step, part_index,3:], translated_points)
        #translated_points = Transform3d(device=device).compose(tr_c).transform_points(translated_points)
#        translated_points = Transform3d(device=device).compose(pred_rotate).transform_points(translated_points)
        #translated_points = Transform3d(device=device).compose(tr_c_r).transform_points(translated_points)
        #translated_points = Transform3d(device=device).compose(pred_trans).transform_points(translated_points)



        '''
        rr = RotateAxisAngle(angle=45, axis="Y", device=device)
        if step % 4 == 0:
            temp_t = Transform3d(device=device)
            translated_points = temp_t.transform_points(translated_points)#.to(torch.float).to(device)
        elif step % 4 == 1:
            temp_t = Transform3d(device=device).compose(tr_c)
            translated_points = temp_t.transform_points(translated_points)#.to(torch.float).to(device)
        elif step % 4 == 2:
            translated_points = Transform3d(device=device).compose(tr_c).transform_points(translated_points)#.to(torch.float).to(device)
            translated_points = Transform3d(device=device).compose(rr).transform_points(translated_points)#.to(torch.float).to(device)
            
        else:
            translated_points = Transform3d(device=device).compose(tr_c).transform_points(translated_points)#.to(torch.float).to(device)
            translated_points = Transform3d(device=device).compose(rr).transform_points(translated_points)#.to(torch.float).to(device)
            translated_points = Transform3d(device=device).compose(tr_c_r).transform_points(translated_points)#.to(torch.float).to(device)

        '''
        '''
        temp_t = Transform3d(device=device) \
            .compose(init_trans_reverse).compose(init_rotate_reverse) \
            .compose(gt_trans_reverse).compose(gt_rotate_reverse) \
            .compose(pred_rotate).compose(pred_trans) \
            .compose(init_trans).compose(init_rotate)
        '''
        
        
        '''
        init_rotate = init_pose[3:]
        init_rotate = slice_util.invert_rotation_quaternion(init_pose[3:])
        #print("init_rotate",slice_util.quaternion_to_matrix(torch.FloatTensor(init_rotate)))

        
        _mean = torch.mean(translated_points, axis=0)
        tr_c = Translate(-_mean[0],-_mean[1],-_mean[2], dtype=torch.float32, device=device)
        rr = Rotate(slice_util.quaternion_to_matrix(torch.FloatTensor(init_rotate)), dtype=torch.float32, device=device)  
        tr_c_r = Translate(_mean[0],_mean[1],_mean[2], dtype=torch.float32, device=device)
        

        temp_t = Transform3d(device=device).compose(tr_c).compose(rr).compose(tr_c_r).compose(tr)
        #temp_t = Transform3d(device=device).compose(tr)
        translated_points = temp_t.transform_points(translated_points)#.to(torch.float).to(device)
        transformation_elem.extend([tr_c,rr,tr_c_r,tr])
        
        gt_trans = gt[part_index,:3]
        gt_rotate = gt[part_index,3:]
        #gt_rotate = slice_util.invert_rotation_quaternion(gt[part_index,3:])
        #print("gt_rotate",slice_util.quaternion_to_matrix(torch.FloatTensor(gt_rotate)))

        _mean = torch.mean(translated_points, axis=0)
        tr_c = Translate(-_mean[0],-_mean[1],-_mean[2], dtype=torch.float32, device=device)
        rr = Rotate(slice_util.quaternion_to_matrix(torch.FloatTensor(gt_rotate)), dtype=torch.float32, device=device)  
        tr_c_r = Translate(_mean[0],_mean[1],_mean[2], dtype=torch.float32, device=device)
        tr = Translate(torch.FloatTensor([gt_trans]), dtype=torch.float32, device=device)

        temp_t = Transform3d(device=device).compose(tr_c).compose(rr).compose(tr_c_r).compose(tr)
        translated_points = temp_t.transform_points(translated_points)#.to(torch.float).to(device)
        transformation_elem.extend([tr_c,rr,tr_c_r,tr])
        
        trans = trans_rotate[step, part_index,:3]
        rotate =trans_rotate[step, part_index,3:]
        
        _mean = torch.mean(translated_points, axis=0)
        tr_c = Translate(-_mean[0],-_mean[1],-_mean[2], dtype=torch.float32, device=device)
        #print("rotate",rotate.shape, slice_util.quaternion_to_matrix(torch.FloatTensor(rotate)))
        #print("360",get_euler_angles_from_quaternion_gpu(torch.FloatTensor(rotate)))
        rr = Rotate(slice_util.quaternion_to_matrix(torch.FloatTensor(rotate)), dtype=torch.float32, device=device)  
        tr_c_r = Translate(_mean[0],_mean[1],_mean[2], dtype=torch.float32, device=device)
        tr = Translate(torch.FloatTensor([trans]), dtype=torch.float32, device=device)

        temp_t = Transform3d(device=device).compose(tr_c).compose(rr).compose(tr_c_r).compose(tr)
        translated_points = temp_t.transform_points(translated_points)#.to(torch.float).to(device)
        
        transformation_elem.extend([tr_c,rr,tr_c_r,tr])
        total_temp_t = Transform3d(device=device)
        for t in transformation_elem:
            total_temp_t = total_temp_t.compose(t)
        '''

    return translated_points, total_temp_t

def get_renderer(device, xlim=(-2, 2),ylim=(-2, 2),zlim=(-2, 2)):
    image_size = 512
    radius = 0.002
    points_per_pixel = 50
    R, T = look_at_view_transform(20, 45, 45) #10,30,90
    cameras = FoVOrthographicCameras( device=device, R=R, T=T, znear=zlim[0], zfar=zlim[1], min_x=xlim[0], max_x=xlim[1], min_y=ylim[0], max_y=ylim[1])

    R, T = look_at_view_transform(20, 0, 0)
    cameras_front = FoVOrthographicCameras( device=device, R=R, T=T, znear=zlim[0], zfar=zlim[1], min_x=xlim[0], max_x=xlim[1], min_y=ylim[0], max_y=ylim[1])
    
    R, T = look_at_view_transform(20, 90, 0)
    cameras_down = FoVOrthographicCameras( device=device, R=R, T=T, znear=zlim[0], zfar=zlim[1], min_x=xlim[0], max_x=xlim[1], min_y=ylim[0], max_y=ylim[1])

    raster_settings = PointsRasterizationSettings(
        image_size=image_size, 
        radius = radius,
        points_per_pixel = points_per_pixel
    )

    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    rasterizer_front = PointsRasterizer(cameras=cameras_front, raster_settings=raster_settings)
    rasterizer_down = PointsRasterizer(cameras=cameras_down, raster_settings=raster_settings)

    renderer = pytorch3d.renderer.PointsRenderer(
        rasterizer=rasterizer,
        compositor=pytorch3d.renderer.AlphaCompositor(background_color=(0, 0, 0))
        #compositor=NormWeightedCompositor()
    )

    renderer_front = pytorch3d.renderer.PointsRenderer(
        rasterizer=rasterizer_front,
        compositor=pytorch3d.renderer.AlphaCompositor(background_color=(0, 0, 0))
        #compositor=NormWeightedCompositor()
    )

    renderer_down = pytorch3d.renderer.PointsRenderer(
        rasterizer=rasterizer_down,
        compositor=pytorch3d.renderer.AlphaCompositor(background_color=(0, 0, 0))
        #compositor=NormWeightedCompositor()
    )
    return renderer, renderer_front, renderer_down

def axis_draw(renderer,renderer_front,renderer_down, device):




    axis_X_xyz = []
    axis_Y_xyz = []
    axis_Z_xyz = []
    #x_axis_pcd = torch.ones_like((512,512,3))
    for i in range(1000):
        axis_X_xyz.append([i/1000,0,0])
        axis_X_xyz.append([i/1000,0.001,0])
        axis_X_xyz.append([i/1000,0.0,0.001])

        axis_Y_xyz.append([0,i/1000,0])
        axis_Y_xyz.append([0,i/1000,0.001])
        axis_Y_xyz.append([0.001,i/1000,0])
        
        axis_Z_xyz.append([0,0,i/1000])
        axis_Z_xyz.append([0,0.001,i/1000])
        axis_Z_xyz.append([0.001,0,i/1000])
    axis_X_xyz = torch.from_numpy(np.array(axis_X_xyz)).to(device).float()
    axis_Y_xyz = torch.from_numpy(np.array(axis_Y_xyz)).to(device).float()
    axis_Z_xyz = torch.from_numpy(np.array(axis_Z_xyz)).to(device).float()


    colors = torch.ones_like(axis_X_xyz) * torch.tensor([1.0,0.0,0.0]).to(device)  # blue points
    axis_X_pcd = pytorch3d.structures.Pointclouds(points=[axis_X_xyz], features=[colors.to(torch.float)])

    colors = torch.ones_like(axis_Y_xyz) * torch.tensor([0.0,1.0,0.0]).to(device)  # blue points
    axis_Y_pcd = pytorch3d.structures.Pointclouds(points=[axis_Y_xyz], features=[colors.to(torch.float)])

    colors = torch.ones_like(axis_Z_xyz) * torch.tensor([0.0,0.0,1.0]).to(device)  # blue points
    axis_Z_pcd = pytorch3d.structures.Pointclouds(points=[axis_Z_xyz], features=[colors.to(torch.float)])


    axis_pcd_rendered = [(renderer(axis_X_pcd)[0].cpu().numpy()*255).astype(np.uint8),
                        (renderer(axis_Y_pcd)[0].cpu().numpy()*255).astype(np.uint8),
                        (renderer(axis_Z_pcd)[0].cpu().numpy()*255).astype(np.uint8)]

    axis_pcd_rendered_front = [(renderer_front(axis_X_pcd)[0].cpu().numpy()*255).astype(np.uint8),
                        (renderer_front(axis_Y_pcd)[0].cpu().numpy()*255).astype(np.uint8),
                        (renderer_front(axis_Z_pcd)[0].cpu().numpy()*255).astype(np.uint8)]

    axis_pcd_rendered_down =  [(renderer_down(axis_X_pcd)[0].cpu().numpy()*255).astype(np.uint8),
                        (renderer_down(axis_Y_pcd)[0].cpu().numpy()*255).astype(np.uint8),
                        (renderer_down(axis_Z_pcd)[0].cpu().numpy()*255).astype(np.uint8)]

    return axis_pcd_rendered, axis_pcd_rendered_front, axis_pcd_rendered_down

def pcd_list_2_img(device,
    part_pcs_gt11,original_vertices,
    output_img_file_name_original,
    output_img_file_name_original_front,
    output_img_file_name_original_down,
    output_img_file_name_init,
    output_img_file_name_init_front,
    output_img_file_name_init_down,
    output_img_file_name_init_gt,
    output_img_file_name_init_gt_front,
    output_img_file_name_init_gt_down,
    alpha=.8,
    max_points=10000,
    xlim=(-1, 1),
    ylim=(-1.5, 1),
    zlim=(-2, 2),
    init_pose = None,
    gt = None):  
    
    #renderer, renderer_front = get_renderer(device, xlim, ylim, zlim)
    renderer, renderer_front , renderer_down= get_renderer(device)




    img_data_list_original= []    
    img_data_list_original_front = []
    img_data_list_original_down = []
    img_data_list_init = []
    img_data_list_init_front = []
    img_data_list_init_down = []
    img_data_list_init_gt = []
    img_data_list_init_gt_front = []
    img_data_list_init_gt_down = []
    for _i, vertices in enumerate(tqdm(original_vertices, desc='gen snapshot images')):
        #print(pcd_file_name)
        #vertices = vertices.cpu()
        if isinstance(vertices, np.ndarray):
            vertices = torch.from_numpy(vertices).to(device).to(torch.float)
        else:
            vertices = vertices.to(device).to(torch.float)
        translated_points = vertices
        #N, dim = vertices.shape

        #tr = Translate(torch.FloatTensor([-gt[_i,:3]]))
        #rr = Rotate(quaternion_to_matrix(torch.FloatTensor([-gt[_i,3:]])))
        #if _i < 3 or _i > len(pcd_file_name_list) - 3:
        #    print(pcd_file_name, torch.min(vertices, axis=0)[0][1].item(), torch.max(vertices, axis=0)[0][1].item())
        #translated_points = vertices.to(device)

        #print(_i,torch.min(translated_points,axis=0)[0][1],torch.max(translated_points,axis=0)[0][1])
        #print(torch.FloatTensor([transform[_i,3:]]))
        
        
        translated_points_init, _ = transform_pc(device, translated_points, init_pose, None, None, -1, _i)
        translated_points_init_gt, _ = transform_pc(device, translated_points, init_pose, gt, None, -1, _i)

        
        #print(_i, torch.FloatTensor([transform[_i,3:]]), torch.min(translated_points,axis=0)[0],torch.max(translated_points,axis=0)[0])

        colors = torch.ones_like(translated_points) * torch.tensor(tab10_r.colors[(_i)%len(tab10_r.colors)][:3]).to(device)  # blue points

        pcd_original = pytorch3d.structures.Pointclouds(points=[vertices], features=[colors.to(torch.float)])
        pcd_init = pytorch3d.structures.Pointclouds(points=[translated_points_init], features=[colors.to(torch.float)])
        pcd_init_gt = pytorch3d.structures.Pointclouds(points=[translated_points_init_gt], features=[colors.to(torch.float)])
        

        #print('pcd_original', torch.min(vertices,axis=0)[0], torch.max(vertices,axis=0)[0])
        #print('pcd_init', torch.min(translated_points_init,axis=0)[0], torch.max(translated_points_init,axis=0)[0])
        #print('pcd_init_gt', torch.min(translated_points_init_gt,axis=0)[0], torch.max(translated_points_init_gt,axis=0)[0])
        
        #exit()
        img_data_list_original.append((renderer(pcd_original)[0].cpu().numpy()*255).astype(np.uint8))
        img_data_list_original_front.append((renderer_front(pcd_original)[0].cpu().numpy()*255).astype(np.uint8))
        img_data_list_original_down.append((renderer_down(pcd_original)[0].cpu().numpy()*255).astype(np.uint8))

        img_data_list_init.append((renderer(pcd_init)[0].cpu().numpy()*255).astype(np.uint8))
        img_data_list_init_front.append((renderer_front(pcd_init)[0].cpu().numpy()*255).astype(np.uint8))
        img_data_list_init_down.append((renderer_down(pcd_init)[0].cpu().numpy()*255).astype(np.uint8))

        img_data_list_init_gt.append((renderer(pcd_init_gt)[0].cpu().numpy()*255).astype(np.uint8))
        img_data_list_init_gt_front.append((renderer_front(pcd_init_gt)[0].cpu().numpy()*255).astype(np.uint8))
        img_data_list_init_gt_down.append((renderer_down(pcd_init_gt)[0].cpu().numpy()*255).astype(np.uint8))

    axis_pcd_rendered, axis_pcd_rendered_front, axis_pcd_rendered_down = axis_draw(renderer, renderer_front, renderer_down, device)
    img_data_list_original.extend(axis_pcd_rendered)
    img_data_list_original_front.extend(axis_pcd_rendered_front)
    img_data_list_original_down.extend(axis_pcd_rendered_down)
    img_data_list_init.extend(axis_pcd_rendered)
    img_data_list_init_front.extend(axis_pcd_rendered_front)
    img_data_list_init_down.extend(axis_pcd_rendered_down)
    img_data_list_init_gt.extend(axis_pcd_rendered)
    img_data_list_init_gt_front.extend(axis_pcd_rendered_front)
    img_data_list_init_gt_down.extend(axis_pcd_rendered_down)

    imageio.mimsave(output_img_file_name_original, [sum_arrays(img_data_list_original)], format='PNG')
    imageio.mimsave(output_img_file_name_original_front, [sum_arrays(img_data_list_original_front)], format='PNG')
    imageio.mimsave(output_img_file_name_original_down, [sum_arrays(img_data_list_original_down)], format='PNG')
    imageio.mimsave(output_img_file_name_init, [sum_arrays(img_data_list_init)], format='PNG')
    imageio.mimsave(output_img_file_name_init_front, [sum_arrays(img_data_list_init_front)], format='PNG')
    imageio.mimsave(output_img_file_name_init_down, [sum_arrays(img_data_list_init_down)], format='PNG')
    imageio.mimsave(output_img_file_name_init_gt, [sum_arrays(img_data_list_init_gt)], format='PNG')
    imageio.mimsave(output_img_file_name_init_gt_front, [sum_arrays(img_data_list_init_gt_front)], format='PNG')
    imageio.mimsave(output_img_file_name_init_gt_down, [sum_arrays(img_data_list_init_gt_down)], format='PNG')

def gt_img(device, part_pcs_gt, original_vertices, inference_result_dir, output_dir, obj_id_list):

    data_id = inference_result_dir.split("/")[-1]
    os.makedirs(f'{output_dir}/{data_id}',exist_ok=True)


    gt = np.load(f'{inference_result_dir}/gt.npy')
    init_pose = np.load(f'{inference_result_dir}/init_pose.npy')

    pcd_list_2_img(device, part_pcs_gt, original_vertices,
        f'{output_dir}/{data_id}/{"original"}.png',f'{output_dir}/{data_id}/{"original_front"}.png',f'{output_dir}/{data_id}/{"original_down"}.png',
        f'{output_dir}/{data_id}/{"init"}.png', f'{output_dir}/{data_id}/{"init_front"}.png', f'{output_dir}/{data_id}/{"init_down"}.png', 
        f'{output_dir}/{data_id}/{"init_gt"}.png',f'{output_dir}/{data_id}/{"init_gt_front"}.png',f'{output_dir}/{data_id}/{"init_gt_down"}.png',
        init_pose = init_pose,
        gt = gt)



    


def plot_pointcloud2_same_with_part_acc(
    device,
    output_dir,
    part_pcs_gt11, original_vertices,
    obj_id_list,
    init_pose, gt,
    predict_0,
    alpha=.8,
    title=None,
    max_points=10000,
    xlim=(-1, 1),
    ylim=(-1, 1),
    zlim=(-1, 1)
    ):
    """Plot a pointcloud tensor of shape (N, coordinates)
    """

   


    renderer, renderer_front,renderer_down = get_renderer(device)


    os.makedirs(output_dir+"/iteration/data", exist_ok = True)

    os.makedirs(output_dir+"/trans_diff", exist_ok = True)

    os.makedirs(output_dir+"/trans", exist_ok = True)
    os.makedirs(output_dir+"/init_gt", exist_ok = True)
    #fig = plt.figure(figsize=(25,20))
    #predict_0[:,:,5:] = np.zeros((predict_0.shape[0],predict_0.shape[1],2))
    #predict_0[:,:,4:5] = np.ones((predict_0.shape[0],predict_0.shape[1],1))
    #print('predict_0',predict_0.shape)
  
  
  
  
  
    translated_points = _rotate_whole_part_xyz(original_vertices, init_pose[3:], True)
    #translated_points = Transform3d(device=device).compose(init_rotate).transform_points(translated_points)#
    #for i in range(10):
    #    print(i, np.mean(translated_points[i], axis=0))
    translated_points = _recenter_centroid(translated_points,init_pose[:3])

    #print("===========================")
    #for i in range(10):
    #    print(i, np.mean(translated_points[i], axis=0))
    new_pc = []
    for i in range(10):
        pc = translated_points[i]
        #print(i,gt[i,:3])
        pc = _recenter_pc(pc,gt[i,:3])
        pc = _rotate_pc_xyz(pc,gt[i,3:], False)
        new_pc.append(pc)
    #new_pc = np.array(new_pc)

    axis_pcd_rendered, axis_pcd_rendered_front, axis_pcd_rendered_down = axis_draw(renderer, renderer_front,renderer_down, device)


    for _step in tqdm(range(20)):
            
        pts_pred = []
        mse_r_list = []
        mse_t_list = []
        for i in range(10):
            pc = new_pc[i]
            #print(i,gt[i,:3])
            #pc = _rotate_pc_xyz(pc,gt[i,3:], True)
            #pc = _recenter_pc(pc,-gt[i,:3])
            pc = _rotate_pc_xyz(pc,predict_0[_step][i,3:], True)
            pc = _recenter_pc(pc,-predict_0[_step][i,:3])


            mse_t = np.mean((-gt[i,:3] - predict_0[_step][i,:3]) ** 2)
            mse_r = np.mean((-gt[i,3:] - predict_0[_step][i,3:]) ** 2)
            mse_r_list.append(mse_r)
            mse_t_list.append(mse_t)
            pts_pred.append(pc)

        #print(_step,np.mean(mse_r_list),np.mean(mse_t_list))
        try:
            pts_pred = np.stack(pts_pred)
        except:
            pass

        pts_pred = _recenter_centroid(pts_pred,init_pose[:3], inverted=True)
        pts_pred = _rotate_whole_part_xyz(pts_pred, init_pose[3:], False)

        #pts_pred = torch.from_numpy(pts_pred).to(device).float()
        images = []
        images_front = []
        images_down = []
        image_index_2_text = {}
        image_index_2_color = {}
        image_index_2_dice_score = {}
        for _part_idx in range(len(pts_pred)):
            if isinstance(pts_pred[_part_idx], np.ndarray):
                _cur_part_pcd = torch.from_numpy(pts_pred[_part_idx]).to(device).float()
            else:
                _cur_part_pcd = pts_pred[_part_idx].to(device).float()

            colors = torch.ones_like(_cur_part_pcd).to(device) * torch.tensor(tab10_r.colors[_part_idx%20][:3]).to(device)  # blue points

            point_cloud = pytorch3d.structures.Pointclouds(points=[_cur_part_pcd], features=[colors.to(torch.float)])

            image_index_2_text[_part_idx] = torch.min(_cur_part_pcd,axis=0)[0][1].item()
            #print(_s, _i, torch.min(translated_points,axis=0))


            if _part_idx >= 1:
                
                shape_cd_min = slice_util.calculate_dice_score_from_point_clouds((translated_points_pre - torch.mean(translated_points_pre, axis=0)).cpu().numpy(),
                (_cur_part_pcd - torch.mean(_cur_part_pcd, axis=0)).cpu().numpy() )
                image_index_2_dice_score[_part_idx] = shape_cd_min
            translated_points_pre = _cur_part_pcd


            if _step == 19: # the last iteration
                with open(f'{output_dir}/trans_diff/{_part_idx}.obj', 'w') as outfile:
                    for _arr in _cur_part_pcd:
                        outfile.write(f'v {_arr[0]} {_arr[1]} {_arr[2]}\n')

            if _step == predict_0.shape[0]-1: # the last iteration

                #translated_points_init_gt, _ = transform_pc(device, vertices, init_pose, gt, None, _s, _i)
                #with open(f'{output_dir}/init_gt/{_i}.obj', 'w') as outfile:
                #    for _arr in translated_points_init_gt.cpu().numpy():
                #        outfile.write(f'v {_arr[0]} {_arr[1]} {_arr[2]}\n')

                with open(f'{output_dir}/trans/{_part_idx}.obj', 'w') as outfile:
                    for _arr in _cur_part_pcd:
                        outfile.write(f'v {_arr[0]} {_arr[1]} {_arr[2]}\n')


            #translated_points = vertices.to(device)
            image_index_2_color[_part_idx] = tab10_r.colors[_part_idx%20][:3]


            images.append((renderer(point_cloud).cpu().numpy()*255).astype(np.uint8))
            images_front.append((renderer_front(point_cloud).cpu().numpy()*255).astype(np.uint8))
            images_down.append((renderer_down(point_cloud).cpu().numpy()*255).astype(np.uint8))
        images.extend(axis_pcd_rendered)
        images_front.extend(axis_pcd_rendered_front)
        images_down.extend(axis_pcd_rendered_down)

        images = sum_arrays(images)
        images_front = sum_arrays(images_front)
        images_down = sum_arrays(images_down)
        imageio.mimsave(f'{output_dir}/iteration/{_step}.png', [images[0]], format='PNG')
        imageio.mimsave(f'{output_dir}/iteration/{_step}_front.png', [images_front[0]], format='PNG')
        imageio.mimsave(f'{output_dir}/iteration/{_step}_down.png', [images_down[0]], format='PNG')
        





        sorted_image_index_2_text = sorted(image_index_2_text.items(), key=lambda item: item[1] ,reverse=True)
        
        slice_pos_text = []
        slice_pos_text.append(f'iteration {_step}')
        slice_pos_text.append(f'id : y : dice')
        for imaage_id, text in sorted_image_index_2_text:


            
            if imaage_id >= 1: # and _step > 19 :                
                #if _step - 19 >= imaage_id:
                #    slice_pos_text.append(f'{obj_id_list[imaage_id]} : {text:.3f} : {image_index_2_dice_score[imaage_id]:.3f} **')
                #else:
                slice_pos_text.append(f'{obj_id_list[imaage_id]} : {text:.3f} : {image_index_2_dice_score[imaage_id]:.3f} / {imaage_id}|{imaage_id-1}')
            else:
                slice_pos_text.append(f'{obj_id_list[imaage_id]} : {text:.3f}')
            
        add_text_to_png(f'{output_dir}/iteration/{_step}.png', slice_pos_text, f'{output_dir}/iteration/{_step}.png')
        add_text_to_png(f'{output_dir}/iteration/{_step}_front.png', slice_pos_text, f'{output_dir}/iteration/{_step}_front.png')
        add_text_to_png(f'{output_dir}/iteration/{_step}_down.png', slice_pos_text, f'{output_dir}/iteration/{_step}_down.png')
        




    #plt.show(fig)
    _last_obj_flies  = glob.glob(f'{output_dir}/trans_diff/{"*"}.obj')
    slice_util.combine_obj_files(_last_obj_flies,f'{output_dir}/combined_diff.obj')
    _last_obj_flies  = glob.glob(f'{output_dir}/trans/{"*"}.obj')
    slice_util.combine_obj_files(_last_obj_flies,f'{output_dir}/combined_diff_rotate.obj')

    w = iio.get_writer(f'{output_dir}/video.mp4', format='FFMPEG', mode='I', fps=2,
                        #codec='h264_vaapi',
                        pixelformat='yuv420p')
    
    for _step in range(predict_0.shape[0]):
        w.append_data(iio.imread(f'{output_dir}/iteration/{_step}.png'))

    w.close()


    w = iio.get_writer(f'{output_dir}/video_front.mp4', format='FFMPEG', mode='I', fps=2,
                        #codec='h264_vaapi',
                        pixelformat='yuv420p')
    
    for _step in range(predict_0.shape[0]):
        w.append_data(iio.imread(f'{output_dir}/iteration/{_step}_front.png'))

    w.close()

    w = iio.get_writer(f'{output_dir}/video_down.mp4', format='FFMPEG', mode='I', fps=2,
                        #codec='h264_vaapi',
                        pixelformat='yuv420p')
    
    for _step in range(predict_0.shape[0]):
        w.append_data(iio.imread(f'{output_dir}/iteration/{_step}_down.png'))

    w.close()

    return

    axis_pcd_rendered, axis_pcd_rendered_front, axis_pcd_rendered_down = axis_draw(renderer, renderer_front,renderer_down, device)

    for _step  in tqdm(range(predict_0.shape[0]), desc='gen pngs'):



        images = []
        image_data_list = []
        images_front = []
        images_down = []
        image_index_2_dice_score = {}
        image_index_2_text = {}
        image_index_2_color = {}
        
        for _part_idx  in range(predict_0.shape[1]):
            translated_points_part = puzzle_transform.transform_pc(predict_0[_step,_part_idx,:3], predict_0[_step,_part_idx,3:], translated_points[_part_idx])
            #print(translated_points.shape)
            colors = torch.ones_like(translated_points_part) * torch.tensor(tab10_r.colors[_part_idx%20][:3]).to(device)  # blue points
            #image_index_2_color[_step] = tab10_r.colors[_part_idx%20][:3]
            
            point_cloud = pytorch3d.structures.Pointclouds(points=[translated_points_part], features=[colors.to(torch.float)])

            translated_points_np = translated_points_part.cpu().numpy()

            sample_indices = np.random.choice(translated_points_np.shape[0], size=1000, replace=False)
            translated_points_np = translated_points_np[sample_indices, :]


            

            #print(np.max(translated_points_np,axis=0)[0])
            #print(np.min(translated_points_np,axis=0)[0])
            image_data_list.append(translated_points_np)
            images.append((renderer(point_cloud).cpu().numpy()*255).astype(np.uint8))
            images_front.append((renderer_front(point_cloud).cpu().numpy()*255).astype(np.uint8))
            images_down.append((renderer_down(point_cloud).cpu().numpy()*255).astype(np.uint8))

        images.extend(axis_pcd_rendered)
        images_front.extend(axis_pcd_rendered_front)
        images_down.extend(axis_pcd_rendered_down)

        images_ = sum_arrays(images)
        images_front = sum_arrays(images_front)
        images_down = sum_arrays(images_down)



        '''
        print(len(image_data_list))
        for w, imageg in enumerate(image_data_list):
            print(imageg.shape)
        '''
        np.save(f'{output_dir}/iteration/data/{_step}',np.array(image_data_list) )

        
        imageio.mimsave(f'{output_dir}/iteration/{_step}.png', [images_[0]], format='PNG')
        imageio.mimsave(f'{output_dir}/iteration/{_step}_front.png', [images_front[0]], format='PNG')
        imageio.mimsave(f'{output_dir}/iteration/{_step}_down.png', [images_down[0]], format='PNG')

        sorted_image_index_2_text = sorted(image_index_2_text.items(), key=lambda item: item[1] ,reverse=True)
        
        slice_pos_text = []
        slice_pos_text.append(f'iteration {_step}')
        slice_pos_text.append(f'id : y : dice')
        for imaage_id, text in sorted_image_index_2_text:

            if _step > 19 and imaage_id >= 1:                
                if _step - 19 >= imaage_id:
                    slice_pos_text.append(f'{obj_id_list[imaage_id]} : {text:.3f} : {image_index_2_dice_score[imaage_id]:.3f} **')
                else:
                    slice_pos_text.append(f'{obj_id_list[imaage_id]} : {text:.3f} : {image_index_2_dice_score[imaage_id]:.3f} ')
            else:
                slice_pos_text.append(f'{obj_id_list[imaage_id]} : {text:.3f}')
            
        add_text_to_png(f'{output_dir}/iteration/{_step}.png', slice_pos_text, f'{output_dir}/iteration/{_step}.png')
        add_text_to_png(f'{output_dir}/iteration/{_step}_front.png', slice_pos_text, f'{output_dir}/iteration/{_step}_front.png')
        add_text_to_png(f'{output_dir}/iteration/{_step}_down.png', slice_pos_text, f'{output_dir}/iteration/{_step}_down.png')
        #ax.imshow(images_[0, ..., :3])



def plot_pointcloud2(
    device,
    output_dir,
    part_pcs_gt,
    obj_id_list,
    init_pose, gt,
    predict_0,
    alpha=.8,
    title=None,
    max_points=10000,
    xlim=(-1, 1),
    ylim=(-1, 1),
    zlim=(-1, 1)
    ):
    """Plot a pointcloud tensor of shape (N, coordinates)
    """

   


    renderer, renderer_front,renderer_down = get_renderer(device)


    os.makedirs(output_dir+"/iteration/data", exist_ok = True)

    os.makedirs(output_dir+"/trans_diff", exist_ok = True)

    os.makedirs(output_dir+"/trans", exist_ok = True)
    os.makedirs(output_dir+"/init_gt", exist_ok = True)
    #fig = plt.figure(figsize=(25,20))
    #predict_0[:,:,5:] = np.zeros((predict_0.shape[0],predict_0.shape[1],2))
    #predict_0[:,:,4:5] = np.ones((predict_0.shape[0],predict_0.shape[1],1))
    #print('predict_0',predict_0.shape)
    for _s  in tqdm(range(predict_0.shape[0]), desc='gen pngs'):
            

        #ax = fig.add_subplot(4, 5, _s+1)
        # ax.set_axis_off()
        frames = []
        images = []
        image_data_list = []
        images_front = []
        images_down = []
        image_index_2_dice_score = {}
        image_index_2_text = {}
        image_index_2_color = {}
        translated_points_pre = None
        for _i, vertices  in enumerate(part_pcs_gt):

            print(vertices)
            #vertices = vertices.cpu()

            N, dim = vertices.shape
            #print(type(vertices))

            
            #tr = Translate(torch.FloatTensor([-gt[_i,:3]]))
            #rr = Rotate(quaternion_to_matrix(torch.FloatTensor([-gt[_i,3:]])))
            #print(predict_0[_s])
            translated_points, _ = transform_pc(device, vertices, init_pose, gt, predict_0, _s, _i)
            #print(translated_points.shape)
            if _i >= 1:
                
                shape_cd_min = slice_util.calculate_dice_score_from_point_clouds((translated_points_pre - torch.mean(translated_points_pre, axis=0)).cpu().numpy(),
                (translated_points - torch.mean(translated_points, axis=0)).cpu().numpy() )
                image_index_2_dice_score[_i] = shape_cd_min

            translated_points_pre = translated_points

            if _s == 19: # the last iteration
                with open(f'{output_dir}/trans_diff/{_i}.obj', 'w') as outfile:
                    for _arr in translated_points.cpu().numpy():
                        outfile.write(f'v {_arr[0]} {_arr[1]} {_arr[2]}\n')

            if _s == predict_0.shape[0]-1: # the last iteration

                translated_points_init_gt, _ = transform_pc(device, vertices, init_pose, gt, None, _s, _i)
                with open(f'{output_dir}/init_gt/{_i}.obj', 'w') as outfile:
                    for _arr in translated_points_init_gt.cpu().numpy():
                        outfile.write(f'v {_arr[0]} {_arr[1]} {_arr[2]}\n')

                with open(f'{output_dir}/trans/{_i}.obj', 'w') as outfile:
                    for _arr in translated_points.cpu().numpy():
                        outfile.write(f'v {_arr[0]} {_arr[1]} {_arr[2]}\n')

                #print(f'{output_dir}/trans/{_i}.obj')
            #print("--------------")
            #print(torch.min(translated_points,axis=0)[0][1])
            #print(torch.min(translated_points,axis=1))
            #print(torch.min(translated_points,axis=0)[0][1].item())
            image_index_2_text[_i] = torch.min(translated_points,axis=0)[0][1].item()
            #print(_s, _i, torch.min(translated_points,axis=0))

            #translated_points = vertices.to(device)
            colors = torch.ones_like(translated_points) * torch.tensor(tab10_r.colors[_i%20][:3]).to(device)  # blue points
            image_index_2_color[_i] = tab10_r.colors[_i%20][:3]
            
            point_cloud = pytorch3d.structures.Pointclouds(points=[translated_points], features=[colors.to(torch.float)])

            translated_points_np = translated_points.cpu().numpy()
            sample_indices = np.random.choice(translated_points_np.shape[0], size=1000, replace=False)
            translated_points_np = translated_points_np[sample_indices, :]
            #print(np.max(translated_points_np,axis=0)[0])
            #print(np.min(translated_points_np,axis=0)[0])
            image_data_list.append(translated_points_np)
            images.append((renderer(point_cloud).cpu().numpy()*255).astype(np.uint8))
            images_front.append((renderer_front(point_cloud).cpu().numpy()*255).astype(np.uint8))
            images_down.append((renderer_down(point_cloud).cpu().numpy()*255).astype(np.uint8))


        axis_pcd_rendered, axis_pcd_rendered_front, axis_pcd_rendered_down = axis_draw(renderer, renderer_front,renderer_down, device)
        images.extend(axis_pcd_rendered)
        images_front.extend(axis_pcd_rendered_front)
        images_down.extend(axis_pcd_rendered_down)

        images_ = sum_arrays(images)
        images_front = sum_arrays(images_front)
        images_down = sum_arrays(images_down)

        '''
        print(len(image_data_list))
        for w, imageg in enumerate(image_data_list):
            print(imageg.shape)
        '''
        np.save(f'{output_dir}/iteration/data/{_s}',np.array(image_data_list) )

        
        imageio.mimsave(f'{output_dir}/iteration/{_s}.png', [images_[0]], format='PNG')
        imageio.mimsave(f'{output_dir}/iteration/{_s}_front.png', [images_front[0]], format='PNG')
        imageio.mimsave(f'{output_dir}/iteration/{_s}_down.png', [images_down[0]], format='PNG')

        sorted_image_index_2_text = sorted(image_index_2_text.items(), key=lambda item: item[1] ,reverse=True)
        
        slice_pos_text = []
        slice_pos_text.append(f'iteration {_s}')
        slice_pos_text.append(f'id : y : dice')
        for imaage_id, text in sorted_image_index_2_text:

            if _s > 19 and imaage_id >= 1:                
                if _s - 19 >= imaage_id:
                    slice_pos_text.append(f'{obj_id_list[imaage_id]} : {text:.3f} : {image_index_2_dice_score[imaage_id]:.3f} **')
                else:
                    slice_pos_text.append(f'{obj_id_list[imaage_id]} : {text:.3f} : {image_index_2_dice_score[imaage_id]:.3f} ')
            else:
                slice_pos_text.append(f'{obj_id_list[imaage_id]} : {text:.3f}')
            
        add_text_to_png(f'{output_dir}/iteration/{_s}.png', slice_pos_text, f'{output_dir}/iteration/{_s}.png')
        add_text_to_png(f'{output_dir}/iteration/{_s}_front.png', slice_pos_text, f'{output_dir}/iteration/{_s}_front.png')
        add_text_to_png(f'{output_dir}/iteration/{_s}_down.png', slice_pos_text, f'{output_dir}/iteration/{_s}_down.png')
        #ax.imshow(images_[0, ..., :3])





    #plt.show(fig)
    _last_obj_flies  = glob.glob(f'{output_dir}/trans_diff/{"*"}.obj')
    slice_util.combine_obj_files(_last_obj_flies,f'{output_dir}/combined_diff.obj')
    _last_obj_flies  = glob.glob(f'{output_dir}/trans/{"*"}.obj')
    slice_util.combine_obj_files(_last_obj_flies,f'{output_dir}/combined_diff_rotate.obj')

    w = iio.get_writer(f'{output_dir}/video.mp4', format='FFMPEG', mode='I', fps=2,
                        #codec='h264_vaapi',
                        pixelformat='yuv420p')
    
    for _step in range(predict_0.shape[0]):
        w.append_data(iio.imread(f'{output_dir}/iteration/{_step}.png'))

    w.close()


    w = iio.get_writer(f'{output_dir}/video_front.mp4', format='FFMPEG', mode='I', fps=2,
                        #codec='h264_vaapi',
                        pixelformat='yuv420p')
    
    for _step in range(predict_0.shape[0]):
        w.append_data(iio.imread(f'{output_dir}/iteration/{_step}_front.png'))

    w.close()

    w = iio.get_writer(f'{output_dir}/video_down.mp4', format='FFMPEG', mode='I', fps=2,
                        #codec='h264_vaapi',
                        pixelformat='yuv420p')
    
    for _step in range(predict_0.shape[0]):
        w.append_data(iio.imread(f'{output_dir}/iteration/{_step}_down.png'))

    w.close()

def add_text_to_png(image_path, text_lines, output_path):
    """
    PNG 이미지에 여러 줄의 텍스트를 추가하는 함수입니다.

    :param image_path: 텍스트를 추가할 PNG 이미지 파일의 경로
    :param text_lines: 이미지에 추가할 텍스트 리스트
    :param output_path: 텍스트가 추가된 이미지를 저장할 경로
    """
    try:
        # 이미지 열기
        img = Image.open(image_path).convert("RGBA")
        draw = ImageDraw.Draw(img)

        # 사용할 폰트와 크기 설정
        font_size = 30
        try:
            font = ImageFont.truetype("Arial.ttf", font_size)
        except IOError:
            #print("Arial.ttf 폰트를 찾을 수 없습니다. 기본 폰트를 사용합니다.")
            font = ImageFont.load_default()

        # 텍스트를 그릴 시작 위치 (왼쪽 상단에 약간의 여백 추가)
        margin = 20
        y_position = margin

        for line in text_lines:
            # 현재 줄의 텍스트 크기 계산 (textsize() 대신 textbbox() 사용)
            bbox = draw.textbbox((0, 0), line, font=font)
            text_height = bbox[3] - bbox[1]

            # 이미지에 텍스트 그리기
            draw.text((margin, y_position), line, font=font, fill=(255, 255, 255, 255))

            # 다음 줄을 위해 y_position 업데이트
            y_position += text_height + 10 # 줄 간격 10픽셀 추가

        # 결과 이미지 저장
        img.save(output_path, "PNG")
        #print(f"텍스트가 성공적으로 추가되어 {output_path}에 저장되었습니다.")

    except Exception as e:
        print(f"오류가 발생했습니다: {e}")

def gen_final_image(device, vertice_list,  df_trasformation, final_image_output_dir):


    
    renderer, renderer_front = get_renderer(device)

    img_data_list_original_data = []
    img_data_list_original = []
    img_data_list_original_front = []
    for _i, vertices in enumerate(tqdm(vertice_list)):
        #print(pcd_file_name)
        #vertices = vertices.cpu()
        translated_points = vertices.to(device).to(torch.float)

        trans_rot_mat = df_trasformation[df_trasformation['part_index']==_i]['transformation_matrix'].values[0].T
        trans_rot_mat = torch.from_numpy(trans_rot_mat).to(device).to(torch.float)
        #translated_points_init, _ = transform_pc(device, translated_points, init_pose, None, None, -1, _i)
        
        _trans_rot_mat = Transform3d(matrix=trans_rot_mat)
        
        temp_t = Transform3d(device=device).compose(_trans_rot_mat)
        translated_points = temp_t.transform_points(translated_points)#.to(torch.float).to(device)


        colors = torch.ones_like(translated_points) * torch.tensor(tab10_r.colors[(_i)%len(tab10_r.colors)][:3]).to(device)  # blue points

        pcd_original = pytorch3d.structures.Pointclouds(points=[translated_points], features=[colors.to(torch.float)])
        

        img_data_list_original_data.append(translated_points.cpu().numpy())
        img_data_list_original.append((renderer(pcd_original)[0].cpu().numpy()*255).astype(np.uint8))
        img_data_list_original_front.append((renderer_front(pcd_original)[0].cpu().numpy()*255).astype(np.uint8))

    np.save(final_image_output_dir+"/0/final",np.array(img_data_list_original_data))

    imageio.mimsave(final_image_output_dir+"/0/final.png", [sum_arrays(img_data_list_original)], format='PNG')
    imageio.mimsave(final_image_output_dir+"/0/final_front.png", [sum_arrays(img_data_list_original_front)], format='PNG')
    return np.array(img_data_list_original_data)


def get_y_rotation_angle_360(q) -> float:
    """
    (w, 0, 1, 0) 형식의 쿼터니언에서 Y축 회전 각도를 360도 형식으로 계산합니다.

    Args:
        q (tuple or list): (w, x, y, z) 형식의 쿼터니언.

    Returns:
        float: Y축 회전각 (0 ~ 360도).
    """
    w, _, y, _ = q
    
    # 쿼터니언의 벡터 부분이 정규화되지 않았을 수 있으므로 y 성분만 사용
    # atan2(y, x) -> atan2(sin(theta/2), cos(theta/2))
    angle_rad = 2 * np.arctan2(y, w)
    
    # 라디안을 도로 변환
    angle_deg = np.degrees(angle_rad)

    # 각도를 0 ~ 360도 범위로 정규화
    normalized_angle = angle_deg % 360
    if normalized_angle < 0:
        normalized_angle += 360
        
    return normalized_angle

def make_video(device, part_pcs_gt11, original_vertices, inference_dir, output_dir, obj_id_list, dice_process = False):
    
        
    data_id = inference_dir.rsplit("/",1)[1]

    gt = np.load(f'{inference_dir}/gt.npy')
    #print(gt.shape)

    init_pose = np.load(f'{inference_dir}/init_pose.npy')
    #print(init_pose.shape)
    
    predict_file_name = glob.glob(f'{inference_dir}/predict*')[0]
    predict_0 = np.load(predict_file_name)
    



    #pts_list = []
    if dice_process:
        #predict_last = predict_0.copy()
        predict_last = np.zeros((1,predict_0.shape[1],predict_0.shape[2]))
        #print('predict_0',predict_0.shape) # [20, 18, 7]
        metric = ChamferDistance()
        last_step_index = predict_0.shape[0]-1
        for _i, vertices  in tqdm(enumerate(original_vertices),desc='rotate by dice score'):
            #ertices torch.Size([10000, 3])
            #init_pose (7,)
            #gt (12, 7)
            #predict_0 (20, 12, 7)
            #print('vertices',vertices.shape)
            #print('init_pose',init_pose.shape)
            #print('gt',gt.shape)
            #print('predict_0',predict_0.shape)
            translated_points_cur, _ = transform_pc(device, vertices, init_pose, gt, predict_0, last_step_index, _i)
            #print(torch.max(translated_points_cur, axis=0)[0][1], torch.min(translated_points_cur, axis=0)[0][1])
            if _i >= 1:

                y_rotation_min = 0
                y_rotation = 1
                _iter = 0
                
                shape_cd_min_init = shape_cd_min = slice_util.calculate_dice_score_from_point_clouds((translated_points_pre - torch.mean(translated_points_pre, axis=0)).cpu().numpy(),
                (translated_points_cur - torch.mean(translated_points_cur, axis=0)).cpu().numpy() )
                '''
                shape_cd_min = metric(
                    translated_points_pre.unsqueeze(0), 
                    translated_points_cur.unsqueeze(0), 
                    bidirectional=False, 
                    point_reduction='mean', 
                    batch_reduction=None
                ).item()
                '''
                translated_points_cur_min = translated_points_cur
                #print(_i, _iter, shape_cd_min)
                while _iter <= 360:
                    #y_rotation = torch.rand(1)
                    translated_points_cur_rotated, quat_gt = slice_util._rotate_pc_y(translated_points_cur, y_rotation, device)
                    #print(torch.max(translated_points_cur_rotated, axis=0)[0][1], torch.min(translated_points_cur_rotated, axis=0)[0][1])
                    #print(translated_points_cur_rotated.shape)
                    '''
                    shape_cd = metric(
                        translated_points_pre.unsqueeze(0), 
                        translated_points_cur_rotated.unsqueeze(0), 
                        bidirectional=False, 
                        point_reduction='mean', 
                        batch_reduction=None
                    ).item()
                    '''
                    shape_cd = slice_util.calculate_dice_score_from_point_clouds((translated_points_pre - torch.mean(translated_points_pre, axis=0)).cpu().numpy(),
                            (translated_points_cur_rotated - torch.mean(translated_points_cur_rotated, axis=0)).cpu().numpy() )
                    
                    #print(_i, _iter, shape_cd, y_rotation,get_y_rotation_angle_360(quat_gt))
                    if shape_cd_min < shape_cd:
                        y_rotation_min = y_rotation
                        
                        #print(shape_cd, y_rotation_min.shape, y_rotation)
                        translated_points_cur_min = translated_points_cur_rotated
                        shape_cd_min = shape_cd

                        new_trans_rotate = predict_0[predict_0.shape[0]-1,_i,:].copy()
                        new_trans_rotate[3:] = quat_gt
                        predict_last[0,_i,:] = new_trans_rotate

                        #predict_last = np.insert(predict_last, predict_0.shape[0], new_element)
                    else:
                        y_rotation += 1
                    _iter += 1
                #print(f'{_i}, {shape_cd_min_init:.3f}, {shape_cd_min:.3f}, {y_rotation_min}')
                translated_points_pre = translated_points_cur_min
                
            else:
                predict_last[0,_i,:] = predict_0[predict_0.shape[0]-1,_i,:]
                translated_points_pre = translated_points_cur
                
            
            #pts_list.append(translated_points_pre  
        #print(predict_0.shape)

        #/data/jhahn/data/shape_dataset/data/mouse_brain_50mm/50_tickness_20_sllices_test/fractured_0

        #mesh_file_dir = '/data/jhahn/data/shape_dataset/data/mouse_brain_50mm/50_tickness_20_sllices_test/fractured_0/'

        #verts_list = get_vertices(mesh_file_dir, device)
        predict_alg = np.zeros((predict_0.shape[1],predict_0.shape[1],predict_0.shape[2]))
        for _alg_step  in range(original_vertices.shape[0]):
            predict_alg[_alg_step,:,:] = predict_0[predict_0.shape[0]-1,:,:]
            predict_alg[_alg_step, :_alg_step+1, :] = predict_last[0,:_alg_step+1,:]

            
        #plot_pointcloud2(verts,xlim=(0, 1), ylim=(0, 0.9), zlim=(0, 1))
        #plot_pointcloud2(device, f'{output_dir}/{data_id}',verts_list, obj_id_list, init_pose, gt, predict_0, xlim=(-2, 2), ylim=(-2, 2), zlim=(-2, 2))
        
        predict_rotated = np.concatenate((predict_0, predict_alg), axis=0)
    else:
        predict_rotated = predict_0

    #print(predict_last)
    os.makedirs(f'{output_dir}/{data_id}', exist_ok=True)

    #plot_pointcloud2(device, f'{output_dir}/{data_id}',part_pcs_gt, obj_id_list, init_pose, gt, predict_rotated, xlim=(-2, 2), ylim=(-2, 2), zlim=(-2, 2))
    plot_pointcloud2_same_with_part_acc(device, f'{output_dir}/{data_id}',part_pcs_gt11,original_vertices, obj_id_list, init_pose, gt, predict_rotated, xlim=(-2, 2), ylim=(-2, 2), zlim=(-2, 2))




                
if __name__ == "__main__":
        
    test_array1 = [ 
        [[255, 255, 255],[255, 255, 255]],
        [[255, 11, 255],[255, 255, 255]]
    ]
    test_array2 = [ 
        [[33, 11, 255],[255, 255, 255]],
        [[255, 13, 255],[255, 255, 255]]
    ]

    total = []
    total.append(np.array(test_array1))
    total.append(np.array(test_array2))
    #print(np.array(test_array1).shape)
    #sum_arrays(total)
'''

i_10 = np.load(render_output_dir+"/0/iteration/data/19.npy")
i_final = np.load(render_output_dir+"/0/final.npy")
are_close = np.isclose(i_10, i_final)
not_close_indices = np.where(~are_close)
print("Elements not close:")
print("Array 1:", i_10[not_close_indices])
print("Array 2:", i_final[not_close_indices])
'''
