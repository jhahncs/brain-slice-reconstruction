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
from pytorch3d.transforms import Transform3d
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
        min_list.append(torch.min(vertices,axis=0)[0].cpu().numpy())
        max_list.append(torch.max(vertices,axis=0)[0].cpu().numpy())

    #print(np.min(min_list,axis=0),np.max(max_list,axis=0))
    _min = np.min(min_list,axis=0)
    _max = np.max(max_list,axis=0)
    
    #xlim = (float(_min[0]),  float(_max[0]))
    #ylim = (float(_min[1]), float(_max[1]))
    #zlim = (float(_min[2]),  float(_max[2]))
    #print(xlim, ylim, zlim)
    #print(type(_max[0]))

    return vertice_list

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

def _transform_pc(translated_points, trans, rotate, device):
    tr = Translate(torch.FloatTensor([trans]), dtype=torch.float32, device=device)
    _mean = torch.mean(translated_points, axis=0)
    tr_c = Translate(-_mean[0],-_mean[1],-_mean[2], dtype=torch.float32, device=device)
    tr_c_r = Translate(_mean[0],_mean[1],_mean[2], dtype=torch.float32, device=device)
    rr = Rotate(slice_util.quaternion_to_matrix(torch.FloatTensor(rotate)), dtype=torch.float32, device=device)    
    return [tr_c, rr, tr_c_r, tr]

def transform_pc(device, vertices, init_pose, gt, trans_rotate, step, part_index):

    translated_points = vertices.clone()

    transformation_elem = []
    total_temp_t = None
    if init_pose is not None and gt is None and trans_rotate is None:
        init_trans = init_pose[:3]
        init_rotate = init_pose[3:]
        init_trans = -init_pose[:3]
        init_rotate = slice_util.invert_rotation_quaternion(init_pose[3:])
        

        _mean = torch.mean(translated_points, axis=0)
        tr_c = Translate(-_mean[0],-_mean[1],-_mean[2], dtype=torch.float32, device=device)
        rr = Rotate(slice_util.quaternion_to_matrix(torch.FloatTensor(init_rotate)), dtype=torch.float32, device=device)  
        tr_c_r = Translate(_mean[0],_mean[1],_mean[2], dtype=torch.float32, device=device)
        tr = Translate(torch.FloatTensor([init_trans]), dtype=torch.float32, device=device)

        temp_t = Transform3d(device=device).compose(tr_c).compose(rr).compose(tr_c_r).compose(tr)
        translated_points = temp_t.transform_points(translated_points)#.to(torch.float).to(device)

    elif init_pose is not None and gt is not None and trans_rotate is None:
        init_trans = init_pose[:3]
        init_rotate = init_pose[3:]
        init_trans = -init_pose[:3]
        init_rotate = slice_util.invert_rotation_quaternion(init_pose[3:])
        

        
        _mean = torch.mean(translated_points, axis=0)
        tr_c = Translate(-_mean[0],-_mean[1],-_mean[2], dtype=torch.float32, device=device)
        rr = Rotate(slice_util.quaternion_to_matrix(torch.FloatTensor(init_rotate)), dtype=torch.float32, device=device)  
        tr_c_r = Translate(_mean[0],_mean[1],_mean[2], dtype=torch.float32, device=device)
        tr = Translate(torch.FloatTensor([init_trans]), dtype=torch.float32, device=device)

        temp_t = Transform3d(device=device).compose(tr_c).compose(rr).compose(tr_c_r).compose(tr)
        translated_points = temp_t.transform_points(translated_points)#.to(torch.float).to(device)


        gt_trans = gt[part_index,:3]
        gt_rotate = gt[part_index,3:]
        gt_trans = -gt[part_index,:3]
        gt_rotate = slice_util.invert_rotation_quaternion(gt[part_index,3:])


        _mean = torch.mean(translated_points, axis=0)
        tr_c = Translate(-_mean[0],-_mean[1],-_mean[2], dtype=torch.float32, device=device)
        rr = Rotate(slice_util.quaternion_to_matrix(torch.FloatTensor(gt_rotate)), dtype=torch.float32, device=device)  
        tr_c_r = Translate(_mean[0],_mean[1],_mean[2], dtype=torch.float32, device=device)
        tr = Translate(torch.FloatTensor([gt_trans]), dtype=torch.float32, device=device)

        temp_t = Transform3d(device=device).compose(tr_c).compose(rr).compose(tr_c_r).compose(tr)
        translated_points = temp_t.transform_points(translated_points)#.to(torch.float).to(device)

    elif init_pose is not None and gt is not None and trans_rotate is not None:

        init_trans = init_pose[:3]
        init_rotate = init_pose[3:]
        init_trans = -init_pose[:3]
        init_rotate = slice_util.invert_rotation_quaternion(init_pose[3:])
        

        
        _mean = torch.mean(translated_points, axis=0)
        tr_c = Translate(-_mean[0],-_mean[1],-_mean[2], dtype=torch.float32, device=device)
        rr = Rotate(slice_util.quaternion_to_matrix(torch.FloatTensor(init_rotate)), dtype=torch.float32, device=device)  
        tr_c_r = Translate(_mean[0],_mean[1],_mean[2], dtype=torch.float32, device=device)
        tr = Translate(torch.FloatTensor([init_trans]), dtype=torch.float32, device=device)

        temp_t = Transform3d(device=device).compose(tr_c).compose(rr).compose(tr_c_r).compose(tr)
        translated_points = temp_t.transform_points(translated_points)#.to(torch.float).to(device)
        transformation_elem.extend([tr_c,rr,tr_c_r,tr])

        gt_trans = gt[part_index,:3]
        gt_rotate = gt[part_index,3:]
        gt_trans = -gt[part_index,:3]
        gt_rotate = slice_util.invert_rotation_quaternion(gt[part_index,3:])


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
        rr = Rotate(slice_util.quaternion_to_matrix(torch.FloatTensor(rotate)), dtype=torch.float32, device=device)  
        tr_c_r = Translate(_mean[0],_mean[1],_mean[2], dtype=torch.float32, device=device)
        tr = Translate(torch.FloatTensor([trans]), dtype=torch.float32, device=device)

        temp_t = Transform3d(device=device).compose(tr_c).compose(rr).compose(tr_c_r).compose(tr)
        translated_points = temp_t.transform_points(translated_points)#.to(torch.float).to(device)
        
        transformation_elem.extend([tr_c,rr,tr_c_r,tr])
        total_temp_t = Transform3d(device=device)
        for t in transformation_elem:
            total_temp_t = total_temp_t.compose(t)


    return translated_points, total_temp_t

def get_renderer(device, xlim=(-0.5, 1),ylim=(-2, 2),zlim=(-0.5, 1)):
    image_size = 512
    radius = 0.002
    points_per_pixel = 50
    R, T = look_at_view_transform(10, 30, 90)
    cameras = FoVOrthographicCameras( device=device, R=R, T=T, znear=zlim[0], zfar=zlim[1], min_x=xlim[0], max_x=xlim[1], min_y=ylim[0], max_y=ylim[1])

    R, T = look_at_view_transform(10, 0, 90)
    cameras_front = FoVOrthographicCameras( device=device, R=R, T=T, znear=zlim[0], zfar=zlim[1], min_x=xlim[0], max_x=xlim[1], min_y=ylim[0], max_y=ylim[1])

    raster_settings = PointsRasterizationSettings(
        image_size=image_size, 
        radius = radius,
        points_per_pixel = points_per_pixel
    )

    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    rasterizer_front = PointsRasterizer(cameras=cameras_front, raster_settings=raster_settings)

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
    return renderer, renderer_front
def pcd_list_2_img(device,
    vertice_list,
    output_img_file_name_original,
    output_img_file_name_original_front,
    output_img_file_name_init,
    output_img_file_name_init_front,
    output_img_file_name_init_gt,
    output_img_file_name_init_gt_front,
    alpha=.8,
    max_points=10000,
    xlim=(-1, 1),
    ylim=(-1.5, 1),
    zlim=(-2, 2),
    init_pose = None,
    gt = None):  
    
    xlim = (-0.5, 1)
    zlim = (-0.5, 1)
    ylim = (-2, 2)
    
    #renderer, renderer_front = get_renderer(device, xlim, ylim, zlim)
    renderer, renderer_front = get_renderer(device)

    img_data_list_original = []
    img_data_list_original_front = []
    img_data_list_init = []
    img_data_list_init_front = []
    img_data_list_init_gt = []
    img_data_list_init_gt_front = []
    for _i, vertices in enumerate(tqdm(vertice_list)):
        #print(pcd_file_name)
        #vertices = vertices.cpu()
        translated_points = vertices.to(device).to(torch.float)
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
        

        img_data_list_original.append((renderer(pcd_original)[0].cpu().numpy()*255).astype(np.uint8))
        img_data_list_original_front.append((renderer_front(pcd_original)[0].cpu().numpy()*255).astype(np.uint8))

        img_data_list_init.append((renderer(pcd_init)[0].cpu().numpy()*255).astype(np.uint8))
        img_data_list_init_front.append((renderer_front(pcd_init)[0].cpu().numpy()*255).astype(np.uint8))

        img_data_list_init_gt.append((renderer(pcd_init_gt)[0].cpu().numpy()*255).astype(np.uint8))
        img_data_list_init_gt_front.append((renderer_front(pcd_init_gt)[0].cpu().numpy()*255).astype(np.uint8))

    
    imageio.mimsave(output_img_file_name_original, [sum_arrays(img_data_list_original)], format='PNG')
    imageio.mimsave(output_img_file_name_original_front, [sum_arrays(img_data_list_original_front)], format='PNG')
    imageio.mimsave(output_img_file_name_init, [sum_arrays(img_data_list_init)], format='PNG')
    imageio.mimsave(output_img_file_name_init_front, [sum_arrays(img_data_list_init_front)], format='PNG')
    imageio.mimsave(output_img_file_name_init_gt, [sum_arrays(img_data_list_init_gt)], format='PNG')
    imageio.mimsave(output_img_file_name_init_gt_front, [sum_arrays(img_data_list_init_gt_front)], format='PNG')

def gt_img(device, vertice_list, inference_result_dir, output_dir):

    data_id = inference_result_dir.split("/")[-1]
    os.makedirs(f'{output_dir}/{data_id}',exist_ok=True)


    gt = np.load(f'{inference_result_dir}/gt.npy')
    init_pose = np.load(f'{inference_result_dir}/init_pose.npy')

    pcd_list_2_img(device, vertice_list, 
        f'{output_dir}/{data_id}/{"original"}.png',f'{output_dir}/{data_id}/{"original_front"}.png',
        f'{output_dir}/{data_id}/{"init"}.png', f'{output_dir}/{data_id}/{"init_front"}.png', 
        f'{output_dir}/{data_id}/{"init_gt"}.png',f'{output_dir}/{data_id}/{"init_gt_front"}.png',
        init_pose = init_pose,
        gt = gt)




    


def plot_pointcloud2(
    device,
    output_dir,
    vertices_list,
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




    renderer, renderer_front = get_renderer(device)


    os.makedirs(output_dir+"/iteration/data", exist_ok = True)

    os.makedirs(output_dir+"/trans", exist_ok = True)
    os.makedirs(output_dir+"/init_gt", exist_ok = True)
    #fig = plt.figure(figsize=(25,20))
    #predict_0[:,:,5:] = np.zeros((predict_0.shape[0],predict_0.shape[1],2))
    #predict_0[:,:,4:5] = np.ones((predict_0.shape[0],predict_0.shape[1],1))
    for _s  in range(predict_0.shape[0]):
            

        #ax = fig.add_subplot(4, 5, _s+1)
        # ax.set_axis_off()
        frames = []
        images = []
        image_data_list = []
        images_front = []
        for _i, vertices  in enumerate(vertices_list):

            
            #vertices = vertices.cpu()

            N, dim = vertices.shape
            #print(type(vertices))

            
            #tr = Translate(torch.FloatTensor([-gt[_i,:3]]))
            #rr = Rotate(quaternion_to_matrix(torch.FloatTensor([-gt[_i,3:]])))
            translated_points, _ = transform_pc(device, vertices, init_pose, gt,predict_0, _s, _i)

            if _s == predict_0.shape[0]-1:

                translated_points_init_gt, _ = transform_pc(device, vertices, init_pose, gt, None, _s, _i)
                with open(f'{output_dir}/init_gt/{_i}.obj', 'w') as outfile:
                    for _arr in translated_points_init_gt.cpu().numpy():
                        outfile.write(f'v {_arr[0]} {_arr[1]} {_arr[2]}\n')

                with open(f'{output_dir}/trans/{_i}.obj', 'w') as outfile:
                    for _arr in translated_points.cpu().numpy():
                        outfile.write(f'v {_arr[0]} {_arr[1]} {_arr[2]}\n')

                print(f'{output_dir}/trans/{_i}.obj')
                
            #print(torch.max(translated_points,axis=0)[0])
            #print(torch.min(translated_points,axis=0)[0])
            #translated_points = vertices.to(device)
            colors = torch.ones_like(translated_points) * torch.tensor(tab10_r.colors[_i%20][:3]).to(device)  # blue points
            point_cloud = pytorch3d.structures.Pointclouds(points=[translated_points], features=[colors.to(torch.float)])
            image_data_list.append(translated_points.cpu().numpy())
            images.append((renderer(point_cloud).cpu().numpy()*255).astype(np.uint8))
            images_front.append((renderer_front(point_cloud).cpu().numpy()*255).astype(np.uint8))

        images_ = sum_arrays(images)
        images_front = sum_arrays(images_front)

        #print(np.array(image_data_list).shape)
        np.save(f'{output_dir}/iteration/data/{_s}',np.array(image_data_list) )

        
        imageio.mimsave(f'{output_dir}/iteration/{_s}.png', [images_[0]], format='PNG')
        imageio.mimsave(f'{output_dir}/iteration/{_s}_front.png', [images_front[0]], format='PNG')
        #ax.imshow(images_[0, ..., :3])

    #plt.show(fig)
    _last_obj_flies  = glob.glob(f'{output_dir}/trans/{"*"}.obj')
    slice_util.combine_obj_files(_last_obj_flies,f'{output_dir}/combined.obj')

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

def make_video(device, verts_list, inference_dir, output_dir):
    
        
    data_id = inference_dir.rsplit("/",1)[1]

    gt = np.load(f'{inference_dir}/gt.npy')
    #print(gt.shape)

    init_pose = np.load(f'{inference_dir}/init_pose.npy')
    #print(init_pose.shape)
    
    predict_file_name = glob.glob(f'{inference_dir}/predict*')[0]
    predict_0 = np.load(predict_file_name)
    
    #print(predict_0.shape)

#/data/jhahn/data/shape_dataset/data/mouse_brain_50mm/50_tickness_20_sllices_test/fractured_0

    #mesh_file_dir = '/data/jhahn/data/shape_dataset/data/mouse_brain_50mm/50_tickness_20_sllices_test/fractured_0/'

    #verts_list = get_vertices(mesh_file_dir, device)

    #plot_pointcloud2(verts,xlim=(0, 1), ylim=(0, 0.9), zlim=(0, 1))
    os.makedirs(f'{output_dir}/{data_id}', exist_ok=True)
    plot_pointcloud2(device, f'{output_dir}/{data_id}',verts_list, init_pose, gt, predict_0, xlim=(-2, 2), ylim=(-2, 2), zlim=(-2, 2))




                
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
