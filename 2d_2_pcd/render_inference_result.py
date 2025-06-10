import os
import json
import torch
import trimesh
import numpy as np
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
import glob
import copy
from utils import *


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


def sum_arrays(total):

    summed_array = copy.deepcopy(total[0])
    if len(total) ==  1:
        return summed_array

    for arr in total[1:]:
        zero_mask = np.ones(total[0].shape, dtype=bool)
        zero_mask &= (summed_array == 255)
        summed_array = np.where(zero_mask, arr, summed_array)

    return summed_array


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def pcd_2_img(
    device,
    pcd_file_name,
    output_img_file_name,
    alpha=.8,
    max_points=10000,
    xlim=(-1, 1),
    ylim=(-1, 1),
    zlim=(-1, 1)
    ):
    """Plot a pointcloud tensor of shape (N, coordinates)
    """

    


    image_size = 512
    radius = 0.002
    points_per_pixel = 100
    R, T = look_at_view_transform(10, 20, 90)
    cameras = FoVOrthographicCameras( device=device, R=R, T=T, znear=-2, zfar=2, min_x=0, max_x=1, min_y=-1, max_y=1)
    #cameras = FoVOrthographicCameras( device=device, R=R, T=T)
    #tensor([0.6651, 0.7305, 0.4894], device='cuda:0')
    #tensor([ 0.1199,  0.1349, -0.0800], device='cuda:0')

    raster_settings = PointsRasterizationSettings(
        image_size=image_size, 
        radius = radius,
        points_per_pixel = points_per_pixel
    )

    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)

    renderer = pytorch3d.renderer.PointsRenderer(
        rasterizer=rasterizer,
        compositor=pytorch3d.renderer.AlphaCompositor(background_color=(1, 1, 1))
        #compositor=NormWeightedCompositor()
    )
    
    vertices = pytorch3d.io.load_obj(pcd_file_name, device=device)[0]
    #vertices = vertices.cpu()

    #N, dim = vertices.shape

    #tr = Translate(torch.FloatTensor([-gt[_i,:3]]))
    #rr = Rotate(quaternion_to_matrix(torch.FloatTensor([-gt[_i,3:]])))
    translated_points = vertices.to(device).to(torch.float)
    #print(torch.max(translated_points, axis=0)[0])
    #print(torch.min(translated_points, axis=0)[0])
    #translated_points = vertices.to(device)
    colors = torch.ones_like(translated_points) * torch.tensor(tab10_r.colors[0][:3]).to(device)  # blue points
    point_cloud = pytorch3d.structures.Pointclouds(points=[translated_points], features=[colors.to(torch.float)])

    _img = renderer(point_cloud)[0].cpu().numpy()*255
    _img = _img.astype(np.uint8)
    #print(_img.shape)

    imageio.mimsave(output_img_file_name, [_img], format='PNG')


def plot_pointcloud2(
    device,
    output_dir,
    vertices_list,
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


    image_size = 512
    radius = 0.002
    points_per_pixel = 100
    R, T = look_at_view_transform(10, 20, 90)
    #cameras = FoVOrthographicCameras( device=device, R=R, T=T, znear=100, zfar=-1200, min_x=-12000, max_x=1000, min_y=-15000, max_y=5000)
    cameras = FoVOrthographicCameras( device=device, R=R, T=T, znear=5, zfar=-5, min_x=-1, max_x=1, min_y=-5, max_y=5)
    #cameras = FoVOrthographicCameras( device=device, R=R, T=T)

    raster_settings = PointsRasterizationSettings(
        image_size=image_size, 
        radius = radius,
        points_per_pixel = points_per_pixel
    )

    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)

    renderer = pytorch3d.renderer.PointsRenderer(
        rasterizer=rasterizer,
        compositor=pytorch3d.renderer.AlphaCompositor(background_color=(1, 1, 1))
        #compositor=NormWeightedCompositor()
    )
    
    #fig = plt.figure(figsize=(25,20))

    for _s  in range(20):
            

        #ax = fig.add_subplot(4, 5, _s+1)
        # ax.set_axis_off()
        frames = []
        images = []
        for _i, vertices  in enumerate(vertices_list):

            
            vertices = vertices.cpu()

            N, dim = vertices.shape
            #print(type(vertices))
            if N > max_points:
                #print(N)
                vertices = torch.from_numpy(np.random.default_rng().choice(vertices, max_points, replace=False))
            
            #tr = Translate(torch.FloatTensor([-gt[_i,:3]]))
            #rr = Rotate(quaternion_to_matrix(torch.FloatTensor([-gt[_i,3:]])))
            tr = Translate(torch.FloatTensor([-predict_0[_s, _i,:3]]))
            rr = Rotate(quaternion_to_matrix(torch.FloatTensor([-predict_0[_s, _i,3:]])))
            t = Transform3d().compose(tr).compose(rr)
            translated_points = t.transform_points(vertices).to(torch.float).to(device)
            #print(torch.max(translated_points,axis=0)[0])
            #print(torch.min(translated_points,axis=0)[0])
            #translated_points = vertices.to(device)
            colors = torch.ones_like(translated_points) * torch.tensor(tab10_r.colors[_i%20][:3]).to(device)  # blue points
            point_cloud = pytorch3d.structures.Pointclouds(points=[translated_points], features=[colors.to(torch.float)])

            _img = renderer(point_cloud).cpu().numpy()*255
            _img = _img.astype(np.uint8)
            #print(_img.shape)
            images.append(_img)

        images_ = sum_arrays(images)
        imageio.mimsave(f'{output_dir}/{_s}.png', [images_[0]], format='PNG')
        #ax.imshow(images_[0, ..., :3])

    #plt.show(fig)

    w = iio.get_writer(f'{output_dir}/video.mp4', format='FFMPEG', mode='I', fps=2,
                        #codec='h264_vaapi',
                        pixelformat='yuv420p')
    
    for _step in range(20):
        w.append_data(iio.imread(f'{output_dir}/{_step}.png'))

    w.close()


def make_video(device, data_dir, inference_dir, output_dir):
    
    with open(f'{inference_dir}/mesh_file_path.txt') as f:
        _mesh_file_dir = f.read()
    mesh_file_dir = data_dir+"/"+_mesh_file_dir

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
    _files = os.listdir(mesh_file_dir)
    mesh_file_list = []
    for f in _files:
        if f.endswith(".obj"):
            mesh_file_list.append(mesh_file_dir+"/"+f)

    mesh_file_list = sorted(mesh_file_list, key=lambda x: int(x.split("/")[-1].split(".")[-2]))
    verts_list = []
    for mesh_file in mesh_file_list[:20]:
        #print(mesh_file)
        mesh = pytorch3d.io.load_obj(mesh_file, device=device)
        verts_list.append(mesh[0])

    #plot_pointcloud2(verts,xlim=(0, 1), ylim=(0, 0.9), zlim=(0, 1))
    os.makedirs(f'{output_dir}/{data_id}', exist_ok=True)
    plot_pointcloud2(device, f'{output_dir}/{data_id}',verts_list, predict_0, xlim=(-9000, 0), ylim=(-9000, 0), zlim=(-9000, 0))



                
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