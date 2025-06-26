import numpy as np
import os
from pytorch3d import transforms
from scipy.spatial.transform import Rotation as R
import torch
import pytorch3d
import random
from chamferdist import ChamferDistance
chamferDist = ChamferDistance()

import time
from torch.nn.functional import normalize

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

def obj_files(_dir):
    obj_file_list = []
    for f in os.listdir(_dir):
        obj_file_list.append(_dir+"/"+f)
    obj_file_list = sorted(obj_file_list, key=lambda x: int(x.split("/")[-1].split(".")[-2]))
    return obj_file_list


def combine_obj_files(obj_file_list, output_filename):
    _obj_list = []

    obj_file_list.sort(key=lambda x: int(x.split("/")[-1].split("_")[-1].replace(".obj",'')), reverse=False)
    
    _pc_list = []
    f_2_last = []
    with open(output_filename, 'w') as outfile:

        f_2_last.append(0)
        for _,fname in enumerate(obj_file_list):
            #print(fname)
            _c = 0
            with open(fname) as infile:
                if fname.endswith('piece.obj'):
                    continue
                #print(fname)
                _pcs = []
                for line in infile:
                    if line.lower().startswith('v'):
                        _c += 1
                        _arr = line[2:].split()
                        _arr = np.array([float(a) for a in _arr])                    
                        _pcs.append(_arr)
                
                _pcs = torch.from_numpy(np.array(_pcs)).type(torch.float32)
                #_pcs = np.array(_pcs)
                #_pcs = trans_pc(_pcs)
                #_pcs = rotate_pc(_pcs)
                #print(_pcs.shape)                                
                _pc_list.append(_pcs)
                
                for _arr in _pcs.numpy():
                    outfile.write(f'v {_arr[0]} {_arr[1]} {_arr[2]}\n')

            f_2_last.append(_c)

        _delta = 0
        for fi, fname in enumerate(obj_file_list):
            _delta += f_2_last[fi]
            with open(fname) as infile:
                if fname.endswith('piece.obj'):
                    continue                
                for line in infile:
                    if line.lower().startswith('f'):
                        _arr = line[2:].split()
                        outfile.write(f'f {int(_arr[0])+_delta} {int(_arr[1])+_delta} {int(_arr[2])+_delta}\n')
    return _obj_list
'''
def combine_obj_files(_dir_list, output_dir):
    _obj_list = []
    for _dir in _dir_list:
        files = [_dir+"/"+f for f in os.listdir(_dir) if os.path.isfile(_dir+"/"+f)]

        _obj_list = combine_obj_files_2(files,output_dir+"/"+_dir.split("/")[-1]+".obj")
        
    return _obj_list
'''


'''
# very slow
def pcs_matched(pc_1, pc_2, point_dist_threshold = 0.03, matched_threshold = 0.5):
    _matched = 0
    #_bottom_flag = torch.zeros(len(pc_2), dtype=torch.bool)
    _b_idx_last = 0
    for _t in pc_1:
        for _b_idx in range(_b_idx_last, len(pc_2)):
            _dist = torch.cdist(_t.unsqueeze(dim=0), pc_2[_b_idx].unsqueeze(dim=0))[0][0]
            #print(_dist)
            if torch.lt(_dist, point_dist_threshold):
                #_bottom_flag[_b_idx] = True
                _matched += 1
                _b_idx_last = _b_idx
                break

    if _matched >= len(pc_1)*matched_threshold:
        return True, _matched
    else:
        return False, _matched
'''

def invert_rotation_quaternion(q):
  """
  Inverts a quaternion representing a rotation.

  Args:
    q: A list or tuple representing the quaternion (w, x, y, z).

  Returns:
    A list representing the inverted quaternion.
  """
  w, x, y, z = q
  return [w, -x, -y, -z]

def rotation_matrix_to_quaternion(R):
    """
    Convert a 3x3 rotation matrix to a quaternion.

    Parameters
    ----------
    R : np.ndarray
        A 3x3 NumPy array representing the rotation matrix.

    Returns
    -------
    np.ndarray
        A 4-element NumPy array representing the quaternion (w, x, y, z).
    """
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s

    return np.array([w, x, y, z])
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






def random_rotation(pc, angle = None):
    """Rotate point cloud by random angle around a random axis"""


    #P, N, _ = pc.shape
    pc = pc.reshape(-1, 3)
    #pc, guat_gt = _rotate_pc(pc)
    

    pc = torch.from_numpy(pc).float()

    _mean = torch.mean(pc, axis=0)
    tr = Translate(-_mean[0],-_mean[1],-_mean[2], dtype=torch.float32)
    tr_r = Translate(_mean[0],_mean[1],_mean[2], dtype=torch.float32)

    if angle is None:
        quat_gt = torch.tensor([torch.rand(1),0,1,0])
    else:
        quat_gt = torch.tensor([angle,0,1,0])
    quat_gt = normalize(quat_gt, p=1.0, dim = 0)
    rr = Rotate(quaternion_to_matrix(quat_gt), dtype=torch.float32)
    t = Transform3d().compose(tr).compose(rr).compose(tr_r)
    pc = t.transform_points(pc)#.to(torch.float).to(device)

    
    return pc.cpu().numpy(), quat_gt.cpu().numpy()

    
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





def top_bottom_pcs(_obj_list, max_num_of_pcs = 1000):
    '''
    _obj_list: [parts, the max number of points, 3]
    '''
    _top_pc_list_batch = None
    _bottom_pc_list_batch = None

    for _pc_list in _obj_list:
            
        _top_pc_list  = None
        _bottom_pc_list  = None
        for _part_pcs in _pc_list:
            #_top_pcs = torch.index_select(_part_pcs, 0, _part_pcs[:,1] >= torch.max(_part_pcs, axis=0).values[1].item() - 0.001)
            #print(_part_pcs.shape)
            #print(torch.max(_part_pcs, axis=0).values)
            _top_pcs = _part_pcs[_part_pcs[:,1] >= torch.max(_part_pcs, axis=0).values[1].item() - 0.001]
            _top_pcs = _top_pcs[np.random.choice(len(_top_pcs), max_num_of_pcs)]
            _top_pcs.sort()
            _top_pcs = _top_pcs.unsqueeze(dim=0)
            #_top_pc_list.append(_top_pcs)
            if _top_pc_list is None:
                _top_pc_list = _top_pcs
            else:
                #print('_top_pc_list',_top_pc_list.shape)
                #print('_top_pcs',_top_pcs.shape)
                _top_pc_list = torch.cat((_top_pc_list,_top_pcs),0)
                #print('_top_pc_list',_top_pc_list.shape)

            _bottom_pcs = _part_pcs[_part_pcs[:,1] <= torch.min(_part_pcs, axis=0).values[1].item() + 0.001]
            _bottom_pcs = _bottom_pcs[np.random.choice(len(_bottom_pcs), max_num_of_pcs)]
            _bottom_pcs.sort()
            _bottom_pcs = _bottom_pcs.unsqueeze(dim=0)
            if _bottom_pc_list is None:
                _bottom_pc_list = _bottom_pcs
            else:
                _bottom_pc_list = torch.cat((_bottom_pc_list,_bottom_pcs),0)
        #_top_pc_list = np.array(_top_pc_list)
        #_bottom_pc_list = np.array(_bottom_pc_list)
        _top_pc_list = _top_pc_list.unsqueeze(dim=0)
        _bottom_pc_list = _bottom_pc_list.unsqueeze(dim=0)
        
        if _top_pc_list_batch is None:
            _top_pc_list_batch = _top_pc_list
        else:
            _top_pc_list_batch = torch.cat((_top_pc_list_batch,_top_pc_list),0)

        if _bottom_pc_list_batch is None:
            _bottom_pc_list_batch = _bottom_pc_list
        else:
            _bottom_pc_list_batch = torch.cat((_bottom_pc_list_batch,_bottom_pc_list),0)
   



    #_top_pc_list_batch = np.array(_top_pc_list_batch)
    #_bottom_pc_list_batch = np.array(_bottom_pc_list_batch)

    return _top_pc_list_batch, _bottom_pc_list_batch
def pc_centroid(_part_pcs):
    shape_len = len(_part_pcs.shape)
    if shape_len == 4 or shape_len == 3: # [Batch, Parts, num of points, (x,y,z)]
        _centroid_min = torch.min(_part_pcs, axis = shape_len - 2).values
        _centroid_max = torch.max(_part_pcs, axis = shape_len - 2).values
        _centroid = (_centroid_max- _centroid_min)/2 +  _centroid_min
        _centroid = torch.repeat_interleave(_centroid.unsqueeze(shape_len - 2), _part_pcs.shape[shape_len - 2], dim=shape_len - 2)        
    else:
        _centroid = (torch.max(_part_pcs, axis=0).values - torch.min(_part_pcs, axis=0).values)/2 +  torch.min(_part_pcs, axis=0).values
    return _centroid
def rotate_pc(_part_pcs, rot_quat = None):
    shape_len = len(_part_pcs.shape)
    _centroid = pc_centroid(_part_pcs)

    _part_pcs = _part_pcs - _centroid
    torch.manual_seed(int(str(time.time())[-1]))
    if rot_quat is None:
        rot_quat = torch.from_numpy(np.array([torch.rand(1).item(), 0, 1.0, 0]))
        print('rot_quat',rot_quat)
    elif rot_quat is not None and (shape_len == 4 or shape_len == 3):
        rot_quat = torch.repeat_interleave(rot_quat.unsqueeze(shape_len - 2),  _part_pcs.shape[shape_len - 2], dim= shape_len - 2)   

    rot_quat = rot_quat / rot_quat.norm(dim=-1, keepdim=True)
    #else:
    #print(rot_quat.shape)
    #print(_part_pcs.shape)
    #real_parts = _part_pcs.new_zeros(_part_pcs.shape[:-1] + (1,))
    #print(real_parts.shape)
    #point_as_quaternion = torch.cat((real_parts, _part_pcs), -1)
    #print('_centroid',_centroid.shape)
    #print('rot_quat',rot_quat.shape)
    #print('_part_pcs',_part_pcs.shape)
    
    _part_pcs = pytorch3d.transforms.quaternion_apply(rot_quat,_part_pcs)
    #print('_part_pcs',_part_pcs.shape)
    _part_pcs = _part_pcs + _centroid

    return _part_pcs, _centroid, rot_quat

def trans_pc(_part_pcs, trans_vec = None):
    shape_len = len(_part_pcs.shape)
    if trans_vec is None:
        torch.manual_seed(int(str(time.time())[-1]))
        trans_vec = torch.rand(3)
        #trans_vec[1] = 0
    elif trans_vec is not None and (shape_len == 4 or shape_len == 3): # [Batch, Parts, num of points, (x,y,z)]
        trans_vec = torch.repeat_interleave(trans_vec.unsqueeze(shape_len - 2),  _part_pcs.shape[shape_len - 2], dim= shape_len - 2)   


    _part_pcs = torch.sub(_part_pcs[...,:], trans_vec)

    return _part_pcs, trans_vec