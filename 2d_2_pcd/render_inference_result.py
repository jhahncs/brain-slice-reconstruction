import os
import json
import torch
import trimesh
import numpy as np
import slice_util
import open3d as o3d
import cv2
import random
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
import cupy as cp

import puzzlefusion_plusplus.denoiser.evaluation.transform as puzzle_transform
from matplotlib.colors import LinearSegmentedColormap


import imageio
viridis = plt.get_cmap('tab20b_r')
tab10_r = ListedColormap(viridis(np.arange(20)))
tab10_r.colors[0][:3]
tab10_r


from moviepy.editor import VideoFileClip, concatenate_videoclips
import os

def concatenate_mp4_files(file_paths, output_path, method="compose"):
    """
    MP4 파일 목록을 연결하여 하나의 파일로 저장합니다.
    
    :param file_paths: 연결할 MP4 파일 경로의 리스트 (순서 중요)
    :param output_path: 연결된 비디오를 저장할 출력 파일 경로 (예: "combined_output.mp4")
    :param method: 클립을 연결하는 방법 ("compose"가 기본값이며 가장 일반적임)
    """
    '''
    for video_filename in ['video_down','video_front','video']:

        input_files = [
            f"{files_root}/{data_ids[0]}/0/render/0/{video_filename}.mp4",  # 실제 파일 경로로 대체하세요
            f"{files_root}/{data_ids[0]}/1/render/0/{video_filename}.mp4",
        ]
        output_file = f"{files_root}/{data_ids[0]}/{video_filename}.mp4"
        concatenate_mp4_files(input_files, output_file)
    '''
    
    video_clips = []
    
    for file_path in file_paths:
        try:
            # 파일 경로가 존재하는지 확인
            if not os.path.exists(file_path):
                print(f"경고: 파일이 존재하지 않습니다. - {file_path}")
                continue
                
            # 비디오 파일 로드
            clip = VideoFileClip(file_path)
            video_clips.append(clip)
            print(f"파일 추가됨: {file_path} (길이: {clip.duration:.2f}초)")
            
        except Exception as e:
            print(f"오류 발생 ({file_path}): {e}")
            
    # 유효한 비디오 클립이 있는지 확인
    if not video_clips:
        print("연결할 유효한 비디오 클립이 없습니다.")
        return
        
    # 모든 클립을 연결
    # method="compose"는 일반적으로 비디오 트랙과 오디오 트랙을 모두 병합하는 데 사용됩니다.
    final_clip = concatenate_videoclips(video_clips, method=method)
    
    # 연결된 비디오를 MP4 형식으로 내보내기
    # "libx264" 코덱을 사용하며, 이는 MP4 표준입니다.
    # fps (프레임 속도)는 첫 번째 클립의 fps를 따릅니다.
    print("\n비디오 연결 및 내보내기 시작...")
    
    final_clip.write_videofile(
        output_path,
        codec="libx264", 
        audio_codec="aac",  # MP4 오디오 표준 코덱
        temp_audiofile='temp-audio.m4a', # 임시 오디오 파일 경로
        remove_temp=True, 
        fps=final_clip.fps,
        verbose=False, 
        logger=None # 콘솔 출력을 깔끔하게 하기 위해 logger를 비활성화합니다.
    )
    
    # 메모리 해제
    for clip in video_clips:
        clip.close()
    final_clip.close()
    
    print(f"\n성공적으로 연결되어 저장되었습니다: **{output_path}**")





def get_vertices(inference_result_dir, objs_dir, device = None, max_points = 10000):

    
    with open(f'{inference_result_dir}/0/mesh_file_path.txt') as f:
        _mesh_file_dir = f.read()
    #mesh_file_dir = objs_dir+"/"+_mesh_file_dir
    mesh_file_dir = _mesh_file_dir


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
            #vertices = torch.tensor(combined_vertices, dtype=torch.float32, device=device)
            vertices = trimesh.PointCloud(vertices=combined_vertices).vertices
            vertices = torch.from_numpy(vertices).float().to(device)
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
'''
def sum_arrays(total, zero_value ):

    summed_array = copy.deepcopy(total[0])
    if len(total) ==  1:
        return summed_array
    #print('total',len(total))
    for arr in total[1:]:
        #print((summed_array > 0 ).sum())
        zero_mask = np.ones(total[0].shape, dtype=bool)
        zero_mask &= (summed_array == zero_value)
        #print((zero_mask == 1 ).sum())

        summed_array = np.where(zero_mask, arr, summed_array)

    return summed_array
'''

def sum_arrays(total, zero_value, device='cuda'):
    """
    Args:
        total: numpy array 리스트 혹은 tensor 리스트
        zero_value: 빈 값으로 간주할 값 (예: 0)
        device: 'cuda' (GPU) 또는 'cpu'
    """
    if not total:
        return None

    # 첫 번째 배열을 GPU 텐서로 변환 및 복사
    # (이미 텐서라면 clone, numpy라면 변환됨)
    summed_array = torch.as_tensor(total[0], device=device).clone()

    if len(total) == 1:
        return summed_array

    # 루프 수행
    for arr in total[1:]:
        # 현재 배열을 GPU로 이동
        curr_arr = torch.as_tensor(arr, device=device)
        
        # 최적화: np.ones를 만들 필요 없이 바로 조건 비교
        # 현재 누적된 결과가 zero_value인 곳(빈 곳)을 찾음
        is_empty_mask = (summed_array == zero_value)
        
        # 빈 곳(is_empty_mask가 True)이면 curr_arr 값을 넣고, 아니면 기존 summed_array 유지
        summed_array = torch.where(is_empty_mask, curr_arr, summed_array)

    return summed_array

# 사용 예시
# result = sum_arrays_gpu(image_list, 0)
# result_numpy = result.cpu().numpy() # 다시 numpy로 필요할 경우

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
                new_pc.append(_part + centroid)
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



    init_trans_reverse = Translate(torch.FloatTensor([-init_pose[:3]]), dtype=torch.float32, device=device)
    init_trans = Translate(torch.FloatTensor([init_pose[:3]]), dtype=torch.float32, device=device)
    init_rotate = Rotate(slice_util.quaternion_to_matrix(torch.FloatTensor(init_pose[3:])), dtype=torch.float32, device=device)
    init_rotate_reverse = Rotate(slice_util.quaternion_to_matrix(torch.FloatTensor(slice_util.invert_rotation_quaternion(init_pose[3:]))), dtype=torch.float32, device=device)
    translated_points = Transform3d(device=device).compose(init_rotate).transform_points(translated_points)
    translated_points = Transform3d(device=device).compose(init_trans_reverse).transform_points(translated_points)


    transformation_elem = []
    total_temp_t = None
    if init_pose is not None and gt is None and trans_rotate is None:
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

    elif init_pose is not None and gt is not None and (step == -1 or trans_rotate is None):

        gt_trans_reverse = Translate(torch.FloatTensor([-gt[part_index,:3]]), dtype=torch.float32, device=device)
        gt_rotate_reverse = Rotate(slice_util.quaternion_to_matrix(torch.FloatTensor(slice_util.invert_rotation_quaternion(gt[part_index,3:]))), dtype=torch.float32, device=device)
        gt_rotate = Rotate(slice_util.quaternion_to_matrix(torch.FloatTensor(gt[part_index,3:])), dtype=torch.float32, device=device)
        gt_trans = Translate(torch.FloatTensor([gt[part_index,:3]]), dtype=torch.float32, device=device)



        translated_points = Transform3d(device=device).compose(gt_trans_reverse).transform_points(translated_points)
        translated_points = Transform3d(device=device).compose(gt_rotate_reverse).transform_points(translated_points)

        #translated_points = Transform3d(device=device).compose(init_trans).transform_points(translated_points)
        #translated_points = Transform3d(device=device).compose(init_rotate_reverse).transform_points(translated_points)


    elif init_pose is not None and gt is not None and trans_rotate is not None and step != -1:
        
        gt_trans_reverse = Translate(torch.FloatTensor([-gt[part_index,:3]]), dtype=torch.float32, device=device)
        gt_rotate_reverse = Rotate(slice_util.quaternion_to_matrix(torch.FloatTensor(slice_util.invert_rotation_quaternion(gt[part_index,3:]))), dtype=torch.float32, device=device)
        
        pred_trans = Translate(torch.FloatTensor([trans_rotate[step, part_index,:3]]), dtype=torch.float32, device=device)
        pred_rotate = Rotate(slice_util.quaternion_to_matrix(torch.FloatTensor(trans_rotate[step, part_index,3:])), dtype=torch.float32, device=device)


        translated_points = Transform3d(device=device).compose(gt_trans_reverse).transform_points(translated_points)
        translated_points = Transform3d(device=device).compose(gt_rotate_reverse).transform_points(translated_points)
        

        translated_points = Transform3d(device=device).compose(pred_rotate).transform_points(translated_points)
        translated_points = Transform3d(device=device).compose(pred_trans).transform_points(translated_points)


        translated_points = Transform3d(device=device).compose(init_trans).transform_points(translated_points)
        translated_points = Transform3d(device=device).compose(init_rotate_reverse).transform_points(translated_points)

  
    return translated_points, total_temp_t
 
import torch

def translate_image_no_interpolation(image_tensor: torch.Tensor, tx: int, ty: int) -> torch.Tensor:
    """
    (H, W) 또는 (C, H, W) 형태의 이미지를 정수 픽셀만큼 평행 이동합니다 (Nearest Neighbor 방식).

    Args:
        image_tensor (torch.Tensor): 평행 이동할 이미지 텐서. (H, W) 또는 (C, H, W) 형태.
        tx (int): X축(가로, 너비) 이동량. 양수: 오른쪽, 음수: 왼쪽.
        ty (int): Y축(세로, 높이) 이동량. 양수: 아래, 음수: 위.

    Returns:
        torch.Tensor: 평행 이동된 이미지 텐서.
    """
    
    # 1. 텐서 차원 확인 및 H, W 인덱스 설정
    # Matplotlib이나 NumPy에서 이미지의 차원은 보통 (H, W) 또는 (C, H, W) 입니다.
    if image_tensor.ndim == 2:
        # (H, W) 형태의 2D 이미지
        # 롤링할 축 (Y축: 0, X축: 1)
        dims_to_roll = [0, 1] 
    elif image_tensor.ndim == 3:
        # (C, H, W) 형태의 3D 이미지
        # 롤링할 축 (Y축: 1, X축: 2)
        dims_to_roll = [1, 2]
    else:
        raise ValueError("이미지 텐서의 차원은 2차원 (H, W) 또는 3차원 (C, H, W)이어야 합니다.")

    # 2. 이동량 설정
    # ty는 H축(0/1)의 이동량, tx는 W축(1/2)의 이동량입니다.
    # ty를 양수로 주면 아래로 이동(인덱스 증가), tx를 양수로 주면 오른쪽으로 이동(인덱스 증가).
    shifts = (ty, tx)
    #print(shifts)
    # 3. torch.roll을 사용하여 평행 이동 (경계는 자동으로 순환/순환 이동 처리됨)
    # roll은 경계를 벗어난 픽셀을 반대편으로 순환시킵니다.
    rolled_tensor = torch.roll(image_tensor, shifts=shifts, dims=dims_to_roll)
    
    # 4. 빈 공간 0으로 채우기 (가장 중요한 부분)
    
    # torch.roll의 순환된 부분을 0으로 덮어쓰기 위해 마스크를 생성합니다.
    # 마스크의 기본값은 0(True)이며, 픽셀이 이동된 영역만 1(False)로 설정합니다.
    mask = torch.ones_like(image_tensor, dtype=torch.bool)
    
    # Y축(세로) 마스킹
    if ty > 0: # 아래로 이동 (상단 ty 픽셀이 빈 공간)
        mask[:ty, ...] = False
    elif ty < 0: # 위로 이동 (하단 abs(ty) 픽셀이 빈 공간)
        mask[ty:, ...] = False
        
    # X축(가로) 마스킹
    if tx > 0: # 오른쪽으로 이동 (좌측 tx 픽셀이 빈 공간)
        # H, W 텐서의 경우: mask[:, :tx] = False
        # C, H, W 텐서의 경우: mask[..., :tx] = False
        if image_tensor.ndim == 2:
            mask[:, :tx] = False
        else: # ndim == 3
            mask[..., :tx] = False
            
    elif tx < 0: # 왼쪽으로 이동 (우측 abs(tx) 픽셀이 빈 공간)
        if image_tensor.ndim == 2:
            mask[:, tx:] = False
        else: # ndim == 3
            mask[..., tx:] = False

    # 5. 마스크를 반전하여 이동된 픽셀(False)만 남기고, 0(True)인 픽셀에 0을 할당
    # 마스크가 False인 위치만 1로 채워진 '유효 영역 마스크'를 만듭니다.
    # 이 과정에서 roll의 순환된 부분이 0으로 덮어쓰여집니다.
    
    # H, W 텐서의 경우:
    # 1. 롤링된 텐서를 0으로 초기화된 텐서에 복사합니다.
    # 2. 유효 영역 마스크를 만듭니다.
    
    # 전체 텐서를 0으로 초기화
    translated_tensor = torch.zeros_like(image_tensor)
    
    # 유효 영역 마스크가 True인 부분(이동된 픽셀이 있는 곳)만 값을 복사
    # 마스크를 논리적으로 반전하여 순환되지 않은 유효 픽셀만 선택해야 합니다.
    # 그러나 위에서 만든 마스크는 '빈 공간'을 False로 표시했습니다. 
    # 따라서, 순환된 부분을 0으로 만들고, 나머지 부분은 rolled_tensor의 값을 유지해야 합니다.
    
    # torch.roll을 사용할 때의 가장 깔끔한 빈 공간 채우기 로직은 다음과 같습니다:
    
    # 새로운 0 텐서를 만들고,
    result = torch.zeros_like(image_tensor)
    
    # roll된 텐서에서 유효한 부분만 마스크를 통해 잘라서 result에 복사합니다.
    # 마스크 로직을 '유효한 영역'을 True로 표시하도록 다시 설계하는 것이 더 직관적입니다.
    
    valid_mask = torch.ones_like(image_tensor, dtype=torch.bool)
    
    # ty 처리
    if ty > 0: # 아래로 이동 (위 ty행 버려짐)
        valid_mask[:ty, ...] = False
    elif ty < 0: # 위로 이동 (아래 abs(ty)행 버려짐)
        valid_mask[ty:, ...] = False
        
    # tx 처리
    if tx > 0: # 오른쪽으로 이동 (왼쪽 tx열 버려짐)
        if image_tensor.ndim == 2:
            valid_mask[:, :tx] = False
        else: 
            valid_mask[..., :tx] = False
            
    elif tx < 0: # 왼쪽으로 이동 (오른쪽 abs(tx)열 버려짐)
        if image_tensor.ndim == 2:
            valid_mask[:, tx:] = False
        else:
            valid_mask[..., tx:] = False

    # rolled_tensor에서 valid_mask가 True인 곳만 result에 복사
    result[valid_mask] = rolled_tensor[valid_mask]
    
    return result
def translate_image_pytorch_no_interpolation(image_tensor: torch.Tensor, tx: float, ty: float) -> torch.Tensor:
    """
    PyTorch 텐서를 NumPy로 변환하여 보간 없이 이동시킨 후, 다시 텐서로 복원합니다.
    """
    
    # 1. PyTorch 텐서를 CPU로 이동하고 NumPy 배열로 변환
    if image_tensor.ndim > 2:
        # (1, H, W) 형태일 경우, (H, W)로 변환
        img_np = image_tensor.squeeze()
    else:
        # (H, W) 형태
        img_np = image_tensor
    tx = int(tx)
    ty = int(ty)
        
    # 2. NumPy 슬라이싱 함수 적용
    translated_np = translate_image_no_interpolation(img_np, tx, ty)
    
    # 3. 다시 PyTorch 텐서로 변환
    translated_tensor = translated_np
    
    # 원래 차원으로 복원 (예: (H, W) -> (1, H, W) 또는 (H, W))
    if image_tensor.ndim > 2:
        translated_tensor = translated_tensor.unsqueeze(0) 
        
    return translated_tensor

def quaternion_to_matrix_cupy(q: cp.ndarray) -> cp.ndarray:
    """
    쿼터니언 [w, x, y, z]를 3x3 회전 행렬로 변환합니다 (CuPy).
    """
    w, x, y, z = q[0], q[1], q[2], q[3]
    
    # 3x3 회전 행렬 공식
    R = cp.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [2*x*y + 2*z*w,     1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x**2 - 2*y**2]
    ], dtype=cp.float32)
    
    return R



import torch
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn.functional as F

def get_center_of_mass(image_tensor):
    """
    이미지 픽셀의 밝기를 질량으로 간주하여 무게 중심(Center of Mass)을 계산합니다.
    
    Args:
        image_tensor: (1, H, W) 또는 (H, W) 형태의 Tensor
        
    Returns:
        center_pos: (2,) 형태의 Tensor [center_x, center_y] (픽셀 좌표계)
    """
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.squeeze(0)
        
    H, W = image_tensor.shape
    device = image_tensor.device
    
    # 좌표 그리드 생성 (미분 끊김 방지를 위해 item() 사용 금지)
    y_coords, x_coords = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=device),
        torch.arange(W, dtype=torch.float32, device=device),
        indexing='ij'
    )
    
    # 밝기(질량) 총합 계산 (0으로 나누기 방지 epsilon 추가)
    total_mass = torch.sum(image_tensor) + 1e-8
    
    # 무게 중심 계산: Sum(좌표 * 질량) / 총질량
    center_y = torch.sum(y_coords * image_tensor) / total_mass
    center_x = torch.sum(x_coords * image_tensor) / total_mass
    
    return torch.stack([center_x, center_y])

def rotate_with_center_and_quaternion(image, quaternion):
    """
    (H, W) 이미지를 XZ 평면으로 가정, 객체의 무게 중심을 구한 후 
    쿼터니언의 Y축 회전 성분만을 적용합니다.
    
    Args:
        image: (H, W), (1, H, W), 또는 (B, 1, H, W) Tensor
        quaternion: (w, x, y, z) 형태의 Tensor 또는 리스트
    
    Returns:
        rotated_image: 회전된 이미지 Tensor (입력과 동일한 차원 유지)
        center_pos: 계산된 무게 중심 Tensor (픽셀 좌표)
    """
    # 1. 입력 차원 표준화 (B, C, H, W)
    original_dim = image.dim()
    if original_dim == 2:   # (H, W)
        img_batch = image.unsqueeze(0).unsqueeze(0)
    elif original_dim == 3: # (C, H, W) or (1, H, W)
        img_batch = image.unsqueeze(0)
    else:
        img_batch = image

    B, C, H, W = img_batch.shape
    device = img_batch.device

    # 쿼터니언 텐서 변환
    if not isinstance(quaternion, torch.Tensor):
        quaternion = torch.tensor(quaternion, dtype=torch.float32, device=device)
    else:
        quaternion = quaternion.to(device)

    # 2. 무게 중심(Center of Mass) 계산 (Grayscale 기준)
    # 배치 처리가 필요하다면 반복문 혹은 배치 연산으로 변경 가능. 여기선 단일 이미지/첫 배치 기준
    if C > 1:
        gray_img = img_batch[0].mean(dim=0)
    else:
        gray_img = img_batch[0, 0]
        
    cx_px, cy_px = get_center_of_mass(gray_img) # Returns Tensor(scalar)
    center_tensor = torch.stack([cx_px, cy_px])

    # 3. 좌표 정규화 (-1 ~ 1 범위로 변환)
    # grid_sample은 중심이 (0,0), 좌상단(-1,-1), 우하단(1,1)
    cx_norm = (cx_px / (W - 1)) * 2 - 1
    cy_norm = (cy_px / (H - 1)) * 2 - 1

    # 4. 쿼터니언에서 Y축 회전 성분 추출
    # q = (w, x, y, z). Y축 회전은 w(실수부)와 y(허수부 j)에 의존
    w, x, y, z = quaternion[0], quaternion[1], quaternion[2], quaternion[3]
    
    # Project to Y-axis rotation (ignore x, z) and Normalize
    norm = torch.sqrt(w*w + y*y) + 1e-8
    w_y = w / norm
    y_y = y / norm
    
    # Quaternion to Rotation Angle (Theta)
    # cos(theta) = w^2 - y^2, sin(theta) = 2wy
    cos_theta = w_y**2 - y_y**2
    sin_theta = 2 * w_y * y_y

    # 5. Affine Matrix 구성: Translate(Center->Origin) -> Rotate -> Translate(Origin->Center)
    # PyTorch affine_grid는 역변환(Inverse Mapping)을 수행하므로, 회전은 -Theta를 적용해야 함
    # cos(-t) = cos(t), sin(-t) = -sin(t)
    
    # Rotation Matrix R
    rot_mat = torch.zeros((3, 3), device=device)
    rot_mat[0, 0] = cos_theta
    rot_mat[0, 1] = sin_theta
    rot_mat[1, 0] = -sin_theta
    rot_mat[1, 1] = cos_theta
    rot_mat[2, 2] = 1.0

    # Translation Matrix T (To Origin)
    t_origin = torch.eye(3, device=device)
    t_origin[0, 2] = -cx_norm
    t_origin[1, 2] = -cy_norm

    # Translation Matrix T_inv (Back to Position)
    t_back = torch.eye(3, device=device)
    t_back[0, 2] = cx_norm
    t_back[1, 2] = cy_norm

    # Final Matrix M = T_back @ R @ T_origin
    final_mat = t_back @ rot_mat @ t_origin
    
    # affine_grid용 2x3 행렬 추출 (Batch 차원 추가)
    affine_matrix = final_mat[:2, :].unsqueeze(0).repeat(B, 1, 1) # (B, 2, 3)

    # 6. 이미지 회전 적용
    # align_corners=True: 픽셀의 중심을 좌표로 간주 (회전 시 더 정확)
    grid = F.affine_grid(affine_matrix, img_batch.size(), align_corners=True)
    rotated_batch = F.grid_sample(img_batch, grid, align_corners=True)

    # 차원 복구
    if original_dim == 2:
        rotated_image = rotated_batch.squeeze(0).squeeze(0)
    elif original_dim == 3:
        rotated_image = rotated_batch.squeeze(0)
    else:
        rotated_image = rotated_batch

    return rotated_image, center_tensor


import cupy as cu_np
from cupyx.scipy.ndimage import prewitt, map_coordinates
# Note: CuPy's map_coordinates is less feature-rich than SciPy's,
# but it should work for standard cases like this.

def warp_image(image, p):
    """
    Warps an image according to the motion vector p = [h, v, theta] using CuPy.
    """
    # Ensure p is on the device if it came from the host
    p = cu_np.asarray(p)
    h, v, theta = p
    
    # Use CuPy's np.indices, which returns CuPy arrays (on GPU)
    n2_out, n1_out = cu_np.indices(image.shape)

    cos_t = cu_np.cos(theta)
    sin_t = cu_np.sin(theta)

    # Inverse of Rotate-then-Translate: p_in = R(-t) * (p_out - T)
    n1_rot = n1_out - h
    n2_rot = n2_out - v

    # All arithmetic operations are now performed on the GPU
    n1_in = n1_rot * cos_t + n2_rot * sin_t
    n2_in = -n1_rot * sin_t + n2_rot * cos_t

    coords = cu_np.array([n2_in, n1_in])
    # Use CuPy's map_coordinates
    warped_image = map_coordinates(image, coords, order=3, cval=0.0)

    return warped_image


def register_iterative(y1, yk, iterations=10, min_update_norm=1e-6, step_size=1.0):
    """
    Iteratively estimates motion using a stable "Lucas-Kanade" style approach on GPU (CuPy).
    """

    # --- Initialization on GPU ---
    # Ensure input images and the parameter vector start on the GPU
    if not isinstance(y1, cu_np.ndarray):
        y1 = cu_np.asarray(y1)
    if not isinstance(yk, cu_np.ndarray):
        yk = cu_np.asarray(yk)
        
    p_total = cu_np.zeros(3)  # [h, v, theta] - now a CuPy array on GPU

    # --- 1. Calculate Gradients and A Matrix (Hessian) ONCE on GPU ---
    print("Calculating constant A matrix from reference image y1...")
    # Use CuPy's prewitt
    # gy1 and gx1 are now CuPy arrays
    gy1 = prewitt(y1, axis=0)  # Gradient in y-direction (rows, n2)
    gx1 = prewitt(y1, axis=1)  # Gradient in x-direction (cols, n1)
    n2, n1 = cu_np.indices(y1.shape) # CuPy arrays

    # This correctly matches the paper's Eq. 27
    g_bar1 = (n1 * gy1) - (n2 * gx1)

    # Flatten gradients for summation (CuPy's ravel/sum)
    gx1_f = gx1.ravel()
    gy1_f = gy1.ravel()
    g_bar1_f = g_bar1.ravel()

    # Construct the constant 3x3 Matrix A on GPU
    # All element-wise products and sums are performed on the GPU
    A11 = cu_np.sum(gx1_f * gx1_f)
    A12 = cu_np.sum(gx1_f * gy1_f)
    A13 = cu_np.sum(gx1_f * g_bar1_f)
    A22 = cu_np.sum(gy1_f * gy1_f)
    A23 = cu_np.sum(gy1_f * g_bar1_f)
    A33 = cu_np.sum(g_bar1_f * g_bar1_f)

    A = (cu_np.array([
        [A11, A12, A13],
        [A12, A22, A23],
        [A13, A23, A33]
    ]))

    print(f"Starting iterative registration (step_size = {step_size})...")
    print(f" iter | {'h (pix)':<10} | {'v (pix)':<10} | {'theta (rad)':<10} | {'Update Norm':<12}")
    print("-" * 58)

    for i in range(iterations):
        yk_warped = warp_image(yk, p_total)

        # 1. The error vector is (current - target)
        diff_image = yk_warped - y1
        diff_image_f = diff_image.ravel()

        # b = G^T * e (Calculations on GPU)
        b1 = cu_np.sum(diff_image_f * gx1_f)
        b2 = cu_np.sum(diff_image_f * gy1_f)
        b3 = cu_np.sum(diff_image_f * g_bar1_f)
        b = cu_np.array([b1, b2, b3]) # CuPy array

        try:
            # p_update = A_inv * b (CuPy's linalg.solve runs on GPU)
            p_update = cu_np.linalg.solve(A, b)
        except cu_np.linalg.LinAlgError:
            # Convert to Python list/scalar for printing if needed, or just print the CuPy error
            print(f"Error: Matrix A is singular at iteration {i}.")
            p_update = cu_np.zeros(3)

        # 2. Update p_total (on GPU)
        p_total = p_total + (step_size * p_update)

        # np.linalg.norm runs on GPU
        update_norm = cu_np.linalg.norm(p_update)
        
        # We fetch only the necessary values from the GPU to the host for printing
        h, v, theta = p_total.get()
        print(f" {i:<4} | {h:<10.4f} | {v:<10.4f} | {theta:<10.6f} | {update_norm.get():<12.8f}")

        if update_norm < min_update_norm:
            print(f"\nConvergence reached in {i + 1} iterations.")
            break

    if i == iterations - 1:
        print(f"\nMaximum iterations reached ({iterations}).")

    # Return the final result as a standard NumPy array (by fetching from GPU)
    return p_total.get()




def get_prewitt_gradients(img_tensor):
    """
    Computes Prewitt gradients using conv2d to match scipy.ndimage.prewitt behavior.
    Input: (H, W) tensor
    Output: gy (H, W), gx (H, W)
    """
    # Reshape to (1, 1, H, W) for conv2d
    img = img_tensor.unsqueeze(0).unsqueeze(0)
    
    # Define Prewitt kernels
    # Gy: Derivative in row (vertical), smoothing in col
    k_y = torch.tensor([[-1.0, -1.0, -1.0],
                        [ 0.0,  0.0,  0.0],
                        [ 1.0,  1.0,  1.0]], device=img.device).view(1, 1, 3, 3)
    
    # Gx: Derivative in col (horizontal), smoothing in row
    k_x = torch.tensor([[-1.0, 0.0, 1.0],
                        [-1.0, 0.0, 1.0],
                        [-1.0, 0.0, 1.0]], device=img.device).view(1, 1, 3, 3)
    
    # Apply convolution (padding=1 to keep same size)
    # Note: scipy.ndimage.prewitt usually correlates. conv2d correlates if kernel is not flipped.
    gy = F.conv2d(img, k_y, padding=1).squeeze()
    gx = F.conv2d(img, k_x, padding=1).squeeze()
    
    return gy, gx

def warp_image_torch(img_tensor, params):
    """
    Warps image using affine transform based on params [h, v, theta].
    img_tensor: (H, W)
    params: [h (pixels), v (pixels), theta (radians)]
    """
    H, W = img_tensor.shape
    h_shift, v_shift, theta = params
    
    # PyTorch grid_sample uses normalized coordinates [-1, 1].
    # We need to construct an affine matrix for the inverse transform (grid generation).
    
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    
    # Normalization factors for translation
    tx = 2.0 * h_shift / W
    ty = 2.0 * v_shift / H
    
    # Affine Matrix (2x3)
    # Note: To shift the image by (h, v), we sample from (x-h, y-v).
    # The matrix below creates the sampling grid.
    theta_mat = torch.tensor([[cos_t, -sin_t, -tx],
                              [sin_t,  cos_t, -ty]], device=img_tensor.device)
    
    # Create grid
    # affine_grid expects batch of 2x3 matrices: (N, 2, 3)
    grid = F.affine_grid(theta_mat.unsqueeze(0), [1, 1, H, W], align_corners=False)
    
    # Sample
    img_input = img_tensor.view(1, 1, H, W)
    warped = F.grid_sample(img_input, grid, align_corners=False, padding_mode='zeros')
    
    return warped.squeeze()

def register_iterative_torch(y1, yk, iterations=10, min_update_norm=1e-6, step_size=1.0, DEBUG = False):
    """
    Iteratively estimates motion using a stable "Lucas-Kanade" style approach on GPU (PyTorch).
    """
    # Select Device (GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Initialization on GPU ---
    # Ensure input images are torch tensors on the correct device
    if not isinstance(y1, torch.Tensor):
        y1 = torch.tensor(y1, dtype=torch.float32, device=device)
    else:
        y1 = y1.to(device=device, dtype=torch.float32)
        
    if not isinstance(yk, torch.Tensor):
        yk = torch.tensor(yk, dtype=torch.float32, device=device)
    else:
        yk = yk.to(device=device, dtype=torch.float32)

    p_total = torch.zeros(3, device=device)  # [h, v, theta]
    # --- 1. Calculate Gradients and A Matrix (Hessian) ONCE on GPU ---
    if DEBUG: print("Calculating constant A matrix from reference image y1...")
    
    # Use custom Prewitt implementation
    gy1, gx1 = get_prewitt_gradients(y1)
    
    # Create indices grid
    # torch.meshgrid with indexing='ij' matches numpy.indices behavior (row, col)
    n2, n1 = torch.meshgrid(torch.arange(y1.shape[0], device=device), 
                            torch.arange(y1.shape[1], device=device), 
                            indexing='ij')

    # This correctly matches the paper's Eq. 27: (x * Gy) - (y * Gx)
    # Note: n1 is x (cols), n2 is y (rows)
    g_bar1 = (n1 * gy1) - (n2 * gx1)

    # Flatten gradients for summation
    gx1_f = gx1.reshape(-1)
    gy1_f = gy1.reshape(-1)
    g_bar1_f = g_bar1.reshape(-1)

    # Construct the constant 3x3 Matrix A on GPU
    A11 = torch.sum(gx1_f * gx1_f)
    A12 = torch.sum(gx1_f * gy1_f)
    A13 = torch.sum(gx1_f * g_bar1_f)
    A22 = torch.sum(gy1_f * gy1_f)
    A23 = torch.sum(gy1_f * g_bar1_f)
    A33 = torch.sum(g_bar1_f * g_bar1_f)

    A = torch.tensor([
        [A11, A12, A13],
        [A12, A22, A23],
        [A13, A23, A33]
    ], device=device)

    if DEBUG: print(f"Starting iterative registration (step_size = {step_size})...")
    if DEBUG: print(f" iter | {'h (pix)':<10} | {'v (pix)':<10} | {'theta (rad)':<10} | {'Update Norm':<12}")
    if DEBUG: print("-" * 58)

    for i in range(iterations):
        # We need to warp yk using current p_total
        yk_warped = warp_image_torch(yk, p_total)

        # 1. The error vector is (current - target)
        diff_image = yk_warped - y1
        diff_image_f = diff_image.reshape(-1)

        # b = G^T * e (Calculations on GPU)
        b1 = torch.sum(diff_image_f * gx1_f)
        b2 = torch.sum(diff_image_f * gy1_f)
        b3 = torch.sum(diff_image_f * g_bar1_f)
        b = torch.tensor([b1, b2, b3], device=device)

        try:
            # p_update = A_inv * b
            p_update = torch.linalg.solve(A, b)
        except RuntimeError as e: # PyTorch raises RuntimeError for singular matrices
            print(f"Error: Matrix A is singular at iteration {i}. {e}")
            p_update = torch.zeros(3, device=device)

        # 2. Update p_total (on GPU)
        p_total = p_total + (step_size * p_update)

        update_norm = torch.linalg.norm(p_update)
        
        # Fetch values to CPU for printing
        h, v, theta = p_total.cpu().numpy()
        norm_val = update_norm.item()
        
        if DEBUG: print(f" {i:<4} | {h:<10.4f} | {v:<10.4f} | {theta:<10.6f} | {norm_val:<12.8f}")

        if norm_val < min_update_norm:
            if DEBUG: print(f"\nConvergence reached in {i + 1} iterations.")
            break

    if i == iterations - 1:
        if DEBUG: print(f"\nMaximum iterations reached ({iterations}).")

    # Return as numpy array (move to CPU)
    return p_total
# --- 사용 예시 ---
# if __name__ == "__main__":
#     # 테스트용 더미 데이터 생성
#     img1 = np.random.rand(256, 256).astype(np.float32)
#     img2 = np.roll(img1, 2, axis=1) # x축으로 2픽셀 이동시킨 이미지
#     res = register_iterative_torch(img1, img2)
#     print("Final Parameters:", res)

def transform_2d_tiff(device, scale_factor, file_path, _init_pose, _gt, _trans_rotate, step, part_index):
    # ----------------- 전제: 변환할 이미지 텐서 준비 -----------------
    # device 설정 및 이미지 텐서 준비
    # 'vertices'는 이제 변환할 2D 이미지 텐서(예: (C, H, W) 형태)
    # PIL을 사용하여 TIFF 파일 로드

    image_top = cv2.imread(file_path, cv2.IMREAD_COLOR)    
    image_array = cv2.cvtColor(image_top, cv2.COLOR_BGR2GRAY)
    #vertices = torch.from_numpy(np.array(vertices, dtype=np.float32)).to(device)

    H_orig, W_orig = image_array.shape
    
    # 1. 축소 (Downsampling)
    # cv2.INTER_AREA는 이미지 축소 시 가장 좋은 품질을 제공합니다.
    interpolation_mode = cv2.INTER_AREA
    #print(file_path,scale_factor)
    # 새 크기 계산
    W_new = int(W_orig * scale_factor)
    H_new = int(H_orig * scale_factor)
    
    resized_image = cv2.resize(
        src=image_array,
        dsize=(W_new, H_new),  # (Width, Height) 순서
        interpolation=interpolation_mode
    ).astype(np.float32) # float32로 통일
    
    # 2. 패딩을 위한 오프셋 계산 (중앙 배치)
    
    # 남은 공간 (여백)
    pad_h = H_orig - H_new
    pad_w = W_orig - W_new
    
    # 패딩 시작 위치 (중앙에 오도록)
    start_y = pad_h // 2
    start_x = pad_w // 2
    
    # 3. 원래 크기의 빈 공간(캔버스) 생성 및 중앙 복사
    # 0으로 초기화된 원본 크기의 배열 (0 패딩 효과)
    padded_array = np.zeros((H_orig, W_orig), dtype=np.float32)
    
    # 축소된 이미지를 중앙 위치에 복사
    padded_array[start_y : start_y + H_new, 
                 start_x : start_x + W_new] = resized_image
    '''
    # 패딩 시작 위치 (중앙에 오도록)
    start_y = pad_h // 2
    start_x = pad_w // 2
    
    # 3. 원래 크기의 빈 공간(캔버스) 생성 및 중앙 복사
    # 0으로 초기화된 원본 크기의 배열 (0 패딩 효과)
    padded_array = np.zeros((H_orig, W_orig), dtype=np.float32)
    
    # 축소된 이미지를 중앙 위치에 복사
    padded_array[start_y : start_y + H_new, 
                 start_x : start_x + W_new] = resized_image
    '''
    init_pose = _init_pose.copy()
    
    #print(init_pose)
    if not isinstance(padded_array, torch.Tensor):
        # TIFF 파일을 읽고 텐서로 변환하는 로직이 이전에 수행되었다고 가정합니다.
        image_tensor = torch.from_numpy(padded_array).to(device).float()
#        if image_tensor.ndim < 3:
#            image_tensor = image_tensor.unsqueeze(0) # (H, W) -> (1, H, W)
    else:
        image_tensor = padded_array

    init_pose[4] = 0
    init_pose[6] = 0
    #print('init_pose_r',H_orig,W_orig,init_pose)
    #print(W_new, int(W_new*1.5))

    if _init_pose is not None and _gt is None and _trans_rotate is None:
        translated = image_tensor
        #translated = rotate_image_y_axis_extended_gpu(image_tensor,init_pose[3:], W_new_rot, H_new_rot)
        #translated = translate_image_pytorch_no_interpolation(translated,-init_pose[0]*H_orig/2,-init_pose[2]*W_orig/2)
    elif _init_pose is not None and _gt is not None and step == -1:
        gt = _gt.copy()
        #translated = rotate_image_y_axis_extended_gpu(image_tensor,init_pose[3:],  W_new_rot, H_new_rot)
        #translated = translate_image_pytorch_no_interpolation(translated,-init_pose[0]*H_orig/2,-init_pose[2]*W_orig/2)
        translated = image_tensor
        gt = gt[part_index,:]
        translated = translate_image_pytorch_no_interpolation(translated,-gt[0]*H_orig/2,-gt[2]*W_orig/2)
        gt[4] = 0
        gt[5] = -gt[5]
        gt[6] = 0
        translated, _ = rotate_with_center_and_quaternion(translated,gt[3:])
    elif _init_pose is not None and _gt is not None and _trans_rotate is not None:
        gt = _gt.copy()
        translated = image_tensor
        gt = gt[part_index,:]
        translated = translate_image_pytorch_no_interpolation(translated,-gt[0]*H_orig/2,-gt[2]*W_orig/2)
        gt[4] = 0
        gt[5] = -gt[5]
        gt[6] = 0
        translated, _ = rotate_with_center_and_quaternion(translated,gt[3:])

        #print(part_index, _trans_rotate[step, part_index,3:])
        translated, _ = rotate_with_center_and_quaternion(translated,_trans_rotate[step, part_index,3:])
        translated = translate_image_pytorch_no_interpolation(translated,_trans_rotate[step, part_index,0]*H_orig/2,
                                                              _trans_rotate[step, part_index,2]*W_orig/2)


    return image_tensor, translated
   
   

def get_renderer(device, xlim=(-2, 2),ylim=(-2, 2),zlim=(-2, 2)):
    image_size = 512
    radius = 0.002
    points_per_pixel = 50
    _background_color = (1,1,1)
    R, T = look_at_view_transform(10, 45, 45) #10,30,90
    cameras = FoVOrthographicCameras( device=device, R=R, T=T, 
                                     znear=zlim[0], zfar=zlim[1], 
                                     min_x=xlim[0], max_x=xlim[1], 
                                     min_y=ylim[0], max_y=ylim[1])

    R, T = look_at_view_transform(10, 0, 0)
    cameras_front = FoVOrthographicCameras( device=device, R=R, T=T, 
                                           znear=zlim[0], zfar=zlim[1], 
                                           min_x=xlim[0] - 0.5, max_x=xlim[1] - 0.5, 
                                           min_y=ylim[0] + 0.5, max_y=ylim[1]  + 0.5)
    
    R, T = look_at_view_transform(10, 90, 0)
    cameras_down = FoVOrthographicCameras( device=device, R=R, T=T, 
                                          znear=zlim[0], zfar=zlim[1], 
                                          min_x=xlim[0]+ 0.5, max_x=xlim[1]+ 0.5, 
                                          min_y=ylim[0]+ 0.5, max_y=ylim[1]+ 0.5)

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
        compositor=pytorch3d.renderer.AlphaCompositor(background_color=_background_color)
        #compositor=NormWeightedCompositor()
    )

    renderer_front = pytorch3d.renderer.PointsRenderer(
        rasterizer=rasterizer_front,
        compositor=pytorch3d.renderer.AlphaCompositor(background_color=_background_color)
        #compositor=NormWeightedCompositor()
    )

    renderer_down = pytorch3d.renderer.PointsRenderer(
        rasterizer=rasterizer_down,
        compositor=pytorch3d.renderer.AlphaCompositor(background_color=_background_color)
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

def save_image_tensor_to_png(image_tensor: torch.Tensor, output_path: str):
    image_tensor = image_tensor.cpu().numpy()
    is_uint8 = image_tensor.dtype == np.uint8
    # Prepare the slice for saving as a grayscale image
    if not is_uint8:
        # Normalize to 0-255 if the data is not already uint8
        if np.max(image_tensor) > 0:
            image_tensor = (image_tensor / np.max(image_tensor) * 255).astype(np.uint8)
        else:
            image_tensor = image_tensor.astype(np.uint8)
    plt.imsave(output_path, image_tensor, cmap='gray')

import torch


def calculate_iou(original, pred, threshold=0.5, epsilon=1e-6):
    """
    두 이미지의 전경에 대해 Intersection over Union (IoU)를 계산합니다.
    
    Args:
        original (torch.Tensor): (H, W) Ground Truth
        pred (torch.Tensor): (H, W) Prediction
        threshold (float): 전경/배경 구분 임계값
        epsilon (float): 0으로 나누기 방지를 위한 작은 값
        
    Returns:
        iou (float): 계산된 IoU 값 (0.0 ~ 1.0)
    """
    # 1. 전경 마스크 생성 (True: 전경, False: 배경)
    mask_orig = original >= threshold
    mask_pred = pred >= threshold

    # 2. 교집합 (Intersection) 계산: 둘 다 True인 영역
    # bitwise AND (&) 연산 사용
    intersection = torch.logical_and(mask_orig, mask_pred)
    intersection_area = torch.sum(intersection).item()

    # 3. 합집합 (Union) 계산: 둘 중 하나라도 True인 영역
    # bitwise OR (|) 연산 사용
    union = torch.logical_or(mask_orig, mask_pred)
    union_area = torch.sum(union).item()

    # 4. IoU 계산
    # 합집합이 0일 경우(둘 다 배경만 있는 경우) 1.0 또는 0.0을 반환하도록 처리
    if union_area == 0:
        return 1.0 if intersection_area == 0 else 0.0
        
    iou = intersection_area / (union_area + epsilon)
    
    return iou



def all_slices_into_grid_layout(mask_2d_list,output_fliename, gt_show = True, title = '',
                                rows = 4, cols = 5):

    translated_tiff_gt_list = [item[f'tiff_gt'] for item in mask_2d_list]
    obj_id_list_gt = [item['id'] for item in mask_2d_list]

    sorted_mask_2d_list = sorted(mask_2d_list, key=lambda item: item['pred_y'])

    translated_tiff_final_list = [item[f'tiff_pred'] for item in sorted_mask_2d_list]
    obj_id_list = [item['id'] for item in sorted_mask_2d_list]
    color_list = [item['color'] for item in sorted_mask_2d_list]
    y_list = [item['pred_y'] for item in sorted_mask_2d_list]


    total_cells = rows * cols
    fig, axes = plt.subplots(rows, cols, figsize=(10, 8)) 

    # axes는 (4, 5) 형태의 NumPy 배열입니다. 이를 평탄화하여 사용합니다.
    axes = axes.flatten()
    # Define the colors and their corresponding positions (0 to 1)
    colors = [(0, 'black'), (0.5, 'grey'), (0.7,'red'), (1.0, 'white')] 
    # (position, color_name_or_hex_code)

    # Create the custom colormap
    custom_cmap = LinearSegmentedColormap.from_list("my_custom_cmap", colors)

    iou_list = []
    num_of_part_correct_y = 0
    # 4. 이미지 배치 및 표시
    for i in range(total_cells):
        # i번째 이미지를 axes 배열의 i번째 셀에 표시
        ax = axes[i]
        ax.set_xticks([])
        ax.set_yticks([])
        if i >= len(translated_tiff_final_list):
            continue
        _img = translated_tiff_final_list[i]

        img_pred = _img.clone()
        img_pred[(img_pred < 255) & (img_pred > 0) ] = 0.5
        img_pred[img_pred > 254] = 1.0


        ax.imshow(img_pred.cpu().numpy(), cmap=custom_cmap, vmin=0.0, vmax=1.0) 
        
        if gt_show:
            _img_ori = translated_tiff_gt_list[i].clone()
            _img_ori[(_img_ori < 255) & (_img_ori > 0) ] = 0.7
            _img_ori[_img_ori > 254] = 0.7
            ax.imshow(_img_ori.cpu().numpy(), cmap=custom_cmap, vmin=0.0, vmax=1.0,alpha=0.25) 

            iou = calculate_iou(_img_ori,img_pred)
            iou_list.append(iou)
            ax.text(
                    2,           # x 좌표 (이미지 왼쪽에서 2픽셀 떨어진 곳)
                    100,           # y 좌표 (이미지 위쪽에서 2픽셀 떨어진 곳)
                    f'IoU: {iou:.2f}',# 표시할 텍스트
                    color='black', # 텍스트 색상
                    fontsize=10,    # 텍스트 크기
                    ha='left',     # 가로 정렬 (horizontal alignment): 'left', 'center', 'right'
                    va='top',      # 세로 정렬 (vertical alignment): 'top', 'center', 'bottom'
                    bbox=dict(facecolor='white', edgecolor='none', pad=1.0) # 배경 박스 (선택 사항)
                )
        # 축(Axis) 라벨 제거 (이미지만 깔끔하게 보이도록)

        # ⭐ 구분선 추가: 각 axes 주변에 테두리(spine)를 그림
        for spine in ax.spines.values():
            spine.set_edgecolor('white') # 테두리 색상
            spine.set_linewidth(1)       # 테두리 두께 (픽셀 단위)
            # (선택 사항) 이미지 제목 추가
            #ax.set_title(f'Slice {i+1}')

        ax.text(
                2,           # x 좌표 (이미지 왼쪽에서 2픽셀 떨어진 곳)
                2,           # y 좌표 (이미지 위쪽에서 2픽셀 떨어진 곳)
                f'{obj_id_list[i]}',# 표시할 텍스트
                color='white', # 텍스트 색상
                fontsize=10,    # 텍스트 크기
                ha='left',     # 가로 정렬 (horizontal alignment): 'left', 'center', 'right'
                va='top',      # 세로 정렬 (vertical alignment): 'top', 'center', 'bottom'
                bbox=dict(facecolor=color_list[i], edgecolor='none', pad=1.0) # 배경 박스 (선택 사항)
            )
        if y_list != None:
            ax.text(
                    2,           # x 좌표 (이미지 왼쪽에서 2픽셀 떨어진 곳)
                    50,           # y 좌표 (이미지 위쪽에서 2픽셀 떨어진 곳)
                    f'Y: {y_list[i]:.3f}',# 표시할 텍스트
                    color='black', # 텍스트 색상
                    fontsize=10,    # 텍스트 크기
                    ha='left',     # 가로 정렬 (horizontal alignment): 'left', 'center', 'right'
                    va='top',      # 세로 정렬 (vertical alignment): 'top', 'center', 'bottom'
                    bbox=dict(facecolor='white', edgecolor='none', pad=1.0) # 배경 박스 (선택 사항)
                )
            if obj_id_list[i] == obj_id_list_gt[i]:
                num_of_part_correct_y += 1

    iou_mean = np.mean(iou_list)
    fig.suptitle(f'{title}\nIoU: {iou_mean:.3f}(1.0 indicates a perfect match), {num_of_part_correct_y} of {len(obj_id_list)} slices in correct Y', fontsize=16, fontweight='bold')

    

    # 5. 레이아웃 조정 및 표시
    # tight_layout()은 플롯 요소들이 겹치는 것을 방지합니다.
    plt.subplots_adjust(
        wspace=0, # 가로(Width) 간격 0
        hspace=0, # 세로(Height) 간격 0
        left=0,   # 왼쪽 여백 0
        right=1,  # 오른쪽 여백 0 (1은 끝까지 채움)
        top=0.90,    # 위쪽 여백 0 (1은 끝까지 채움)
        bottom=0  # 아래쪽 여백 0
    )
    #plt.tight_layout() 
    plt.savefig(output_fliename)
    plt.close()

    return iou_mean

# # 0-255 범위의 uint8 데이터로 가정된 텐서인 경우
# # translated_image_uint8 = (torch.rand(1, H, W) * 255).to(torch.uint8)
# # save_image_tensor_to_png(translated_image_uint8, 'translated_output_uint8.png')


            

def make_video2(device, tiff_dir, part_pcs_gt, original_vertices, part_valids, 
           expanded_part_scale, inference_result_dir, output_dir, obj_id_list):

    data_id = inference_result_dir.split("/")[-1]
    os.makedirs(f'{output_dir}/{data_id}',exist_ok=True)
    os.makedirs(f'{output_dir}/{data_id}/steps',exist_ok=True)


    gt = np.load(f'{inference_result_dir}/gt.npy')
    init_pose = np.load(f'{inference_result_dir}/init_pose.npy')
    predict_file_name = glob.glob(f'{inference_result_dir}/predict*')[0]
    predict_0 = np.load(predict_file_name)

    r_default, r_front, r_down = get_renderer(device) # 기존 함수 호출
    axis_imgs = axis_draw(r_default, r_front, r_down, device) # 리스트 형태로 반환된다고 가정 [def, front, down]
    axis_map = {'_eye': axis_imgs[0], '_front': axis_imgs[1], '_down': axis_imgs[2]}

    renderers = {
        '_eye': r_default,       
        '_front': r_front,
        '_down': r_down
    }
    modes = ['original', 'init', 'init_gt', 'final']
    buffer = {mode: {view: [] for view in renderers} for mode in modes}
    mask_2d_list = []
    _step = predict_0.shape[0] - 1
    for _part_idx, vertices in enumerate(tqdm(original_vertices, desc='gen snapshot images')):
        #print(pcd_file_name)
        #vertices = vertices.cpu()
        tiff_image_path = f'{tiff_dir}/{obj_id_list[_part_idx]}.tif'
        if not part_valids[_part_idx]:
            continue

        if isinstance(vertices, np.ndarray):
            vertices = torch.from_numpy(vertices).to(device).to(torch.float)
        else:
            vertices = vertices.to(device).to(torch.float)
        translated_points = vertices

        pcd_points = {}
        
        # Original
        pcd_points['original'] = vertices
        pts_init, _ = transform_pc(device, translated_points, init_pose, None, None, -1, _part_idx)
        pts_init_gt, _ = transform_pc(device, translated_points, init_pose, gt, None, -1, _part_idx)
        pts_final, _ = transform_pc(device, translated_points, init_pose, gt, predict_0, _step, _part_idx)
        pcd_points['init'] = pts_init
        pcd_points['init_gt'] = pts_init_gt
        pcd_points['final'] = pts_final

        

        original_tiff, translated_tiff_pred = transform_2d_tiff(device,expanded_part_scale[0,_part_idx,0,:].item(), 
                                                              tiff_image_path, init_pose, gt, predict_0, _step, _part_idx)
        #print(_i, torch.FloatTensor([transform[_i,3:]]), torch.min(translated_points,axis=0)[0],torch.max(translated_points,axis=0)[0])


        mask_2d_obj = {}
        mask_2d_obj['tiff_gt'] = original_tiff
        mask_2d_obj['tiff_pred'] = translated_tiff_pred
        mask_2d_obj['pred_y'] = torch.min(pts_final,axis=0)[0][1].item()
        mask_2d_obj['color'] = tab10_r.colors[(_part_idx)%len(tab10_r.colors)][:3]
        mask_2d_obj['id'] = obj_id_list[_part_idx]

        mask_2d_list.append(mask_2d_obj)


        colors = torch.ones_like(translated_points) * torch.tensor(tab10_r.colors[(_part_idx)%len(tab10_r.colors)][:3]).to(device)  # blue points

        for mode in modes:
            pcd = pytorch3d.structures.Pointclouds(points=[pcd_points[mode]], features=[colors.to(torch.float)])
            for view_name, renderer in renderers.items():
                rendered_img = (renderer(pcd)[0] * 255).clamp(0, 255).to(torch.uint8)
                buffer[mode][view_name].append(rendered_img)





    all_slices_into_grid_layout(mask_2d_list, f'{output_dir}/{data_id}/{"mask_2d_final"}.png'
                                , True, f'Step {_step} : Denoising')

    for mode in modes:
        for view_name, axis_data in axis_map.items():
            buffer[mode][view_name].extend(axis_data)
            file_name = f'{output_dir}/{data_id}/{mode}{view_name}.png'
            image_data = sum_arrays(buffer[mode][view_name], buffer[mode][view_name][0][0][0][0],device)
            imageio.mimsave(file_name, [image_data.cpu().numpy()], format='PNG')




    mask_2d_list_last_step = []
    for _step in tqdm(range(-1, predict_0.shape[0]), desc='gen step images'):
        mask_2d_list = []
        mask_2d_list_last_step = mask_2d_list
        buffer = {view: [] for view in renderers}

        for _part_idx, vertices in enumerate(original_vertices):
            #print(pcd_file_name)
            #vertices = vertices.cpu()
            tiff_image_path = f'{tiff_dir}/{obj_id_list[_part_idx]}.tif'
            if not part_valids[_part_idx]:
                continue

            if isinstance(vertices, np.ndarray):
                vertices = torch.from_numpy(vertices).to(device).to(torch.float)
            else:
                vertices = vertices.to(device).to(torch.float)
            translated_points = vertices

            pts_pred = None
            colors = torch.ones_like(translated_points) * torch.tensor(tab10_r.colors[(_part_idx)%len(tab10_r.colors)][:3]).to(device)  # blue points
            for view_name, renderer in renderers.items():
                pts_pred, _ = transform_pc(device, translated_points, init_pose, gt, predict_0, _step, _part_idx)
                pcd = pytorch3d.structures.Pointclouds(points=[pts_pred], features=[colors.to(torch.float)])                
                rendered_img = (renderer(pcd)[0] * 255).clamp(0, 255).to(torch.uint8)
                buffer[view_name].append(rendered_img)

            
            original_tiff, translated_tiff_pred = transform_2d_tiff(device,expanded_part_scale[0,_part_idx,0,:].item(),
                                                                   tiff_image_path, init_pose, gt, predict_0, _step, _part_idx)
            #print(_i, torch.FloatTensor([transform[_i,3:]]), torch.min(translated_points,axis=0)[0],torch.max(translated_points,axis=0)[0])


            mask_2d_obj = {}
            mask_2d_obj['tiff_gt'] = original_tiff
            mask_2d_obj['tiff_pred'] = translated_tiff_pred
            mask_2d_obj['pred_y'] = torch.min(pts_pred,axis=0)[0][1].item()
            mask_2d_obj['color'] = tab10_r.colors[(_part_idx)%len(tab10_r.colors)][:3]
            mask_2d_obj['id'] = obj_id_list[_part_idx]

            mask_2d_list.append(mask_2d_obj)
    





        all_slices_into_grid_layout(mask_2d_list, f'{output_dir}/{data_id}/steps/{_step}_mask.png'
                                        , True, f'Step {_step} : Denoising')


        for view_name, axis_data in axis_map.items():
            buffer[view_name].extend(axis_data)
            file_name = f'{output_dir}/0/steps/{_step}{view_name}.png'
            image_data = sum_arrays(buffer[view_name], buffer[view_name][0][0][0][0],device)
            imageio.mimsave(file_name, [image_data.cpu().numpy()], format='PNG')

    _denoised_part_seq = 1

    max_step = _step + 4
    last_step_idx = max_step - 1

    translated_tiff_gt_list = [item[f'tiff_gt'] for item in mask_2d_list_last_step]

    sorted_mask_2d_list = sorted(mask_2d_list_last_step, key=lambda item: item['pred_y'])
    
    translated_tiff_pred_list = [item[f'tiff_pred'] for item in sorted_mask_2d_list]

    obj_id_list = [item['id'] for item in sorted_mask_2d_list]
    color_list = [item['color'] for item in sorted_mask_2d_list]
    y_list = [item['pred_y'] for item in sorted_mask_2d_list]

    
    for _step_temp in tqdm(range(_step, max_step), desc='adustment'):

    #for _step in tqdm(range(_step, _step+1), desc='small step'):
        



        mask_2d_list = []
        adjusted_img = None
        adjusted_img_pre = None

        _denoised_part_seq_start = len(original_vertices) - 1
        _denoised_part_seq_end = int(len(original_vertices)/2)




        if _denoised_part_seq_start < _denoised_part_seq_end:
            for _denoised_part_seq in range(_denoised_part_seq_start):
                mask_2d_obj = {}
                mask_2d_obj['tiff_gt'] = translated_tiff_gt_list[_denoised_part_seq]
                mask_2d_obj['tiff_pred'] = translated_tiff_pred_list[_denoised_part_seq]
                mask_2d_obj['pred_y'] = y_list[_denoised_part_seq]
                mask_2d_obj['color'] = color_list[_denoised_part_seq]
                mask_2d_obj['id'] = obj_id_list[_denoised_part_seq]

                mask_2d_list.append(mask_2d_obj)

        adjusted_id = []
        for _denoised_part_seq in range(_denoised_part_seq_start, _denoised_part_seq_end, 
                                        -1 if _denoised_part_seq_start > _denoised_part_seq_end else 1):
            
            cur_img = translated_tiff_pred_list[_denoised_part_seq]
            if ((_denoised_part_seq_start < _denoised_part_seq_end) and (_denoised_part_seq > _denoised_part_seq_start)) or ((_denoised_part_seq_start > _denoised_part_seq_end) and (_denoised_part_seq < _denoised_part_seq_start)):
                
                adjusted_id.append(obj_id_list[_denoised_part_seq])
                adjustment = register_iterative_torch(adjusted_img_pre,cur_img, 
                                                      iterations=1000, min_update_norm=1e-7, step_size=0.0001,)
                adjusted_img = warp_image_torch(cur_img, adjustment)
                #print("@@@@@@@@@@@@@@@@@@@@@@@@@")
                #print(adjusted_img_pre.shape,cur_img.shape,adjusted_img.shape)
            else:
                adjusted_img = cur_img
            
            mask_2d_obj = {}
            mask_2d_obj['tiff_gt'] = translated_tiff_gt_list[_denoised_part_seq]
            mask_2d_obj['tiff_pred'] = adjusted_img
            translated_tiff_pred_list[_denoised_part_seq] = adjusted_img
            mask_2d_obj['pred_y'] = y_list[_denoised_part_seq]
            mask_2d_obj['color'] = color_list[_denoised_part_seq]
            mask_2d_obj['id'] = obj_id_list[_denoised_part_seq]

            mask_2d_list.append(mask_2d_obj)
            adjusted_img_pre = cur_img

        if _denoised_part_seq_start > _denoised_part_seq_end:
            for _denoised_part_seq in range(_denoised_part_seq_end,-1,-1):
                mask_2d_obj = {}
                mask_2d_obj['tiff_gt'] = translated_tiff_gt_list[_denoised_part_seq]
                mask_2d_obj['tiff_pred'] = translated_tiff_pred_list[_denoised_part_seq]
                mask_2d_obj['pred_y'] = y_list[_denoised_part_seq]
                mask_2d_obj['color'] = color_list[_denoised_part_seq]
                mask_2d_obj['id'] = obj_id_list[_denoised_part_seq]

                mask_2d_list.append(mask_2d_obj)
        print(f"{_step_temp} / ADJUST / {adjusted_id}")
        
        if _denoised_part_seq_start > _denoised_part_seq_end:
            mask_2d_list.reverse()

        iou_mean = all_slices_into_grid_layout(mask_2d_list, f'{output_dir}/{data_id}/steps/{_step_temp}_mask.png'
                                        , True, f'Step {_step_temp} : Adjusting')

        
        #if True or iou_mean < 0.001:
        #    break


        '''
        for view_name, axis_data in axis_map.items():
            buffer[view_name].extend(axis_data)
            file_name = f'{output_dir}/0/steps/{_step}{view_name}.png'
            image_data = sum_arrays(buffer[view_name], 0)
            imageio.mimsave(file_name, [image_data], format='PNG')
        '''

        
    

    iter_list = list(axis_map.items()) + [('_mask', None)]
    for view_name, axis_data in tqdm(iter_list , desc='gen videos'):
        w = iio.get_writer(f'{output_dir}/{data_id}/video{view_name}.mp4', format='FFMPEG', mode='I', fps=1,
                            #codec='h264_vaapi',
                            pixelformat='yuv420p')
        
        for _step_temp in range(-1, max_step ):
            try:
                _img = iio.imread(f'{output_dir}/{data_id}/steps/{_step_temp}{view_name}.png')
            except:
                continue
            w.append_data(_img)

        w.close()
    

    return last_step_idx

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
