""" Use pre-processed point cloud data for training. """

import os
import numpy as np
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import copy
import torch
from torch.nn.functional import normalize

from pytorch3d.transforms import Transform3d
from pytorch3d.transforms.transform3d import (
    Rotate,
    RotateAxisAngle,
    Scale,
    Transform3d,
    Translate,
)
from puzzlefusion_plusplus.denoiser.evaluation.transform import (
    transform_pc,
    quaternion_to_euler,
    quaternion_to_matrix,
    get_euler_angles_from_quaternion,
    y_axis_rotation_quaternion_from_rad,
)

class GeometryPartDataset(Dataset):
    """Geometry part assembly dataset.

    We follow the data prepared by Breaking Bad dataset:
        https://breaking-bad-dataset.github.io/
    """

    def __init__(
        self,
        cfg,
        data_dir,
        data_fn,
        category='',
        rot_range=-1,
        overfit=-1,
        device = None
    ):
        self.cfg = cfg
        self.category = category if category.lower() != 'all' else ''
        self.data_dir = data_dir
        self.data_fn = data_fn
        self.device = device

        self.disassemble_mode = self.cfg.disassemble_mode
        self.disassemble_jitter_ratio = self.cfg.disassemble_jitter_ratio

        self.data_files = sorted([f for f in os.listdir(self.data_dir) if f.endswith('.npz')])

        self.max_num_part = cfg.data.max_num_part
        self.min_num_part = cfg.data.min_num_part
        self.rotation_1d = True
        if overfit != -1: 
            self.data_files = self.data_files[:overfit] 
        
        self.data_list = []
        self.rot_range = rot_range

        for file_name in tqdm(self.data_files):
            data_dict = np.load(os.path.join(self.data_dir, file_name))

            pc = data_dict['part_pcs_gt']
            data_id = data_dict['data_id'].item()
            part_valids = data_dict['part_valids']
            num_parts = data_dict["num_parts"].item()
            mesh_file_path = data_dict['mesh_file_path'].item()
            category = data_dict["category"]
            
            sample = {
                'part_pcs': pc,
                'data_id': data_id,
                'part_valids': part_valids,
                'mesh_file_path': mesh_file_path,
                'num_parts': num_parts,
            }

            if num_parts > self.max_num_part or num_parts < self.min_num_part:
                continue

            self.data_list.append(sample)
        #print("@@@@@@@@@@@@@@@@@@",len(self.data_list))
    #@staticmethod
    def _recenter_pc(self, pc, ratio=0.001):
        """
        pc: [N, 3] Tensor
        ratio: 전체 크기 대비 이동할 비율
        """

        if self.disassemble_mode == 'jitter':
            # 1. 전체 포인트 클라우드의 범위(Bounding Box) 계산
            # torch.min/max는 (values, indices)를 반환하므로 [0]으로 값만 취함
            p_min = torch.min(pc, dim=0)[0]
            p_max = torch.max(pc, dim=0)[0]

            # 2. 대각선 길이(Scale) 계산
            bbox_diagonal = torch.norm(p_max - p_min)

            # 3. 노이즈의 표준편차(sigma) 설정
            sigma = bbox_diagonal * ratio

            # 4. 가우시안 노이즈 생성 및 적용 (Global Translation)
            # torch.randn: 평균 0, 표준편차 1인 정규분포
            # pc와 같은 device에 생성해야 연산 가능
            noise = torch.randn(3, device=pc.device) * sigma
            
            # Broadcasting: [N, 3] + [3] -> 모든 점에 동일한 noise가 더해짐
            pc = pc + noise
            
            return pc, noise

        elif self.disassemble_mode == 'center':
            # Centroid 계산
            centroid = torch.mean(pc, dim=0)
            
            # 중심점 빼기
            # [N, 3] - [3] 브로드캐스팅이 자동으로 되지만, 
            # 명시적으로 차원을 맞추려면 centroid.unsqueeze(0) 사용
            pc = pc - centroid
            
            return pc, centroid

    '''
    @staticmethod
    def angle_2_quaternion(angle_rad):
        rotation = R.from_rotvec([0, angle_rad, 0])
        quaternion = rotation.as_quat()[[3, 0, 1, 2]]
        
        return torch.from_numpy(quaternion)
    '''
   
   
    def get_limited_y_rotation_quat(max_angle_degree=45):
        """
        max_angle_degree: 제한할 최대 회전 각도 (예: ±10도)
        """
        # 1. 각도 생성 (-max ~ +max 사이의 랜덤 라디안)
        max_rad = torch.deg2rad(torch.tensor(5.0))

        
        # (torch.rand(1) - 0.5) * 2  => -1.0 ~ 1.0 범위 생성
        theta = (torch.rand(1) - 0.5) * 2 * max_rad

        # 2. 반각(Half-angle) 계산
        half_theta = theta / 2

        # 3. Y축 회전 쿼터니언 생성
        # 공식: q = [cos(t/2), 0, sin(t/2), 0] (w, x, y, z 순서 가정)
        w = torch.cos(half_theta)
        x = torch.zeros_like(theta)
        y = torch.sin(half_theta)
        z = torch.zeros_like(theta)

        # 4. 합치기
        quat_gt = torch.cat([w, x, y, z], dim=0)

        # 이미 수식적으로 unit quaternion이므로 별도의 정규화 불필요
        # 하지만 부동소수점 오차 방지를 위해 안전하게 한 번 더 해줄 수 있음
        quat_gt = quat_gt / quat_gt.norm()
        
        return quat_gt

    #@staticmethod
    def _rotate_pc(self, pc, ratio = 0.001):
        """pc: [N, 3]"""
        #pc = torch.from_numpy(pc).float()
         

        if self.disassemble_mode == 'jitter':
            
            #quat_gt = self.get_limited_y_rotation_quat()
            max_rad = torch.deg2rad(torch.tensor(5.0))

            # (torch.rand(1) - 0.5) * 2  => -1.0 ~ 1.0 범위 생성
            theta = (torch.rand(1) - 0.5) * 2 * max_rad

            # 2. 반각(Half-angle) 계산
            half_theta = theta / 2

            quat_gt = torch.rand(4)            
            quat_gt[2] = torch.sin(half_theta)
            quat_gt[1] = 0
            quat_gt[3] = 0
            quat_gt = quat_gt / quat_gt.norm(dim=-1, keepdim=True)
            #quat_gt = torch.rand(4)
        else:
            quat_gt = torch.rand(4)            
            quat_gt[1] = 0
            quat_gt[3] = 0
            quat_gt = quat_gt / quat_gt.norm(dim=-1, keepdim=True)
        #print('_rotate_pc',quat_gt.shape)
        
        #_mean = torch.mean(pc, axis=0)
        #tr = Translate(-_mean[0],-_mean[1],-_mean[2], dtype=torch.float32)
        #tr_r = Translate(_mean[0],_mean[1],_mean[2], dtype=torch.float32)
        rr = Rotate(quaternion_to_matrix(quat_gt), dtype=torch.float32)
        #pc = Transform3d().compose(tr).transform_points(pc)
        pc = Transform3d().compose(rr).transform_points(pc)
        #pc = Transform3d().compose(tr_r).transform_points(pc)
        
        
        return pc, quat_gt


    @staticmethod
    def _rotate_pc_xyz(pc):
        """pc: [N, 3]"""
        rot_mat = R.random().as_matrix()
        pc = (rot_mat @ pc.T).T
        quat_gt = R.from_matrix(rot_mat.T).as_quat()
        # we use scalar-first quaternion
        quat_gt = quat_gt[[3, 0, 1, 2]]
        return pc, quat_gt

    def _pad_data(self, data):
        """Pad data to shape [`self.max_num_part`, data.shape[1], ...]."""
        
        # 2. 목표 형상(Shape) 계산
        # (max_num, D1, D2, ...)
        pad_shape = (self.max_num_part, ) + data.shape[1:]
        
        # 3. 0으로 채워진 텐서 생성
        # ★중요: 입력 데이터(data)와 같은 device(GPU/CPU)에 만들어야 에러가 안 남
        pad_data = torch.zeros(pad_shape, dtype=torch.float32, device=data.device)
        
        # 4. 데이터 복사
        # 입력 데이터의 길이(N)를 구함
        curr_len = data.shape[0]
        
        # 만약 현재 데이터가 max보다 길 경우를 대비해 슬라이싱 처리 (안전장치)
        limit = min(curr_len, self.max_num_part)
        
        pad_data[:limit] = data[:limit]
        
        return pad_data


    def __getitem__(self, idx):
        """
        recenter the fragments, and random rotate it to train ae
        """
        
        #data_dict = copy.deepcopy(self.data_list[idx])
        # 원본 데이터 가져오기
        src_data = self.data_list[idx]

        # 얕은 복사(껍데기) 생성 후, 내부 텐서만 명시적으로 clone
        data_dict = {}

        for key, value in src_data.items():
            if isinstance(value, torch.Tensor):
                # Tensor는 .clone()을 사용해야 안전하게 메모리가 복사됨 (GPU/CPU 유지)
                data_dict[key] = value.clone()
            elif isinstance(value, np.ndarray):
                # 혹시 섞여있을 NumPy 배열 처리
                data_dict[key] = value.copy()
            else:
                # 정수, 문자열 등 불변 객체는 그냥 할당
                data_dict[key] = value

                
        #pcs = data_dict['part_pcs']
        pcs = torch.from_numpy(data_dict['part_pcs']).float()
        num_parts = data_dict['num_parts']

        cur_pts = []
        for i in range(num_parts):
            pc = pcs[i]
            pc, _ = self._recenter_pc(pc, self.disassemble_jitter_ratio)
            if self.rotation_1d:
                pc, _ = self._rotate_pc(pc, self.device)
            else:
                pc, _ = self._rotate_pc_xyz(pc)
            cur_pts.append(pc)
            
        cur_pts = self._pad_data(torch.stack(cur_pts, dim=0))  # [P, N, 3]
        #scale = np.max(np.abs(cur_pts), axis=(1,2), keepdims=True)
        #scale[scale == 0] = 1
        #cur_pts = cur_pts / scale


        scale = torch.amax(torch.abs(cur_pts), dim=(1, 2), keepdim=True)
        scale[scale == 0] = 1
        cur_pts = cur_pts / scale
        


        data_dict['part_pcs'] = cur_pts
        
        return data_dict

    def __len__(self):
        return len(self.data_list)


def build_geometry_dataloader(cfg):




    data_dict = dict(
        cfg=cfg,
        data_dir=cfg.data.data_dir,
        data_fn='train',
        category=cfg.data.category,
        rot_range=cfg.data.rot_range,
        overfit=cfg.data.overfit,
        device = cfg.device
    )
    train_set = GeometryPartDataset(**data_dict)
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        drop_last=False, # jhahn
        persistent_workers=(cfg.data.num_workers > 0),
    )

    data_dict['data_fn'] = 'val'
    data_dict['data_dir'] = cfg.data.data_val_dir
    val_set = GeometryPartDataset(**data_dict)
    val_loader = DataLoader(
        dataset=val_set,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(cfg.data.num_workers > 0),
    )
    return train_loader, val_loader
