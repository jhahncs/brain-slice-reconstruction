import os
import numpy as np
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import copy
from puzzlefusion_plusplus.denoiser.model.modules.custom_diffusers import PiecewiseScheduler
import torch
import torch
from torch.nn.functional import normalize
from puzzlefusion_plusplus.denoiser.evaluation.transform import (
    transform_pc,
    quaternion_to_euler,
    quaternion_to_matrix,
    rotate_y_axis,
    y_axis_rotation_quaternion_from_rad
)
from pytorch3d.transforms import Transform3d
from pytorch3d.transforms.transform3d import (
    Rotate,
    RotateAxisAngle,
    Scale,
    Transform3d,
    Translate,
)



class GeometryLatentDataset(Dataset):
    def __init__(
            self,
            cfg,
            data_dir,
            overfit,
            data_fn,
            denoiser_only_flag = False
    ):
        self.cfg = cfg
        self.mode = data_fn
        self.data_dir = data_dir
        self.data_files = sorted([f for f in os.listdir(self.data_dir) if f.endswith('.npz')])
        self.noise_scheduler = PiecewiseScheduler()
        self.max_num_part = self.cfg.data.max_num_part
        self.denoiser_only_flag = denoiser_only_flag
        self.rotation_1d = True
        if overfit != -1:
            self.data_files = self.data_files[:overfit] 

        if self.mode == "test":
            self.matching_data_path = self.cfg.data.matching_data_path
        

        self.disassemble_mode = self.cfg.disassemble_mode
        self.disassemble_jitter_ratio = self.cfg.disassemble_jitter_ratio

        self.data_list = []
        #self.data_list = self.data_list[:1]
        print("data_dir",self.data_dir)
        #print("@@@@@@@@@@@@@",self.data_files)
        for file_name in tqdm(self.data_files):
            data_dict = np.load(os.path.join(self.data_dir, file_name))
            num_parts = data_dict["num_parts"].item()
            data_id = data_dict['data_id'].item()
            part_valids = data_dict['part_valids']
            part_pcs_gt = data_dict['part_pcs_gt']
            mesh_file_path = data_dict['mesh_file_path'].item()
            graph = data_dict['graph']
            ref_part = data_dict['ref_part']

            #jhahn
            if 'gt' in data_dict:

                sample = {
                    'data_id': data_id,
                    'part_valids': part_valids,
                    'mesh_file_path': mesh_file_path,
                    'num_parts': num_parts,
                    'ref_part': ref_part,
                    'part_pcs_gt': part_pcs_gt,
                    'graph': graph,
                    'gt': data_dict['gt'],
                    'part_pcs': data_dict['part_pcs'],
                    'init_pose': data_dict['init_pose'],
                    'part_scale': data_dict['part_scale'],
                    
                }
            else:
                sample = {
                    'data_id': data_id,
                    'part_valids': part_valids,
                    'mesh_file_path': mesh_file_path,
                    'num_parts': num_parts,
                    'ref_part': ref_part,
                    'part_pcs_gt': part_pcs_gt,
                    'graph': graph,
                    
                }

            if self.mode == "test" and denoiser_only_flag is False:
                matching_data_path = os.path.join(self.matching_data_path, str(data_id) + '.npz')
                if not os.path.exists(matching_data_path):
                    continue
                matching_data = np.load(matching_data_path, allow_pickle=True)
                edges = matching_data['edges']
                correspondences = matching_data['correspondence']
                gt_pc_by_area = matching_data['gt_pcs']
                critical_pcs_idx = matching_data['critical_pcs_idx']
                n_pcs = matching_data['n_pcs']
                n_critical_pcs = matching_data['n_critical_pcs']
                    
                if correspondences.shape[0] != 1:
                    if correspondences.dtype == "O":
                        sample['correspondences'] = correspondences.tolist()
                    else:
                        sample['correspondences'] = [correspondences[i] for i in range(correspondences.shape[0])]
                else:
                    sample['correspondences'] = [correspondences.squeeze()]
                    
                sample['gt_pc_by_area'] = gt_pc_by_area
                sample['critical_pcs_idx'] = critical_pcs_idx
                sample['edges'] = edges
                sample['n_pcs'] = n_pcs
                sample['n_critical_pcs'] = n_critical_pcs

            self.data_list.append(sample)

    
    def _anchor_coords(self, pcs, global_t, global_r):
        global_r = R.from_quat(global_r[[1, 2, 3, 0]]).inv()
        
        pcs = global_r.apply(pcs)
        pcs = pcs - global_t
        return pcs


    def _move_to_init_pose(self, pcs, n_pcs, num_parts, trans, rots):
        final_pose_pts = []
        
        pcs_count = 0
        for i in range(num_parts):
            c_pcs = pcs[pcs_count:pcs_count+n_pcs[i]]
            
            c_pcs = c_pcs - trans[i]



            #c_pcs = R.from_quat(rots[i][[1, 2, 3, 0]]).inv().apply(c_pcs) # [w, x, y, z] -> [x, y, z, w]
            c_pcs = rotate_y_axis(rots[i],c_pcs)


            final_pose_pts.append(c_pcs)
            pcs_count += n_pcs[i]
        
        final_pose_pts = np.concatenate(final_pose_pts, axis=0)
                
        return final_pose_pts
    
    
    def __len__(self):
        return len(self.data_list)           
    
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

            
    
    def _rotate_pc_xyz(self, pc):
        """
        pc: [N, 3]
        """
    
        rot_mat = R.random().as_matrix()
        pc = (rot_mat @ pc.T).T
        quat_gt = R.from_matrix(rot_mat.T).as_quat()
        # we use scalar-first quaternion
        quat_gt = quat_gt[[3, 0, 1, 2]]
        return pc, quat_gt



    def get_limited_y_rotation_quat(max_angle_degree=45):
        """
        max_angle_degree: 제한할 최대 회전 각도 (예: ±10도)
        """
        # 1. 각도 생성 (-max ~ +max 사이의 랜덤 라디안)
        max_rad = np.radians(max_angle_degree)
        
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
        quat_gt = quat_gt / quat_gt.norm(dim=-1, keepdim=True)
        
        return quat_gt
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


    def _rotate_whole_part_xyz(self, pc):
        """
        pc: [P, N, 3]
        """
        P, N, _ = pc.shape
        pc = pc.reshape(-1, 3)
        rot_mat = R.random().as_matrix()
        #print('rot_mat',rot_mat)
        pc = (rot_mat @ pc.T).T
        quat_gt = R.from_matrix(rot_mat.T).as_quat()
        # we use scalar-first quaternion
        quat_gt = quat_gt[[3, 0, 1, 2]]
        return pc.reshape(P, N, 3), quat_gt

    def _rotate_whole_part(self, pc):
        """
        pc: [P, N, 3]
        """
        P, N, _ = pc.shape
        pc = pc.reshape(-1, 3)
        #pc, guat_gt = _rotate_pc(pc)
        

        #pc = torch.from_numpy(pc).float()
        quat_gt = torch.rand(4)
        quat_gt[1] = 0
        quat_gt[3] = 0
        quat_gt = quat_gt / quat_gt.norm(dim=-1, keepdim=True)

        rr = Rotate(quaternion_to_matrix(quat_gt), dtype=torch.float32)
        pc = Transform3d().compose(rr).transform_points(pc)

        #return pc.cpu().numpy().reshape(P, N, 3), quat_gt.cpu().numpy()
        return pc.reshape(P, N, 3), quat_gt

    
    def _recenter_ref(self, pc, ref_part):
        P, N, _ = pc.shape

        # 1. Reference Index 찾기
        # torch.where는 튜플을 반환하므로 [0]으로 인덱스 텐서를 가져옴
        ref_idx = torch.where(ref_part)[0]

        # 2. Centroid 계산
        # ref_idx.item(): 1개의 원소를 가진 텐서에서 정수 값(Python scalar) 추출
        # dim=0: [N, 3] -> [3] (N개의 점들에 대한 평균)
        centroid = torch.mean(pc[ref_idx.item()], dim=0)

        # 3. 빼기 연산 (Broadcasting)
        # [P, N, 3] - [3] 형태로, 마지막 차원이 같으므로 자동 브로드캐스팅 되어 모든 점(P*N)에서 centroid가 빠짐
        pc = pc - centroid

        return pc, centroid

    def _recenter_ref_cpu(self, pc, ref_part):
        """
        pc: [P, N, 3]
        """
        P, N, _ = pc.shape
        ref_idx = np.where(ref_part)[0]
        centroid = np.mean(pc[ref_idx.item()], axis=0)
        pc = pc - centroid
        return pc, centroid
    
    def _pad_data_cpu(self, data):
        """Pad data to shape [`self.max_num_part`, data.shape[1], ...]."""
        data = np.array(data)
        pad_shape = (self.max_num_part, ) + tuple(data.shape[1:])
        pad_data = np.zeros(pad_shape, dtype=np.float32)
        pad_data[:data.shape[0]] = data
        return pad_data

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
        data_dict = copy.deepcopy(self.data_list[idx])
        num_parts = data_dict['num_parts']
        part_pcs_gt = data_dict['part_pcs_gt']
        
        #print("part_pcs_gt=============================")
        #for i in range(num_parts):
        #    print(i,part_pcs_gt[i][:3,:])
        

        ref_part = torch.from_numpy(data_dict['ref_part'])


        part_pcs_gt = torch.from_numpy(part_pcs_gt).float()

        #for i in range(num_parts):
        #    print(i,np.mean(part_pcs_gt[i], axis=0))
        if self.rotation_1d:
            part_pcs_final, pose_gt_r = self._rotate_whole_part(part_pcs_gt)
        else:
            part_pcs_final, pose_gt_r = self._rotate_whole_part_xyz(part_pcs_gt)
        #for i in range(num_parts):
        #    print(i,np.mean(part_pcs_final[i], axis=0))
        part_pcs_final, pose_gt_t = self._recenter_ref(part_pcs_final, ref_part)
        #for i in range(num_parts):
        #    print(i,np.mean(part_pcs_final[i], axis=0))
        #print('pose_gt_t',pose_gt_t)
        #print('pose_gt_r',pose_gt_r)
        #print('ref_part',ref_part)
        cur_pts, cur_quat, cur_trans = [], [], []
        
        for i in range(num_parts):
            pc = part_pcs_final[i]
            pc, gt_trans = self._recenter_pc(pc,self.disassemble_jitter_ratio)
            
            if self.rotation_1d:
                pc, gt_quat = self._rotate_pc(pc)
            else:
                pc, gt_quat = self._rotate_pc_xyz(pc)
            #print(i,gt_quat )
            cur_quat.append(gt_quat)
            cur_trans.append(gt_trans)
            cur_pts.append(pc)
            #print(i,gt_trans,gt_quat)

        #print("gt=============================")
        #for i in range(num_parts):
        #    print(i,cur_pts[i][:3,:])


        cur_pts_tensor = torch.stack(cur_pts, dim=0)
        cur_quat_tensor = torch.stack(cur_quat, dim=0)
        cur_trans_tensor = torch.stack(cur_trans, dim=0)
        #part_pcs_gt_tensor = torch.stack(part_pcs_gt, dim=0)

        # 2. _pad_data 적용 및 float 변환
        # 주의: _pad_data 함수 내부도 torch.nn.functional.pad 등을 쓰도록 수정 필요
        cur_pts = self._pad_data(cur_pts_tensor).float()          # [P, N, 3]
        cur_quat = self._pad_data(cur_quat_tensor).float()        # [P, 4]
        cur_trans = self._pad_data(cur_trans_tensor).float()      # [P, 3]
        part_pcs_gt = self._pad_data(part_pcs_gt).float()


        #cur_pts = self._pad_data(np.stack(cur_pts, axis=0)).astype(np.float32)  # [P, N, 3]
        #cur_quat = self._pad_data(np.stack(cur_quat, axis=0)).astype(np.float32)  # [P, 4]
        #cur_trans = self._pad_data(np.stack(cur_trans, axis=0)).astype(np.float32)  # [P, 3]
        #part_pcs_gt = self._pad_data(np.stack(part_pcs_gt, axis=0)).astype(np.float32) # [P, N, 3]

        
        if self.mode == 'test' and self.denoiser_only_flag is False:        
            gt_pc_by_area = self._anchor_coords(
                data_dict['gt_pc_by_area'], 
                pose_gt_t, 
                pose_gt_r
            )

            part_pcs_by_area = self._move_to_init_pose(
                gt_pc_by_area,
                data_dict['n_pcs'],
                num_parts,
                cur_trans,
                cur_quat
            )
            
            data_dict['part_pcs_by_area'] = part_pcs_by_area.astype(np.float32)
        
        
        # Normalize the part pcs
        # 1. 절댓값 및 최대값 계산
        # np.max(..., axis=(1,2)) -> torch.amax(..., dim=(1, 2))
        scale = torch.amax(torch.abs(cur_pts), dim=(1, 2), keepdim=True)
        scale[scale == 0] = 1
        cur_pts = cur_pts / scale
        
        data_dict['part_pcs_gt'] = part_pcs_gt


        if 'gt' in data_dict:
            print('gt, init_pose available')
            #data_dict['part_scale'] = scale.squeeze(-1)
            #data_dict['part_pcs'] = data_dict['gt']
            data_dict['part_rots'] = data_dict['gt'][...,3:]
            data_dict['part_trans'] = data_dict['gt'][...,:3]
            data_dict['init_pose_r'] = data_dict['init_pose'][...,3:]
            data_dict['init_pose_t'] = data_dict['init_pose'][...,:3]
        else:
            data_dict['part_scale'] = scale.squeeze(-1)
            data_dict['part_pcs'] = cur_pts
            data_dict['part_rots'] = cur_quat
            data_dict['part_trans'] = cur_trans            
            data_dict['init_pose_r'] = pose_gt_r
            data_dict['init_pose_t'] = pose_gt_t



        
        # Only one reference part
        if self.cfg.model.multiple_ref_parts is False:
            return data_dict
    
        if self.mode != 'train':
            return data_dict

        num_parts = data_dict['num_parts']
        if num_parts == 2:
            return data_dict
        
        # half of the time, only one reference part
        if True or np.random.rand() < 0.5:
            return data_dict
        
        # Randomly sample more reference parts which connected to the original reference part
        ref_part = data_dict['ref_part']
        graph = data_dict['graph']
        scale = data_dict['part_scale']

        ref_part_idx = np.where(ref_part)[0]
        connect_parts = np.where(graph[ref_part_idx, :])[1]

        larger_connect_parts = [part for part in connect_parts if scale[part] > 0.05]
        if not larger_connect_parts:
            return data_dict

        num_connect_parts = len(larger_connect_parts)

        sample_num = np.random.randint(0, num_connect_parts)
        
        sample_ref_parts = np.random.choice(connect_parts, sample_num, replace=False)
        ref_part[sample_ref_parts] = True

        
        data_dict['ref_part'] = ref_part
        part_trans_ref = data_dict['part_trans'][sample_ref_parts]
        part_rots_ref = data_dict['part_rots'][sample_ref_parts]
        # random perturb the reference part
        noise_trans = torch.randn(part_trans_ref.shape)
        noise_rots = torch.randn(part_rots_ref.shape)
        timesteps = torch.randint(0, 50, (1,)).long()
        '''
        if self.rotation_1d:
            noise_rots[...,1] = torch.repeat_interleave(torch.Tensor([0]), noise_rots.shape[-2], dim=0).unsqueeze(dim=0)
            noise_rots[...,2] = torch.repeat_interleave(torch.Tensor([1]), noise_rots.shape[-2], dim=0).unsqueeze(dim=0)
            noise_rots[...,3] = torch.repeat_interleave(torch.Tensor([0]), noise_rots.shape[-2], dim=0).unsqueeze(dim=0)
        '''

        part_trans_ref = self.noise_scheduler.add_noise(torch.tensor(part_trans_ref), noise_trans, timesteps).numpy()
        part_rots_ref = self.noise_scheduler.add_noise(torch.tensor(part_rots_ref), noise_rots, timesteps).numpy()
        '''
        if self.rotation_1d:
            part_rots_ref[...,1] = torch.repeat_interleave(torch.Tensor([0]), part_rots_ref.shape[-2], dim=0).unsqueeze(dim=0)
            part_rots_ref[...,2] = torch.repeat_interleave(torch.Tensor([1]), part_rots_ref.shape[-2], dim=0).unsqueeze(dim=0)
            part_rots_ref[...,3] = torch.repeat_interleave(torch.Tensor([0]), part_rots_ref.shape[-2], dim=0).unsqueeze(dim=0)
        '''
        data_dict['part_trans'][sample_ref_parts] = part_trans_ref
        data_dict['part_rots'][sample_ref_parts] = part_rots_ref

        
        return data_dict


def build_geometry_dataloader(cfg):
    data_dict = dict(
        cfg=cfg,
        data_dir=cfg.data.data_dir,
        overfit=cfg.data.overfit,
        data_fn="train",
    )
    train_set = GeometryLatentDataset(**data_dict)

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(cfg.data.num_workers > 0),
    )


    data_dict['data_fn'] = "val"
    data_dict['data_dir'] = cfg.data.data_val_dir
    val_set = GeometryLatentDataset(**data_dict)
    val_loader = DataLoader(
        dataset=val_set,
        batch_size=cfg.data.val_batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(cfg.data.num_workers > 0),
    )
    return train_loader, val_loader


def build_test_dataloader(cfg, denoiser_only_flag):
    data_dict = dict(
        cfg=cfg,
        data_dir=cfg.data.data_val_dir,
        overfit=cfg.data.overfit,
        data_fn="test",
        denoiser_only_flag=denoiser_only_flag,
    )

    val_set = GeometryLatentDataset(**data_dict)
    val_loader = DataLoader(
        dataset=val_set,
        batch_size=cfg.data.val_batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(cfg.data.num_workers > 0),
    )
    return val_loader

