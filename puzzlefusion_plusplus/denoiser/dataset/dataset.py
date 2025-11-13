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
        
        self.data_list = []
        #self.data_list = self.data_list[:1]
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
    
    def _recenter_pc(self, pc):
        """pc: [N, 3]"""
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid[None]
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

    @staticmethod
    def _rotate_pc(pc):
        """pc: [N, 3]"""
        
        pc = torch.from_numpy(pc).float()


        
        #quat_gt = torch.tensor([torch.rand(1),0,1,0])
        #quat_gt = normalize(quat_gt, p=1.0, dim = 0)


        #_rotate_pc_rad torch.Size([3, 4]) tensor([-2.6932,  6.0788,  0.7550])
        #_rotate_pc torch.Size([4])
        #random_radian = np.random.uniform(low=0, high=2 * np.pi)
        #random_radian = ( torch.rand(3) ) * 4 * torch.pi - 2 * torch.pi  
        #quat_gt = y_axis_rotation_quaternion_from_rad(random_radian) 
        #quat_gt = torch.squeeze(quat_gt, dim=0)
        #print('_rotate_pc_rad',quat_gt.shape, random_radian)
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
        
        
        return pc.cpu().numpy(), quat_gt.cpu().numpy()


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
        

        pc = torch.from_numpy(pc).float()

        #_mean = torch.mean(pc, axis=0)
        #tr = Translate(-_mean[0],-_mean[1],-_mean[2], dtype=torch.float32)
        #tr_r = Translate(_mean[0],_mean[1],_mean[2], dtype=torch.float32)

        
        #quat_gt = torch.tensor([torch.rand(1),0,1,0])
        #quat_gt = normalize(quat_gt, p=1.0, dim = 0)
        #random_radian = (torch.rand(1) ) * 2 * torch.pi
        #quat_gt = y_axis_rotation_quaternion_from_rad(random_radian) 
        #quat_gt = torch.squeeze(quat_gt, dim=0)

        quat_gt = torch.rand(4)
        quat_gt[1] = 0
        quat_gt[3] = 0
        quat_gt = quat_gt / quat_gt.norm(dim=-1, keepdim=True)



        #r = Rotate(quaternion_to_matrix(quat_gt), dtype=torch.float32)
        #t = Transform3d().compose(rr)
        #pc = t.transform_points(pc)#.to(torch.float).to(device)

        
        #_mean = torch.mean(pc, axis=0)
        #tr = Translate(-_mean[0],-_mean[1],-_mean[2], dtype=torch.float32)
        #tr_r = Translate(_mean[0],_mean[1],_mean[2], dtype=torch.float32)
        rr = Rotate(quaternion_to_matrix(quat_gt), dtype=torch.float32)
        #pc = Transform3d().compose(tr).transform_points(pc)
        pc = Transform3d().compose(rr).transform_points(pc)
        #pc = Transform3d().compose(tr_r).transform_points(pc)

        return pc.cpu().numpy().reshape(P, N, 3), quat_gt.cpu().numpy()

    
    def _recenter_ref(self, pc, ref_part):
        """
        pc: [P, N, 3]
        """
        P, N, _ = pc.shape
        ref_idx = np.where(ref_part)[0]
        centroid = np.mean(pc[ref_idx.item()], axis=0)
        pc = pc - centroid
        return pc, centroid
    
    def _pad_data(self, data):
        """Pad data to shape [`self.max_num_part`, data.shape[1], ...]."""
        data = np.array(data)
        pad_shape = (self.max_num_part, ) + tuple(data.shape[1:])
        pad_data = np.zeros(pad_shape, dtype=np.float32)
        pad_data[:data.shape[0]] = data
        return pad_data
    

    def __getitem__(self, idx):
        data_dict = copy.deepcopy(self.data_list[idx])
        num_parts = data_dict['num_parts']
        part_pcs_gt = data_dict['part_pcs_gt']
        
        #print("part_pcs_gt=============================")
        #for i in range(num_parts):
        #    print(i,part_pcs_gt[i][:3,:])
        

        ref_part = data_dict['ref_part']
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
            pc, gt_trans = self._recenter_pc(pc)
            
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

        cur_pts = self._pad_data(np.stack(cur_pts, axis=0)).astype(np.float32)  # [P, N, 3]
        cur_quat = self._pad_data(np.stack(cur_quat, axis=0)).astype(np.float32)  # [P, 4]
        cur_trans = self._pad_data(np.stack(cur_trans, axis=0)).astype(np.float32)  # [P, 3]
        part_pcs_gt = self._pad_data(np.stack(part_pcs_gt, axis=0)).astype(np.float32) # [P, N, 3]

        
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
        scale = np.max(np.abs(cur_pts), axis=(1,2), keepdims=True)
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

