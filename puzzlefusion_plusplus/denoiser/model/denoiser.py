import torch
from torch.nn import functional as F
import pytorch_lightning as pl
import hydra
from puzzlefusion_plusplus.denoiser.model.modules.denoiser_transformer import DenoiserTransformer
from tqdm import tqdm
from chamferdist import ChamferDistance
from puzzlefusion_plusplus.denoiser.evaluation.evaluator import (
    calc_part_acc,
    trans_metrics,
    rot_metrics,
    calc_shape_cd
)
import math
import numpy as np
from puzzlefusion_plusplus.denoiser.model.modules.custom_diffusers import PiecewiseScheduler
from pytorch3d import transforms
from puzzlefusion_plusplus.denoiser.evaluation.transform import (
    transform_pc,
    quaternion_to_euler,
)
class Denoiser(pl.LightningModule):
    def __init__(self, cfg):
        super(Denoiser, self).__init__()
        self.cfg = cfg
        self.denoiser = DenoiserTransformer(cfg)

        self.save_hyperparameters()

        self.rotation_1d = True

        self.noise_scheduler = PiecewiseScheduler(
            num_train_timesteps=cfg.model.DDPM_TRAIN_STEPS,
            beta_schedule=cfg.model.DDPM_BETA_SCHEDULE,
            prediction_type=cfg.model.PREDICT_TYPE,
            beta_start=cfg.model.BETA_START,
            beta_end=cfg.model.BETA_END,
            clip_sample=False,
            timestep_spacing=self.cfg.model.timestep_spacing
        )

        self.encoder = hydra.utils.instantiate(cfg.ae.ae_name, cfg)

        self.cd_loss = ChamferDistance()
        self.num_points = cfg.model.num_point
        self.num_channels = cfg.model.num_dim

        self.noise_scheduler.set_timesteps(
            num_inference_steps=cfg.model.num_inference_steps
        )

        self.rmse_r_list = []
        self.rmse_t_list = []
        self.acc_list = []
        self.cd_list = []

        self.metric = ChamferDistance()


    def _apply_rots(self, part_pcs, noise_params):
        """
        Apply Noisy rotations to all points
        """
        noise_quat = noise_params[..., 3:]
        noise_quat = noise_quat / noise_quat.norm(dim=-1, keepdim=True)
        part_pcs = transforms.quaternion_apply(noise_quat.unsqueeze(2), part_pcs)
        
        return part_pcs
    

    def _extract_features(self, part_pcs, part_valids, noisy_trans_and_rots):
        B, P , _, _ = part_pcs.shape
        part_pcs = self._apply_rots(part_pcs, noisy_trans_and_rots)
        part_pcs = part_pcs[part_valids.bool()]

        encoder_out = self.encoder.encode(part_pcs)
        latent = torch.zeros(B, P, self.num_points, self.num_channels, device=self.device)
        xyz = torch.zeros(B, P, self.num_points, 3, device=self.device)

        latent[part_valids.bool()] = encoder_out["z_q"]
        xyz[part_valids.bool()] = encoder_out["xyz"]
        return latent, xyz


    def forward(self, data_dict):
        gt_trans = data_dict['part_trans']
        gt_rots = data_dict['part_rots']
        gt_trans_and_rots = torch.cat([gt_trans, gt_rots], dim=-1)
        ref_part = data_dict["ref_part"]
        noise = torch.randn(gt_trans_and_rots.shape, device=self.device)
        if self.rotation_1d:
            noise[...,4] = torch.repeat_interleave(torch.Tensor([0]), noise.shape[-2], dim=0).unsqueeze(dim=0)
            noise[...,5] = torch.repeat_interleave(torch.Tensor([1]), noise.shape[-2], dim=0).unsqueeze(dim=0)
            noise[...,6] = torch.repeat_interleave(torch.Tensor([0]), noise.shape[-2], dim=0).unsqueeze(dim=0)

        B, P, N, C = data_dict["part_pcs"].shape

        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (B,),
                                  device=self.device).long()
        
        noisy_trans_and_rots = self.noise_scheduler.add_noise(gt_trans_and_rots, noise, timesteps)


        noisy_trans_and_rots[ref_part] = gt_trans_and_rots[ref_part]
        if self.rotation_1d:
            noisy_trans_and_rots[...,4] = torch.repeat_interleave(torch.Tensor([0]), noisy_trans_and_rots.shape[-2], dim=0).unsqueeze(dim=0)
            noisy_trans_and_rots[...,5] = torch.repeat_interleave(torch.Tensor([1]), noisy_trans_and_rots.shape[-2], dim=0).unsqueeze(dim=0)
            noisy_trans_and_rots[...,6] = torch.repeat_interleave(torch.Tensor([0]), noisy_trans_and_rots.shape[-2], dim=0).unsqueeze(dim=0)


        part_pcs = data_dict["part_pcs"]
        part_valids = data_dict["part_valids"]
        latent, xyz = self._extract_features(part_pcs, part_valids, noisy_trans_and_rots)
        
        pred_noise = self.denoiser(
            noisy_trans_and_rots, 
            timesteps, 
            latent, 
            xyz, 
            data_dict['part_valids'],
            data_dict['part_scale'],
            ref_part
        )
        '''
        if self.rotation_1d:
            pred_noise[...,4] = torch.repeat_interleave(torch.Tensor([0]), pred_noise.shape[-2], dim=0).unsqueeze(dim=0)
            pred_noise[...,5] = torch.repeat_interleave(torch.Tensor([1]), pred_noise.shape[-2], dim=0).unsqueeze(dim=0)
            pred_noise[...,6] = torch.repeat_interleave(torch.Tensor([0]), pred_noise.shape[-2], dim=0).unsqueeze(dim=0)
        '''
        #pred_noise[ref_part]  = gt_trans_and_rots[ref_part]

        output_dict = {
            'pred_noise': pred_noise,
            'gt_noise': noise
        }

        return output_dict


    def calculate_dice_score_from_point_clouds(self, pc1: np.ndarray, pc2: np.ndarray, resolution: int = 64) -> float:
        """
        Calculates the Dice score between two point clouds using voxelization.

        Args:
            pc1 (np.ndarray): The first point cloud, shape (N, 3).
            pc2 (np.ndarray): The second point cloud, shape (M, 3).
            resolution (int): The resolution of the 3D voxel grid.

        Returns:
            float: The Dice score (0.0 to 1.0).
        """

        # 1. Normalize point clouds to fit within a [0, 1] cube
        combined_pc = np.concatenate([pc1, pc2], axis=0)
        min_coords = np.min(combined_pc, axis=0)
        max_coords = np.max(combined_pc, axis=0)
        
        # Handle the case of zero-size point clouds or flat point clouds
        if np.all(min_coords == max_coords):
            if pc1.shape[0] > 0 and pc2.shape[0] > 0:
                return 1.0 # Both are single points at the same location
            return 0.0 # One or both are empty
        
        scale = max_coords - min_coords
        
        
        pc1_norm = (pc1 - min_coords) / scale
        pc2_norm = (pc2 - min_coords) / scale

        # 2. Voxelize the point clouds
        # Get voxel indices for each point
        voxel_coords1 = np.floor(pc1_norm * (resolution - 1)).astype(int)
        voxel_coords2 = np.floor(pc2_norm * (resolution - 1)).astype(int)
        #print(voxel_coords1)
        #print(voxel_coords2)
        # Convert coordinates to a single index for a flat array
        voxel_indices1 = (voxel_coords1[:, 0] * resolution * resolution) + (voxel_coords1[:, 1] * resolution) + voxel_coords1[:, 2]
        voxel_indices2 = (voxel_coords2[:, 0] * resolution * resolution) + (voxel_coords2[:, 1] * resolution) + voxel_coords2[:, 2]

        # Create binary voxel sets
        voxel_set1 = set(voxel_indices1)
        voxel_set2 = set(voxel_indices2)

        # 3. Calculate intersection and union
        intersection_size = len(voxel_set1.intersection(voxel_set2))
        total_size = len(voxel_set1) + len(voxel_set2)
        #print('intersection_size',intersection_size)
        # 4. Calculate Dice score
        dice_score = (2.0 * intersection_size) / total_size if total_size > 0 else 0.0
        
        return dice_score

        

    def _loss(self, data_dict, output_dict):
        pred_noise = output_dict['pred_noise']
        part_valids = data_dict['part_valids'].bool()
        noise = output_dict['gt_noise']

        part_valids[data_dict["ref_part"]] = False
        mse_loss = F.mse_loss(pred_noise[part_valids], noise[part_valids])


        return {'mse_loss': mse_loss}

        
        
        gt_trans = data_dict['part_trans']
        gt_rots = data_dict['part_rots']
        gt_trans_and_rots = torch.cat([gt_trans, gt_rots], dim=-1)


        part_valids = data_dict['part_valids'].clone()
        part_scale = data_dict["part_scale"].clone()
        part_pcs = data_dict["part_pcs"].clone()


        pts = data_dict['part_pcs']
        pred_trans = pred_noise[..., :3]
        pred_rots = pred_noise[..., 3:]

        expanded_part_scale = data_dict["part_scale"].unsqueeze(-1).expand(-1, -1, 1000, -1)
        pts = pts * expanded_part_scale
        '''
        acc, _, _ = calc_part_acc(pts, trans1=pred_trans, trans2=gt_trans,
                            rot1=pred_rots, rot2=gt_rots, valids=data_dict['part_valids'], 
                            chamfer_distance=self.metric)
        '''
        shape_cd = calc_shape_cd(pts, trans1=pred_trans, trans2=gt_trans,
                            rot1=pred_rots, rot2=gt_rots, valids=data_dict['part_valids'], 
                            chamfer_distance=self.metric)


        self.dice_score(pts, trans1=pred_trans, trans2=gt_trans,
                            rot1=pred_rots, rot2=gt_rots, valids=data_dict['part_valids'], 
                            chamfer_distance=self.metric)
        shape_cd_svg = shape_cd.sum(-1) / (len(shape_cd))
        return {'mse_loss': mse_loss, 'shape_cd_loss': shape_cd_svg}
    def get_y_rotation_angle_360(self,w):
        """
        (w, 0, 1, 0) 형태의 쿼터니언에서 y축 회전 각도(360도 기준)를 계산합니다.
        
        Args:
            w (float): 쿼터니언의 실수부.
        
        Returns:
            float: 0도에서 360도 사이의 y축 회전 각도.
        """
        # w가 0에 매우 가까울 경우 나누기 0 오류를 방지
        if np.isclose(w, 0):
            angle_rad = np.pi
        else:
            # 회전 각도(라디안) 계산
            angle_rad = 2 * np.arctan(1.0 / w)
            
        # 각도를 도(degree)로 변환
        angle_deg = np.degrees(angle_rad)
        
        # 각도를 0 ~ 360도 범위로 조정
        angle_deg = angle_deg % 360
        if angle_deg < 0:
            angle_deg += 360
        
        return angle_deg

    def euler_from_quaternion(x, y, z, w):
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)

        return roll_x, pitch_y, yaw_z # in radians
    def get_euler_angles_from_quaternion(self,q, degrees=True):
        """
        (w, x, y, z) 쿼터니언에서 x, y, z 각 축의 회전 각도(Roll, Pitch, Yaw)를 계산합니다.
        이 공식은 Z-Y-X 순서의 오일러 각을 가정합니다.
        
        Args:
            q (tuple or list): 4개의 원소를 가진 쿼터니언 (w, x, y, z).
            degrees (bool): 각도를 도로 반환할지 여부. 기본값은 True.
        
        Returns:
            tuple: (roll, pitch, yaw) 튜플.
        """
        w = q[0]
        x = q[1]
        y = q[2]
        z = q[3]

        
        # 쿼터니언 정규화 (안정성 확보)
        norm = np.sqrt(w**2 + x**2 + y**2 + z**2)
        if norm == 0:
            return (0.0, 0.0, 0.0)
            
        w, x, y, z = w / norm, x / norm, y / norm, z / norm

        # Roll (x축 회전) 계산
        t0 = 2.0 * (w * x + y * z)
        t1 = 1.0 - 2.0 * (x**2 + y**2)
        roll_rad = math.atan2(t0, t1)
        




        # Pitch (y축 회전) 계산
        t2 = 2.0 * (w * y - z * x)
        # 짐벌 잠금 방지: t2 값이 -1.0과 1.0 사이로 벗어날 경우를 처리
        t2 = 1.0 if t2 > 1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_rad = math.asin(t2)
        




        # Yaw (z축 회전) 계산
        t3 = 2.0 * (w * z + x * y)
        t4 = 1.0 - 2.0 * (y**2 + z**2)
        yaw_rad = math.atan2(t3, t4)



        
        if degrees:
            roll = np.degrees(roll_rad)
            pitch = np.degrees(pitch_rad)
            yaw = np.degrees(yaw_rad)
            
            # 0~360도 범위로 조정 (선택적)
            # roll = roll % 360
            # pitch = pitch % 360
            # yaw = yaw % 360
            
            return (roll, pitch, yaw)
        else:
            return (roll_rad, pitch_rad, yaw_rad)
    def training_step(self, data_dict, idx):
        output_dict = self(data_dict)

        #print(output_dict['pred_noise'].shape)
        #print(output_dict['pred_noise'][0,0,3:].detach().cpu().numpy())
        #print(type(output_dict['pred_noise'][0,0,3:].detach().cpu().numpy()))
        for b in range(output_dict['pred_noise'].shape[0]):
            _di = {}
            for p in range(output_dict['pred_noise'].shape[1]):
                #print(self.get_euler_angles_from_quaternion(output_dict['pred_noise'][b,p,3:].detach().cpu().numpy()))
                _di[f'360_{b}_{p}_roll'] = self.get_euler_angles_from_quaternion(output_dict['pred_noise'][b,p,3:].detach().cpu().numpy())[0]
                _di[f'360_{b}_{p}_pitch'] = self.get_euler_angles_from_quaternion(output_dict['pred_noise'][b,p,3:].detach().cpu().numpy())[1]
                _di[f'360_{b}_{p}_yaw'] = self.get_euler_angles_from_quaternion(output_dict['pred_noise'][b,p,3:].detach().cpu().numpy())[2]
            #tensor_from_dict = torch.tensor(list(_di.values()))
            #print(_di)
            self.log_dict( _di, on_step=True, on_epoch=False)

        for b in range(output_dict['pred_noise'].shape[0]):
            _di = {}
            for p in range(output_dict['pred_noise'].shape[1]):
                _di[f'q_{b}_{p}_w'] = output_dict['pred_noise'][b,p,3]
                _di[f'q_{b}_{p}_x'] = output_dict['pred_noise'][b,p,4]
                _di[f'q_{b}_{p}_y'] = output_dict['pred_noise'][b,p,5]
                _di[f'q_{b}_{p}_z'] = output_dict['pred_noise'][b,p,6]
            #tensor_from_dict = torch.tensor(list(_di.values()))
            #print(_di)
            self.log_dict( _di, on_step=True, on_epoch=False)
        
        loss_dict = self._loss(data_dict, output_dict)
        
        total_loss = 0
        for loss_name, loss_value in loss_dict.items():
            total_loss += loss_value
            self.log(f"train_loss/{loss_name}", loss_value, on_step=True, on_epoch=False)
        self.log(f"train_loss/total_loss", total_loss, on_step=True, on_epoch=False)
        
        return total_loss
    

    def _calc_val_loss(self, data_dict):
        output_dict = self(data_dict)
        loss_dict = self._loss(data_dict, output_dict)
        # calculate the total loss and logs
        total_loss = 0
        for loss_name, loss_value in loss_dict.items():
            total_loss += loss_value
            self.log(f"val_loss/{loss_name}", loss_value, on_step=False, on_epoch=True)
        self.log(f"val_loss/total_loss", total_loss, on_step=False, on_epoch=True)
                


        

    def validation_step(self, data_dict, idx):
        self._calc_val_loss(data_dict)
        
        gt_trans = data_dict['part_trans']
        gt_rots = data_dict['part_rots']
        gt_trans_and_rots = torch.cat([gt_trans, gt_rots], dim=-1)
        noisy_trans_and_rots = torch.randn(gt_trans_and_rots.shape, device=self.device)
        ref_part = data_dict["ref_part"]        

        reference_gt_and_rots = torch.zeros_like(gt_trans_and_rots, device=self.device)
        reference_gt_and_rots[ref_part] = gt_trans_and_rots[ref_part]

        noisy_trans_and_rots[ref_part] = reference_gt_and_rots[ref_part]

        if self.rotation_1d:
            reference_gt_and_rots[...,4] = torch.repeat_interleave(torch.Tensor([0]), reference_gt_and_rots.shape[-2], dim=0).unsqueeze(dim=0)
            reference_gt_and_rots[...,5] = torch.repeat_interleave(torch.Tensor([1]), reference_gt_and_rots.shape[-2], dim=0).unsqueeze(dim=0)
            reference_gt_and_rots[...,6] = torch.repeat_interleave(torch.Tensor([0]), reference_gt_and_rots.shape[-2], dim=0).unsqueeze(dim=0)
            noisy_trans_and_rots[...,4] = torch.repeat_interleave(torch.Tensor([0]), noisy_trans_and_rots.shape[-2], dim=0).unsqueeze(dim=0)
            noisy_trans_and_rots[...,5] = torch.repeat_interleave(torch.Tensor([1]), noisy_trans_and_rots.shape[-2], dim=0).unsqueeze(dim=0)
            noisy_trans_and_rots[...,6] = torch.repeat_interleave(torch.Tensor([0]), noisy_trans_and_rots.shape[-2], dim=0).unsqueeze(dim=0)


        part_valids = data_dict['part_valids'].clone()
        part_scale = data_dict["part_scale"].clone()
        part_pcs = data_dict["part_pcs"].clone()


        for t in self.noise_scheduler.timesteps:
            timesteps = t.reshape(-1).repeat(len(noisy_trans_and_rots)).cuda()
            latent, xyz = self._extract_features(part_pcs, part_valids, noisy_trans_and_rots)
            pred_noise = self.denoiser(
                noisy_trans_and_rots, 
                timesteps,
                latent,
                xyz,
                part_valids,
                part_scale,
                ref_part
            )
            if self.rotation_1d:
                pred_noise[...,4] = torch.repeat_interleave(torch.Tensor([0]), pred_noise.shape[-2], dim=0).unsqueeze(dim=0)
                pred_noise[...,5] = torch.repeat_interleave(torch.Tensor([1]), pred_noise.shape[-2], dim=0).unsqueeze(dim=0)
                pred_noise[...,6] = torch.repeat_interleave(torch.Tensor([0]), pred_noise.shape[-2], dim=0).unsqueeze(dim=0)

            noisy_trans_and_rots = self.noise_scheduler.step(pred_noise, t, noisy_trans_and_rots).prev_sample
            noisy_trans_and_rots[ref_part] = reference_gt_and_rots[ref_part]  
            if self.rotation_1d:
                noisy_trans_and_rots[...,4] = torch.repeat_interleave(torch.Tensor([0]), noisy_trans_and_rots.shape[-2], dim=0).unsqueeze(dim=0)
                noisy_trans_and_rots[...,5] = torch.repeat_interleave(torch.Tensor([1]), noisy_trans_and_rots.shape[-2], dim=0).unsqueeze(dim=0)
                noisy_trans_and_rots[...,6] = torch.repeat_interleave(torch.Tensor([0]), noisy_trans_and_rots.shape[-2], dim=0).unsqueeze(dim=0)

        pts = data_dict['part_pcs']
        pred_trans = noisy_trans_and_rots[..., :3]
        pred_rots = noisy_trans_and_rots[..., 3:]

        expanded_part_scale = data_dict["part_scale"].unsqueeze(-1).expand(-1, -1, 1000, -1)
        pts = pts * expanded_part_scale

        acc, _, _ = calc_part_acc(pts, trans1=pred_trans, trans2=gt_trans,
                            rot1=pred_rots, rot2=gt_rots, valids=data_dict['part_valids'], 
                            chamfer_distance=self.metric)
        
        shape_cd = calc_shape_cd(pts, trans1=pred_trans, trans2=gt_trans,
                            rot1=pred_rots, rot2=gt_rots, valids=data_dict['part_valids'], 
                            chamfer_distance=self.metric)
        
        rmse_r = rot_metrics(pred_rots, gt_rots, data_dict['part_valids'], 'rmse')
        rmse_t = trans_metrics(pred_trans, gt_trans,  data_dict['part_valids'], 'rmse')
        
        self.acc_list.append(acc)
        self.rmse_r_list.append(rmse_r)
        self.rmse_t_list.append(rmse_t)
        self.cd_list.append(shape_cd)


    def on_validation_epoch_end(self):
        total_acc = torch.mean(torch.cat(self.acc_list))
        total_rmse_t = torch.mean(torch.cat(self.rmse_t_list))
        total_rmse_r = torch.mean(torch.cat(self.rmse_r_list))
        total_shape_cd = torch.mean(torch.cat(self.cd_list))
        
        self.log(f"eval/part_acc", total_acc, sync_dist=True)
        self.log(f"eval/rmse_t", total_rmse_t, sync_dist=True)
        self.log(f"eval/rmse_r", total_rmse_r, sync_dist=True)
        self.log(f"eval/shape_cd", total_shape_cd, sync_dist=True)
        self.acc_list = []
        self.rmse_t_list = []
        self.rmse_r_list = []
        self.cd_list = []
        return total_acc, total_rmse_t, total_rmse_r, total_shape_cd
    

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=2e-4,
            betas=(0.95, 0.999),
            weight_decay=1e-6,
            eps=1e-08,
        )
        lr_scheduler = hydra.utils.instantiate(self.cfg.model.lr_scheduler, optimizer=optimizer)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
