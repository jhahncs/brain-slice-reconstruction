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
#from pytorch3d import transforms
from puzzlefusion_plusplus.denoiser.evaluation.transform import (
    transform_pc,
    quaternion_to_euler,
    quaternion_to_matrix,
    rotate_y_axis,
    get_euler_angles_from_quaternion,
    y_axis_rotation_quaternion_from_rad,
    get_euler_angles_from_quaternion_gpu
)

from pytorch3d.transforms.transform3d import (
    Rotate,
    RotateAxisAngle,
    Scale,
    Transform3d,
    Translate,
)
from pytorch3d import transforms
import torch.nn as nn

# torch.nn.Module을 상속하여 사용자 정의 손실 함수 구현
class SinCosLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, pred_angle: torch.Tensor, true_angle: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_angle (torch.Tensor): 모델이 예측한 회전 각도 (단위: 라디안)
            true_angle (torch.Tensor): 실제 정답 회전 각도 (단위: 라디안)
        
        Returns:
            torch.Tensor: 삼각함수 변환 기반의 MSE 손실 값
        """
        # 각도를 라디안으로 변환 (만약 입력이 '도'라면)
        # pred_rad = pred_angle * math.pi / 180.0
        # true_rad = true_angle * math.pi / 180.0
        
        # 실제 각도를 이용해 sin과 cos 값 계산
        true_sin = torch.sin(true_angle)
        true_cos = torch.cos(true_angle)

        # 예측 각도를 이용해 sin과 cos 값 계산
        pred_sin = torch.sin(pred_angle)
        pred_cos = torch.cos(pred_angle)

        # 예측된 (cos, sin)과 실제 (cos, sin) 사이의 MSE 손실 계산
        loss_cos = self.mse_loss(pred_cos, true_cos)
        loss_sin = self.mse_loss(pred_sin, true_sin)
        
        # 두 손실의 합계를 반환
        total_loss = loss_cos + loss_sin
        return total_loss


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

        self.loss_fn_rots = SinCosLoss()
    # y-axis rotation version
    def _apply_rots(self, part_pcs, noise_params):
        """
        Apply Noisy rotations to all points
        """
        noise_quat = noise_params[..., 3:]
        noise_quat = noise_quat / noise_quat.norm(dim=-1, keepdim=True)

        #part_pcs = transforms.quaternion_apply(noise_quat.unsqueeze(2), part_pcs)

        
        part_pcs = rotate_y_axis(noise_quat, part_pcs)
        

        
        return part_pcs
    
    def _apply_rots_xyz(self, part_pcs, noise_params):
        """
        Apply Noisy rotations to all points
        """
        noise_quat = noise_params[..., 3:]
        noise_quat = noise_quat / noise_quat.norm(dim=-1, keepdim=True)
        part_pcs = transforms.quaternion_apply(noise_quat.unsqueeze(2), part_pcs)
        
        return part_pcs


    def _extract_features(self, part_pcs, part_valids, noisy_trans_and_rots):
        B, P , _, _ = part_pcs.shape
        if self.rotation_1d:
            part_pcs = self._apply_rots(part_pcs, noisy_trans_and_rots)
        else:
            part_pcs = self._apply_rots_xyz(part_pcs, noisy_trans_and_rots)

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
        #print('gt_trans',gt_trans.shape,gt_rots.shape)
        gt_trans_and_rots = torch.cat([gt_trans, gt_rots], dim=-1)
        ref_part = data_dict["ref_part"]
        
        #noise_trans = torch.randn(gt_trans.shape, device=self.device)
        #noise_rotate = self.random_rotation_tensor(gt_rots)
        #noise = torch.cat([noise_trans, noise_rotate], dim=-1)

        noise = torch.randn(gt_trans_and_rots.shape, device=self.device)

        
        if self.rotation_1d:
            for _i in [4,6]: # 4, 6
                noise[...,_i] = torch.repeat_interleave(torch.Tensor([0]), noise.shape[-2], dim=0).unsqueeze(dim=0)
            #noise[...,3] = torch.repeat_interleave(torch.Tensor([1]), noise.shape[-2], dim=0).unsqueeze(dim=0)

        
        B, P, N, C = data_dict["part_pcs"].shape

        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (B,),
                                  device=self.device).long()
        
        noisy_trans_and_rots = self.noise_scheduler.add_noise(gt_trans_and_rots, noise, timesteps)

        #print('gt_trans_and_rots',gt_trans_and_rots[0,0:2,:])
        #print('noise',noise[0,0:2,:])
        
        noisy_trans_and_rots[ref_part] = gt_trans_and_rots[ref_part]
        if self.rotation_1d:
            for _i in [4,6]:
                noisy_trans_and_rots[...,_i] = torch.repeat_interleave(torch.Tensor([0]), noisy_trans_and_rots.shape[-2], dim=0).unsqueeze(dim=0)
            #noisy_trans_and_rots[...,3] = torch.repeat_interleave(torch.Tensor([1]), noisy_trans_and_rots.shape[-2], dim=0).unsqueeze(dim=0)
        '''
        if self.rotation_1d:
            noisy_trans_and_rots[...,4] = torch.repeat_interleave(torch.Tensor([0]), noisy_trans_and_rots.shape[-2], dim=0).unsqueeze(dim=0)
            noisy_trans_and_rots[...,5] = torch.repeat_interleave(torch.Tensor([1]), noisy_trans_and_rots.shape[-2], dim=0).unsqueeze(dim=0)
            noisy_trans_and_rots[...,6] = torch.repeat_interleave(torch.Tensor([0]), noisy_trans_and_rots.shape[-2], dim=0).unsqueeze(dim=0)
        '''

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
        #print('forward',"\n",gt_trans_and_rots[0,0:2,3:],"\n",noise[0,0:2,3:],"\n",noisy_trans_and_rots[0,0:2,3:],"\n",pred_noise[0,0:2,3:])
        '''
        for i in range(1,5):
            gt = gt_trans_and_rots[0,i,3:].detach().cpu().numpy()
            pred = pred_noise[0,i,3:].detach().cpu().numpy()
            print(f'vvv,{i},({gt[0]:.3f},{gt[1]:.3f},{gt[2]:.3f}),({pred[0]:.3f},{pred[1]:.3f},{pred[2]:.3f})')
            print(f'360,{i},{get_euler_angles_from_quaternion(gt_trans_and_rots[0,i,3:].detach().cpu().numpy())[1]:.3f},{get_euler_angles_from_quaternion(pred_noise[0,i,3:].detach().cpu().numpy())[1]:.3f}')
        '''

        
        

        
        if self.rotation_1d:
            for _i in [4,6]: # 4, 6
                pred_noise[...,_i] = torch.repeat_interleave(torch.Tensor([0]), pred_noise.shape[-2], dim=0).unsqueeze(dim=0)        
            #pred_noise[...,3] = torch.repeat_interleave(torch.Tensor([1]), pred_noise.shape[-2], dim=0).unsqueeze(dim=0)
        #pred_noise[ref_part]  = gt_trans_and_rots[ref_part]

        output_dict = {
            'pred_noise': pred_noise,
            'gt_noise': noise
        }


        
        for b in range(output_dict['pred_noise'].shape[0]):
            _di = {}
            for p in [2]:
                #print(self.get_euler_angles_from_quaternion(output_dict['pred_noise'][b,p,3:].detach().cpu().numpy()))
                _di[f'360_{b}_{p}_roll'] = get_euler_angles_from_quaternion(output_dict['pred_noise'][b,p,3:].detach().cpu().numpy())[0]
                _di[f'360_{b}_{p}_pitch'] = get_euler_angles_from_quaternion(output_dict['pred_noise'][b,p,3:].detach().cpu().numpy())[1]
                _di[f'360_{b}_{p}_yaw'] = get_euler_angles_from_quaternion(output_dict['pred_noise'][b,p,3:].detach().cpu().numpy())[2]

                _di[f'gt360_{b}_{p}_roll'] = get_euler_angles_from_quaternion(output_dict['gt_noise'][b,p,3:].detach().cpu().numpy())[0]
                _di[f'gt360_{b}_{p}_pitch'] = get_euler_angles_from_quaternion(output_dict['gt_noise'][b,p,3:].detach().cpu().numpy())[1]
                _di[f'gt360_{b}_{p}_yaw'] = get_euler_angles_from_quaternion(output_dict['gt_noise'][b,p,3:].detach().cpu().numpy())[2]
                #gt360 = _di[f'gt360_{b}_{p}_pitch']
                #a360 = _di[f'360_{b}_{p}_pitch']

                #print(f'gt360_{b}_{p}_pitch : {gt360:.3f}')
                #print(f'360_{b}_{p}_pitch : {a360:.3f}')
            #tensor_from_dict = torch.tensor(list(_di.values()))
            #print(_di)
            self.log_dict( _di, on_step=True, on_epoch=False)
            if True:
                break

        for b in range(output_dict['pred_noise'].shape[0]):
            _di = {}
            for p in [2]:
                _di[f'q_{b}_{p}_w'] = output_dict['pred_noise'][b,p,3]
                _di[f'q_{b}_{p}_x'] = output_dict['pred_noise'][b,p,4]
                _di[f'q_{b}_{p}_y'] = output_dict['pred_noise'][b,p,5]
                _di[f'q_{b}_{p}_z'] = output_dict['pred_noise'][b,p,6]

                _di[f'gtq_{b}_{p}_w'] = output_dict['gt_noise'][b,p,3]
                _di[f'gtq_{b}_{p}_x'] = output_dict['gt_noise'][b,p,4]
                _di[f'gtq_{b}_{p}_y'] = output_dict['gt_noise'][b,p,5]
                _di[f'gtq_{b}_{p}_z'] = output_dict['gt_noise'][b,p,6]
                #print(f'gtq_{b}_{p}_w : {output_dict["gt_noise"][b,p,3]:.3f}')
                #print(f'q_{b}_{p}_w : {output_dict["pred_noise"][b,p,3]:.3f}')
                #print(f'gtq_{b}_{p}_y : {output_dict["gt_noise"][b,p,5]:.3f}')
                #print(f'q_{b}_{p}_y : {output_dict["pred_noise"][b,p,5]:.3f}')

            #tensor_from_dict = torch.tensor(list(_di.values()))
            #print(_di)
            self.log_dict( _di, on_step=True, on_epoch=False)
            if True:
                break

        

        return output_dict

    def _loss(self, data_dict, output_dict):
        pred_noise = output_dict['pred_noise']
        part_valids = data_dict['part_valids'].bool()
        noise = output_dict['gt_noise']

        part_valids[data_dict["ref_part"]] = False
        mse_loss_trans = F.mse_loss(pred_noise[part_valids][...,:3], noise[part_valids][...,:3])
        #print("@@@@@@@@@@@@@@@@@@@@")
        #print(pred_noise[part_valids][...,3:].shape)

        #print('pred_noise',pred_noise[:3,3:])
        #print('noise',noise[:3,3:])
        #angles_pred = get_euler_angles_from_quaternion_gpu(pred_noise[part_valids][...,3:])
        #angles_noise = get_euler_angles_from_quaternion_gpu(noise[part_valids][...,3:])

        #angles_pred[...,0] = 0
        #angles_pred[...,2] = 0
        #angles_noise[...,0] = 0
        #angles_noise[...,2] = 0
        #print('angles_pred',angles_pred[:3,:])
        #print('angles_noise',angles_noise[:3,:])
        #mse_loss_rots = F.mse_loss(angles_pred/360,angles_noise/360)
        #mse_loss_rots = self.loss_fn_rots(angles_pred,angles_noise)
        mse_loss_rots = F.mse_loss(pred_noise[part_valids][...,3:], noise[part_valids][...,3:])
        #print(mse_loss_rots)


        return {'mse_loss_trans': mse_loss_trans,'mse_loss_rots': mse_loss_rots}

        
        
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
  
    
    def training_step(self, data_dict, idx):
        output_dict = self(data_dict)
        #noise = output_dict['gt_noise']

        #print(output_dict['pred_noise'].shape)
        #print(output_dict['pred_noise'][0,0,3:].detach().cpu().numpy())
        #print(type(output_dict['pred_noise'][0,0,3:].detach().cpu().numpy()))

        
        loss_dict = self._loss(data_dict, output_dict)
        
        total_loss = 0
        weight = 1
        for loss_name, loss_value in loss_dict.items():
            if self.rotation_1d:
                if loss_name == 'mse_loss_rots':
                    weight = idx*0.1
                else:
                    weight = 1
            total_loss += (weight*loss_value)
            self.log(f"train_loss/{loss_name}", loss_value, on_step=True, on_epoch=False)
        self.log(f"train_loss/total_loss", total_loss, on_step=True, on_epoch=False)
        
        return total_loss
    

    def _calc_val_loss(self, data_dict):
        output_dict = self(data_dict)
        loss_dict = self._loss(data_dict, output_dict)
        # calculate the total loss and logs
        total_loss = 0
        weight = 0
        for loss_name, loss_value in loss_dict.items():
            if self.rotation_1d:
                if loss_name == 'mse_loss_rots':
                    weight = 2
                else:
                    weight = 1
            total_loss += (weight*loss_value)
            self.log(f"val_loss/{loss_name}", loss_value, on_step=False, on_epoch=True)
        self.log(f"val_loss/total_loss", total_loss, on_step=False, on_epoch=True)
                




    def validation_step(self, data_dict, idx):
        self._calc_val_loss(data_dict)
        
        gt_trans = data_dict['part_trans']
        gt_rots = data_dict['part_rots']
        gt_trans_and_rots = torch.cat([gt_trans, gt_rots], dim=-1)



        #noise_trans = torch.randn(gt_trans.shape, device=self.device)
        #noise_rotate = self.random_rotation_tensor(gt_rots)
        #noisy_trans_and_rots = torch.cat([noise_trans, noise_rotate], dim=-1)

        noisy_trans_and_rots = torch.randn(gt_trans_and_rots.shape, device=self.device)
        
        if self.rotation_1d:
            noisy_trans_and_rots[...,4] = torch.repeat_interleave(torch.Tensor([0]), noisy_trans_and_rots.shape[-2], dim=0).unsqueeze(dim=0)
            noisy_trans_and_rots[...,6] = torch.repeat_interleave(torch.Tensor([0]), noisy_trans_and_rots.shape[-2], dim=0).unsqueeze(dim=0)
        

        ref_part = data_dict["ref_part"]        


        reference_gt_and_rots = torch.zeros_like(gt_trans_and_rots, device=self.device)
        reference_gt_and_rots[ref_part] = gt_trans_and_rots[ref_part]

        noisy_trans_and_rots[ref_part] = reference_gt_and_rots[ref_part]

        '''
        if self.rotation_1d:
            reference_gt_and_rots[...,4] = torch.repeat_interleave(torch.Tensor([0]), reference_gt_and_rots.shape[-2], dim=0).unsqueeze(dim=0)
            reference_gt_and_rots[...,5] = torch.repeat_interleave(torch.Tensor([1]), reference_gt_and_rots.shape[-2], dim=0).unsqueeze(dim=0)
            reference_gt_and_rots[...,6] = torch.repeat_interleave(torch.Tensor([0]), reference_gt_and_rots.shape[-2], dim=0).unsqueeze(dim=0)
            noisy_trans_and_rots[...,4] = torch.repeat_interleave(torch.Tensor([0]), noisy_trans_and_rots.shape[-2], dim=0).unsqueeze(dim=0)
            noisy_trans_and_rots[...,5] = torch.repeat_interleave(torch.Tensor([1]), noisy_trans_and_rots.shape[-2], dim=0).unsqueeze(dim=0)
            noisy_trans_and_rots[...,6] = torch.repeat_interleave(torch.Tensor([0]), noisy_trans_and_rots.shape[-2], dim=0).unsqueeze(dim=0)

        '''

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
            #print('pred_noise',pred_noise[0,0,3:])
            
            if self.rotation_1d:
                pred_noise[...,4] = torch.repeat_interleave(torch.Tensor([0]), pred_noise.shape[-2], dim=0).unsqueeze(dim=0)
                #pred_noise[...,5] = torch.repeat_interleave(torch.Tensor([1]), pred_noise.shape[-2], dim=0).unsqueeze(dim=0)
                pred_noise[...,6] = torch.repeat_interleave(torch.Tensor([0]), pred_noise.shape[-2], dim=0).unsqueeze(dim=0)
            
            #noisy_trans_and_rots_before = noisy_trans_and_rots.clone()
            #print('noisy_trans_and_rots',noisy_trans_and_rots[0,0,3:])
            noisy_trans_and_rots = self.noise_scheduler.step(pred_noise, t, noisy_trans_and_rots).prev_sample
            noisy_trans_and_rots[ref_part] = reference_gt_and_rots[ref_part]  
            #print('step',pred_noise[0,:3,3:], noisy_trans_and_rots_before[0,:3,3:], noisy_trans_and_rots[0,0:3,3:])
            
            if self.rotation_1d:
                noisy_trans_and_rots[...,4] = torch.repeat_interleave(torch.Tensor([0]), noisy_trans_and_rots.shape[-2], dim=0).unsqueeze(dim=0)
                #noisy_trans_and_rots[...,5] = torch.repeat_interleave(torch.Tensor([1]), noisy_trans_and_rots.shape[-2], dim=0).unsqueeze(dim=0)
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
