import torch.nn as nn
import torch
from torch.nn import functional as F
from utils.model_utils import (
    PositionalEncoding,
    EmbedderNerf

from puzzlefusion_plusplus.denoiser.evaluation.transform import (
    transform_pc,
    quaternion_to_euler,
    quaternion_to_matrix,
    rotate_y_axis,
    y_axis_rotation_quaternion_from_rad
)
from puzzlefusion_plusplus.denoiser.model.modules.attention import EncoderLayer
from scipy.spatial.transform import Rotation as R
import numpy as np
class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func
    
    def forward(self, x):
        return self.func(x)

class DenoiserTransformer(nn.Module):

    def __init__(self, cfg):
        super(DenoiserTransformer, self).__init__()
        self.cfg = cfg

        self.model_channels = cfg.model.embed_dim
        self.out_channels = cfg.model.out_channels
        self.num_layers = cfg.model.num_layers
        self.num_heads = cfg.model.num_heads
        self.ref_part_emb = nn.Embedding(2, cfg.model.embed_dim)
        self.activation = nn.SiLU()

        num_embeds_ada_norm = 6 * self.model_channels

        self.transformer_layers = nn.ModuleList([
            EncoderLayer(
                dim=self.model_channels,
                num_attention_heads=self.num_heads,
                attention_head_dim=self.model_channels // self.num_heads,
                dropout=0.2,
                activation_fn='geglu',
                num_embeds_ada_norm=num_embeds_ada_norm, 
                attention_bias=False,
                norm_elementwise_affine=True,
                final_dropout=False,
            )
            for _ in range(self.num_layers)
        ])

        multires = 10
        embed_kwargs = {
            'include_input': True,
            'input_dims': 7,
            'max_freq_log2': multires - 1,
            'num_freqs': multires,
            'log_sampling': True,
            'periodic_fns': [torch.sin, torch.cos],
        }
        
        embedder_obj = EmbedderNerf(**embed_kwargs)
        self.param_embedding = lambda x, eo=embedder_obj: eo.embed(x)

        embed_pos_kwargs = {
            'include_input': True,
            'input_dims': 3,
            'max_freq_log2': multires - 1,
            'num_freqs': multires,
            'log_sampling': True,
            'periodic_fns': [torch.sin, torch.cos],
        }
        embedder_pos = EmbedderNerf(**embed_pos_kwargs)
        # Pos embedding for positions of points xyz
        self.pos_embedding = lambda x, eo=embedder_pos: eo.embed(x)

        embed_scale_kwargs = {
            'include_input': True,
            'input_dims': 1,
            'max_freq_log2': multires - 1,
            'num_freqs': multires,
            'log_sampling': True,
            'periodic_fns': [torch.sin, torch.cos],
        }
        embedder_scale = EmbedderNerf(**embed_scale_kwargs)
        self.scale_embedding = lambda x, eo=embedder_scale: eo.embed(x)

        self.shape_embedding = nn.Linear(
            cfg.model.num_dim + embedder_scale.out_dim + embedder_pos.out_dim, 
            self.model_channels
        )

        self.param_fc = nn.Linear(embedder_obj.out_dim, self.model_channels)

        # Pos encoding for indicating the sequence. 
        self.pos_encoding = PositionalEncoding(self.model_channels)

        # mlp out for translation N, 256 -> N, 3
        self.mlp_out_trans = nn.Sequential(
            nn.Linear(self.model_channels, self.model_channels),
            nn.SiLU(),
            nn.Linear(self.model_channels, self.model_channels // 2),
            nn.SiLU(),
            nn.Linear(self.model_channels // 2, 3),
        )

        #self.hardtanh = nn.Hardtanh(min_val=-300, max_val=300)
        # mlp out for rotation N, 256 -> N, 4
        self.mlp_out_rot = nn.Sequential(
            nn.Linear(self.model_channels, self.model_channels),
            nn.SiLU(),
            nn.Linear(self.model_channels, self.model_channels // 2),
            nn.SiLU(),
            nn.Linear(self.model_channels // 2, 1) ,
            nn.Tanh(),
            Lambda(lambda x: x * torch.pi * 2)

            #nn.Hardtanh(min_val=-300, max_val=300)
              # jhahn
        )


    # def _gen_mask(self, L, N, B, mask):
    #     self_block = torch.ones(L, L, device=mask.device)  # Each L points should talk to each other
    #     self_mask = torch.block_diag(*([self_block] * N))  # Create block diagonal tensor
    #     self_mask = self_mask.unsqueeze(0).repeat(B, 1, 1)  # Expand dimensions to [B, N*L, N*L]

    #     flattened_mask = mask.unsqueeze(-1).repeat(1, 1, L).flatten(1, 2)  # shape [B, N*L]
    #     flattened_mask = flattened_mask.unsqueeze(1)  # shape [B, 1, N*L]
    #     gen_mask = flattened_mask * flattened_mask.transpose(-1, -2)  # shape [B, N*L, N*L]
    #     return self_mask, gen_mask
    

    def _gen_cond(self, x, xyz, latent, scale):

        x = x.flatten(0, 1)  # (B*N, 7)

        xyz = xyz.flatten(0, 1)  # (B*N, L, 3)

        latent = latent.flatten(0, 1)  # (B*N, L, 64)

        scale = scale.flatten(0, 1)  # (B*N, 1)
        scale_emb = self.scale_embedding(scale).unsqueeze(1) # (B*N, 1, C)
        scale_emb = scale_emb.repeat(1, latent.shape[1], 1) # (B*N, L, C)

        xyz_pos_emb = self.pos_embedding(xyz)

        latent = torch.cat((latent, xyz_pos_emb, scale_emb), dim=-1)
        shape_emb = self.shape_embedding(latent)

        x_emb = self.param_fc(self.param_embedding(x))
        return x_emb, shape_emb
    @staticmethod
    def y_axis_rotation_quaternion_from_rad(angle_radians):
        """
        y축 회전각(라디안)에 대한 쿼터니언 텐서를 생성합니다.
        
        Args:
            angle_radians (torch.Tensor): 회전 각도를 담은 스칼라 또는 배치 텐서.
                                        단위는 라디안.
        Returns:
            torch.Tensor: [x, y, z, w] 형식을 따르는 쿼터니언 텐서.
        """
        # 1. 반각(half-angle) 계산
        half_angle = angle_radians / 2.0
        
        # 2. 사인, 코사인 값 계산
        sin_half_angle = torch.sin(half_angle)
        cos_half_angle = torch.cos(half_angle)

        # 3. 차원 확장 및 쿼터니언 텐서 생성
        #    .unsqueeze(-1)로 차원을 확장하여 배치 처리 가능하게 함
        sin_half_angle = sin_half_angle.unsqueeze(-1)
        cos_half_angle = cos_half_angle.unsqueeze(-1)
        
        # 쿼터니언의 x, z 성분은 0이므로, 0으로 채워진 텐서를 생성
        zeros = torch.zeros_like(sin_half_angle)
        
        # [x, y, z, w] 순서로 텐서를 결합 (concatenate)
        #quaternion = torch.cat([zeros, sin_half_angle, zeros, cos_half_angle], dim=-1)
        quaternion = torch.cat([cos_half_angle, zeros, sin_half_angle, zeros ], dim=-1)
        
        return quaternion
    def _out(self, data_emb, B, N, L):
        out = data_emb.reshape(B, N, L, self.model_channels)

        # Avg pooling
        out = out.mean(dim=2)

        trans = self.mlp_out_trans(out)
        rots = self.mlp_out_rot(out)

        #rots = self.angle_2_quaternion(rots)
        rots = y_axis_rotation_quaternion_from_rad(rots)
        rots = torch.squeeze(rots, dim=2)

        #print('rots',rots.shape)
        # tanh 함수를 적용하여 출력 범위를 (-1, 1)로 제한
        #tanh_output = self.tanh(rots)
        
        # (-1, 1) 범위의 값을 원하는 범위(-300, 300)로 스케일링
        #scaled_output = tanh_output * 300

        #print(rots.shape)
        #print(rots[...,3])
        #print(rots[...,3].shape)
        #print('@@',torch.min(rots[...,3], axis=0))
        '''
        y_min = torch.min(torch.min(rots[...,3], axis=0)[0], axis=0)[0]
        y_max = torch.max(torch.max(rots[...,3], axis=0)[0], axis=0)[0]
        #print('y_min',y_min)
        #print('y_max',y_max)
        # Avoid division by zero if y_max and y_min are equal
        if y_max != y_min:
            rots[...,3] = ((rots[...,3] - y_min) / (y_max - y_min))*100
        else:
            rots[...,3] = rots[...,3]  # or some default value if all outputs are the same
        '''

        

        return torch.cat([trans, rots], dim=-1)


    def _add_ref_part_emb(self, B, x_emb, ref_part):
        x_emb = x_emb.reshape(B, -1, self.model_channels)
        ref_part_emb = self.ref_part_emb.weight[0].repeat(B, x_emb.shape[1], 1)
        ref_part_emb[ref_part.to(torch.bool)] = self.ref_part_emb.weight[1]

        x_emb = x_emb + ref_part_emb
        return x_emb.reshape(-1, self.model_channels)

    def _gen_mask(self, B, N, L, part_valids):
        self_block = torch.ones(L, L, device=part_valids.device)  # Each L points should talk to each other
        self_mask = torch.block_diag(*([self_block] * N))  # Create block diagonal tensor
        self_mask = self_mask.unsqueeze(0).repeat(B, 1, 1)  # Expand dimensions to [B, N*L, N*L]
        self_mask = self_mask.to(torch.bool)
        gen_mask = part_valids.unsqueeze(-1).repeat(1, 1, L).flatten(1, 2)
        gen_mask = gen_mask.to(torch.bool)
        
        return self_mask, gen_mask


    def forward(self, x, timesteps, latent, xyz, part_valids, scale, ref_part):

        B, N, L, _ = latent.shape

        x_emb, shape_emb = self._gen_cond(x, xyz, latent, scale)

        x_emb = self._add_ref_part_emb(B, x_emb, ref_part)

        x_emb = x_emb.reshape(B, N, 1, -1)
        x_emb = x_emb.repeat(1, 1, L, 1)
        condition_emb = shape_emb.reshape(B, N*L, -1) 

        # B, N*L, C
        data_emb = x_emb.reshape(B, N*L, -1) 

        data_emb = data_emb + condition_emb
        data_emb = self.pos_encoding(data_emb.reshape(B, N, L, -1)).reshape(B, N*L, -1)


        self_mask, gen_mask = self._gen_mask(B, N, L, part_valids)


        for i, layer in enumerate(self.transformer_layers):
            data_emb = layer(
                hidden_states=data_emb, 
                self_mask=self_mask,
                gen_mask=gen_mask,
                timestep=timesteps 
            )

        # data_emb (B, N*L, C)
        out_dec = self._out(data_emb, B, N, L)

        return out_dec

