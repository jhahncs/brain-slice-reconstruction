from pytorch3d.transforms import quaternion_apply
from pytorch3d import transforms
import torch
import numpy as np
from pytorch3d.transforms.transform3d import (
    Rotate,
    RotateAxisAngle,
    Scale,
    Transform3d,
    Translate,
)
import math

rotation_1d = True

def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    quaternions = quaternions / quaternions.norm(dim=-1, keepdim=True)
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

# y-axis rotation version
def qrot(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
        where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    # repeat to e.g. apply the same quat for all points in a point cloud
    # [4] --> [N, 4], [B, 4] --> [B, N, 4], [B, P, 4] --> [B, P, N, 4]

    if rotation_1d:
        q = rotate_y_axis(q,v)
        return q
    else:
        if len(q.shape) == len(v.shape) - 1:
            q = q.unsqueeze(-2).repeat_interleave(v.shape[-2], dim=-2)
        assert q.shape[:-1] == v.shape[:-1]
        return quaternion_apply(q, v)


def qtransform(t, q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q,
        and then translate it by the translation described by t.
    Expects a tensor of shape (*, 3) for t, a tensor of shape (*, 4) for q and
        a tensor of shape (*, 3) for v, where * denotes any dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert t.shape[-1] == 3

    # repeat to e.g. apply the same trans for all points in a point cloud
    # [3] --> [N, 3], [B, 3] --> [B, N, 3], [B, P, 3] --> [B, P, N, 3]
    if len(t.shape) == len(v.shape) - 1:
        t = t.unsqueeze(-2).repeat_interleave(v.shape[-2], dim=-2)
        

    assert t.shape == v.shape

    qv = qrot(q, v)
    tqv = qv + t
    return tqv

def rotate_y_axis(rot, pc):
    if len(pc.shape) == 4:


        #print('rotate_y_axis', rot.shape, pc.shape)
        
        #new_pc = torch.zeros_like(pc).to(pc.device)
        new_object_list = []
        for index_in_a_batch in range(pc.shape[0]):
            new_part_list = []
            for index_in_a_object in range(pc.shape[1]):
                pcd = pc[index_in_a_batch,index_in_a_object,...]
                #_mean = torch.mean(pcd, axis=0)
                #print('index_in_a_object',  pcd.shape, _mean.shape)
                #tr = Translate(-_mean[...,0],-_mean[...,1],-_mean[...,2], dtype=torch.float32)
                #tr_r = Translate(_mean[...,0],_mean[...,1],_mean[...,2], dtype=torch.float32)
                #t = Transform3d().compose(rr).to(pc.device)
                #print(new_pc.device, t.device, pcd.device)
                


                #_mean = torch.mean(pcd, axis=0)
                #tr = Translate(-_mean[...,0],-_mean[...,1],-_mean[...,2], dtype=torch.float32).to(pcd.device)
                #tr_r = Translate(_mean[...,0],_mean[...,1],_mean[...,2], dtype=torch.float32).to(pcd.device)
                rr = Rotate(quaternion_to_matrix(rot[index_in_a_batch][index_in_a_object]), dtype=torch.float32).to(pcd.device)
                
                #new_pcd = Transform3d(device=pc.device).compose(tr).transform_points(pcd)
                new_pcd = Transform3d(device=pcd.device).compose(rr).transform_points(pcd)
                #new_pcd = Transform3d(device=pc.device).compose(tr_r).transform_points(new_pcd)

                
                
                #print(new_pcd.shape, pcd.shape)
                new_part_list.append(new_pcd)
            new_object = torch.stack(new_part_list)   
            #print('new_object',new_object.shape, len(new_part_list))
            new_object_list.append(new_object)     
        new_pc = torch.stack(new_object_list)
        #print('new_pc',new_pc.shape)
        return new_pc       
        
    else:
        #_mean = torch.mean(pc, axis=0)
        #print('rotate_y_axis', rot.shape, pc.shape, _mean.shape)
        #tr = Translate(-_mean[0],-_mean[1],-_mean[2], dtype=torch.float32)
        #tr_r = Translate(_mean[0],_mean[1],_mean[2], dtype=torch.float32)
        #rr = Rotate(quaternion_to_matrix(rot), dtype=torch.float32)
        #t = Transform3d().compose(rr)
        #pc = t.transform_points(pc)#.to(torch.float).to(device)
    
        #_mean = torch.mean(pc, axis=0)
        #tr = Translate(-_mean[...,0],-_mean[...,1],-_mean[...,2], dtype=torch.float32).to(pc.device)
        #tr_r = Translate(_mean[...,0],_mean[...,1],_mean[...,2], dtype=torch.float32).to(pc.device)
        rr = Rotate(quaternion_to_matrix(rot), dtype=torch.float32).to(pc.device)
        #pc = Transform3d(device=pc.device).compose(tr).transform_points(pc)
        pc = Transform3d(device=pc.device).compose(rr).transform_points(pc)
        #pc = Transform3d(device=pc.device).compose(tr_r).transform_points(pc)


    return pc

# y-axis rotation version
def transform_pc(trans, rot, pc, rot_type=None):
    """Rotate and translate the 3D point cloud.

    Args:
        rot (torch.Tensor): quat
    """
    
    return qtransform(trans, rot, pc)


def quaternion_to_euler(quat, to_degree=True):
    """Convert quaternion to euler angle.

    Args:
        quat: [B, 4], quat
        to_degree: bool, whether to convert to degree

    Returns:
        [B, 3], euler angle
    """

    r_mat = transforms.quaternion_to_matrix(quat)
    euler = transforms.matrix_to_euler_angles(r_mat, convention="XYZ")
    if to_degree:
        euler = torch.rad2deg(euler)

    return euler


def qeuler(q, order='xyz', epsilon=0, to_degree=True):
    """
    Convert quaternion(s) q to Euler angles.
    Expects a tensor of shape (*, 4), where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4

    original_shape = list(q.shape)
    original_shape[-1] = 3
    q = q.view(-1, 4)

    q0 = q[:, 0]
    q1 = q[:, 1]
    q2 = q[:, 2]
    q3 = q[:, 3]

    if order == 'xyz':
        x = torch.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
        y = torch.asin(
            torch.clamp(2 * (q1 * q3 + q0 * q2), -1 + epsilon, 1 - epsilon))
        z = torch.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
    elif order == 'yzx':
        x = torch.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
        y = torch.atan2(2 * (q0 * q2 - q1 * q3), 1 - 2 * (q2 * q2 + q3 * q3))
        z = torch.asin(
            torch.clamp(2 * (q1 * q2 + q0 * q3), -1 + epsilon, 1 - epsilon))
    elif order == 'zxy':
        x = torch.asin(
            torch.clamp(2 * (q0 * q1 + q2 * q3), -1 + epsilon, 1 - epsilon))
        y = torch.atan2(2 * (q0 * q2 - q1 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
        z = torch.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2 * (q1 * q1 + q3 * q3))
    elif order == 'xzy':
        x = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
        y = torch.atan2(2 * (q0 * q2 + q1 * q3), 1 - 2 * (q2 * q2 + q3 * q3))
        z = torch.asin(
            torch.clamp(2 * (q0 * q3 - q1 * q2), -1 + epsilon, 1 - epsilon))
    elif order == 'yxz':
        x = torch.asin(
            torch.clamp(2 * (q0 * q1 - q2 * q3), -1 + epsilon, 1 - epsilon))
        y = torch.atan2(2 * (q1 * q3 + q0 * q2), 1 - 2 * (q1 * q1 + q2 * q2))
        z = torch.atan2(2 * (q1 * q2 + q0 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
    elif order == 'zyx':
        x = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
        y = torch.asin(
            torch.clamp(2 * (q0 * q2 - q1 * q3), -1 + epsilon, 1 - epsilon))
        z = torch.atan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
    else:
        raise NotImplementedError

    euler = torch.stack((x, y, z), dim=1).view(original_shape)
    if to_degree:
        euler = euler * 180. / np.pi
    return euler

def get_euler_angles_from_quaternion_gpu(q_tensor, degrees=True):
    """
    (N, 4) 텐서 형태의 쿼터니언에서 N개의 회전 각도(Roll, Pitch, Yaw)를
    GPU에서 병렬로 계산합니다.

    Args:
        q_tensor (torch.Tensor): 쿼터니언 배치의 (N, 4) 텐서.
                                 디바이스는 'cuda'여야 합니다.
        degrees (bool): 각도를 도로 반환할지 여부. 기본값은 True.

    Returns:
        torch.Tensor: (N, 3) 형태의 (roll, pitch, yaw) 텐서.
    """
    # 텐서 디바이스 확인
    if not q_tensor.is_cuda:
        print("경고: 입력 텐서가 GPU에 있지 않습니다. GPU로 이동시킵니다.")
        q_tensor = q_tensor.to('cuda')

    # 쿼터니언 각 성분 추출
    w = q_tensor[..., 0]
    x = q_tensor[..., 1]
    y = q_tensor[..., 2]
    z = q_tensor[..., 3]
    
    # 정규화 (안정성 확보)
    norm = torch.sqrt(w**2 + x**2 + y**2 + z**2)
    w = w / norm
    x = x / norm
    y = y / norm
    z = z / norm

    # Roll (x축 회전) 계산
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x**2 + y**2)
    roll_rad = torch.atan2(t0, t1)

    # Pitch (y축 회전) 계산
    t2 = 2.0 * (w * y - z * x)
    # 짐벌 잠금 방지: -1.0과 1.0 사이로 값 클램핑
    t2 = torch.clamp(t2, -1.0, 1.0)
    pitch_rad = torch.asin(t2)

    # Yaw (z축 회전) 계산
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y**2 + z**2)
    yaw_rad = torch.atan2(t3, t4)

    # (N, 3) 텐서로 결과 결합
    #print("roll",roll_rad.shape,roll_rad,pitch_rad,yaw_rad)
    if len(roll_rad.shape) == 0:
        euler_angles_rad =  torch.stack((roll_rad, pitch_rad, yaw_rad), dim=-1)
    else:
        euler_angles_rad = torch.stack((roll_rad, pitch_rad, yaw_rad), dim=1)
    if degrees:
        return torch.rad2deg(euler_angles_rad)
    else:
        return euler_angles_rad

def get_euler_angles_from_quaternion(q, degrees=True):
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
            roll = np.abs(np.degrees(roll_rad))
            pitch = np.abs(np.degrees(pitch_rad))
            yaw = np.abs(np.degrees(yaw_rad))
            
            # 0~360도 범위로 조정 (선택적)
            # roll = roll % 360
            # pitch = pitch % 360
            # yaw = yaw % 360
            
            return (roll, pitch, yaw)
        else:
            return (roll_rad, pitch_rad, yaw_rad)
