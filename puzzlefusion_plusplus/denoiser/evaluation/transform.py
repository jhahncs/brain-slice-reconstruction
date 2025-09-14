#from pytorch3d.transforms import quaternion_apply
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
    if len(q.shape) == len(v.shape) - 1:
        q = q.unsqueeze(-2).repeat_interleave(v.shape[-2], dim=-2)
    assert q.shape[:-1] == v.shape[:-1]


    q = rotate_y_axis(v,q)
    
    return q

    #return quaternion_apply(q, v)

# y-axis rotation version
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
    _mean = torch.mean(pc, axis=0)
    tr = Translate(-_mean[0],-_mean[1],-_mean[2], dtype=torch.float32)
    tr_r = Translate(_mean[0],_mean[1],_mean[2], dtype=torch.float32)
    rr = Rotate(quaternion_to_matrix(rot), dtype=torch.float32)
    t = Transform3d().compose(tr).compose(rr).compose(tr_r)
    pc = t.transform_points(pc)#.to(torch.float).to(device)
    
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
