import torch
import itertools
import numpy as np
from scipy.spatial.transform import Rotation as R



def combine_gt_pred(gt, pred, part_valid,step=19):
    
    new_row_data = np.array([0, 0, 0, 1, 0, 0, 0])
    is_zero = (gt == 0)

    # 3-2. 각 행의 모든 요소가 0인지 확인 (결과 모양: (20,) Bool 배열)
    # axis=1을 따라 all()을 적용하여 각 행의 논리합을 계산합니다.
    all_zeros_mask = is_zero.all(axis=1)

    # 4. Bool 마스크를 사용하여 해당 행 수정
    # all_zeros_mask가 True인 모든 행을 new_row_data로 대체
    gt[all_zeros_mask] = new_row_data

    
    
    
    #rotate_gt = R.from_quat(gt[part_valid[0].astype(bool),3:], scalar_first=True)
    rotate_gt = R.from_quat(gt[...,3:], scalar_first=True)
    rotate_pred = R.from_quat(pred[step,:,3:], scalar_first=True).inv()

    R_combined = rotate_pred * rotate_gt
    #rotated_t_gt_1 = rotate_gt.apply(-gt[part_valid[0].astype(bool),:3])
    rotated_t_gt_1 = rotate_gt.apply(-gt[...,:3])
    rotated_t_gt_2 = rotate_pred.apply(rotated_t_gt_1)
    t_combined = rotated_t_gt_2 + pred[step,:,:3]
    
    return np.concatenate([t_combined, R_combined.as_quat(scalar_first=True)], axis=1).astype(float)



def calculate_part_pair_distances_gpu(data_tensor: torch.Tensor, part_valids:torch.Tensor) -> torch.Tensor:
    """
    3D 객체 데이터 (B, P, N, 3)에서 모든 Part Pair의 최소 거리를 GPU 기반으로 계산합니다.

    Args:
        data_tensor: (B, P, N, 3) 형태의 텐서 (B: 배치 크기, P: 파트 개수, N: 파트당 점 개수, 3: xyz 좌표).

    Returns:
        (B, P, P) 형태의 텐서. Part i와 j 사이의 최소 거리를 포함합니다. 
        (i=j일 때는 0입니다).
    """
    
    # 텐서를 GPU로 이동 (CUDA 사용 가능 시)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_tensor = data_tensor.to(device)

    B, P, N, D = data_tensor.shape
    
    # 결과 텐서 초기화: (B, P, P) - 각 배치의 Part 쌍 거리를 저장
    # 대각선 (i=j)은 거리가 0이므로 미리 0으로 초기화
    distances = torch.zeros((B, P, P), device=device, dtype=data_tensor.dtype)

    # 1. Part 쌍 인덱스 생성 (중복 없이, 대칭성을 고려하여 i < j만 계산)
    part_indices = list(itertools.combinations(range(P), 2))
    
    # 모든 Part 쌍에 대해 병렬 계산
    for i, j in part_indices:
        if part_valids[0][i] == 1 and part_valids[0][j] == 1:
                
            # Part i와 Part j의 점들을 추출: (B, N, 3)
            # B: 배치 인덱스, i/j: 파트 인덱스, N: 점 인덱스, 3: 좌표
            part_i_points = data_tensor[:, i, :, :]  
            part_j_points = data_tensor[:, j, :, :]

            # **핵심 연산: torch.cdist (Pairwise Distance Matrix)**
            # Part i의 N개 점과 Part j의 N개 점 사이의 모든 N x N 쌍별 유클리드 거리 행렬을 계산
            # shape: (B, N, N)
            distance_matrix = torch.cdist(part_i_points, part_j_points, p=2) # p=2는 유클리드 거리

            # **최소 거리 (Part Pair Distance)**
            # distance_matrix에서 모든 값 중 가장 작은 값을 찾음
            # shape: (B,)
            avg_dist_batch = torch.mean(distance_matrix.reshape(B, -1), dim=1)
            
            # 결과 텐서에 저장 (대칭적으로)
            distances[:, i, j] = avg_dist_batch
            distances[:, j, i] = avg_dist_batch
            
    return distances.cpu() # CPU로 이동하여 반환

def find_min_non_zero_distance_index(distances: torch.Tensor):
    """
    Part Pair 거리 행렬 (B, P, P)에서 0보다 크면서 가장 작은 거리와 인덱스를 찾습니다.

    Args:
        distances: (B, P, P) 형태의 거리 텐서. GPU에 있을 수 있습니다.

    Returns:
        min_distance: 각 배치별 0보다 큰 최소 거리 (shape: (B,))
        min_indices: 각 배치별 최소 거리에 해당하는 원본 (P, P) 행렬에서의 인덱스 (shape: (B, 2))
    """
    B, P, P = distances.shape
    
    # 텐서를 1차원으로 펼칩니다. (B, P*P)
    # GPU에서 연산하도록 현재 device를 유지합니다.
    flattened_distances = distances.reshape(B, -1)
    
    # 0보다 큰 값만 남기고 나머지는 무한대(inf)로 설정하여 0이 선택되는 것을 방지합니다.
    # distances >= 1e-6 대신 distances > 0을 사용하는 것이 일반적입니다.
    # 부동소수점 오차를 고려하여 작은 임계값(예: 1e-6)을 사용하는 것이 더 안전할 수 있습니다.
    is_positive = (flattened_distances > 1e-6) 
    
    # 0보다 큰 값만 유지하고, 나머지는 PyTorch의 가장 큰 값으로 대체합니다.
    # 이렇게 하면 min() 연산 시 0인 값들은 무시됩니다.
    masked_distances = torch.where(is_positive, 
                                   flattened_distances, 
                                   torch.full_like(flattened_distances, float('inf')))
    
    # 각 배치(0번째 차원)별로 최솟값과 그 인덱스를 찾습니다.
    # min_distance_values: (B,)
    # min_flattened_indices: (B,)
    min_distance_values, min_flattened_indices = torch.min(masked_distances, dim=1)
    
    # 1차원 인덱스 (min_flattened_indices)를 2차원 (P, P) 인덱스로 복원합니다.
    # P로 나눈 몫이 행 인덱스, 나머지가 열 인덱스가 됩니다.
    min_row_indices = min_flattened_indices // P
    min_col_indices = min_flattened_indices % P
    
    # 결과 인덱스를 (B, 2) 형태로 결합합니다.
    min_indices = torch.stack((min_row_indices, min_col_indices), dim=1)
    
    return min_distance_values, min_indices


def merge_two_pcd(point_cloud_A, point_cloud_B, num_of_points = 1000):
    # 2. 두 포인트 클라우드 결합 (Concatenation)
    # np.vstack을 사용하여 두 배열을 수직으로 쌓아 올립니다.
    combined_cloud = np.vstack((point_cloud_A, point_cloud_B))
    if num_of_points < 0 :
        return combined_cloud

    #print(f"결합된 클라우드 모양: {combined_cloud.shape} (2000, 3)")

    # ---

    # 3. 다운샘플링 (Downsampling)을 통해 1000개 포인트 선택

    # 총 포인트 수 (2000개)에서 1000개를 무작위로 선택하기 위한 인덱스 생성
    total_points = combined_cloud.shape[0] # 2000
    target_points = point_cloud_A.shape[0] 

    # 0부터 total_points-1까지의 정수 중에서 target_points만큼 무작위로 선택합니다.
    # replace=False는 중복 선택을 허용하지 않음을 의미합니다.
    sampling_indices = np.random.choice(total_points, target_points, replace=False)

    # 무작위로 선택된 인덱스를 사용하여 결합된 클라우드에서 포인트 추출
    merged_cloud = combined_cloud[sampling_indices]

    #print(f"최종 병합된 클라우드 모양: {merged_cloud.shape} (1000, 3)")
    return merged_cloud

def init_gt_pred_seq(pcd, init_pose, gt, pred):
    
    rotated_points = R.from_quat(init_pose[3:], scalar_first=True).inv().apply(pcd)
    translated_points = rotated_points - init_pose[:3]

    translated_points = translated_points - gt[3,:3]
    rotate_gt = R.from_quat(gt[3,3:], scalar_first=True)
    translated_points = rotate_gt.apply(translated_points)

    rotate_pred = R.from_quat(pred[19,3,3:], scalar_first=True).inv()
    translated_points = rotate_pred.apply(translated_points)
    translated_points = translated_points + pred[19,3,:3]

    return translated_points



def init_gt_pred_combined(pcd, init_pose, gt, pred):

    rotated_points = R.from_quat(init_pose[3:], scalar_first=True).inv().apply(pcd)
    pcd = rotated_points - init_pose[:3]

    rotate_gt = R.from_quat(gt[3,3:], scalar_first=True)
    rotate_pred = R.from_quat(pred[19,3,3:], scalar_first=True).inv()


    R_combined = rotate_pred * rotate_gt
    rotated_t_gt_1 = rotate_gt.apply(-gt[3,:3])
    rotated_t_gt_2 = rotate_pred.apply(rotated_t_gt_1)
    t_combined = rotated_t_gt_2 + pred[19,3,:3]

    pcd = R_combined.apply(pcd) + t_combined
    print(R_combined.as_quat(scalar_first=True))
    print(t_combined)
    return pcd, np.concatenate([t_combined, R_combined.as_quat(scalar_first=True)])


#print( init_gt_pred_seq(obj_3,init_pose_0,gt_0,predict_0) )
#pcd, gt =  init_gt_pred_combined(obj_3,init_pose_0,gt_0,predict_0) 
#print(pcd)
#print(gt)


'''
gt_next = combine_gt_pred(gt_20parts, predict_0, part_valid)
print(gt_next.shape)
for i in range(10):
    if i == 3:
        print(gt_next[i,3:])
        print(gt_next[i,:3])
        rotated_points = R.from_quat(init_pose_0[3:], scalar_first=True).inv().apply(obj_3)
        pcd = rotated_points - init_pose_0[:3]

        rt = R.from_quat(gt_next[i,3:] , scalar_first=True)
        pcd = rt.apply(pcd) + gt_next[i,:3]
        print(pcd)
'''