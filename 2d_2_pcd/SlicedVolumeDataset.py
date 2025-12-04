from torchvision import datasets, transforms
from typing import List, Dict, Tuple, Any
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import pandas as pd
DELIMITER = '-'

default_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

class SlicedVolumeDataset(Dataset):
    """
    멀티 레이블 CSV 메타데이터를 로드하는 커스텀 데이터셋
    """
    def __init__(self, annotation_file, transform=None):
        """
        Args:
            annotation_file (str): 'data.csv' 파일 경로.
            transform (callable, optional): 각 슬라이스 이미지에 적용할 변환.
        """
        
        # CSV 파일 로드
        self.data = pd.read_csv(annotation_file)
        self.transform = transform
        print(f'data length: {len(self.data)}')
        # 'Unnamed: 0' 컬럼은 인덱스이므로 제거합니다.
        if 'Unnamed: 0' in self.data.columns:
            self.data = self.data.drop(columns=['Unnamed: 0'])

    def __len__(self):
        """총 데이터 샘플(Volume/Sequence)의 개수를 반환합니다."""
        return len(self.data)
    def get_obj_dir_name(self, row, idx):
        if row['is_curvature']:
            name = f"{row['obj_dir_root']}/{row['slice_angle']}_{'CURV'}_{row['from_index']}_{row['to_index']}_{row['num_of_slices']}_{row['num_of_missing_slices_dist']}_{idx}"
        else:
            name = f"{row['obj_dir_root']}/{row['slice_angle']}_{'FLAT'}_{row['from_index']}_{row['to_index']}_{row['num_of_slices']}_{row['num_of_missing_slices_dist']}_{idx}"

        return name

    def __getitem__(self, idx):
        # 1. 해당 인덱스의 모든 데이터 로드
        row = self.data.iloc[idx]
        obj_dir_name = self.get_obj_dir_name(row, idx)
        # 2. 이미지 파일 경로 리스트 파싱
        # 긴 문자열을 쉼표를 기준으로 분리합니다.
        path_string = row['image_filename_list_for_one_sub_brain']
        image_paths = path_string.split(',')
        
        # 3. 이미지 로드 및 변환
        slices = []
        for path in image_paths:
            try:
                slices.append(path)
                
            except FileNotFoundError:
                print(f"경고: 파일 경로를 찾을 수 없습니다: {path}. 해당 슬라이스는 건너뜁니다.")
                continue
            except Exception as e:
                print(f"경고: 이미지 로드 또는 변환 중 오류 발생: {path}. 오류: {e}")
                continue

        # 4. 메타데이터 파싱 및 준비
        
        # _num_of_missing_slices_list 파싱 (문자열 -> 정수 리스트)
        # 이전 질문의 TypeError 방지를 위해 문자열로 변환 후 분리하고 정수화합니다.
        missing_slices_str = str(row['_num_of_missing_slices_list'])
        missing_slices_list = [int(n.strip()) for n in missing_slices_str.split(',')]
        
        # 다른 메타데이터는 그대로 반환
        metadata = {
            'obj_dir_root': row['obj_dir_root'],
            'slice_angle': row['slice_angle'], # String/Object 타입
            'tickness': row['tickness'],       # Float
            'from_index': row['from_index'],       # Float
            'to_index': row['to_index'],       # Float
            'num_of_missing_slices_dist': row['num_of_missing_slices_dist'],       # Float
            'missing_slices_list': missing_slices_list,
            'is_curvature': row['is_curvature'], # Boolean
            'num_of_slices': row['num_of_slices'] # Int
        }
        
        # 5. 최종 반환
        # (이미지 슬라이스 리스트, 메타데이터 딕셔너리)
        return slices, metadata, obj_dir_name
