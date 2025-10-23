import random
import numpy as np

import os
import shutil
import argparse
from exp_util import ExpInfo

if __name__ == "__main__":

    '''

    parser = argparse.ArgumentParser(
        description="convert tiff to obj",
        formatter_class=argparse.RawTextHelpFormatter)
        
    #parser.add_argument("--tickness",required=False, default='0.005')
    #parser.add_argument("--is_no_gap_between_slices",required=False, default='True')
    #parser.add_argument("--is_curvature",required=False, default='True')
    #parser.add_argument("--num_of_missing_slices",required=False, default='19')
    parser.add_argument("--obj_dir",required=True, default = '/data/jhahn/data/shape_dataset/data/brain_lightsheet')
    parser.add_argument("--datalist_file_dir",required=False, default='/data/jhahn/data/shape_dataset/data')
    parser.add_argument("--dataname",required=True)


    args = parser.parse_args()
    dataname = args.dataname
    #dataname = '0.001_10_HIP_CUR'
    datalist_file_dir = args.datalist_file_dir
    dir = args.obj_dir  
    '''
    date_type = 'atlas_mouse_brain_50mm_mesh'
    date_type = 'brain_lightsheet'
    
    dataname = 'LS_10s_1000n'
    dir = f'/data/jhahn/data/shape_dataset/data/{date_type}'
    datalist_file_dir = '/data/jhahn/data/shape_dataset/data'


    slice_angle_to_range_map = {}


    slice_angle_to_range_map['sliced_on_1_0_0'] = (0,366)
    slice_angle_to_range_map['sliced_on_1_0_1'] = (0,490)
    slice_angle_to_range_map['sliced_on_0_0_1'] = (61,398)
    slice_angle_to_range_map['sliced_on_1.0_1.0_0.0'] = (5,608)
    slice_angle_to_range_map['sliced_on_0_1_1'] = (6,569)
    slice_angle_to_range_map['sliced_on_0_1_0'] = (5,797)
    slice_angle_to_range_map['sliced_on_1_1_0'] = (5,608)
    slice_angle_to_range_map['sliced_on_1_1_1'] = (0,640)


    _dir_list = os.listdir(dir)
    _dir_list_filtered = []

    tickness = '0.004'
    is_no_gap_between_slices = 'True'
    is_curvature = 'True'
    num_of_missing_slices = "5"
    to_index = "700"

    '''
    tickness = str(dataname.split("_")[0])
    is_no_gap_between_slices = str(dataname.split("_")[1])
    is_curvature = str(dataname.split("_")[2])
    num_of_missing_slices = str(dataname.split("_")[3])
    '''
    for _dir in _dir_list:
        if not os.path.isdir(dir+"/"+_dir):
            continue

        exp_info = ExpInfo(_dir)
        if exp_info.cut_y == 1 and exp_info.cut_y == 1 and exp_info.cut_y == 0:
            slice_dirname = f'sliced_on_1.0_1.0_0.0'
        else:
            slice_dirname = f'sliced_on_{int(exp_info.cut_x)}_{int(exp_info.cut_y)}_{int(exp_info.cut_z)}'
        if  exp_info.num_of_slices == 10:
            _dir_list_filtered.append(_dir)
            #print(slice_dirname, exp_info.num_of_slices)
        
        #if exp_info.start_data_id == slice_angle_to_range_map[slice_dirname][0] and exp_info.end_data_id == slice_angle_to_range_map[slice_dirname][1]-1:
        #    _dir_list_filtered.append(_dir)
    
    #data_name = f'{tickness}_{is_no_gap_between_slices}_{is_curvature}_{num_of_missing_slices}'
    #print(data_name)
    #_dir_list_filtered = _dir_list 
    # 그룹 비율을 설정합니다 (총합이 1이 되어야 합니다).
    print(len(_dir_list_filtered))
    random.shuffle(_dir_list_filtered)
    _dir_list_filtered = _dir_list_filtered[:1000]
    group_ratios = [0.8, 0.1, 0.1] 

    # 폴더 목록을 무작위로 섞습니다.
    

    # 총 폴더 수를 계산합니다.
    total_folders = len(_dir_list_filtered)

    # 각 그룹의 크기를 계산하고, 리스트를 나눌 인덱스를 결정합니다.
    split_indices = np.cumsum([int(total_folders * r) for r in group_ratios[:-1]])

    # 계산된 인덱스를 기준으로 리스트를 3개의 그룹으로 나눕니다.
    groups = np.split(_dir_list_filtered, split_indices)
    # 결과를 출력합니다.
    group1 = list(groups[0])
    group2 = list(groups[1])
    group3 = list(groups[2])

    print(f"📂 그룹 1 ({len(group1)}개, {group_ratios[0]*100:.0f}%):", group1[:2])
    print(f"📂 그룹 2 ({len(group2)}개, {group_ratios[1]*100:.0f}%):", group2[:2])
    print(f"📂 그룹 3 ({len(group3)}개, {group_ratios[2]*100:.0f}%):", group3[:2])


    
    
    for data_type, data in [('train',group1),('test',group2),('val',group3)]:
        output_file_name = f'{datalist_file_dir}/{dataname}.{data_type}.txt'
        print(output_file_name)
        with open(output_file_name,"w") as output_file:
                    
            for f in sorted(data):
                output_file.write(f'{date_type}/{f}\n')
                #output_file.write(f'brain_lightsheet/{f}\n')
    
    