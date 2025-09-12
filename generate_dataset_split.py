import random
import numpy as np

import os
import shutil
import argparse



if __name__ == "__main__":



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



    _dir_list = os.listdir(dir)
    _dir_list_filtered = []

    tickness = '0.003'
    is_no_gap_between_slices = 'True'
    is_curvature = 'True'
    num_of_missing_slices = "10"
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
        #print(_dir)
        parts = _dir.split('_')
        _cut_xyz = (parts[2], parts[3], parts[4])
        _tickness = parts[5]
        _is_no_gap_between_slices = parts[6]
        _num_of_missing_slices = parts[7]
        _begin_index = parts[8]
        _is_curvature = parts[9]
        _to_index = parts[10]
        try:
            _desc = parts[11]
        except:
            continue
        if _tickness == tickness and _is_no_gap_between_slices == is_no_gap_between_slices \
            and _is_curvature == is_curvature \
            and  _desc == "HIP":
            _dir_list_filtered.append(_dir)

    #data_name = f'{tickness}_{is_no_gap_between_slices}_{is_curvature}_{num_of_missing_slices}'
    #print(data_name)
    #_dir_list_filtered = _dir_list 
    # 그룹 비율을 설정합니다 (총합이 1이 되어야 합니다).
    group_ratios = [0.80, 0.1, 0.1] 

    # 폴더 목록을 무작위로 섞습니다.
    #_dir_list_filtered = _dir_list_filtered[:2000]
    random.shuffle(_dir_list_filtered)
    print(len(_dir_list_filtered))

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
                output_file.write(f'brain_lightsheet/{f}\n')
