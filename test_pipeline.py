import os
import hydra
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf,open_dict
import omegaconf
import shutil
import importlib
import sys
sys.path.insert(0, "2d_2_pcd")
from torch.utils.data import Dataset, DataLoader
from puzzlefusion_plusplus.vqvae.dataset.dataset import build_geometry_dataloader, GeometryPartDataset
import os
import numpy as np
from tqdm import tqdm
import zipfile
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

#torch.nn.parallel.DistributedDataParallel.no_sync()
#torch._dynamo.config.optimize_ddp = False
import puzzlefusion_plusplus.vqvae.dataset.dataset
from puzzlefusion_plusplus.vqvae.dataset.dataset import GeometryPartDataset
import render_inference_result
from puzzlefusion_plusplus.denoiser.dataset.dataset import build_test_dataloader
from puzzlefusion_plusplus.auto_aggl import AutoAgglomerative
import obj_2_pcd


def load_cfg(config_dir):
        
    cfg_auto_aggl = omegaconf.OmegaConf.load(f'{config_dir}/auto_aggl.yaml')
    cfg_denoiser_data = omegaconf.OmegaConf.load(f'{config_dir}/denoiser/data.yaml')
    cfg_denoiser_encode = omegaconf.OmegaConf.load(f'{config_dir}/denoiser/encoder.yaml')
    cfg_denoiser_global_config = omegaconf.OmegaConf.load(f'{config_dir}/denoiser/global_config.yaml')
    cfg_denoiser_model = omegaconf.OmegaConf.load(f'{config_dir}/denoiser/model.yaml')

    cfg_ae_vq_vae = omegaconf.OmegaConf.load(f'{config_dir}/ae/vq_vae.yaml')
    cfg_ae_global_config = omegaconf.OmegaConf.load(f'{config_dir}/ae/global_config.yaml')
    cfg_ae_model = omegaconf.OmegaConf.load(f'{config_dir}/ae/model.yaml')
    cfg_ae_data = omegaconf.OmegaConf.load(f'{config_dir}/ae/data.yaml')

    cfg = OmegaConf.merge(
        cfg_auto_aggl,
        {"denoiser": cfg_denoiser_data},
        {"denoiser": cfg_denoiser_encode},
        {"denoiser": cfg_denoiser_global_config},
        {"denoiser": cfg_denoiser_model},
        {"ae": cfg_ae_vq_vae},
        {"ae": cfg_ae_global_config},
        {"ae": cfg_ae_model},
        cfg_ae_data
        #{"verifier": omegaconf.OmegaConf.load('config/verifier/global_config.yaml')},
        #{"verifier": omegaconf.OmegaConf.load('config/verifier/model.yaml')},
    )
    return cfg



def init_dir(files_root,data_ids):
        
    test_root = f'{files_root}/{data_ids[0]}'

    tiff_dir_root = f'{test_root}/tiff'
    obj_dir_root = f'{test_root}/objs'
    pc_dir_root = f'{test_root}/pc'
    inference_dir_root = f'{test_root}/inference'
    render_output_dir = test_root+'/render'

    os.makedirs(tiff_dir_root, exist_ok=True)
    os.makedirs(obj_dir_root, exist_ok=True)
    os.makedirs(pc_dir_root, exist_ok=True)
    os.makedirs(render_output_dir, exist_ok=True)
    os.makedirs(render_output_dir+"/objs", exist_ok=True)

    return tiff_dir_root, obj_dir_root, pc_dir_root, inference_dir_root, render_output_dir



def _gen_pc_data(cfg, loader, data_type):
    save_path = cfg.data.save_pc_data_path
    os.makedirs(save_path, exist_ok=True)
    print('_gen_pc_data',save_path)
    for i, data_dict in tqdm(enumerate(loader), total=len(loader), desc=f"Processing {data_type} data"):
        data_id = data_dict['data_id'][0].item()
        part_valids = data_dict['part_valids'][0]
        num_parts = data_dict['num_parts'][0].item()
        mesh_file_path = data_dict['mesh_file_path'][0]
        print(mesh_file_path)
        graph = data_dict['graph'][0]
        category = data_dict['category'][0]
        part_pcs_gt = data_dict['part_pcs_gt'][0]
        #print(part_pcs_gt.shape)



        ref_part = data_dict['ref_part'][0]

        np.savez(
            os.path.join(save_path, f'{data_id:05}.npz'),
            data_id=data_id,
            part_valids=part_valids.cpu().numpy(),
            num_parts=num_parts,
            mesh_file_path=mesh_file_path,
            graph=graph.cpu().numpy(),
            category=category,
            part_pcs_gt=part_pcs_gt.cpu().numpy(),
            ref_part=ref_part.cpu().numpy()
        )
        # print(f"Saved {data_id:05}.npz in {data_type} data.")


import trimesh
def get_obj_from_testdataset(data_type,data_id,tiff_dir,tiff_dir_root,obj_dir_root,render_output_dir):
    # get from test data
    obj_files = []
    glb_dir = f'/data/jhahn/data/shape_dataset/data/{data_type}/{data_id}/fractured_0'
    os.makedirs(f'{obj_dir_root}/test/fractured', exist_ok=True)
    for f in os.listdir(glb_dir):
        _id = int(f.split(".")[-2])
        #shutil.copyfile(f'{tiff_dir}/{_id}.tif', f'{tiff_dir_root}/{_id}.tif')    
        shutil.copyfile(f'{glb_dir}/{f}', f'{obj_dir_root}/test/fractured/{f}')
        _glb = trimesh.load(f'{obj_dir_root}/test/fractured/{f}')
        _new_obj_filename = render_output_dir+"/objs/"+f.replace(".glb",".obj")
        _glb.export(_new_obj_filename)
        print(_new_obj_filename)
        obj_files.append(_new_obj_filename)
    print(f"combining {len(obj_files)} files")
    slice_util.combine_obj_files(obj_files, render_output_dir+f"/gt_{data_id}.obj")

def obj_2_pc(cfg,   obj_dir_root, pc_dir_root):
        
    with open(obj_dir_root+"/test.txt",'w') as f:
        #f.write(obj_dir_list_relative[0])
        f.write("test\n")

    
    
    data_dict = dict(
        data_dir=obj_dir_root,
        data_fn= "test.txt",
        data_keys=None,
        cfg=cfg,
        category='all',
        num_points=cfg.data.num_pc_points,
        min_num_part=cfg.data.min_num_part,
        max_num_part=cfg.data.max_num_part,
        shuffle_parts=cfg.data.shuffle_parts,
        rot_range=cfg.data.rot_range,
        overfit=cfg.data.overfit,
    )


    #data_dict['data_fn'] = cfg.data.data_fn.format('test')
    data_dict['shuffle_parts'] = False
    test_set = GeometryPartDataset(**data_dict)
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=cfg.data.val_batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(cfg.data.num_workers > 0),
    )


    cfg.data.batch_size = 1
    cfg.data.val_batch_size = 1
    cfg.data.num_workers: 64 
    #cfg.data.save_pc_data_path = pc_dir_root + "/"+obj_dir_list_relative[0]
    cfg.data.save_pc_data_path = pc_dir_root

    _gen_pc_data(cfg, test_loader, 'test')

    #return obj_dir_list_relative


def inference(cfg, pc_dir_root, ckpt_path, inference_dir_root):       
    

    with open_dict(cfg):
        #cfg.experiment_output_path = data_home_dir+'experiment_output/'
        cfg.denoiser.data.data_val_dir = pc_dir_root# + "/"+obj_dir_list_relative[0]
        #cfg.denoiser.data.data_val_dir = obj_dir_list_relative[0]
        #cfg.denoiser.ckpt_path= data_home_dir+f'output/denoiser/everyday_epoch100_bs64/training/last.ckpt'
        cfg.denoiser.ckpt_path= ckpt_path
        cfg.inference_dir= inference_dir_root
        cfg.denoiser.data.val_batch_size=1
        cfg.verifier.max_iters = 1
        #cfg.experiment_output_path = project_root
        #cfg.verifier.ckpt_path= '/disk2/data/breaking-bad-dataset/output/verifier/everyday_epoch100_bs64/training/last.ckpt'

    print(cfg)

    denoiser_only_flag = cfg.verifier.max_iters == 1


    #print(OmegaConf.to_yaml(cfg))
    #print(cfg.experiment_output_path)
    # initialize data
    test_loader = build_test_dataloader(cfg.denoiser, denoiser_only_flag)

    # load denoiser weights
    model = AutoAgglomerative(cfg)
    
    denoiser_weights = torch.load(cfg.denoiser.ckpt_path)['state_dict']

    model.denoiser.load_state_dict(
        {k.replace('denoiser.', ''): v for k, v in denoiser_weights.items() 
            if k.startswith('denoiser.')}
    )

    model.encoder.load_state_dict(
        {k.replace('encoder.', ''): v for k, v in denoiser_weights.items() 
            if k.startswith('encoder.')}
    )

    #if cfg.verifier.max_iters > 1:
        # load verifier weights    
    #    verifier_weights = torch.load(cfg.verifier.ckpt_path)['state_dict']
    #    model.verifier.load_state_dict({k.replace('verifier.', ''): v for k, v in verifier_weights.items()})

    # initialize trainer
    trainer = pl.Trainer(accelerator=cfg.accelerator, devices=1, max_epochs=1, logger=False)
    print(trainer)
    # start inference
    r = trainer.test(model=model, dataloaders=test_loader)
    
    print("test done",r)

def zip_and_download_folder(folder_path):
    """
    폴더를 .zip 파일로 압축하고, 다운로드 링크를 생성합니다.
    """
    # 폴더가 존재하지 않으면 에러 메시지 출력
    if not os.path.exists(folder_path):
        print(f"오류: 폴더 '{folder_path}'가 존재하지 않습니다.")
        return

    # 압축 파일 이름 설정
    zip_file_name = f"{folder_path}.zip"

    # 폴더를 .zip 파일로 압축
    print(f"폴더 '{folder_path}' 압축 중...")
    with zipfile.ZipFile(zip_file_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                # zip 파일 내의 경로 설정
                arcname = os.path.relpath(file_path, os.path.join(folder_path, os.pardir))
                zipf.write(file_path, arcname)

    print(f"'{zip_file_name}' 압축 완료.")

    # 다운로드 링크 생성 및 표시
    
    print("위 링크를 클릭하여 다운로드하세요.")
    return zip_file_name

def render(inference_dir_root, obj_id_list ,part_pcs_gt, original_vertices,render_output_dir):
        
    result_dir_list = []
    for f in os.listdir(inference_dir_root):
        if os.path.isdir(inference_dir_root+"/"+f):
            result_dir_list.append(inference_dir_root+"/"+f)
    
    os.makedirs(render_output_dir, exist_ok=True)

    _result_dir = result_dir_list[0]

    
    render_inference_result.gt_img(device, part_pcs_gt,original_vertices, _result_dir, render_output_dir, obj_id_list)

    render_inference_result.make_video(device, part_pcs_gt,original_vertices, _result_dir,render_output_dir , obj_id_list)

from chamferdist import ChamferDistance

def eval(vertices_gt,inference_dir_root, render_output_dir):
    
    df_original_pos, df_init, df_trasformation , init_gt_pcd_list, pred_pcd_list = get_transformations(vertices_gt,inference_dir_root)

    
    df_original_pos.to_csv(render_output_dir+"/0/BoundingBoxOfInputParts.csv")
    df_init.to_csv(render_output_dir+"/0/BoundingBoxOfInputParts_init.csv")
    df_trasformation.to_csv(render_output_dir+"/0/Transformation.csv")

    pred_pos = render_inference_result.gen_final_image(device, vertices_gt,  df_trasformation, render_output_dir)
    #init_gt_pcd_list = torch.tensor(init_gt_pcd_list)
    #pred_pcd_list = torch.tensor(pred_pcd_list)
    #print(init_gt_pcd_list.shape)
    #print(pred_pcd_list.shape)
    
    #init_gt_pcd_list = torch.from_numpy(init_gt_pcd_list).to(device)
    #pred_pcd_list = torch.from_numpy(pred_pcd_list).to(device)

    #print(init_gt_pcd_list.shape)
    #print(pred_pcd_list.shape)
    #shape1 = init_gt_pcd_list.flatten(1, 2)
    #shape2 = pred_pcd_list.flatten(1, 2)
    shape1 = init_gt_pcd_list
    shape2 = pred_pcd_list
    shape1 = torch.stack(shape1, dim=0) 
    shape2 = torch.stack(shape2, dim=0) 
    #print(len(shape1), shape1[0].shape)
    #print(len(shape2), shape2[0].shape)
    metric = ChamferDistance()
    shape_cd = metric(
        shape1, 
        shape2, 
        bidirectional=False, 
        point_reduction='mean', 
        batch_reduction=None
    )
    


    return shape_cd

import glob
import pandas as pd
import slice_util
def get_transformations(vertecis_list, inference_dir):
    gt = np.load(f'{inference_dir}/0/gt.npy')
    init_pose = np.load(f'{inference_dir}/0/init_pose.npy')
    predict_file_name = glob.glob(f'{inference_dir}/0/predict*')[0]
    predict_0 = np.load(predict_file_name)
    #vertecis_list = render_inference_result.get_vertices(f'{obj_dir_root}/{obj_dir_list_relative}/fractured_0',device)

    row_list = []
    for part_index in range(predict_0.shape[1]): 
        _max = torch.max(vertecis_list[part_index], axis=0)
        _min = torch.min(vertecis_list[part_index], axis=0)
        row = {}
        row['part_index'] = part_index
        row['x_min'] = _min.values[0].item()
        row['x_max'] = _max.values[0].item()
        row['y_min'] = _min.values[1].item()
        row['y_max'] = _max.values[1].item()
        row['z_min'] = _min.values[2].item()
        row['z_max'] = _max.values[2].item()
        row_list.append(row)

    bx_original = pd.DataFrame(row_list)

    init_gt_pcd_list = []
    row_list = []
    for part_index in range(predict_0.shape[1]): 

        translated_points, _ = render_inference_result.transform_pc(device, vertecis_list[part_index], 
                                                                    init_pose, gt, None, None, part_index)

        init_gt_pcd_list.append(translated_points)
        _max = torch.max(translated_points, axis=0)
        _min = torch.min(translated_points, axis=0)
        row = {}
        row['part_index'] = part_index
        row['x_min'] = _min.values[0].item()
        row['x_max'] = _max.values[0].item()
        row['y_min'] = _min.values[1].item()
        row['y_max'] = _max.values[1].item()
        row['z_min'] = _min.values[2].item()
        row['z_max'] = _max.values[2].item()
        row_list.append(row)

    bx_init = pd.DataFrame(row_list)


    row_list = []
    pred_pcd_list = []
    step = predict_0.shape[0] - 1
    #step = 0
    for part_index in range(predict_0.shape[1]):        
        translated_points, temp_t = render_inference_result.transform_pc(device, vertecis_list[part_index], init_pose, gt, predict_0, step, part_index)
        pred_pcd_list.append(translated_points)
        _t_m = temp_t.get_matrix()[0].T.cpu().numpy()
        row = {}
        row['part_index'] = part_index
        row['tX'] = _t_m[:3,3][0]
        row['tY'] = _t_m[:3,3][1]
        row['tZ'] = _t_m[:3,3][2]
        _rr = slice_util.rotation_matrix_to_quaternion(_t_m[:3,:3])       
        row['rW'] = _rr[0]
        row['rX'] = _rr[1]
        row['rY'] = _rr[2]
        row['rZ'] = _rr[3]
        row['transformation_matrix'] = _t_m
        
        row_list.append(row)
        
    trans_matrix_df = pd.DataFrame(row_list)

    return bx_original, bx_init, trans_matrix_df, init_gt_pcd_list, pred_pcd_list