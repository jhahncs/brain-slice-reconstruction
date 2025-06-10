from myrenderer import MyRenderer
import os
import hydra
import json
import bpy
import sys
from contextlib import contextmanager

import logging
import os
from datetime import datetime


# 로그 생성
logger = logging.getLogger()

# 로그의 출력 기준 설정
logger.setLevel(logging.INFO)

# log 출력 형식
formatter = logging.Formatter('%(asctime)s[%(levelname)s]: %(message)s')

while logger.hasHandlers():
    logger.removeHandler(logger.handlers[0])
    
    
# log 출력
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# log를 파일에 출력
file_handler = logging.FileHandler('my.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)




def render_results(cfg, renderer: MyRenderer):
    save_dir = cfg.renderer.output_path
    #print(cfg.inference_path)
    sampled_files = renderer.sample_data_files()
    print('sampled_files',sampled_files)
    # sampled_files = ["1"]

    for file in sampled_files:
        transformation, gt_transformation, acc, init_pose = renderer.load_transformation_data(file)
        #print(file)
        parts = renderer.load_mesh_parts(file, gt_transformation, init_pose)
        if parts is None:
            continue
        #print("@",parts[0])
        save_path = f"./BlenderToolBox_render/{save_dir}/{file}"
        os.makedirs(save_path, exist_ok=True)

        renderer.save_img(parts, gt_transformation, gt_transformation, init_pose, os.path.join(save_path, "gt.png"))

        frame = 0

        # bpy.ops.wm.save_mainfile(filepath=save_path + "test" + '.blend')
        
        for i in range(transformation.shape[0]):
            renderer.render_parts(
                parts, 
                gt_transformation, 
                transformation[i], 
                init_pose, 
                frame,
            )
            frame += 1


        imgs_path = os.path.join(save_path, "imgs")
        os.makedirs(imgs_path, exist_ok=True)
        renderer.save_video(imgs_path=imgs_path, video_path=os.path.join(save_path, "video.mp4"), frame=frame)
        
        renderer.clean()
        

@hydra.main(config_path="../config", config_name="auto_aggl", version_base="1.3")
def main(cfg):
    #print(cfg)
    #cfg.renderer.output_path='/data/jhahn/data/shape_dataset/results_tg'
    renderer = MyRenderer(cfg)
    render_results(cfg, renderer)



if __name__ == "__main__":
    main()