python test.py experiment_name=everyday_epoch2000_bs64 \
    denoiser.data.val_batch_size=20 \
    denoiser.data.data_val_dir=/data/jhahn/data/shape_dataset/pc_data/brain_lightsheet/test \
    denoiser.ckpt_path=/home/jhahn/puzzlefusion-plusplus/brain_lightsheet/denoiser/everyday_2000epoch/training/last.ckpt \
    inference_dir=brain_lightsheet_from_100 \
    verifier.max_iters=1 \
	

_brain_10_parts
	

python render_results.py \
    experiment_name=everyday_epoch2000_bs64 \
    inference_dir=brain_lightsheet_from_100 \
    renderer.num_samples=20 \
    renderer.mesh_path=/data/jhahn/data/shape_dataset/data/ \
    renderer.output_path=brain_lightsheet_from_100 \
    renderer.blender.imgRes_x=512 \
    renderer.blender.imgRes_y=512 \
    renderer.blender.numSamples=50

python render_results.py \
    experiment_name=everyday_epoch2000_bs64 \
    inference_dir=denoiser_by_brain_block_for_brain_block_full \
    renderer.num_samples=20 \
    renderer.mesh_path=/data/jhahn/data/shape_dataset/data/ \
    renderer.output_path=denoiser_by_brain_block_for_brain_block_full \
    renderer.blender.imgRes_x=512 \
    renderer.blender.imgRes_y=512 \
    renderer.blender.numSamples=50

python render_results.py \
    experiment_name=everyday_epoch2000_bs64 \
    inference_dir=denoiser_by_brain_block_for_brain_block \
    renderer.num_samples=20 \
    renderer.mesh_path=/data/jhahn/data/shape_dataset/data/ \
    renderer.output_path=denoiser_by_brain_block_for_brain_block \
    renderer.blender.imgRes_x=512 \
    renderer.blender.imgRes_y=512 \
    renderer.blender.numSamples=50

python render_results.py \
    experiment_name=everyday_epoch2000_bs64 \
    inference_dir=denoiser_by_brain_block_for_tg \
    renderer.num_samples=20 \
    renderer.mesh_path=/data/jhahn/data/shape_dataset/data/ \
    renderer.output_path=denoiser_by_brain_block_for_tg \
    renderer.blender.imgRes_x=512 \
    renderer.blender.imgRes_y=512 \
    renderer.blender.numSamples=50

# /data/jhahn/data/breaking_bad/everyday/pc_data/val/