python decompress.py --data_root /data/jhahn/data/breaking_bad/everyday --subset everyday

python generate_pc_data.py +data.save_pc_data_path=/data/jhahn/data/breaking_bad/everyday/pc_data


python train_vqvae.py \
    experiment_name=everyday_2000epoch \
    data.batch_size=45 \
    data.val_batch_size=45 \
    +trainer.devices=2 \
    +trainer.strategy=ddp
	
	
python train_denoiser.py \
    experiment_name=everyday_epoch2000_bs64 \
    data.batch_size=64 \
    data.val_batch_size=64 \
    model.encoder_weights_path=output_original/autoencoder/everyday_2000epoch/training/last.ckpt \
    +trainer.devices=2 \
    +trainer.strategy=ddp
	
	
#/data/jhahn/data/shape_dataset/pc_data/shape/val
output_original/denoiser/everyday_epoch2000_bs64/training/last.ckpt 
python test.py experiment_name=everyday_epoch2000_bs64 \
    denoiser.data.val_batch_size=20 \
    denoiser.data.data_val_dir=/data/jhahn/data/breaking_bad/everyday/pc_data/val/ \
    denoiser.ckpt_path=/home/jhahn/puzzlefusion-plusplus/output_p/denoiser/everyday_epoch2000_bs64/training/epoch69.ckpt \
    inference_dir=denoiser_only \
    verifier.max_iters=1
	
	
python renderer/render_results.py \
    experiment_name=everyday_epoch2000_bs64 \
    inference_dir=denoiser_only \
    renderer.num_samples=20 \
    renderer.output_path=results \
    renderer.blender.imgRes_x=512 \
    renderer.blender.imgRes_y=512 \
    renderer.blender.numSamples=50
