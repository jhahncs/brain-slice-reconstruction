python test.py \
    experiment_name=everyday_epoch2000_bs64 \
    denoiser.data.val_batch_size=20 \
    denoiser.data.data_val_dir=./data/pc_data/everyday/val/ \
    denoiser.ckpt_path=output/denoiser/everyday_epoch2000_bs64/training/last.ckpt \
    inference_dir=denoiser_only \
    verifier.max_iters=1 \
