python train_vqvae.py \
    experiment_name=everyday_2000epoch \
    data.batch_size=45 \
    data.val_batch_size=45 \
    +trainer.devices=1 \
    +trainer.strategy=ddp
	
	
python train_denoiser.py \
    experiment_name=everyday_epoch2000_bs64 \
    data.batch_size=64 \
    data.val_batch_size=64 \
    model.encoder_weights_path=output_p/autoencoder/everyday_2000epoch/training/last.ckpt \
    +trainer.devices=1 \
    +trainer.strategy=ddp
	
	