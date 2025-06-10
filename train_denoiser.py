import os
import torch
import hydra
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from puzzlefusion_plusplus.denoiser.dataset.dataset import build_geometry_dataloader

import setproctitle
setproctitle.setproctitle('train_denoiser')    

def init_callbacks(cfg):
    checkpoint_monitor = hydra.utils.instantiate(cfg.checkpoint_monitor)
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    # print_callback = PrintCallback()
    return [checkpoint_monitor, lr_monitor]


@hydra.main(version_base=None, config_path="config/denoiser", config_name="global_config")
def main(cfg):
    # fix the seed
    print(cfg)
    pl.seed_everything(cfg.train_seed, workers=True)

    # create directories for training outputs
    os.makedirs(os.path.join(cfg.experiment_output_path, "training"), exist_ok=True)

    # initialize data
    train_loader, val_loader = build_geometry_dataloader(cfg)
    
    # initialize model
    model = hydra.utils.instantiate(cfg.model.model_name, cfg)

    if cfg.model.encoder_weights_path is not None:
        encoder_weights = torch.load(cfg.model.encoder_weights_path)['state_dict']
        model.encoder.load_state_dict({k.replace('ae.', ''): v for k, v in encoder_weights.items()})
        # freeze the encoder
        for param in model.encoder.parameters():
            param.requires_grad = False

    # initialize logger
    logger = hydra.utils.instantiate(cfg.logger)

    # initialize callbacks
    callbacks = init_callbacks(cfg)
    
    # initialize trainer
    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=logger,
        **cfg.trainer
    )

    # check the checkpoint
    if cfg.ckpt_path is not None:
        assert os.path.exists(cfg.ckpt_path), "Error: Checkpoint path does not exist."

    # start training
    trainer.fit(
        model=model, 
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=cfg.ckpt_path
    )


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]= "MIG-e5a11831-23aa-537c-9ebe-5e6cce8a2bce,MIG-6b03ab11-3f04-52a2-952d-e8d9df38ee72" 

    main()