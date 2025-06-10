import os
import torch
import hydra
import pytorch_lightning as pl
from puzzlefusion_plusplus.vqvae.data.data_module import DataModule
from pytorch_lightning.callbacks import LearningRateMonitor
import setproctitle
setproctitle.setproctitle('train_vqvae')    

def init_callbacks(cfg):
    checkpoint_monitor = hydra.utils.instantiate(cfg.checkpoint_monitor)
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    return [checkpoint_monitor, lr_monitor]


@hydra.main(version_base=None, config_path="config/ae", config_name="global_config")
def main(cfg):
    # fix the seed
    print(cfg)
    pl.seed_everything(cfg.train_seed, workers=True)

    # create directories for training outputs
    os.makedirs(os.path.join(cfg.experiment_output_path, "training"), exist_ok=True)

    # initialize data
    data_module = DataModule(cfg)

    # initialize model
    model = hydra.utils.instantiate(cfg.model.model_name, cfg)

    # initialize logger
    logger = hydra.utils.instantiate(cfg.logger)

    # initialize callbacks
    callbacks = init_callbacks(cfg)

    # initialize trainer
    trainer = pl.Trainer(callbacks=callbacks, logger=logger, **cfg.trainer)

    # check the checkpoint
    if cfg.ckpt_path is not None:
        assert os.path.exists(cfg.ckpt_path), "Error: Checkpoint path does not exist."
    
    # start training
    trainer.fit(model=model, datamodule=data_module, ckpt_path=cfg.ckpt_path)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]= "MIG-e5a11831-23aa-537c-9ebe-5e6cce8a2bce,MIG-6b03ab11-3f04-52a2-952d-e8d9df38ee72" 

    main()