import torch
import pytorch_lightning as pl
import os
from CDRNet import *
from dataset import mads_3d
from tools.load import load_data

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from pytorch_lightning.callbacks import ModelCheckpoint

@hydra.main(version_base=None, config_path='configs', config_name='mads_3d')
def train(cfg):

    model = instantiate(cfg.MODEL.init)

    rain_dataset, valid_dataset, train_loader, val_loader \
        = load_data(cfg)

    os.makedirs(cfg.SAVE_DIR, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
                            dirpath=cfg.SAVE_DIR,
                             # Metric to monitor
                            monitor='val/loss',
                            # 'min' for lower is better
                            mode='min',
                            filename=cfg.MODEL.NAME + '_{epoch:02d}',
                            save_top_k=1,  # Save only the best model
                            verbose=True,  # Print checkpoints to console
    )
    trainer = instantiate(cfg.TRAINER, callbacks=[checkpoint_callback])

    trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
    train()