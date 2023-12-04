import os
from pytorch_lightning.callbacks import ModelCheckpoint
from tools.load import load_data

import hydra
from hydra.utils import instantiate

@hydra.main(version_base=None, config_path='configs', config_name='mads_3d')
def train(cfg):

    model = instantiate(cfg.MODEL.init, LOSS=instantiate(cfg.LOSS))

    rain_dataset, valid_dataset, train_loader, val_loader \
        = load_data(cfg)

    os.makedirs(cfg.SAVE_DIR, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
                            dirpath=cfg.SAVE_DIR,
                            # Metric to monitor
                            monitor='val/total_loss',
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