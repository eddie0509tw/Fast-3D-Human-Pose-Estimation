import os
import torch
import hydra
from omegaconf import DictConfig
from lightning.pytorch import LightningModule, Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from tools.load import HumanPoseDataModule
from models.cdrnet import CDRNet
from models.loss import JointsMSESmoothLoss

# Faster, but less precise
torch.set_float32_matmul_precision("highest")
# sets seeds for numpy, torch and python.random.
seed_everything(42, workers=True)


class CDRNetModule(LightningModule):
    def __init__(self,
                 cfg, num_joints, base_joint, use_target_weight,
                 batch_size, lr, lr_step, lr_factor, warmup_epoch, clip_grad):
        super().__init__()

        self.model = CDRNet(cfg.num_layers, num_joints)

        if len(cfg.pretrained) > 0:
            print("Load pretrained weights from '{}'"
                  .format(cfg.pretrained))
            self.model.init_weights(cfg.pretrained)

        self.criterion = JointsMSESmoothLoss(use_target_weight)

        self.batch_size = batch_size
        self.lr = lr
        self.lr_step = lr_step
        self.lr_factor = lr_factor
        self.warmup_epoch = warmup_epoch
        self.clip_grad = clip_grad
        self.num_joints = num_joints
        self.base_joint = base_joint

        self.train_step_loss = []
        self.val_step_loss = []

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), self.lr)

        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, self.lr_step, self.lr_factor)

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def configure_gradient_clipping(self,
                                    optimizer,
                                    gradient_clip_val,
                                    gradient_clip_algorithm):
        if self.current_epoch >= self.warmup_epoch:
            gradient_clip_val = self.clip_grad

        # Lightning will handle the gradient clipping
        self.clip_gradients(
            optimizer,
            gradient_clip_val=gradient_clip_val,
            gradient_clip_algorithm=gradient_clip_algorithm)

    def forward(self, batch, batch_idx):
        (image_left, image_right, target_3d,
         target_left, target_right, meta) = batch

        P_left = meta["P_left"]
        P_right = meta["P_right"]

        target_weight = meta["joints_vis"]

        imgs = [image_left, image_right]
        Ps = [P_left, P_right]
        targets = [target_left, target_right]

        pred_2ds, pred_3ds = self.model(imgs, Ps)

        pred_3ds[:, torch.arange(self.num_joints) != self.base_joint] \
            -= pred_3ds[:, self.base_joint:self.base_joint+1]
        target_3d[:, torch.arange(self.num_joints) != self.base_joint] \
            -= target_3d[:, self.base_joint:self.base_joint+1]

        loss = 0
        if self.current_epoch >= self.warmup_epoch:
            loss += self.criterion(
                pred_3ds * 0.1, target_3d * 0.1, target_weight) * 4

        for pred, target in zip(pred_2ds, targets):
            loss += self.criterion(pred, target, target_weight)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.forward(batch, batch_idx)

        self.train_step_loss.append(loss)

        self.log_dict(
            {
                'train/loss': loss
            },
            on_epoch=True,
            batch_size=self.batch_size)

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.forward(batch, batch_idx)

        self.val_step_loss.append(loss)

        self.log_dict(
            {
                'val/loss': loss
            },
            on_epoch=True,
            batch_size=self.batch_size)

        return loss

    def on_train_epoch_end(self):
        train_mean_loss = torch.stack(self.train_step_loss).mean()
        val_mean_loss = torch.stack(self.val_step_loss).mean()

        print(f"train_mean_loss: {train_mean_loss}, "
              f"val_mean_loss: {val_mean_loss}")

        self.train_step_loss.clear()
        self.val_step_loss.clear()


@hydra.main(config_path="conf", config_name="pose3d", version_base="1.3")
def run(cfg: DictConfig):

    model_name = (f"{cfg.model.name}_{cfg.model.num_layers}"
                  f"_{cfg.train.img_size[0]}_{cfg.dataset.name}")
    model_path = os.path.join(cfg.path.weight_dir, model_name)

    dm = HumanPoseDataModule(
        cfg.dataset,
        cfg.load.num_workers,
        cfg.train.batch_size,
        cfg.train.img_size,
        cfg.target.heatmap_size,
        cfg.target.sigma)

    model = CDRNetModule(
         cfg.model,
         cfg.dataset.num_joints,
         cfg.dataset.base_joint,
         cfg.loss.use_target_weight,
         cfg.train.batch_size,
         cfg.train.lr,
         cfg.train.lr_step,
         cfg.train.lr_factor,
         cfg.train.warmup_epoch,
         cfg.train.clip_grad)

    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    ckpt_cb = ModelCheckpoint(
        dirpath=model_path,
        filename="best",
        monitor='val/loss',
        mode='min',
        save_top_k=2,
        save_last=True,
        save_weights_only=True)
    callbacks = [lr_monitor, ckpt_cb]

    logger = TensorBoardLogger(
        save_dir=cfg.path.log_dir,
        name=model_name)

    trainer = Trainer(accelerator='gpu',
                      devices=1,
                      precision=32,
                      max_epochs=cfg.train.epoch,
                      deterministic=True,
                      num_sanity_val_steps=1,
                      logger=logger,
                      callbacks=callbacks)

    trainer.fit(model, dm)


if __name__ == "__main__":
    run()
