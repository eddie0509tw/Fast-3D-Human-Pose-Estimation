import os
import torch
import hydra
from omegaconf import DictConfig
from lightning.pytorch import LightningModule, Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from tools.load import HumanPoseDataModule
from models.poseresnet import PoseResNet
from models.loss import JointsMSELoss
from models.metrics import accuracy

# Faster, but less precise
torch.set_float32_matmul_precision("high")
# sets seeds for numpy, torch and python.random.
seed_everything(42, workers=True)


class SimpleBaselineModule(LightningModule):
    def __init__(self,
                 cfg, num_joints, use_target_weight,
                 batch_size, lr, lr_step, lr_factor):
        super().__init__()

        self.model = PoseResNet(cfg.num_layers, num_joints)

        if len(cfg.pretrained) > 0:
            print("Load pretrained weights from '{}'"
                  .format(cfg.pretrained))
            self.model.init_weights(cfg.pretrained)

        self.criterion = JointsMSELoss(use_target_weight)

        self.batch_size = batch_size
        self.lr = lr
        self.lr_step = lr_step
        self.lr_factor = lr_factor

        self.train_step_loss = []
        self.train_step_acc = []
        self.val_step_loss = []
        self.val_step_acc = []

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), self.lr)

        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, self.lr_step, self.lr_factor)

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def forward(self, batch, batch_idx):
        img, target, target_weight, meta = batch

        out = self.model(img)
        loss = self.criterion(out, target, target_weight)

        acc, _ = accuracy(out.detach().cpu().numpy(),
                          target.detach().cpu().numpy())

        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self.forward(batch, batch_idx)

        self.train_step_loss.append(loss)
        self.train_step_acc.append(torch.tensor(acc))

        self.log_dict(
            {
                'train/loss': loss,
                'train/acc': acc[0]
            },
            on_epoch=True,
            batch_size=self.batch_size)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self.forward(batch, batch_idx)

        self.val_step_loss.append(loss)
        self.val_step_acc.append(torch.tensor(acc))

        self.log_dict(
            {
                'val/loss': loss,
                'val/acc': acc[0]
            },
            on_epoch=True,
            batch_size=self.batch_size)

        return loss

    def on_train_epoch_end(self):
        train_mean_loss = torch.stack(self.train_step_loss).mean()
        train_mean_acc = torch.stack(self.train_step_acc).mean()
        val_mean_loss = torch.stack(self.val_step_loss).mean()
        val_mean_acc = torch.stack(self.val_step_acc).mean()

        print(f"train_mean_loss: {train_mean_loss}, "
              f"train_mean_acc: {train_mean_acc}")
        print(f"val_mean_loss: {val_mean_loss}, "
              f"val_mean_acc: {val_mean_acc}")

        self.train_step_loss.clear()
        self.train_step_acc.clear()
        self.val_step_loss.clear()
        self.val_step_acc.clear()


@hydra.main(config_path="conf", config_name="pose2d", version_base="1.3")
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

    model = SimpleBaselineModule(
         cfg.model,
         cfg.dataset.num_joints,
         cfg.loss.use_target_weight,
         cfg.train.batch_size,
         cfg.train.lr,
         cfg.train.lr_step,
         cfg.train.lr_factor)

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
