import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import pytorch_lightning as pl
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate


class Net(pl.LightningModule):
    def __init__(self, net: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler,
                 config):
        super(Net, self).__init__()
        self.model = net
        self.loss = instantiate(config.LOSS.init)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.start_3d = config.START_3D

        self.training_step_outputs = []
        self.training_losses_epoch = 0
        self.validation_step_outputs = []
        self.validation_losses_epoch = 0
        # self.init_weights()

    def init_weights(self):
        # Initialize the weights of your model here
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                if 'conv' in name:
                    init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
                elif 'fc' in name:
                    init.xavier_normal_(param)
            elif 'bias' in name:
                init.constant_(param, 0.0)

    def forward(self, imgs_list, proj_mat_list=None):
        return self.model(imgs_list, proj_mat_list)

    def configure_optimizers(self):
        self.optimizer = self.optimizer(
            params=self.trainer.model.parameters())
        self.scheduler = self.scheduler(
            optimizer=self.optimizer)
        return {
                "optimizer": self.optimizer,
                "lr_scheduler": {
                    "scheduler": self.scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch"
                },
            }

    def training_step(self, batch, batch_idx):
        images, tgt_3d, proj_out, tgt_2d, _ = batch
        pred_2ds, pred_3ds = self.forward(images, proj_out)
        loss_2d = 0
        loss_3d = 0
        if self.current_epoch <= self.start_3d:
            loss_2d = self.loss(pred_2ds, tgt_2d)

        else:
            loss_3d = self.loss(pred_3ds, tgt_3d)

        total = loss_2d + loss_3d
        # self.epoch_train_losses.append(loss.item())
        self.log(
            "train/loss_2d", loss_2d,
            on_step=False, on_epoch=True, prog_bar=True)

        self.log(
            "train/loss_3d", loss_3d,
            on_step=False, on_epoch=True, prog_bar=True)

        self.log(          
            "train/total_loss", total,
            on_step=False, on_epoch=True, prog_bar=True)

        self.training_step_outputs.append(total)

        return total

    def validation_step(self, batch, batch_idx):
        images, tgt_3d, proj_out, tgt_2d, _ = batch
        pred_2ds, pred_3ds = self.forward(images, proj_out)
        loss_2d = 0
        loss_3d = 0
        if self.current_epoch <= self.start_3d:
            loss_2d = self.loss(pred_2ds, tgt_2d)

        else:
            loss_3d = self.loss(pred_3ds, tgt_3d)

        total = loss_2d + loss_3d

        self.log(
            "val/loss_2d", loss_2d,
            on_step=False, on_epoch=True, prog_bar=True)

        self.log(
            "val/loss_3d", loss_3d,
            on_step=False, on_epoch=True, prog_bar=True)

        self.log(    
            "val/total_loss", total,
            on_step=False, on_epoch=True, prog_bar=True)

        self.validation_step_outputs.append(total)

    def on_train_epoch_start(self):
        self.training_step_outputs = []

    def on_validation_epoch_start(self):
        self.validation_step_outputs = []

    def on_train_epoch_end(self):
        self.training_losses_epoch = torch.stack(
            self.training_step_outputs).mean().item()

    def on_validation_epoch_end(self):
        self.validation_losses_epoch = torch.stack(
            self.validation_step_outputs).mean().item()

    def on_save_checkpoint(self, checkpoint):
        # Save the training losses in the checkpoint
        checkpoint['training_losses'] = self.training_losses_epoch
        checkpoint['validation_loss'] = self.validation_losses_epoch
