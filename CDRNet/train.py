import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from Net import CDRNet


class Net(pl.LightningModule):
    def __init__(self, net: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler, cfg):
        super(Net, self).__init__()
        self.model = net
        self.loss = cfg.loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epoch_miou = []
        self.miou_results = []
        self.epoch_train_losses = []
        self.train_losses_results = []
        self.init_weights()

    def init_weights(self):
        pass

    def forward(self, imgs_list, proj_matrices_list=None):
        return self.model(imgs_list, proj_matrices_list)

    def configure_optimizers(self):
        return {
                "optimizer": self.optimizer,
                'scheduler': self.scheduler,
                'monitor': 'avg_train_loss'
                }


    def training_step(self, batch, batch_idx):
        images, targets = batch
        predictions = self.forward(images)
        loss = self.loss(predictions, targets)
        self.epoch_train_losses.append(loss.item())
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        (img_list, proj_mats_list), targets = batch
        predictions = self.forward(img_list, proj_mats_list)
        # self.epoch_miou.append(miou.item())
        # self.log('miou', miou, prog_bar=True)


    def on_train_epoch_end(self):
        avg_train_loss = sum(self.epoch_train_losses) / len(self.epoch_train_losses)
        self.train_losses_results.append(avg_train_loss)
        self.log('avg_train_loss', avg_train_loss,prog_bar=True)
        self.epoch_train_losses = []  # reset for the next epoch

    def on_validation_epoch_end(self):
        # avg_miou = sum(self.epoch_miou) / len(self.epoch_miou)
        # self.miou_results.append(avg_miou)
        # self.log('avg_miou', avg_miou,prog_bar=True)
        # self.epoch_miou = []  # reset for the next epoch
        pass

    def on_save_checkpoint(self, checkpoint):
        # Save the training losses in the checkpoint
        checkpoint['training_losses'] = self.train_losses_results
        # checkpoint['miou'] = self.miou_results
