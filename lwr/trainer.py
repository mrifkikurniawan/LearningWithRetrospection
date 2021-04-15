from argparse import ArgumentParser
from typing import Optional, Union, List, Dict
import yaml
from easydict import EasyDict as edict

import torch
import torchvision
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import timm 
import torchmetrics


from lwr import datasets
from lwr.utils import load_pretrained_weight, initialize_dataset, initialize_optimizer
from lwr.loss import aggregate_loss, generate_loss
from lwr.utils.activation import SoftmaxT
from lwr.datasets.softlabels import SoftLabelsDataset

class LWR(pl.LightningModule):
    def __init__(self, 
                 model_cfg: edict,
                 trainer_cfg: edict,
                 loss_cfg: edict,
                 dataloader_cfg: edict):

        super().__init__()
        self.save_hyperparameters()
        self.model = timm.create_model(**model_cfg)
        self.softmaxt = SoftmaxT(temperature=trainer_cfg.temperature)
        self.optimizer = trainer_cfg.optimizer

        # load pre-traned weight
        if model_cfg.checkpoint_path and model_cfg.pretrained:
            load_pretrained_weight(model=self.model, weight_path=model_cfg.checkpoint_path, device=self.device)
        
        # loss
        self.loss = generate_loss(loss_cfg=loss_cfg)

        # logits buffer
        self.logits_buffer = SoftLabelsDataset()

        # metrics
        self.metrics = edict()
        for mode in ['train', 'val', 'test']:
            self.metrics[mode] = edict()
            for metric in trainer_cfg.metrics:
                try:
                    self.metrics[mode][metric] = getattr(torchmetrics, metric)()
                except:
                    self.metrics[mode][metric] = getattr(torchmetrics, metric)(num_classes=model_cfg.num_classes)
        print(self.metrics, end='\r')


    def forward(self, x):
        # use forward for inference/predictions
        logits = self.model(x)
        return logits


    def training_step(self, dataloader, batch_idx):
        x, y = dataloader
        logits = self.forward(x)
        preds = torch.softmax(logits, dim=1)

        # calculate loss
        prefix = 'train'

        if self.current_epoch <= self.hparams.trainer_cfg.k:

            # generate soft labels
            if self.current_epoch == self.hparams.trainer_cfg.k:
                output = self.softmaxt(logits, forward_softmax=False)
                self.logits_buffer.append(output)
            
            # calculate loss
            loss = aggregate_loss(loss_fs=self.loss, 
                                  preds=dict(preds=logits), 
                                  targets=dict(targets=y), 
                                  prefix=prefix,
                                  retros=False)
        
        else:
            # calculate loss
            output = self.softmaxt(logits, forward_softmax=False)
            
            # retros buffer loader
            softlabels = self.logits_buffer[batch_idx]
            softlabels = softlabels.type_as(output)
            
            loss = aggregate_loss(loss_fs=self.loss, 
                                  preds=dict(preds=logits,
                                             preds_T=output), 
                                  targets=dict(targets=y,
                                               soft_targets=softlabels), 
                                  prefix=prefix,
                                  retros=True)

            # update soft labels
            if self.current_epoch%self.hparams.trainer_cfg.k == 0:
                self.logits_buffer.update(output, batch_idx) 

        # log 
        for log in loss:
            self.log(f'{log}', loss.get(log), on_epoch=True)
        
        # calculate metrics & log metrics
        for metric in self.metrics.train:
            self.log(f'train_{metric}_step', self.metrics.train[metric](preds.cpu(), y.cpu()), 
                     on_epoch=True, prog_bar=True, logger=True, on_step=True)
        
        return loss[f'{prefix}_total_loss']


    def validation_step(self, dataloader, batch_idx):
        x, y = dataloader
        logits = self.forward(x)
        preds = torch.softmax(logits, dim=1)
        

        # calculate loss
        prefix = 'val'
        loss = aggregate_loss(loss_fs=self.loss, preds=dict(preds=logits), targets=dict(targets=y), prefix=prefix, retros=False)

        # log 
        for log in loss:
            self.log(f'{log}', loss.get(log), on_epoch=True)

        # calculate metrics & log metrics
        for metric in self.metrics.val:
            self.log(f'val_{metric}_step', self.metrics.val[metric](preds.cpu(), y.cpu()), 
                     on_epoch=True, prog_bar=True, logger=True, on_step=True)


    def test_step(self, dataloader, batch_idx):
        x, y = dataloader
        logits = self.forward(x)
        preds = torch.softmax(logits, dim=1)

        # calculate loss
        prefix = 'test'
        loss = aggregate_loss(loss_fs=self.loss, preds=dict(preds=logits), targets=dict(targets=y), prefix=prefix, retros=False)

        # log 
        for log in loss:
            self.log(f'{log}', loss.get(log), on_epoch=True)

        # calculate metrics & log metrics
        for metric in self.metrics.test:
            self.log(f'test_{metric}_step', self.metrics.test[metric](preds.cpu(), y.cpu()), 
                     on_epoch=True, prog_bar=True, logger=True, on_step=True)


    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        optimizer = initialize_optimizer(torch.optim, self.optimizer)
        return optimizer(self.parameters(), lr=self.hparams.trainer_cfg.learning_rate)

    def training_epoch_end(self, outputs):
        for metric in self.metrics.train:
        self.log(f'train_{metric}_epoch', self.metrics.train[metric].compute(), 
                on_epoch=True, prog_bar=True, logger=True)


    def val_epoch_end(self):
        for metric in self.metrics.val:
        self.log(f'val_{metric}_epoch', self.metrics.val[metric].compute(), 
                on_epoch=True, prog_bar=True, logger=True)
        print(self.log)

    def test_epoch_end(self, outputs):
        for metric in self.metrics.test:
        self.log(f'test_{metric}_epoch', self.metrics.test[metric].compute(), 
                on_epoch=True, prog_bar=True, logger=True) 