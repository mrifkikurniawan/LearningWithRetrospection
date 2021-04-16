from typing import Optional, Union, List, Dict
from easydict import EasyDict as edict

import torch
import pytorch_lightning as pl
from torch.nn import functional as F
import timm 
import torchmetrics

from lwr.utils import load_pretrained_weight, initialize_optimizer
from lwr.loss import *
from lwr.utils.activation import SoftmaxT

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
        self.epoch_i = 0

        # load pre-traned weight
        if model_cfg.checkpoint_path and model_cfg.pretrained:
            load_pretrained_weight(model=self.model, weight_path=model_cfg.checkpoint_path, device=self.device)
        
        # loss
        self.cross_entropy = CrossEntropyLoss()
        self.kldiv = KLDivLoss(temperature=self.hparams.trainer_cfg.temperature)

        # logits buffer
        # create logits buffer
        # dim(batch indexs, batch size, num classes)
        self.logits_buffer = torch.zeros((dataloader_cfg.train.len, *(dataloader_cfg.train.batch_size, model_cfg.num_classes)))

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
        
    def train_naive(self, logits, y):
        loss = self.cross_entropy(logits, y)
        self.log("train_CrossEntropy_loss_step", loss, on_step=True)
        return loss
    
    def train_lwr(self, logits, y, softlabels):
        self.ce_weight = self.ce_weight.type_as(logits)
        self.kl_weight = self.kl_weight.type_as(logits)
        
        loss_ce = self.ce_weight * self.cross_entropy(logits,y)
        loss_kl = self.kl_weight * self.kldiv(logits, softlabels)
        
        loss = loss_ce + loss_kl
        
        self.log("train_CrossEntropy_loss_step", loss_ce, on_step=True)
        self.log("train_KLDiv_loss_step", loss_kl, on_step=True)
        return loss

    def training_step(self, dataloader, batch_idx):
        x, y = dataloader
        logits = self.forward(x)
        preds = F.softmax(logits, dim=1)
        self.epoch_i = (self.current_epoch+1)/self.hparams.trainer_cfg.k

        # calculate loss
        prefix = 'train'

        if self.current_epoch+1 <= self.hparams.trainer_cfg.k:

            # generate soft labels
            if self.current_epoch+1 == self.hparams.trainer_cfg.k:
                output = self.softmaxt(logits, act_f="softmax")
                self.logits_buffer[batch_idx, ...] = output.detach().clone().cpu()
            
            # calculate loss
            # without retros
            loss = self.train_naive(logits, y)
        
        else:
            # calculate loss
            # with retros
                        
            # retros buffer loader
            softlabels = self.logits_buffer[batch_idx]

            # set loss functions state
            for loss_f in self.hparams.loss_cfg:
                module = loss_f.module
                if module == "KLDivLoss":
                    kl_weight = loss_f.weight      
            
            epoch_ikm = self.epoch_i * self.hparams.trainer_cfg.k / self.hparams.trainer_cfg.max_epochs
            alpha = 1 - (kl_weight * epoch_ikm)
            beta = kl_weight * epoch_ikm

            self.ce_weight = torch.Tensor([alpha])
            self.kl_weight = torch.Tensor([beta])
            
            loss = self.train_lwr(logits, y, softlabels)

            # update soft labels
            if (self.current_epoch+1) % self.hparams.trainer_cfg.k == 0:
                soft_labels = self.softmaxt(logits, act_f='softmax') 
                self.logits_buffer[batch_idx, ...] = soft_labels.detach().clone().cpu()

        # log 
        self.log("loss_step", loss, on_step=True)
        
        # calculate metrics & log metrics
        for metric in self.metrics.train:
            self.log(f'train_{metric}_step', self.metrics.train[metric](preds.cpu(), y.cpu()), 
                     on_epoch=True, prog_bar=True, logger=True, on_step=True)
        
        return loss


    def validation_step(self, dataloader, batch_idx):
        x, y = dataloader
        logits = self.forward(x)
        preds = F.softmax(logits, dim=1)
        
        # calculate loss
        loss = self.cross_entropy(logits, y)

        # log 
        self.log("val_CrossEntropy_loss_step", loss, on_step=True)

        # calculate metrics & log metrics
        for metric in self.metrics.val:
            self.log(f'val_{metric}_step', self.metrics.val[metric](preds.cpu(), y.cpu()), 
                     on_epoch=True, prog_bar=True, logger=True, on_step=True)

    def test_step(self, dataloader, batch_idx):
        x, y = dataloader
        logits = self.forward(x)
        preds = F.softmax(logits, dim=1)

        # calculate loss
        loss = self.cross_entropy(logits, y)

        # log 
        self.log("test_CrossEntropy_loss_step", loss, on_step=True)

        # calculate metrics & log metrics
        for metric in self.metrics.test:
            self.log(f'test_{metric}_step', self.metrics.test[metric](preds.cpu(), y.cpu()), 
                     on_epoch=True, prog_bar=True, logger=True, on_step=True)


    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        optimizer = initialize_optimizer(torch.optim, self.optimizer.module)
        return optimizer(self.parameters(), **self.optimizer.args)

    def training_epoch_end(self, outputs):
        if self.current_epoch+1 > self.hparams.trainer_cfg.k:
            print(f"weight kl: {self.kl_weight}, weight ce: {self.ce_weight}")

        for metric in self.metrics.train:
            self.log(f'train_{metric}_epoch', self.metrics.train[metric].compute(), 
                    on_epoch=True, prog_bar=True, logger=True, on_step=False)

    def val_epoch_end(self):
        for metric in self.metrics.val:
            self.log(f'val_{metric}_epoch', self.metrics.val[metric].compute(), 
                    on_epoch=True, prog_bar=True, logger=True, on_step=False)
            print(self.log, end="\r")

    def test_epoch_end(self, outputs):
        for metric in self.metrics.test:
            self.log(f'test_{metric}_epoch', self.metrics.test[metric].compute(), 
                    on_epoch=True, prog_bar=True, logger=True, on_step=False) 