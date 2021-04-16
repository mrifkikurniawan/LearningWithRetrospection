from easydict import EasyDict as edict
from argparse import ArgumentParser
import yaml

from torchvision import transforms
from torchvision import datasets
from torch.utils.data import random_split, DataLoader
import pytorch_lightning as pl
import pytorch_lightning.loggers as loggers 

from lwr.trainer import LWR 
from lwr.utils.initialize_instance import initialize_dataset

def cli_main():
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('-c', '--cfg', default='', type=str, help='path to trainer config')

    # trainer
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    config = edict(yaml.safe_load(open(args.cfg, 'r')))
    print(f"config: {config}")

    # seed everything
    pl.seed_everything(config.trainer.seed)

    # ------------
    # data
    # ------------
    
    # dataset
    dataset_name = str(config.datasets.name)

    dataset_train = initialize_dataset(datasets, dataset_name, **config.datasets.train)
    dataset_test =  initialize_dataset(datasets, dataset_name, **config.datasets.test)
    
    len_train = len(dataset_train)
    dataset_train, dataset_val = random_split(dataset_train, [int(len_train*config.datasets.train_val_ratio.train), 
                                                              int(len_train*config.datasets.train_val_ratio.val)])
    
    print(f"Dataset Train: {len(dataset_train)}")
    print(f"Dataset Val: {len(dataset_val)}")
    print(f"Dataset Test: {len(dataset_test)}")

    # dataloader
    train_loader = DataLoader(dataset_train, **config.dataloaders.train)
    val_loader = DataLoader(dataset_val, **config.dataloaders.val)
    test_loader = DataLoader(dataset_test, **config.dataloaders.test)
    config.dataloaders.train.len = len(train_loader)

    # ------------
    # model
    # ------------    
    model = LWR(model_cfg=config.model,
                trainer_cfg=config.trainer,
                loss_cfg=config.loss,
                dataloader_cfg=config.dataloaders)

    # ------------
    # training
    # ------------
    logger = getattr(loggers, config.trainer.logger)
    config.trainer.logger = logger('tb_logs', name=f"{config.trainer.module}_{config.datasets.name}_{config.model.model_name}")

    trainer = pl.Trainer.from_argparse_args(config.trainer)
    trainer.fit(model, train_loader, val_loader)

    # ------------
    # testing
    # ------------
    result = trainer.test(test_dataloaders=test_loader)
    print(result)


if __name__ == '__main__':
    cli_main()