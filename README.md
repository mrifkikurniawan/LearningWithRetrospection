<div align="center">    
 
# Learning With Retrospection     

[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://arxiv.org/abs/2012.13098)
[![Conference](http://img.shields.io/badge/AAAI-2021-4b44ce.svg)](https://www.aminer.cn/pub/5fe5b80991e011e85bd96943/learning-with-retrospection?conf=aaai2021)  
</div>
 
## Description   
Unofficial implementation of learning on top of PytorchLightning. 

## How to run   
First, install dependencies   
```bash
# clone project   
git clone https://github.com/mrifkikurniawan/LearningWithRetrospection.git

# install project   
cd LearningWithRetrospection 
pip install -e .   
 ```   
 Next, navigate to any file and run it.   
 ```bash

# run training module (example: train lwr on mnist dataset)   
python train.py --cfg config/lwr_mnist_resnet18.yaml  
```

## Imports
This project is setup as a package which means you can now easily import any file into any other file like so:
```python
from lwr.trainer import LWR
from pytorch_lightning import Trainer

# model
model = LWR()

# data
train, val, test = mnist()

# train
trainer = Trainer()
trainer.fit(model, train, val)

# test using the best model!
trainer.test(test_dataloaders=test)
```

## Experiments Results
To Do