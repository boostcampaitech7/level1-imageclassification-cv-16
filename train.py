import os
import random
import wandb
import torch
import torch.nn as nn

import numpy as np

from tqdm import tqdm
from model import modelSelection

from util.data import CustomDataset



# cuda 적용
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# seed값 설정
seed = 2024
deterministic = True

random.seed(seed) # random seed 고정
np.random.seed(seed) # numpy random seed 고정
torch.manual_seed(seed) # torch random seed 고정
if device == 'cuda':
    torch.cuda.manual_seed_all(seed)
if deterministic:
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False


if __name__=='__main__':
    # 학습 데이터의 경로와 정보를 가진 파일의 경로를 설정.
    traindata_dir = "./data/train"
    traindata_info_file = "./data/train.csv"
    save_result_path = "./train_result"
    
    epochs = 100
    batch_size = 32
    lr = 0.01
    num_classes = 500 
 
    config = {'epoches': epochs, 'batch_size': batch_size, 'learning_rate': lr}
    wandb.init(project='my-test-project', config=config)
    
    train_dataloader = None
    test_dataloader = None
    
    model = None
    loss = None
    Optimizer = None
    
    # 스케줄러 초기화
    scheduler_step_size = None # int 30
    scheduler_gamma = None # float 0.1
    
    steps_per_epoch = len(train_dataloader)
    
    scheduler = None
    
    model.to(device)
    
    trainer = None
    
    trainer.train()
    
    matrics_info = None
    
    