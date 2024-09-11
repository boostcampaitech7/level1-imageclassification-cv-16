import os
import random
import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

import numpy as np
import pandas as pd
from tqdm import tqdm

from model import modelSelection

from util.data import CustomDataset
from util.augmentation import TransformSelector
from util.optimizers import get_optimizer
from trainer import Trainer

from model.modelSelection import ModelSelector


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
    train_data_dir = "./data/train"
    val_data_dir = "./data/val"
    train_data_info_file = "./data/train.csv"
    val_data_info_file = "./data/val.csv"
    save_result_path = "./train_result"
    
    epochs = 100
    batch_size = 32
    lr = 0.01
    num_classes = 500 
 
    config = {'epoches': epochs, 'batch_size': batch_size, 'learning_rate': lr}
    wandb.init(project='my-test-project', config=config)
    
    transform_selector = TransformSelector(transform_type = "albumentations")
    
    train_df = pd.read_csv(train_data_info_file)
    val_df = pd.read_csv(val_data_info_file)
    
    
    train_transform = transform_selector.get_transform(is_train=True)
    val_transform = transform_selector.get_transform(is_train=False)
    
    ## <추후 수정 예정>
    train_dataset = CustomDataset(
        root_dir = train_data_dir,
        data_df = train_df,
        transform = train_transform
    )
    
    val_dataset = CustomDataset(
        root_dir = val_data_dir,
        data_df = val_df,
        transform = val_transform
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False
    )
    ## </추후 수정 예정>
    

    model_selector = ModelSelector("cnn", num_classes)    
    model = model_selector.get_model()
    optimizer = get_optimizer(model, 'adam', lr)
    loss = nn.CrossEntropyLoss()
    
    
    # 스케줄러 초기화
    scheduler_step_size = 30 # int 30
    scheduler_gamma = 0.1 # float 0.1
    
    steps_per_epoch = len(train_dataloader)
    
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=scheduler_step_size,
        gamma=scheduler_gamma
    )
    
    model.to(device)
    
    trainer = Trainer(
        model=model,
        device=device,
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss,
        epochs=epochs,
        result_path=save_result_path
    )
    
    trainer.train()
    
    matrics_info = None
    
    