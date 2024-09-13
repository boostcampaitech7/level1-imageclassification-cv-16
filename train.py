import os
import argparse
import random
import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
from tqdm import tqdm

from model import modelSelection

from util.data import CustomDataset, HoDataLoad   # hobbang: Dataset, DataLoader 코드 하나로 합체
from util.augmentation import TransformSelector
from util.optimizers import get_optimizer
from util.losses import CustomLoss
from trainer import Trainer

from model.modelSelection import ModelSelector

def run_train():
    # 학습 데이터의 경로와 정보를 가진 파일의 경로를 설정.
    train_data_dir = "./data/train"
    train_data_info_file = "./data/train.csv"
    val_data_info_file = "./data/val.csv"
    save_result_path = "./train_result"
    
    epochs = 10
    batch_size = 64
    lr = 0.001
    num_classes = 500
    r_epoch = 2
    
    config = {'epoches': epochs, 'batch_size': batch_size, 'learning_rate': lr}
    wandb.init(project='my-test-project', config=config)
    
    transform_selector = TransformSelector(transform_type = "albumentations")
    
    # train_df = pd.read_csv(train_data_info_file)
    # val_df = pd.read_csv(val_data_info_file)
    
    train_info = pd.read_csv(train_data_info_file)
    
    train_df, val_df = train_test_split(
        train_info, 
        test_size=0.2,
        stratify=train_info['target'],
        random_state=42
    )
    
    train_transform = transform_selector.get_transform(augment=False)
    val_transform = transform_selector.get_transform(augment=False)
    
    train_dataset = CustomDataset(
        root_dir = train_data_dir,
        data_df = train_df,
        transform = train_transform
    )
    
    val_dataset = CustomDataset(
        root_dir = train_data_dir,
        data_df = val_df,
        transform = val_transform
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    model_selector = ModelSelector("timm", num_classes, 
                                    model_name='resnet18', pretrained=True)    
    model = model_selector.get_model()
    optimizer = get_optimizer(model, 'adam', lr)
    loss = CustomLoss()
    
    # 스케줄러 초기화
    scheduler_step_size = 30 # int 30
    scheduler_gamma = 0.1 # float 0.1
    
    steps_per_epoch = len(train_dataloader)
    
    epochs_per_lr_decay = 2
    scheduler_step_size = steps_per_epoch * epochs_per_lr_decay
    
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
        result_path=save_result_path,
        train_total=train_df.shape[0],
        val_total=val_df.shape[0],
        r_epoch=r_epoch
    )
    
    trainer.train()
    
    matrics_info = None

def parse_args_and_config():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_root', type=str, default='./data', help='Path to data root', action='store')
    parser.add_argument('--train_csv', type=str, default='./data/train.csv', help='Path to train csv', action='store')
    parser.add_argument('--val_csv', type=str, default='./data/val.csv', help='Path to val csv', action='store')
    parser.add_argument('--auto_split', type=bool, default=True, help='Set auto_split, requires train & val csv if False', action='store')
    parser.add_argument('--')
    

if __name__=='__main__':
    
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

    run_train()
    
    