import os
import argparse
from argparse import Namespace
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

from model import model_selection

from util.data import CustomDataset, HoDataLoad   # hobbang: Dataset, DataLoader 코드 하나로 합체
from util.augmentation import TransformSelector
from util.optimizers import get_optimizer
from util.losses import CustomLoss
from trainer import Trainer

from model.model_selection import ModelSelector

def run_train(args:Namespace) -> None:
    ## device와 seed 설정
    device = torch.device(args.device)
    
    ## 데이터 경로 및 CSV 파일 경로
    data_root = args.data_root
    train_data_dir = data_root + "/train/"
    train_data_info_file = data_root + '/train.csv'
    val_data_info_file = data_root + '/val.csv'
    save_result_path = "./train_result"
    
    ## 데이터 증강
    transform_type = args.transform_type
    
    ## 모델, 옵티마이저, 손실 함수(로스 함수)
    model_type = args.model
    loss_type = args.loss
    optimizer_type = args.optim
    
    ## 학습률, 클래스 개수, 에포크, 배치 크기, 돌려서 학습
    epochs = args.epochs
    batch_size = args.batch
    lr = args.lr
    num_classes = args.num_classes
    r_epoch = args.r_epochs
    
    ## 학습 재개 정보
    resume = args.resume
    weights_path = args.weights_path
    
    config = {'epoches': epochs, 'batch_size': batch_size, 'learning_rate': lr}
    # wandb.init(project='my-test-project', config=config)
    
    ## 데이터 증강 및 세팅
    transform_selector = TransformSelector(transform_type=transform_type)
    
    # train_df = pd.read_csv(train_data_info_file)
    # val_df = pd.read_csv(val_data_info_file)
    
    train_info = pd.read_csv(train_data_info_file)
    
    train_df, val_df = train_test_split(
        train_info, 
        test_size=0.2,
        stratify=train_info['target'],
        random_state=42
    )
    
    train_transform = transform_selector.get_transform(augment=True)
    val_transform = transform_selector.get_transform(augment=False)
    
    train_dataset = CustomDataset(train_data_dir, train_df, transform=train_transform)
    val_dataset = CustomDataset(train_data_dir, val_df, transform=val_transform)
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    ## 학습 모델
    if model_type == "cnn":
        model_selector = ModelSelector(
            "cnn", 
            num_classes, 
        )
    elif model_type == 'timm-resnet18':
        model_selector = ModelSelector(
            "timm", 
            num_classes, 
            model_name='resnet18', 
            pretrained=True
        )
    else:
        raise Exception('모델을 찾을 수 없습니다.')
    
    model = model_selector.get_model()
    
    ## 옵티마이저
    optimizer = get_optimizer(model, optimizer_type, lr)

    ## 손실 함수
    if loss_type == 'CE':
        loss = CustomLoss()
    
    ## Scheduler 관련
    scheduler_gamma = args.lr_scheduler_gamma # float 0.1
    steps_per_epoch = len(train_dataloader)
    
    epochs_per_lr_decay = args.lr_scheduler_epochs_per_decay
    scheduler_step_size = steps_per_epoch * epochs_per_lr_decay
    
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=scheduler_step_size,
        gamma=scheduler_gamma
    )
    
    model.to(device)
    
    ## 학습 시작
    trainer = Trainer(
        model=model,
        device=device,
        resume=resume,
        weights_path=weights_path,
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

def parse_args_and_config() -> Namespace:
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--mode', type=str, default='train', help='Select mode train or test default is train', action='store')
    parser.add_argument('--device', type=str, default='cpu', help='Select device to run, default is cpu', action='store')
    parser.add_argument('--data_root', type=str, default='./data', help='Path to data root', action='store')
    parser.add_argument('--train_csv', type=str, default='./data/train.csv', help='Path to train csv', action='store')
    parser.add_argument('--val_csv', type=str, default='./data/val.csv', help='Path to val csv', action='store')
    parser.add_argument('--test_csv', type=str, default='./data/test.csv', help='Path to test csv', action='store')
    parser.add_argument('--num_classes', type=int, default=500, help="Select number of classes, default is 500", action='store')
    parser.add_argument('--auto_split', type=bool, default=True, help='Set auto_split, requires train & val csv if False', action='store')
    parser.add_argument('--split_seed', type=int, default=42, help='Set split_seed, default is 42', action='store')
    parser.add_argument('--stratify', type=bool, default=True, help='Set auto_split, requires train & val csv if False', action='store')
    parser.add_argument('--model', type=str, default='cnn', help='Select a model to train, default is cnn', action='store')
    parser.add_argument('--lr', type=float, default=0.01, help='Select Learning Rate, default is 0.01', action='store')
    parser.add_argument('--lr_scheduler', type=str, default="stepLR", help='Select LR scheduler, default is stepLR', action='store')
    parser.add_argument('--lr_scheduler_gamma', type=float, default=0.1, help='Select LR scheduler gamma, default is 0.1', action='store')
    parser.add_argument('--lr_scheduler_epochs_per_decay', type=int, default=2, help='Select LR scheduler epochs_per_decay, default is 2', action='store')
    parser.add_argument('--batch', type=int, default=64, help='Select batch_size, default is 64', action='store')
    parser.add_argument('--loss', type=str, default='CE', help='Select Loss, default is Cross Entropy(CE)', action='store')
    parser.add_argument('--optim', type=str, default='adam', help='Select a optimizer, default is adam', action='store')
    parser.add_argument('--epochs', type=int, default='100', help='Select total epochs to train, default is 100 epochs', action='store')
    parser.add_argument('--r_epochs', type=int, default='2', help='Select total data swap epochs, default is last 2 epochs', action='store')
    parser.add_argument('--seed', type=int, default=2024, help='Select seed, default is 2024', action='store')
    parser.add_argument('--transform_type', type=str, default='albumentations', help='Select transform type, default is albumentation', action='store')
    
    parser.add_argument('--resume', type=bool, default=False, help='resuming training, default is False meaning new training (requires weights_path for checkpoints)', action='store')
    parser.add_argument('--weights_path', type=str, default=None, help='Path to resuming weight_path, default is None', action='store')
    
    return parser.parse_args()

if __name__=='__main__':
    
    ## 설정 및 하이퍼파라미터 가져오기
    args = parse_args_and_config()
    
    # cuda 적용
    if args.device.lower() == 'cuda':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        assert device == 'cuda', 'cuda로 수행하려고 하였으나 cuda를 찾을 수 없습니다.'
    else:
        device = 'cpu'

    # seed값 설정
    seed = args.seed
    deterministic = True

    random.seed(seed) # random seed 고정
    np.random.seed(seed) # numpy random seed 고정
    torch.manual_seed(seed) # torch random seed 고정
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    run_train(args)
    
    