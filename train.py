import os
from argparse import Namespace

from args import Custom_arguments_parser
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

from util.data import CustomDataset
from util.augmentation import TransformSelector
from util.optimizers import get_optimizer
from util.losses import CustomLoss
from trainer import Trainer

from model.model_selection import ModelSelector
from util.data import HoDataLoader

def run_train(args:Namespace) -> None:
    ## device와 seed 설정
    device = torch.device(args.device)
    early_stopping = args.early_stopping
    
    ## 데이터 경로 및 CSV 파일 경로
    data_root = args.data_root
    train_data_dir = data_root + "/train/"
    train_data_info_file = args.csv_path
    val_data_info_file = args.val_csv
    save_result_path = "./train_result"
    
    ## 데이터 증강
    transform_type = args.transform_type
    stratify_column = args.stratify
    height = args.height
    width = args.width
    
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
    weights_path = args.checkpoint_path
    
    # 출력 관련 (progress bar)
    verbose = args.verbose
    
    ## 데이터 증강 및 세팅
    transform_selector = TransformSelector(transform_type=transform_type)
    
    # train_df = pd.read_csv(train_data_info_file)
    # val_df = pd.read_csv(val_data_info_file)
    
    train_info = pd.read_csv(train_data_info_file)
    
    train_df, val_df = train_test_split(
        train_info, 
        test_size=0.2,
        stratify=train_info[stratify_column],
        random_state=42
    )
    
    train_transform = transform_selector.get_transform(augment=True, height=height, width=width, augment_list=args.augmentations, adjust_ratio=args.adjust_ratio)
    val_transform = transform_selector.get_transform(augment=False, height=height, width=width, augment_list=args.augmentations, adjust_ratio=args.adjust_ratio)
    
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

    X = pd.read_csv(train_data_info_file).iloc[:, 1]
    y = pd.read_csv(train_data_info_file).iloc[:, -1]
    num_folds = 5
    custom_loader = HoDataLoader(
        X, 
        y, 
        train_transform, 
        val_transform, 
        batch_size=batch_size, 
        csv_path=train_data_info_file,
        num_folds=num_folds
    )

    ## 학습 모델
    if 'timm' in model_type:
        model_selector = ModelSelector(
            "timm", 
            num_classes, 
            model_name=model_type.split("-")[-1], 
            pretrained=True
        )
    else:
        model_selector = ModelSelector(
            model_type,
            num_classes,
        )
    
    model = model_selector.get_model()
    model.to(device)
    
    ## 옵티마이저
    optimizer = get_optimizer(model, optimizer_type, lr)

    ## 손실 함수
    if loss_type == 'CE':
        loss = CustomLoss()
    
    ## Scheduler 관련
    if args.lr_scheduler == 'stepLR':
        scheduler_gamma = args.lr_scheduler_gamma # float 0.1
        steps_per_epoch = len(train_dataloader)
        
        epochs_per_lr_decay = args.lr_scheduler_epochs_per_decay
        scheduler_step_size = steps_per_epoch * epochs_per_lr_decay
        
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_step_size,
            gamma=scheduler_gamma
        )
    elif args.lr_scheduler == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=args.lr_scheduler_gamma,
            patience=2,
            verbose=True
        )

    model.to(device)    

    config = {'epoches': epochs, 'batch_size': batch_size, 'learning_rate': lr, 
              'model': model, 'device': device, 
              'optimizer': optimizer, 'scheduler': scheduler, 'loss_fn': loss}
    wandb.init(project='Project1', config=config)    

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
        r_epoch=r_epoch,
        early_stopping=early_stopping,
        verbose=verbose,
        args=args,
        custom_loader=custom_loader
    )
    is_k_fold = True
    if is_k_fold:
        trainer.k_fold_train()
    else:
        trainer.train()
    
    matrics_info = None

if __name__=='__main__':
    
    ## 설정 및 하이퍼파라미터 가져오기
    train_parser = Custom_arguments_parser(mode='train')
    args = train_parser.get_parser()
    
    # cuda 적용
    if args.device.lower() == 'cuda':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        assert device == 'cuda', 'cuda로 수행하려고 하였으나 cuda를 찾을 수 없습니다.'
    elif args.device.lower() == 'mps':
        device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        assert device == 'mps', 'mps를 찾을 수 없습니다.'
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
    
    