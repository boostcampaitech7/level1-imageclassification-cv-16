# 필요 library들을 import합니다.
import os
from typing import Tuple, Any, Callable, List, Optional, Union

import cv2
import timm
import torch
import numpy as np
import pandas as pd
import albumentations as A
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models, datasets, transforms
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from albumentations.pytorch import ToTensorV2
from util.checkpoints import load_checkpoint, save_checkpoint

class Trainer: # 변수 넣으면 바로 학습되도록
    def __init__( # 여긴 config로 나중에 빼야하는지 이걸 유지하는지
        self, 
        model: nn.Module, 
        device: torch.device, 
        train_loader: DataLoader, 
        val_loader: DataLoader, 
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler,
        loss_fn: torch.nn.modules.loss._Loss, 
        epochs: int,
        result_path: str
    ):
        # 클래스 초기화: 모델, 디바이스, 데이터 로더 등 설정
        self.model = model  # 훈련할 모델
        self.device = device  # 연산을 수행할 디바이스 (CPU or GPU)
        self.train_loader = train_loader  # 훈련 데이터 로더
        self.val_loader = val_loader  # 검증 데이터 로더
        self.optimizer = optimizer  # 최적화 알고리즘
        self.scheduler = scheduler # 학습률 스케줄러
        self.loss_fn = loss_fn  # 손실 함수
        self.epochs = epochs  # 총 훈련 에폭 수
        self.result_path = result_path  # 모델 저장 경로
        self.best_models = [] # 가장 좋은 상위 3개 모델의 정보를 저장할 리스트
        self.lowest_loss = float('inf') # 가장 낮은 Loss를 저장할 변수

    def _save_checkpoint(self, epoch, val_loss):
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            checkpoint_filepath = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch + 1}.pth')
            save_checkpoint(self.model, self.optimizer, epoch, val_loss, checkpoint_filepath)
            print(f"Checkpoint updated at epoch {epoch + 1} and saved as {checkpoint_filepath}")
            
    # save model과 체크포인트의 차이는? 아예 다른 코드인지
    def final_save_model(self, epoch, loss) -> None:
        # checkpoints 폴더가 없으면 생성
        checkpoint_dir = 'checkpoints'
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 체크포인트 저장
        final_checkpoint_filepath = os.path.join(checkpoint_dir, 'final_checkpoint.pth')
        save_checkpoint(self.model, self.optimizer, epoch, loss, final_checkpoint_filepath)
        print(f"Final checkpoint saved as {final_checkpoint_filepath}")

    def train_epoch(self) -> float:
        # 한 에폭 동안의 훈련을 진행
        self.model.train()
        
        total_loss = 0.0
        train_correct = 0
        progress_bar = tqdm(self.train_loader, desc="Training", leave=False)
        
        for images, targets in progress_bar:
            images, targets = images.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images) # pred
            
            loss = self.loss_fn(outputs, targets)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step() # 스케줄러 더 알아보기
            
            total_loss += loss.item() # 손실 계산
            
            # 정확도 계산
            outputs = torch.argmax(outputs, dim = 1)
            acc = (outputs == targets).sum().item()
            train_correct += acc
            progress_bar.set_postfix(loss=loss.item())
        
        # 이거 멘토링 때 데이터셋으로 불러야 총 데이터 개수라지않았나??
        # train_loader.dataset 전체 데이터셋에 대한 정확도 계산
        return total_loss / len(self.train_loader), train_correct/len(self.train_loader.dataset)

    def validate(self) -> float:
        # 모델의 검증을 진행
        self.model.eval()
        val_correct = 0
        total_loss = 0.0
        progress_bar = tqdm(self.val_loader, desc="Validating", leave=False)
        
        with torch.no_grad():
            for images, targets in progress_bar:
                images, targets = images.to(self.device), targets.to(self.device)
                
                outputs = self.model(images)    
                
                loss = self.loss_fn(outputs, targets)
                
                total_loss += loss.item()
                outputs = torch.argmax(outputs, dim = 1)
                val_correct += (outputs == targets).sum().item()
                progress_bar.set_postfix(loss=loss.item())
        
        return total_loss / len(self.val_loader), val_correct / len(self.val_loader.dataset) # 전체 데이터셋에 대한 정확도

    def train(self) -> None:
        # 전체 훈련 과정을 관리
        for epoch in range(self.epochs):
            print(f"Epoch {epoch+1}/{self.epochs}")
            
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()
            # 볼 때 너무 길면 자르기
            print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.3f}, Validation Loss: {val_loss:.4f} | Val Acc: {val_acc:.3f}\n")

            # wandb code 추가
            # wandb.log({'Train Accuracy': train_acc, 'Train Loss': avg_train_loss, "Epoch": epoch + 1})
            
            # 체크포인트 trainloss에 대해 찍어야하?
            # self._save_checkpoint(epoch, val_loss)
            self._save_checkpoint(epoch, train_loss)
            
            self.scheduler.step()
        # 최종 체크포인트
        # self.final_save_model(epoch, val_loss)
        self.final_save_model(epoch, train_loss)