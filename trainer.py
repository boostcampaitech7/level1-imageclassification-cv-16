# 필요 library들을 import합니다.
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from util.checkpoints import save_checkpoint
from typing import Tuple, Any, Callable, List, Optional, Union



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
        result_path: str,
        train_total: int,
        val_total: int,
        r_epoch: int = 0

        #wandb 관련 부분
        #log_val_metrics_every_epoch: bool = False  # 검증 메트릭을 매 epoch마다 계산할지 여부
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
        self.train_total = train_total
        self.val_total = val_total
        self.r_epoch = r_epoch
        
        self.best_val_loss = float('inf')
        self.checkpoint_dir = "./checkpoints"

        # wandb 익명 모드로 초기화
        #wandb.init(project="Project1", anonymous="allow")
        wandb.watch(self.model, log="all")  # 모델을 모니터링하도록 설정


    def save_checkpoint_tmp(self, epoch, val_loss):
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


    def train_epoch(self, train_loader) -> float:
        # 한 에폭 동안의 훈련을 진행
        self.model.train()
        
        total_loss = 0.0
        train_correct = 0
        progress_bar = tqdm(train_loader, desc="Training", leave=False)

        # Train step에서 추가 메트릭 계산 (accuracy, F1 score, confusion matrix)
        #all_preds = []  # 모든 예측을 저장할 리스트
        #all_labels = []  # 모든 실제 라벨을 저장할 리스트
        
        for images, targets in progress_bar:
            images, targets = images.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            loss = self.loss_fn(outputs, targets)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item() * targets.shape[0]

            outputs = torch.argmax(outputs, dim = 1)
            acc = (outputs == targets).sum().item()
            train_correct += acc
            progress_bar.set_postfix(loss=loss.item())
        

        avg_train_loss = total_loss / len(self.train_loader)
        avg_train_acc = train_correct / len(self.train_loader.dataset)

        # wandb에 학습 손실 및 정확도 로깅
        #wandb.log({'Train Loss': avg_train_loss, 'Train Accuracy': avg_train_acc})
        
        #return avg_train_loss, avg_train_acc
    
        # 이거 멘토링 때 데이터셋으로 불러야 총 데이터 개수라지않았나??
        # train_loader.dataset 전체 데이터셋에 대한 정확도 계산
        return total_loss, train_correct


    def validate(self, val_loader) -> float:
        # 모델의 검증을 진행
        self.model.eval()
        val_correct = 0
        total_loss = 0.0
        progress_bar = tqdm(val_loader, desc="Validating", leave=False)
        
        global log_images
        log_images= []
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(progress_bar):
                images, targets = images.to(self.device), targets.to(self.device)
                
                # 모델 예측 출력 (로짓)
                outputs = self.model(images)    
                
                # 손실 계산
                loss = self.loss_fn(outputs, targets)
                total_loss += loss.item() * targets.shape[0]

                # 예측 값 계산
                outputs = torch.argmax(outputs, dim = 1)
                val_correct += (outputs == targets).sum().item()
                
                # 진행 상황 표시
                progress_bar.set_postfix(loss=loss.item())

                #if (outputs == targets).sum().item() == 0: #틀린 거면
                    #log_images.append(wandb.Image(images[0], caption="Pred: {} Truth: {}".format(outputs[0].item(), targets[0])))    
                
                # 예측과 실제 값이 다른 경우만 이미지 로그
                for i in range(len(images)):
                    if outputs[i].item() != targets[i].item():
                        caption = f"Batch: {batch_idx}, Index: {i}, Pred: {outputs[i].item()}, Truth: {targets[i].item()}"
                        log_images.append(wandb.Image(images[i], caption=caption))
                
                

        #wandb.log({"Test Images": log_images})
        #wandb
        #avg_val_loss = total_loss / len(self.val_loader)
        #avg_val_acc = val_correct / len(self.val_loader.dataset)
        # wandb에 검증 손실 및 정확도 로깅 (필요할 때만)
        #wandb.log({'Val Loss': avg_val_loss, 'Val Accuracy': avg_val_acc})
        #return avg_val_loss, avg_val_acc        # 전체 데이터셋에 대한 정확도     
        #wandb.log({"Test Images": log_images})
        
        return total_loss, val_correct



    def train(self) -> None:
        # 전체 훈련 과정을 관리
        for epoch in range(self.epochs):
            train_loss, train_acc = 0.0, 0.0
            val_loss, val_acc = 0.0, 0.0
            print(f"Epoch {epoch+1}/{self.epochs}")
            
            if epoch < self.epochs - self.r_epoch:
                train_loss, train_acc = self.train_epoch(self.train_loader)
                val_loss, val_acc = self.validate(self.val_loader)

                train_loss, train_acc = train_loss / self.train_total, train_acc / self.train_total
                val_loss, val_acc = val_loss / self.val_total, val_acc / self.val_total
            else:
                train_loss, train_acc = self.train_epoch(self.val_loader)
                val_loss, val_acc = self.validate(self.train_loader)
            
                train_loss, train_acc = train_loss / self.val_total, train_acc / self.val_total
                val_loss, val_acc = val_loss / self.train_total, val_acc / self.train_total
            
            print(f"Epoch {epoch+1}, Train Loss: {train_loss:.8f} | Train Acc: {train_acc:.8f} \nValidation Loss: {val_loss:.8f} | Val Acc: {val_acc:.8f}\n")

            # wandb code 추가
            wandb.log({'Epoch': epoch+1, 'Train Accuracy': train_acc, 'Train Loss': train_loss, 'Val Accuracy': val_acc, 'Val Loss': val_loss, 'Test Images': log_images}, step=epoch)
            #andb.log({'Epoch': epoch+1, 'Train Accuracy': train_acc, 'Train Loss': train_loss, 'Val Accuracy': val_acc, 'Val Loss': val_loss}, step=epoch)

        
            # 체크포인트 trainloss에 대해 찍어야하?
            # self.save_checkpoint_tmp(epoch, val_loss)
            self.save_checkpoint_tmp(epoch, train_loss)
            
            self.scheduler.step()
        # 최종 체크포인트
        # self.final_save_model(epoch, val_loss)
        self.final_save_model(epoch, train_loss)
        wandb.finish()
