# 필요 library들을 import합니다.
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from util.checkpoints import save_checkpoint

class Trainer: # 변수 넣으면 바로 학습되도록
    def __init__( # 여긴 config로 나중에 빼야하는지 이걸 유지하는지
        self, 
        model: nn.Module,
        resume: bool,
        weights_path: str,
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
        r_epoch: int,
        early_stopping: int,
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
        self.best_val_acc = 0.0
        self.checkpoint_dir = "./checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.start_epoch = 0
        self.resume = resume # 학습 재개를 위한 것인지
        self.weights_path = weights_path # 학습 재개를 위해 불러와야할 가중치 주소
        
        self.early_stopping = early_stopping

    def save_checkpoint_tmp(self, epoch, val_loss, val_acc) -> None:
        if val_acc >= self.best_val_acc+0.01:
            self.best_val_loss = val_loss
            self.best_acc_loss = val_acc
            checkpoint_filepath = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch + 1}.pth')
            save_checkpoint(self.model, self.optimizer, self.scheduler, epoch, val_loss, checkpoint_filepath)
            print(f"Checkpoint updated at epoch {epoch + 1} and saved as {checkpoint_filepath}")
            
    # save model과 체크포인트의 차이는? 아예 다른 코드인지
    def final_save_model(self, epoch, loss) -> None:
        # 체크포인트 저장
        final_checkpoint_filepath = os.path.join(self.checkpoint_dir, 'final_checkpoint.pth')
        save_checkpoint(self.model, self.optimizer, epoch, loss, final_checkpoint_filepath)
        print(f"Final checkpoint saved as {final_checkpoint_filepath}")

    def train_epoch(self, train_loader) -> float:
        # 한 에폭 동안의 훈련을 진행
        self.model.train()
        
        total_loss = 0.0
        train_correct = 0
        progress_bar = tqdm(train_loader, desc="Training", leave=False)
        
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
        
        # 이거 멘토링 때 데이터셋으로 불러야 총 데이터 개수라지않았나??
        # train_loader.dataset 전체 데이터셋에 대한 정확도 계산
        return total_loss, train_correct

    def validate(self, val_loader) -> float:
        # 모델의 검증을 진행
        self.model.eval()
        val_correct = 0
        total_loss = 0.0
        progress_bar = tqdm(val_loader, desc="Validating", leave=False)
        
        with torch.no_grad():
            for images, targets in progress_bar:
                images, targets = images.to(self.device), targets.to(self.device)
                
                outputs = self.model(images)    
                
                loss = self.loss_fn(outputs, targets)
                
                total_loss += loss.item() * targets.shape[0]
                outputs = torch.argmax(outputs, dim = 1)
                val_correct += (outputs == targets).sum().item()
                progress_bar.set_postfix(loss=loss.item())
        
        return total_loss, val_correct

    def train(self) -> None:
        if self.resume:
            self.load_settings()

        # 전체 훈련 과정을 관리
        count = 0
        for epoch in range(self.start_epoch, self.epochs):
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
            # wandb.log({'Train Accuracy': train_acc, 'Train Loss': avg_train_loss, "Epoch": epoch + 1})
            
            # self.save_checkpoint_tmp(epoch, val_loss)
            self.save_checkpoint_tmp(epoch, val_loss, val_acc)
            
            self.scheduler.step()
            
            if val_acc > self.best_val_acc:
                count = 0
            else:
                count += 1
                if count == self.early_stopping:
                    print(f"{self.early_stopping} 에포크 동안 개선이 없어 학습이 중단됩니다.")
                    break
        # 최종 체크포인트
        # self.final_save_model(epoch, val_loss)
        self.final_save_model(epoch, train_loss)
        
    def load_settings(self) -> None:
        ## 학습 재개를 위한 모델, 옵티마이저, 스케줄러 가중치 및 설정을 불러옵니다.
        print("loading prev training setttings")
        try:
            setting_info = torch.load(
                self.weights_path,
                map_location='cpu'
            )
            self.start_epoch = setting_info['epoch']
            self.model.load_state_dict(setting_info['model_state_dict'])
            self.optimizer.load_state_dict(setting_info['optimizer_state_dict'])
            self.scheduler.load_state_dict(setting_info['scheduler_state_dict'])
            print("loading successful")
        except:
            raise Exception('학습 재개를 위한 정보를 불러오는데 문제가 발생하였습니다')