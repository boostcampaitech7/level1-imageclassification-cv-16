import os
import torch
import pandas as pd
import numpy as np
from typing import Callable
import cv2 # <= image load를 위해 필요함. 사용하기 위해 pip install opencv-python 필요
from torch.utils.data import Dataset
from torchvision import transforms, utils
from PIL import Image

class CustomDataset(Dataset):
    def __init__(
        self, 
        root_dir: str,
        data_df: pd.DataFrame, 
        transform: Callable = None, 
        is_inference: bool = False  
        ):
        # CustomDataset 초기화
        # 매개변수 
        #  - root_dir : 데이터 저장 위치
        #  - data_df : 이미지 정보 관련 데이터셋
        #  - transform : 적용 이미지 변환 처리
        #  - is_inference : True => 추론(inference) False => 학습(Training)
        
        self.root_dir = root_dir
        self.data_df = data_df
        self.transform = transform
        self.is_inference = is_inference
        self.image_paths = data_df['image_path'].tolist()
        
        if not self.is_inference:
            self.targets = data_df['target'].tolist()

    def __len__(self):
        # 데이터 개수 반환 함수
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # 전달 받은 인덱스에 해당하는 이미지 로드 및 변환 적용 후 반환
        # 반환 값 : is_inference=False => 이미지, 레이블, is_inference=True => 이미지
        img_path = os.path.join(self.root_dir, self.image_paths[idx]) # 완전한 이미지 path로 만들기
        img = cv2.imread(img_path, cv2.IMREAD_COLOR) # BGR 컬러 포맷인 numpy 배열로 읽어옴
        img = cv2.cvtcolor(img, cv2.COLOR_BGR2RGB) # BGR 포맷을 RGB 포맷으로 변환
        img = self.transform(img) # 이미지 변환 수행
        
        if self.is_inference: # 추론 시
            return img # 이미지만 반환
        else: # 학습 시
            target = self.targets[idx] # 해당 이미지의 레이블 
            return img, target # 이미지와 레이블 반환
               
# # 테스트용
# def test(csv_file=None, root_dir=None, transform=None):
#     CD = CustomDataset(csv_file, root_dir, transform)
    
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     # transforms.Normalize((0.485,0.456,0.406), (0.229, 0.224, 0.225))
# ])

# test("", "./data", transform)