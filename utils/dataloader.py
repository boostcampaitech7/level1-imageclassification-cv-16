import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms, utils
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        ## 초기화 함수
        ## 사용할 변수 : self.labels, self.root_dir, self.transform
        self.labels = os.listdir(root_dir)
        self.root_dir = root_dir
        self.transform = transform
        
        self.label_to_idx = {label:i for i, label in enumerate(self.labels)}
        
        self.data_dir = []
        
        for label in self.labels:
            data_path = os.path.join(root_dir, label)
            data_files = os.listdir(data_path)
            for file_name in data_files: 
                self.data_dir.append([os.path.join(data_path, file_name), label])

    def __len__(self):
        ## 데이터 개수 반환 함수
        return len(self.data_dir)
    
    def __getitem__(self, idx):
        ## 데이터 증강 기법을 적용한 특정 인덱스의 입력 x와 레이블 y 반환 함수
        ## 반환 값 : dict형의 {x, y}
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # 텐서  -> 리스트로 변환
        
        img_path, label = self.data_dir[idx]

        # 이미지 채널 수 유지하니까 1로 만들어서 흑백
        image = Image.open(img_path).convert('RGB')

        label_idx = self.label_to_idx[label]

        # 데이터 증강
        if self.transform:
            image = self.transform(image)

        return image, label_idx
               
# # 테스트용
# def test(csv_file=None, root_dir=None, transform=None):
#     CD = CustomDataset(csv_file, root_dir, transform)
    
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     # transforms.Normalize((0.485,0.456,0.406), (0.229, 0.224, 0.225))
# ])

# test("", "./data", transform)