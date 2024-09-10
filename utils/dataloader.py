import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms, utils
from PIL import Image
import matplotlib.pyplot as plt

class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, ho_path='../data'):
        ## 초기화 함수
        ## 사용할 변수 : self.labels, self.root_dir, self.transform
        self.labels = os.listdir(root_dir)
        self.root_dir = root_dir
        self.transform = transform
        
        self.label_to_idx = {label:i for i, label in enumerate(self.labels)}
        
        self.data_dir = []
        
        
    ################################################################## ho_change
        # for label in self.labels:
        #     data_path = os.path.join(root_dir, label)
        #     data_files = os.listdir(data_path)
        #     for file_name in data_files: 
        #         self.data_dir.append([os.path.join(data_path, file_name), label])

        self.train_path = os.path.join(ho_path, 'train')
        self.df_path = os.path.join(ho_path, 'train.csv')
        self.trans = transforms.Compose(
            [
                transforms.Resize([224,224]), 
                transforms.ToTensor(),
                # transforms.Normalize((0.485,0.456,0.406), (0.229, 0.224, 0.225))     ######## transforms에 normalize 켜면 그림 형체 알아보기 힘듦 : 논의할 내용
                ]
            )
        df = pd.read_csv(self.df_path)
        self.img_path = {i:[] for i in df['target'].unique()}
        for i in df['class_name'].unique():
            rows = df[df['class_name']==i]
            for image_path, target in zip(rows['image_path'], rows['target']):
                self.img_path[target].append(os.path.join(self.train_path, image_path))

        self.tensors={i:[] for i in range(500)}
        for i in self.img_path.keys():
            for path in self.img_path[i]:
                image=Image.open(path).convert('RGB')
                image = self.trans(image).transpose(1,2).transpose(0,2)
                self.tensors[i].append(image)
    #################################################################

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
    
    ################################################################## ho_change
    def print_images(self):
        fig, axes = plt.subplots(25,20, figsize=[25,20])
        for i in range(25):
            for j in range(20):
                if i==24 and j==19:
                    axes[i][j].imshow(self.tensors[0][0])
                    axes[i][j].set_xticks([])
                    axes[i][j].set_yticks([])
                    break
                axes[i][j].imshow(self.tensors[(i+1)*(j+1)][0])
                axes[i][j].set_xticks([])
                axes[i][j].set_yticks([])
    ##################################################################


               
# # 테스트용
# def test(csv_file=None, root_dir=None, transform=None):
#     CD = CustomDataset(csv_file, root_dir, transform)
    
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     # transforms.Normalize((0.485,0.456,0.406), (0.229, 0.224, 0.225))
# ])

# test("", "./data", transform)