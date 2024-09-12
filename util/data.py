import os
import torch
import pandas as pd
import numpy as np
from typing import Callable
import cv2 # <= image load를 위해 필요함. 사용하기 위해 pip install opencv-python 필요
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, utils
from PIL import Image
import matplotlib.pyplot as plt

class CustomDataset(Dataset):
    def __init__(
        self, 
        root_dir: str,
        data_df: pd.DataFrame, 
        transform: Callable = None, 
        is_inference: bool = False,
        one_hot: bool = False  
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
        self.one_hot = one_hot
        if self.one_hot:
            # class 개수 크기의 one_hot template 생성 
            # 비효율적일 수 있음 / target이 정수인 경우에만 동작함
            self.one_hot_template = [0 for i in range(len(data_df['target'].unique()))]
        
        if not self.is_inference:
            self.targets = data_df['target'].tolist()

    def __len__(self) -> int: ## 이런걸 전부 포함하도록 하는게 좋음
        # 데이터 개수 반환 함수
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # 전달 받은 인덱스에 해당하는 이미지 로드 및 변환 적용 후 반환
        # 반환 값 : is_inference=False => 이미지, 레이블, is_inference=True => 이미지
        img_path = os.path.join(self.root_dir, self.image_paths[idx]) # 완전한 이미지 path로 만들기
        img = cv2.imread(img_path, cv2.IMREAD_COLOR) # BGR 컬러 포맷인 numpy 배열로 읽어옴
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # BGR 포맷을 RGB 포맷으로 변환
        img = self.transform(img) # 이미지 변환 수행
        
        if self.is_inference: # 추론 시
            return img # 이미지만 반환
        else: # 학습 시
            target = self.targets[idx] # 해당 이미지의 레이블
            if self.one_hot: # One_hot으로 반환하고자 하는 경우
                temp = self.one_hot_template.copy() # template을 가져옴
                temp[target] = 1 # 레이블에 해당하는 위치에 1 나머지는 0
                target = temp # 반환을 위해 target에 할당
            return img, target # 이미지와 레이블 반환


def print_image(idx:list, train:bool = True):
    image_paths= pd.read_csv('./data/train.csv').iloc[:, 1] if train else pd.read_csv('./data/test.csv').iloc[:, 0]
    folder_path= './data/train' if train else './data/test'
    if type(idx) == list:
        # # 리스트로 2개의 index 받았을 때 예시
        # fig, ax = plt.subplots(1,2, figsize=[12,4])
        # ax[0].imshow(Image.open(os.path.join(folder_path, image_paths[idx[0]])))
        # ax[1].imshow(Image.open(os.path.join(folder_path, image_paths[idx[1]])))
        row_size=(len(idx)+1)//2
        fig, ax = plt.subplots(row_size, 2)
        num=0
        for r in range(row_size):
            for c in range(2):
                if num == len(idx):
                    break
                if row_size == 1:
                    ax[num].set_xticks([])
                    ax[num].set_yticks([])
                    ax[num].imshow(Image.open(os.path.join(folder_path, image_paths[idx[num]])))
                else:
                    ax[r][c].set_xticks([])
                    ax[r][c].set_yticks([])
                    ax[r][c].imshow(Image.open(os.path.join(folder_path, image_paths[idx[num]])))
                num+=1
    else:
        fig, ax = plt.subplots(1,1, figsize=[12,4])
        ax.imshow(Image.open(os.path.join(folder_path, image_paths[idx])))
        ax.set_xticks([])
        ax.set_yticks([])

class HoDataset(Dataset):
    def __init__(self, csv_file, root_dir, batch_size=32, is_train:bool=False):
        '''
        Args:
            csv_file (string): csv 파일 경로
            root_dir (string): 모든 이미지가 존재하는 디렉토리 경로
            transform (callable, optional): 샘플에 적용될 Optional transform
        '''

        self.mode = 'train' if is_train else 'test'
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.batch_size=batch_size
        ## test: root_dir = ./data/test

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.mode == 'train':
            img_name = os.path.join(self.root_dir, self.data.iloc[idx,1])
            label = self.data.iloc[idx, 2]
        else:
            img_name = os.path.join(self.root_dir, self.data.iloc[idx,0])
        image = Image.open(img_name)

        img_np=np.array(image)
        if img_np.ndim != 3:
            img_np = np.expand_dims(img_np, axis=2)
            img_np = np.repeat(img_np, 3, axis=2)


        img_tensor = self.transform(transforms.ToPILImage()(img_np))

        
        if self.mode == 'train':
            return img_tensor, label
        else:
            return img_tensor
    
def HoDataLoad(csv_path:str='./data', root_dir:str='./data/test', is_train:bool=False, batch_size:int=32, shuffle:bool=False) -> DataLoader:
    mode = 'train' if is_train else 'test'
    dataset = HoDataset(csv_file=csv_path, root_dir=root_dir, batch_size=batch_size, is_train=is_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

##################################dataloader 사용 예시
# dataloader = HoDataLoad()
# for batch, (img, label) in enumerate(dataloader):
#     print(batch, img.shape, label.shape)
#####################################################
# # 테스트용
# def test(csv_file=None, root_dir=None, transform=None):
#     df = pd.read_csv(csv_file)
#     train = CustomDataset(root_dir, df, transform)
#     test = CustomDataset(root_dir, df, transform, True)
    
#     print("testing for train & test")
#     img, target = train.__getitem__(0)
#     print(f"train idx 0 : {img.shape}, {target}")
#     img = test.__getitem__(0)
#     print(f"train idx 0 : {img.shape}")
    
#     one_hot_train = CustomDataset(root_dir, df, transform, one_hot=True)

#     print("testing for train one_hot")
#     img, target = one_hot_train.__getitem__(0)
#     print(f"train idx 0 : {img.shape}, {target}")
#     img, target = one_hot_train.__getitem__(1)
#     print(f"train idx 1 : {img.shape}, {target}")
    
    
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     # transforms.Normalize((0.485,0.456,0.406), (0.229, 0.224, 0.225))
# ])

# test("data/test_sample.csv", "./data", transform)