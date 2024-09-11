import torch
import random
import torchvision.transforms as T
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Union, Tuple


#기본 트랜스폼 클래스
class BasicTransforms:
    def __init__(self, augment=False) -> None:
        self.augment = augment

        # 기본 트랜스폼 (증강 없음)
        self.base_transform = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
        ])

        # 기본 증강을 사용할 때
        self.augment_transform = T.Compose([
            T.RandomHorizontalFlip(p=0.5), #50% 확률로 이미지 뒤집
            T.RandomRotation(degrees=30), #최대 30도로 회전함
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2), #색깔 변경
            T.Resize((256, 256)), #리사이즈
            T.ToTensor(), #이미지를 텐서로 변환
        ])

    def __call__(self, image) -> torch.tensor: #이미지에 트랜스폼 적용함
        if self.augment:
            return self.augment_transform(image)
        return self.base_transform(image)


# Albumentation 기반 트랜스폼
class AlbumentationsTransforms:
    def __init__(self, augment=False) -> None: #True면 증강을 포함한 트랜스폼 적용, False면 기본 트랜스폼 적용
        self.augment = augment

        # 증강 없는 기본 트랜스폼
        self.base_transform = A.Compose([
            A.Resize(256, 256),
            ToTensorV2() #이미지를 텐서로 변환
        ])

        # Albumentations 증강을 사용한 트랜스폼 (랜덤 자르기, 플립, 회전...)
        self.augment_transform = A.Compose([
            A.RandomResizedCrop(256, 256), #랜덤하게 크롭하고 256x256로 리사이즈
            A.HorizontalFlip(p=0.5), #수평 플립
            A.VerticalFlip(p=0.5), #수직 플립
            A.Rotate(limit=45), #45도 제한 랜덤 회전
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3), #색깔 변경
            ToTensorV2(), #이미지를 텐서로 변환
        ])

    def __call__(self, image) -> torch.Tensor: #이미지에 트랜스폼 적용
        if self.augment:
            augmented = self.augment_transform(image=image)
            return augmented['image']
        else:
            base = self.base_transform(image=image)
            return base['image']


#컷믹스 트랜스폼
class CutMixTransforms:
    def __init__(self, alpha=1.0) -> None: #알파: CutMix에서 사용할 베타 분포 파라미터
        self.alpha = alpha

    def __call__(self, image1, image2, label1, label2) -> Tuple[torch.Tensor, torch.Tensor]: #두 개 이미지를 컷믹스
        lam = random.beta(self.alpha, self.alpha) #베타 분포로 람다 값 생성
        bbx1, bby1, bbx2, bby2 = self.rand_bbox(image1.size(), lam) #랜덤 바운딩 박스 생성
        image1[:, bbx1:bbx2, bby1:bby2] = image2[:, bbx1:bbx2, bby1:bby2] #바운딩 박스 내 영역 교체함
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (image1.size(-1) * image1.size(-2))) #새로운 람다 계산
        mixed_label = lam * label1 + (1 - lam) * label2 #레이블 섞음
        return image1, mixed_label

    @staticmethod
    def rand_bbox(size, lam) -> Tuple[int, int, int, int]: #람다 값을 기준으로 랜덤한 바운딩 박스 생성
        W = size[2]
        H = size[1]
        cut_rat = torch.sqrt(1. - lam) #자를 비율 계산
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        cx = random.randint(0, W)
        cy = random.randint(0, H)

        bbx1 = torch.clamp(cx - cut_w // 2, 0, W)
        bby1 = torch.clamp(cy - cut_h // 2, 0, H)
        bbx2 = torch.clamp(cx + cut_w // 2, 0, W)
        bby2 = torch.clamp(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2


# 믹스업 트랜스폼
class MixUpTransforms:
    def __init__(self, alpha=1.0) -> None: #알파: 믹스업에서 사용할 베타 분포의 파라미터
        self.alpha = alpha

    def __call__(self, image1, image2, label1, label2) -> Tuple[torch.Tensor, torch.Tensor]: #두 개 이미지를 믹스업
        lam = random.beta(self.alpha, self.alpha)  # 베타 분포로 람다 값 생성
        mixed_image = lam * image1 + (1 - lam) * image2  # 이미지를 섞음
        mixed_label = lam * label1 + (1 - lam) * label2  # 레이블을 섞음
        return mixed_image, mixed_label
    

# 트랜스폼 데이터 증강 라이브러리와 기법을 선택하는 클래스
class TransformSelector: #사용자가 지정한 transform_type에 따라 서로 다른 변환 객체를 반환하는 구조

    def __init__(self, transform_type: str) -> None:
        self.transform_type = transform_type
        #인자 확인해서 변환 라이브러리를 선택함

    def get_transform(self, augment=False, alpha=1.0) -> Union[BasicTransforms, AlbumentationsTransforms, CutMixTransforms, MixUpTransforms]:
        """
        augment: 데이터 증강 여부
        alpha: MixUp이나 CutMix의 파라미터
        """
        if self.transform_type == 'basic':
            return BasicTransforms(augment=augment)

        elif self.transform_type == 'albumentations':
            return AlbumentationsTransforms(augment=augment)
        
        elif self.transform_type == 'cutmix':
            return CutMixTransforms(alpha=alpha)

        elif self.transform_type == 'mixup':
            return MixUpTransforms(alpha=alpha)


        else:
            raise ValueError(f"Unknown transform type: {self.transform_type}")


"""
사용 예시

# 기본 변환을 사용할 때
transform_selector = TransformSelector(transform_type='basic')
transform = transform_selector.get_transform(augment=True)

# Albumentations 기반 변환을 사용할 때
albumentations_selector = TransformSelector(transform_type='albumentations')
albumentations_transform = albumentations_selector.get_transform(augment=True)


# CutMix을 사용할 때
cutmix_selector = TransformSelector(transform_type='cutmix')
cutmix_transform = cutmix_selector.get_transform(alpha=0.5)


# MixUp을 사용할 때
mixup_selector = TransformSelector(transform_type='mixup')
mixup_transform = mixup_selector.get_transform(alpha=0.4)



"""
