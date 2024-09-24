"""
import torch
import torch.nn as nn
import timm

class EfficientNetModel(nn.Module):
    def __init__(self, model_name: str, num_classes: int, pretrained: bool = True):
        super(EfficientNetModel, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
"""

import torch
import torch.nn as nn
import math

class Swish(nn.Module):
    """
    Swish 활성화 함수: x * sigmoid(x)
    EfficientNet에서 사용되는 비선형성을 제공
    """
    def forward(self, x):
        return x * torch.sigmoid(x)

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation 블록
    채널 간의 상호의존성을 모델링하여 특성 재조정을 수행
    중요한 특성을 강조
    """
    def __init__(self, in_channels, se_ratio=0.25):
        super().__init__()
        se_channels = max(1, int(in_channels * se_ratio))
        self.squeeze = nn.AdaptiveAvgPool2d(1)  # 전역 평균 풀링
        self.excitation = nn.Sequential(
            nn.Conv2d(in_channels, se_channels, kernel_size=1),
            Swish(),
            nn.Conv2d(se_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.squeeze(x)
        y = self.excitation(y)
        return x * y  # 스케일링된 특성

class MBConvBlock(nn.Module):
    """
    MBConv (Mobile Inverted Residual Bottleneck) 블록
    EfficientNet의 주요 구성 요소
    모바일 인버티드 병목 컨볼루션 블록
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio):
        super().__init__()
        self.has_se = se_ratio is not None and 0 < se_ratio <= 1
        self.id_skip = stride == 1 and in_channels == out_channels

        expanded_channels = in_channels * expand_ratio

        # 확장 단계 (선택적)
        if expand_ratio != 1:
            self.expand_conv = nn.Conv2d(in_channels, expanded_channels, kernel_size=1, bias=False)
            self.bn0 = nn.BatchNorm2d(expanded_channels)

        # 깊이별 컨볼루션
        self.depthwise_conv = nn.Conv2d(
            expanded_channels, expanded_channels, kernel_size=kernel_size, stride=stride, 
            groups=expanded_channels, padding=(kernel_size - 1) // 2, bias=False
        )
        self.bn1 = nn.BatchNorm2d(expanded_channels)

        # Squeeze-and-Excitation 블록 (선택적)
        if self.has_se:
            self.se = SEBlock(expanded_channels, se_ratio)

        # 프로젝션 단계
        self.project_conv = nn.Conv2d(expanded_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.swish = Swish()

    def forward(self, x):
        identity = x

        # 확장
        if hasattr(self, 'expand_conv'):
            x = self.swish(self.bn0(self.expand_conv(x)))

        # 깊이별 컨볼루션
        x = self.swish(self.bn1(self.depthwise_conv(x)))

        # Squeeze-and-Excitation
        if self.has_se:
            x = self.se(x)

        # 프로젝션
        x = self.bn2(self.project_conv(x))

        # 스킵 연결 (선택적)
        if self.id_skip:
            x += identity

        return x

class EfficientNet(nn.Module):
    """
    EfficientNet 모델
    확장 및 깊이 계수를 사용하여 모델의 크기를 조절할 수 있음
    """
    def __init__(self, width_coefficient, depth_coefficient, num_classes):
        super().__init__()
        # 기본 설정
        channels = [32, 16, 24, 40, 80, 112, 192, 320, 1280]
        repeats = [1, 2, 2, 3, 3, 4, 1]
        strides = [1, 2, 2, 2, 1, 2, 1]
        kernel_sizes = [3, 3, 5, 3, 5, 5, 3]
        expand_ratios = [1, 6, 6, 6, 6, 6, 6]

        # 너비와 깊이 조정
        channels = [int(x * width_coefficient) for x in channels]
        repeats = [int(math.ceil(x * depth_coefficient)) for x in repeats]

        # 스템 (첫 번째 컨볼루션 레이어)
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            Swish()
        )

        # 블록 구성
        self.blocks = nn.Sequential()
        for i in range(7):
            for j in range(repeats[i]):
                self.blocks.add_module(f"block_{i}_{j}", MBConvBlock(
                    in_channels=channels[i] if j == 0 else channels[i+1],
                    out_channels=channels[i+1],
                    kernel_size=kernel_sizes[i],
                    stride=strides[i] if j == 0 else 1,
                    expand_ratio=expand_ratios[i],
                    se_ratio=0.25
                ))

        # 헤드 (마지막 레이어들)
        self.head = nn.Sequential(
            nn.Conv2d(channels[-2], channels[-1], kernel_size=1, bias=False),
            nn.BatchNorm2d(channels[-1]),
            Swish(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(channels[-1], num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x

class EfficientNetB0(nn.Module):
    """
    EfficientNet-B0 모델
    EfficientNet의 기본 버전
    """
    def __init__(self, num_classes: int):
        super().__init__()
        self.model = EfficientNet(width_coefficient=1.0, depth_coefficient=1.0, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)