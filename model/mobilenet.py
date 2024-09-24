import torch
import torch.nn as nn

class DepthwiseSeparableConv(nn.Module):
    """
    깊이별 분리 컨볼루션 (Depthwise Separable Convolution)
    표준 컨볼루션을 두 단계로 나누어 계산량을 줄임
    1. Depthwise Convolution: 각 입력 채널에 대해 별도의 필터 적용
    2. Pointwise Convolution: 1x1 컨볼루션으로 채널 간 정보 결합
    """
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

class MobileNet(nn.Module):
    """
    MobileNet 모델
    경량화된 컨볼루션 신경망으로, 모바일 및 임베디드 비전 응용 프로그램에 적합
    """
    def __init__(self, num_classes=1000, width_multiplier=1.0):
        super().__init__()
        
        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return DepthwiseSeparableConv(inp, oup, stride)

        # width_multiplier를 사용하여 채널 수 조정
        def adjust_channels(channels):
            return int(channels * width_multiplier)

        self.model = nn.Sequential(
            conv_bn(3, adjust_channels(32), 2),
            conv_dw(adjust_channels(32), adjust_channels(64), 1),
            conv_dw(adjust_channels(64), adjust_channels(128), 2),
            conv_dw(adjust_channels(128), adjust_channels(128), 1),
            conv_dw(adjust_channels(128), adjust_channels(256), 2),
            conv_dw(adjust_channels(256), adjust_channels(256), 1),
            conv_dw(adjust_channels(256), adjust_channels(512), 2),
            *[conv_dw(adjust_channels(512), adjust_channels(512), 1) for _ in range(5)],
            conv_dw(adjust_channels(512), adjust_channels(1024), 2),
            conv_dw(adjust_channels(1024), adjust_channels(1024), 1),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(adjust_channels(1024), num_classes)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, x.shape[1])
        x = self.fc(x)
        return x

class MobileNetV1(nn.Module):
    """
    MobileNet V1 모델
    width_multiplier를 통해 모델의 크기와 계산량을 조절할 수 있음
    """
    def __init__(self, num_classes: int, width_multiplier: float = 1.0):
        super().__init__()
        self.model = MobileNet(num_classes=num_classes, width_multiplier=width_multiplier)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)