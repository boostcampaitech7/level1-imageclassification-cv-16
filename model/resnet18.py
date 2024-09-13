import torch
import torch.nn as nn
from torchvision import models

class ResNetModel(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True):
        super(ResNetModel, self).__init__()
        self.model = models.resnet18(pretrained=pretrained)  # ResNet18을 사용

        # ResNet의 마지막 fc 레이어를 클래스 수에 맞게 재정의
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)