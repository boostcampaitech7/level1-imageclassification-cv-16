import timm
import torch
import torch.nn as nn
from model.CNN import SimpleCNN
from model.mlp import MLP
from model.torchvisionModel import TorchvisionModel
from model.timm import TimmModel
from model.resnet18 import ResNetModel

class ModelSelector:
    """
    사용할 모델 유형을 선택하는 클래스.
    """
    def __init__(
        self, 
        model_type: str, 
        num_classes: int, 
        **kwargs
    ):
        # 모델 유형을 소문자로 변환
        model_type = model_type.lower()
        
        if model_type == 'cnn':
            self.model = SimpleCNN(num_classes=num_classes)
        
        elif model_type == 'torchvision':
            self.model = TorchvisionModel(num_classes=num_classes, **kwargs)
        
        elif model_type == 'timm':
            self.model = TimmModel(num_classes=num_classes, **kwargs)

            """
        elif model_type == 'mlp':
            input_size = kwargs.get('input_size', 784)  # 예: 28x28 이미지의 경우
            hidden_size = kwargs.get('hidden_size', 128)
            self.model = MLP(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes)"""

        elif model_type == 'resnet':
            self.model = ResNetModel(num_classes=num_classes, **kwargs)    
        
        else:
            raise ValueError("Unknown model type specified.")

    def get_model(self) -> nn.Module:

        # 생성된 모델 객체 반환
        return self.model