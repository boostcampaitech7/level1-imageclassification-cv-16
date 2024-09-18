import timm
import torch
import torch.nn as nn
from model.cnn import SimpleCNN
from model.mlp import MLP
from model.torchvision_model import TorchvisionModel
from model.timm import TimmModel
from model.resnet18 import ResNetModel
from model.efficientnet import EfficientNetB0
from model.vit import VisionTransformer
from model.mobilenet import MobileNetV1

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

        elif model_type == 'efficientnet':
            self.model = EfficientNetB0(num_classes=num_classes)

        elif model_type == 'vit':
            # ViT의 기본 설정값들. 필요에 따라 kwargs에서 오버라이드 가능
            img_size = kwargs.get('img_size', 224)
            patch_size = kwargs.get('patch_size', 16)
            in_channels = kwargs.get('in_channels', 3)
            embed_dim = kwargs.get('embed_dim', 768)
            depth = kwargs.get('depth', 12)
            n_heads = kwargs.get('n_heads', 12)
            
            self.model = VisionTransformer(
                img_size=img_size,
                patch_size=patch_size,
                in_channels=in_channels,
                n_classes=num_classes,
                embed_dim=embed_dim,
                depth=depth,
                n_heads=n_heads
            )

        
        elif model_type == 'mobilenet':
            # MobileNet의 기본 설정값. 필요에 따라 kwargs에서 오버라이드 가능
            width_multiplier = kwargs.get('width_multiplier', 1.0)
            self.model = MobileNetV1(num_classes=num_classes, width_multiplier=width_multiplier)
                
        
        else:
            raise ValueError("Unknown model type specified.")

    def get_model(self) -> nn.Module:

        # 생성된 모델 객체 반환
        return self.model
    

"""
사용 예시

# EfficientNet 모델 선택
model_selector = ModelSelector(model_type='efficientnet', num_classes=10)
efficientnet_model = model_selector.get_model()

# ViT 모델 선택 (추가 매개변수 지정)
model_selector = ModelSelector(model_type='vit', num_classes=10, img_size=224, patch_size=16, embed_dim=512)
vit_model = model_selector.get_model()

# MobileNet 모델 선택 (width_multiplier 지정)
model_selector = ModelSelector(model_type='mobilenet', num_classes=10, width_multiplier=0.75)
mobilenet_model = model_selector.get_model()
    
    
"""