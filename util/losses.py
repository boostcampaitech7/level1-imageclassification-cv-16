# losses.py
import torch
import torch.nn as nn

def cross_entropy_loss(output, target):
    return nn.CrossEntropyLoss()(output, target)

def mse_loss(output, target):
    return nn.MSELoss()(output, target)

# 커스텀 손실 함수 예시
class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(
        self, 
        outputs: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
    
        return self.loss_fn(outputs, targets)