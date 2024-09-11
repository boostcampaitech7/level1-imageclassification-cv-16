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
        # 필요한 초기화 코드

    def forward(self, output, target):
        # 손실 함수 계산 코드
        return loss