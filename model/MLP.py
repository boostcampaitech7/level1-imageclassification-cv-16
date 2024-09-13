import torch
import torch.nn as nn

class MLP(nn.Module):

    def __init__(self, input_size: int, hidden_size: int, num_classes: int):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(x, 1)  # 입력을 일렬로 평탄화
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x