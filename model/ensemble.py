# ensemble.py
import torch
import torch.nn.functional as F
import torch.nn as nn

class VotingEnsemble(nn.Module):  # torch.nn.Module 상속받기
    def __init__(self, models, voting_type="soft"):
        super(VotingEnsemble, self).__init__()
        self.models = models
        self.voting_type = voting_type
        if voting_type not in ["soft", "hard"]:
            raise ValueError("voting_type must be 'soft' or 'hard'")

    def forward(self, x):
        # 각 모델을 forward하고 그 결과를 받아오는 방식
        outputs = [model(x) for model in self.models]

        if self.voting_type == "soft":
            # 소프트 보팅: 각 모델의 출력 확률 평균
            return torch.mean(torch.stack(outputs), dim=0)
        elif self.voting_type == "hard":
            # 하드 보팅: 각 모델의 예측 결과(가장 높은 확률) 중 다수결을 사용
            outputs = torch.stack([torch.argmax(output, dim=1) for output in outputs])
            # 다수결로 최종 예측 결정
            return torch.mode(outputs, dim=0)[0]
    """
    def predict(self, x):
        if self.voting == 'soft':
            # 소프트 보팅: 확률의 평균을 계산
            probabilities = []
            for model in self.models:
                model.eval()
                with torch.no_grad():
                    probs = F.softmax(model(x), dim=1)  # 모델의 확률 예측값
                    probabilities.append(probs)
            avg_probs = torch.mean(torch.stack(probabilities), dim=0)
            return avg_probs.argmax(dim=1)  # 최종 예측 클래스 반환
        
        elif self.voting == 'hard':
            # 하드 보팅: 각 모델의 예측 결과를 모아 다수결로 결합
            predictions = []
            for model in self.models:
                model.eval()
                with torch.no_grad():
                    pred = model(x).argmax(dim=1)  # 각 모델의 클래스 예측
                    predictions.append(pred)
            # 모델들의 예측을 모아서 다수결 적용
            stacked_preds = torch.stack(predictions, dim=0)
            return torch.mode(stacked_preds, dim=0)[0]  # 최종 예측 클래스 반환
    """

        
    def parameters(self):
        # 각 모델의 파라미터들을 리스트로 모아서 반환
        params = []
        for model in self.models:
            params.extend(model.parameters())
        return params
