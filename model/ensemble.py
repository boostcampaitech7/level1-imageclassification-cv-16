# ensemble.py
import torch
import torch.nn.functional as F

class VotingEnsemble:
    def __init__(self, models, voting='soft'):
        """
        models: List of models to ensemble
        voting: 'hard' for majority voting, 'soft' for average probability voting
        """
        self.models = models
        self.voting = voting

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
