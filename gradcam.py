import os
import cv2
import torch
import numpy as np
from torchcam.methods import GradCAM
from torch.utils.data import DataLoader

def visualize_gradcam(
        model: torch.nn.Module,
        device: torch.device,
        dataloader: DataLoader,
        target_layer: str,
        image_count: int
    ) -> None:
    cam_extractor = GradCAM(model.model, target_layer)

    save_dir = "./gradcam"
    os.makedirs(save_dir, exist_ok=True)

    model.to(device)
    model.eval()  # 모델을 평가 모드로 설정합니다.
    count = 0
    # 데이터 로더에서 배치를 반복합니다.
    for inputs in dataloader:
        inputs = inputs.to(device)  # 입력 이미지를 장치로 이동합니다.
        outputs = model(inputs)  # 모델을 통해 예측을 수행합니다.
        _, preds = torch.max(outputs, 1)  # 예측된 클래스 인덱스를 가져옵니다.
        # 배치 내의 각 이미지에 대해 처리합니다.
        for j in range(inputs.size()[0]):
            # CAM을 가져옵니다.
            cam = cam_extractor(preds[j].item(), outputs[j].unsqueeze(0),  retain_graph=True)[0]
            # CAM을 1채널로 변환합니다.
            cam = cam.mean(dim=0).cpu().numpy()
            # CAM을 원본 이미지 크기로 리사이즈합니다.
            cam = cv2.resize(cam, (inputs[j].shape[2], inputs[j].shape[1]))
            # CAM을 정규화합니다.
            cam = (cam - cam.min()) / (cam.max() - cam.min())  # 정규화
            # CAM을 0-255 범위로 변환합니다.
            cam = np.uint8(255 * cam)
            # 컬러맵을 적용하여 RGB 이미지로 변환합니다.
            cam = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
            cam = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)  # BGR에서 RGB로 변환
            # 입력 이미지가 1채널 또는 3채널인지 확인하고 처리합니다.
            input_image = inputs[j].cpu().numpy().transpose((1, 2, 0))
            if input_image.shape[2] == 1:  # 1채널 이미지인 경우
                input_image = np.squeeze(input_image, axis=2)  # (H, W, 1) -> (H, W)
                input_image = np.stack([input_image] * 3, axis=-1)  # (H, W) -> (H, W, 3)로 변환하여 RGB처럼 만듭니다.
            else:  # 3채널 이미지인 경우
                input_image = (input_image - input_image.min()) / (input_image.max() - input_image.min())
                input_image = (input_image * 255).astype(np.uint8)  # 정규화된 이미지를 8비트 이미지로 변환합니다.

            # 오버레이된 이미지 생성
            overlay = cv2.addWeighted(input_image, 0.5, cam, 0.5, 0)
            cv2.imwrite(f"./gradcam/gradcam_{count}.jpg", overlay)
            count += 1
            if count > image_count:
                return