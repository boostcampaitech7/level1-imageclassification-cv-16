import os
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

from model import modelSelection

from util.data import CustomDataset
from util.augmentation import TransformSelector
from util.optimizers import get_optimizer
from trainer import Trainer

from model.modelSelection import ModelSelector

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def inference(
        model: nn.Module, 
        device: torch.device, 
        test_loader: DataLoader) -> None:
    
    model.to(device)
    model.eval()

    total_loss = 0.0
    predictions = []
    
    with torch.no_grad():
        for images in tqdm(test_loader):
            images = images.to(device)

            output = model(images)
            output = F.softmax(output, dim=1)
            preds = output.argmax(dim=1)

            predictions.extend(preds.cpu().detach().numpy())
    
    return predictions

if __name__=='__main__':
    # 추론 데이터의 경로와 정보를 가진 파일의 경로를 설정.
    test_data_dir = "./data/test"
    test_data_info_file = "./data/test.csv"
    save_result_path = "./checkpoint" # "./train_result"
    
    # 추론 데이터의 class, image path, target에 대한 정보가 들어있는 csv파일을 읽기.
    test_df = pd.read_csv(test_data_info_file)

    num_classes = 500
    
    transform_selector = TransformSelector(transform_type="albumentations")
    transform = transform_selector.get_transform(augment=False)
    
    test_dataset = CustomDataset(
        root_dir=test_data_dir,
        data_df=test_df,
        transform=transform,
        is_inference=True
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        drop_last=False
    )
    
    model_selector = model_selector = ModelSelector("timm", num_classes, 
                                    model_name='resnet18', pretrained=True)  
    model = model_selector.get_model()
    
    model.load_state_dict(
        torch.load(
            "checkpoints\\final_checkpoint.pth", 
            # os.path.join(save_result_path, "best_model.pt"),
            map_location='cpu'
    )['model_state_dict'])
    
    predictions = inference(
        model=model,
        device=device,
        test_loader=test_dataloader
    )
    
    test_df['target'] = predictions
    test_df = test_df.reset_index().rename(columns = {"index": "ID"})
    test_df.to_csv("output.csv", index=False)