import os
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

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
    
    test_info['target'] = predictions
    test_info = test_info.reset_index().rename(columns = {"index": "ID"})
    test_info.to_csv("output.csv", index=False)

if __name__=='__main__':
    # 추론 데이터의 경로와 정보를 가진 파일의 경로를 설정.
    testdata_dir = "./data/test"
    testdata_info_file = "./data/test.csv"
    save_result_path = "./train_result"
    
    # 추론 데이터의 class, image path, target에 대한 정보가 들어있는 csv파일을 읽기.
    test_df = pd.read_csv(testdata_info_file)

    num_classes = 500
    
    transform = None
    
    test_dataloader = None
    
    model = None
    model.load_state_dict(
        torch.load(
            os.path.join(save_result_path, "best_model.pt"),
            map_location='cpu'
    ))
    
    predictions = inference(
        model=model,
        device=device,
        test_dataloader=test_dataloader
    )
