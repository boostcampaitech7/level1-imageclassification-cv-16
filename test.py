import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

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