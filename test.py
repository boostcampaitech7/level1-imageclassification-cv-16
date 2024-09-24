import os
import torch
import pandas as pd
import argparse
from argparse import Namespace
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from util.data import CustomDataset
from util.augmentation import TransformSelector
from model.model_selection import ModelSelector

def inference(
        model: nn.Module, 
        device: torch.device, 
        test_loader: DataLoader
    ) -> None:
    
    model.to(device)
    model.eval()

    predictions = []
    
    with torch.no_grad():
        for images in tqdm(test_loader):
            images = images.to(device)

            output = model(images)
            output = F.softmax(output, dim=1)
            preds = output.argmax(dim=1)

            predictions.extend(preds.cpu().detach().numpy())
    
    return predictions

def parse_args_and_config() -> Namespace:
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--mode', type=str, default='test', help='Select mode train or test default is train', action='store')
    parser.add_argument('--device', type=str, default='cpu', help='Select device to run, default is cpu', action='store')
    parser.add_argument('--data_root', type=str, default='./data', help='Path to data root', action='store')
    parser.add_argument('--test_csv', type=str, default='./data/test.csv', help='Path to test csv', action='store')
    parser.add_argument('--output_path', type=str, default='output.csv', help='Path for csv result', action='store')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints/final_checkpoint.pth', help='Path to checkpoints of the model', action='store')
    parser.add_argument('--height', type=int, default=224, help="Select input img height, default is 224", action='store')
    parser.add_argument('--width', type=int, default=224, help="Select input img width, default is 224", action='store')
    parser.add_argument('--num_classes', type=int, default=500, help="Select number of classes, default is 500", action='store')
    parser.add_argument('--model', type=str, default='cnn', help='Select a model to train, default is cnn', action='store')
    parser.add_argument('--batch', type=int, default=64, help='Select batch_size, default is 64', action='store')
    parser.add_argument('--transform_type', type=str, default='albumentations', help='Select transform type, default is albumentation', action='store')
    parser.add_argument('--augmentations', type=str, default="hflip_vflip_rotate", help='Select augmentations to use, default is hflip_vflip_rotate', action='store')
    parser.add_argument('--adjust_ratio', help='Turn True to adjust the ratio', action='store_true')
    
    return parser.parse_args()

if __name__=='__main__':
    # 하이퍼파라미터 가져오기
    args = parse_args_and_config()
    
    # 추론 데이터의 경로와 정보를 가진 파일의 경로를 설정.
    test_data_dir = args.data_root + "/test"
    test_data_info_file = args.test_csv
    checkpoint_path = args.checkpoint_path # "./train_result"
    
    # 추론 데이터의 class, image path, target에 대한 정보가 들어있는 csv파일을 읽기.
    test_df = pd.read_csv(test_data_info_file)

    num_classes = args.num_classes
    
    transform_selector = TransformSelector(transform_type=args.transform_type)
    transform = transform_selector.get_transform(augment=False, height=args.height, width=args.width, augment_list=args.augmentations, adjust_ratio=args.adjust_ratio)
    
    test_dataset = CustomDataset(
        root_dir=test_data_dir,
        data_df=test_df,
        transform=transform,
        is_inference=True
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch,
        shuffle=False,
        drop_last=False
    )

    ## 학습 모델
    if 'timm' in args.model:
        model_selector = ModelSelector(
            "timm", 
            num_classes, 
            model_name=args.model.split("-")[-1], 
            pretrained=True
        )
    else:
        model_selector = ModelSelector(
            args.model,
            num_classes,
        )

    model = model_selector.get_model()
    
    model.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location='cpu'
    )['model_state_dict'])
    
    predictions = inference(
        model=model,
        device=args.device,
        test_loader=test_dataloader
    )
    
    test_df['target'] = predictions
    test_df = test_df.reset_index().rename(columns = {"index": "ID"})
    test_df.to_csv(args.output_path, index=False)
    print(f"create {args.output_path}")