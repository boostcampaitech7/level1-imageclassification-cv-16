import os
import torch
import pandas as pd
from args import Custom_arguments_parser
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

if __name__=='__main__':
    # 하이퍼파라미터 가져오기
    parser = Custom_arguments_parser(mode='test')
    args = parser.get_parser()
    
    # 추론 데이터의 경로와 정보를 가진 파일의 경로를 설정.
    test_data_dir = args.data_root + "/test"
    test_data_info_file = args.csv_path
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