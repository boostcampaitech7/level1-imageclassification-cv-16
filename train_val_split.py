import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

root_path = './data'  # 원본 이미지가 저장된 디렉토리
train_dir = './data/train_images'  # 학습 이미지가 저장될 디렉토리
val_dir = './data/val_images'  # 검증 이미지가 저장될 디렉토리
csv_file = "./data/train.csv"


def data_preprocess(root_path, csv_file, train_dir, val_dir):
    train_df = pd.read_csv(csv_file)
    train_df, val_df = train_test_split(train_df, test_size=0.1, stratify=train_df["target"], random_state=42)

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    for image_path in train_df["image_path"]:
        source = os.path.join(root_path, "train", image_path)
        destination = os.path.join(train_dir, image_path)
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        shutil.copy(source, destination)
        print(f"copy from {source} to {destination}.")

    for image_path in val_df["image_path"]:
        source = os.path.join(root_path, "train", image_path)
        destination = os.path.join(val_dir, image_path)
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        shutil.copy(source, destination)
        print(f"copy from {source} to {destination}.")

    train_df.to_csv(os.path.join(root_path, "train_data.csv"), index=False, header=True)
    val_df.to_csv(os.path.join(root_path, "val_data.csv"), index=False, header=True)
    
data_preprocess(root_path, csv_file, train_dir, val_dir)
    

