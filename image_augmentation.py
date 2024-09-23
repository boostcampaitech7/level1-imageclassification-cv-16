import os
import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

root_path = './data/train'
csv_file = "./data/train.csv"

def flip_image(root_path: str, csv_file: str):
    train_df = pd.read_csv(csv_file)
    for file in tqdm(train_df["image_path"]):
        image_path = os.path.join(root_path, file)
        image_path = "/".join(image_path.split("\\"))
        image = cv2.imread(image_path)

        flip_image = cv2.flip(image, 1)
        flip_file = "flip_" + os.path.basename(image_path)
        new_path = os.path.join(os.path.dirname(image_path), flip_file)

        cv2.imwrite(new_path, flip_image)
        # print(f"image augmentation: {new_path}")

    train_df_copy = train_df.copy()
    train_df_copy["image_path"] = train_df_copy["image_path"].replace(r'([^/]+/)', r'\1flip_', regex=True)
    train_df = pd.concat([train_df, train_df_copy], ignore_index=True)
    train_df.to_csv("./data/train1.csv", index=False, header=True)
    print(f"created : train1.csv")

def add_white_image(root_path: str, csv_file: str):
    train_df = pd.read_csv(csv_file)
    new_rows = pd.DataFrame(columns=train_df.columns)

    for i, file in enumerate(tqdm(train_df["image_path"])):
        image_path = os.path.join(root_path, file)
        image_path = "/".join(image_path.split("\\"))
        image = cv2.imread(image_path)

        height, width = image.shape[:2]
        if 0.5 < height / width < 1.5:
            continue
        max_side = max(width, height)
        white_background = np.ones((max_side, max_side, 3), dtype=np.uint8) * 255
        x_offset = (max_side - width) // 2
        y_offset = (max_side - height) // 2
        white_background[y_offset:y_offset + height, x_offset:x_offset + width] = image

        white_file = "white_" + os.path.basename(image_path)
        new_path = os.path.join(os.path.dirname(image_path), white_file)
        cv2.imwrite(new_path, white_background)

        new_row = train_df.iloc[i].copy()
        new_row['image_path'] = "/".join(new_path.split("/")[-2:])
        new_row_df = pd.DataFrame([new_row])
        new_rows = pd.concat([new_rows, new_row_df], ignore_index=True)
        # print(f"image augmentation: {new_path}")
    
    train_df = pd.concat([train_df, new_rows], ignore_index=True)
    train_df.to_csv("./data/train1.csv", index=False, header=True)
    print(f"created : train1.csv")

def reset_augmentation(root_path: str, aug: str):
    for root, dirs, files in tqdm(os.walk(root_path)):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')) and aug in file:
                file_path = os.path.join(root, file)
                os.remove(file_path)
    try:
        os.remove("./data/train1.csv")
        print(f"Deleted: train1.csv")
    except Exception as e:
        pass

# reset_augmentation(root_path, "white_")
# add_white_image(root_path, "./data/train.csv")
flip_image(root_path, "./data/train1.csv")
