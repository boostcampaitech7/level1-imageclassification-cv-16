import os
import cv2
import pandas as pd

root_path = './data/train'
csv_file = "./data/train.csv"

def offlineImageFlip(root_path, csv_file):
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file)
                image_path = "/".join(image_path.split("\\"))
                image = cv2.imread(image_path)

                flip_image = cv2.flip(image, 1)
                flip_file = "flip_" + os.path.basename(image_path)
                new_path = os.path.join(os.path.dirname(image_path), flip_file)

                cv2.imwrite(new_path, flip_image)
                print(f"image augmentation: {new_path}")

    train_df = pd.read_csv(csv_file)
    train_df_copy = train_df.copy()
    train_df_copy["image_path"] = train_df_copy["image_path"].replace(r'([^/]+/)', r'\1flip_', regex=True)
    train_df = pd.concat([train_df, train_df_copy], ignore_index=True)
    train_df.to_csv("./data/train1.csv", index=False, header=True)
    print(f"created : train1.csv")
    

def resetOfflineImageFlip(root_path):
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')) and "flip_" in file:
                file_path = os.path.join(root, file)
                os.remove(file_path)
                print(f"Deleted: {file_path}")
    try:
        os.remove("./data/train1.csv")
        print(f"Deleted: train1.csv")
    except Exception as e:
        pass
    

offlineImageFlip(root_path, csv_file)
# resetOfflineImageFlip(root_path)