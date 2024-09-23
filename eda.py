import os
import numpy as np
import pandas as pd
from glob import glob
from PIL import Image
from collections import defaultdict

root_path = './data/train'
csv_file = "./data/train.csv"

def get_image_info(root_path: str, csv_file: str):
    def extract_image_features(image_path: str):
        try:
            with Image.open(image_path) as img:
                mode = img.mode  # 원래 모드를 저장
                width, height = img.size
                img_array = np.array(img)

                if mode == 'RGB':
                    mean_red = np.mean(img_array[:, :, 0])
                    mean_green = np.mean(img_array[:, :, 1])
                    mean_blue = np.mean(img_array[:, :, 2])
                else:
                    mean_red = mean_green = mean_blue = None
                format = image_path.split('.')[-1].upper()
                return width, height, mode, format, os.path.getsize(image_path), mean_red, mean_green, mean_blue
            
        except Exception as e:
            return None, None, None, None, None, None, None, None

    images = glob(root_path + "/*/*")
    data = pd.read_csv(csv_file)
    image_prop = defaultdict(list)

    for i, path in enumerate(images):
        print(i, path)
        width, height, mode, format, size, mean_red, mean_green, mean_blue = extract_image_features(path)
        image_prop['height'].append(height)
        image_prop['width'].append(width)
        image_prop['mode'].append(mode)
        image_prop['format'].append(format)
        image_prop['size'].append(round(size / 1e6, 2) if size else None)
        image_prop['mean_red'].append(mean_red)
        image_prop['mean_green'].append(mean_green)
        image_prop['mean_blue'].append(mean_blue)
        image_prop['path'].append(path)
        image_prop['image_path'].append(path.split('/')[-2] + "/" + path.split('/')[-1])

    image_data = pd.DataFrame(image_prop)
    image_data['img_aspect_ratio'] = image_data['width'] / image_data['height'] # aspect_ratio = 종횡비 즉, 이미지의 가로 세로 비율

    image_data = image_data.merge(data, on='image_path')
    image_data.sort_values(by='target', inplace=True)
    print("분석완료 ㅎ")
    image_data.to_csv("image_data.csv")

get_image_info(root_path, csv_file)