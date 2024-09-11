import os
import shutil #파일, 디렉토리 작업 관련 라이브러리임
from typing import List, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

class ImageDataPreprocessor:
    def __init__(self, source_dir: str, train_dir: str, val_dir: str, csv_dir: str, val_size: float = 0.1) -> None:
        """
        source_dir: 원본 이미지가 저장된 디렉토리
        train_dir: train 데이터가 저장될 디렉토
        val_dir: val 데이터가 저장될 디렉토리
        csv_dir: CSV 파일 저장할 디렉토리
        val_size: 검증 데이터의 비율
        """
        self.source_dir: str = source_dir
        self.train_dir: str = train_dir
        self.val_dir: str = val_dir
        self.csv_dir: str = csv_dir
        self.val_size: float = val_size
        self._create_dirs()  # 학습 및 검증 디렉토리 생성
        self.image_files: List[str] = self._get_image_files()  # 이미지 파일 리스트 가져오기

    def _create_dirs(self) -> None: #학습, 검증 데이터 및 CSV 파일 디렉토리를 생성
        os.makedirs(self.train_dir, exist_ok=True)
        os.makedirs(self.val_dir, exist_ok=True)
        os.makedirs(self.csv_dir, exist_ok=True)

    def _get_image_files(self) -> List[str]: #원본 dir에서 모든 이미지 파일의 경로 수집하고
        # 이미지 파일 경로 리스트를 리턴함
        image_files: List[str] = []
        for root, _, files in os.walk(self.source_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):  # 이미지 파일 확장자 필터링
                    image_files.append(os.path.join(root, file))
        return image_files

    def _split_data(self) -> Tuple[List[str], List[str]]: #학습 데이터와 val 데이터로 분리
        return train_test_split(self.image_files, test_size=self.val_size, shuffle=True, random_state=42)
        #학습 데이터와 val 데이터로 나눈 파일 경로 리스트 리턴

    def _move_files(self, file_list: List[str], target_dir: str) -> None:
        """
        파일 리스트를 지정한 디렉토리로 이동
        
        file_list: 이동할 파일 경로의 리스트
        target_dir: 파일이 이동할 대상 디렉토리
        """
        for file in file_list:
            shutil.move(file, os.path.join(target_dir, os.path.basename(file)))

    def _save_csv(self, file_list: List[str], file_name: str) -> None:
        """
        파일 리스트를 CSV 파일로 저장
        
        file_list: CSV 파일로 저장할 파일 경로의 리스트
        file_name: 저장할 CSV 파일의 이름
        """
        df: pd.DataFrame = pd.DataFrame(file_list, columns=['file_path'])
        df.to_csv(os.path.join(self.csv_dir, file_name), index=False)

    def process(self) -> None:
        """
        이미지 파일을 학습 데이터와 검증 데이터로 나누고, 각각의 디렉토리에 저장
        학습 데이터와 검증 데이터 CSV 파일 저장
        """
        # 데이터를 학습 데이터와 검증 데이터로 분리
        train_files, val_files = self._split_data()

        # 학습 데이터와 검증 데이터를 각각의 디렉토리로 이동
        self._move_files(train_files, self.train_dir)
        self._move_files(val_files, self.val_dir)

        # 학습 데이터와 검증 데이터에 대한 CSV 파일을 생성
        self._save_csv(train_files, 'train_data.csv')
        self._save_csv(val_files, 'val_data.csv')

        print(f"Train data moved to {self.train_dir}")
        print(f"Validation data moved to {self.val_dir}")
        print(f"CSV files saved to {self.csv_dir}")


"""
사용 예시

source_directory = './data'  # 원본 이미지가 저장된 디렉토리
train_directory = './data/train'  # 학습 이미지가 저장될 디렉토리
val_directory = './data/val'  # 검증 이미지가 저장될 디렉토리
csv_directory = './data/csv'  # CSV 파일이 저장될 디렉토리


preprocessor = ImageDataPreprocessor(source_directory, train_directory, val_directory, csv_directory)
preprocessor.process()

"""

