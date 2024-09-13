import os
import pandas as pd
from sklearn.model_selection import train_test_split

root_path = './data'
csv_file = "./data/train.csv"
val_size = 0.1


def data_preprocess(
        root_path: str,
        csv_file: str, 
        val_size: float=0.1,
    ) -> None:
    train_df = pd.read_csv(csv_file)
    train_df, val_df = train_test_split(train_df, test_size=val_size, stratify=train_df["target"], random_state=42)

    train_df.to_csv(os.path.join(root_path, "train_data.csv"), index=False, header=True)
    val_df.to_csv(os.path.join(root_path, "val_data.csv"), index=False, header=True)
    
data_preprocess(root_path, csv_file, val_size)
    

