import os
from typing import List

def remove_dot_files(folder_path: List[str]) -> None:
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            if file_name.startswith('.'):
                file_path = os.path.join(root, file_name)
                os.remove(file_path)
                print(f"Removed file: {file_path}")
        
        for dir_name in dirs:
            if dir_name.startswith('.'):
                dir_path = os.path.join(root, dir_name)
                try:
                    os.rmdir(dir_path)
                    print(f"Removed directory: {dir_path}")
                except OSError:
                    print(f"Directory not empty, cannot remove: {dir_path}")

folder_path = './data'
remove_dot_files(folder_path)