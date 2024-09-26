![imagenet-sketch](https://github.com/user-attachments/assets/a6307765-05bc-4cc7-9cd6-c4b6d70e9427)

<br/>
<br/>

# 1. Project Overview (프로젝트 개요)
- 프로젝트 이름: Sketch 이미지 데이터 분류
- 프로젝트 설명: Sketch기반 이미지를 분류하여 어떤 객체를 나타내는지 예측하는 대회

<br/>
<br/>

# Team Members (팀원 및 팀 소개)
| 곽기훈 | 김재환 | 양호철 | 오종민 | 조소윤 | 홍유향 |
|:------:|:------:|:------:|:------:|:------:|:------:|
| <img src="https://github.com/user-attachments/assets/fb56b1d0-9c5c-49c0-a274-f5b7ff7ab8b1" alt="곽기훈" width="150"> | <img src="https://github.com/user-attachments/assets/28a7109b-4959-473c-a6e4-5ee736370ab6" alt="김재환" width="150"> | <img src="https://github.com/user-attachments/assets/9007ffff-765c-4ffa-80bf-31668fe199ba" alt="양호철" width="150"> | <img src="https://github.com/user-attachments/assets/8760f7bd-10d8-4397-952b-f1ca562b90d4" alt="오종민" width="150"> | <img src="https://github.com/user-attachments/assets/22baca4a-189a-4bc3-ab1c-8f6256637a16" alt="조소윤" width="150"> | <img src="https://github.com/user-attachments/assets/91f96db7-3137-42d2-9175-8a55f1493b31" alt="홍유향" width="150"> |
| T7102 | T7128 | T7204 | T7207 | T7252 | T7267 |
| [GitHub](https://github.com/kkh090) | [GitHub](https://github.com/Ja2Hw) | [GitHub](https://github.com/hocheol0303) | [GitHub](https://github.com/sejongmin) | [GitHub](https://github.com/whthdbs03) | [GitHub](https://github.com/hyanghyanging) | 

<br/>
<br/>

# Project Structure (프로젝트 구조)
```plaintext
📦level1-imageclassification-cv-16
 ┣ 📂.github
 ┃ ┣ 📂ISSUE_TEMPLATE
 ┃ ┃ ┗ 📜-title----body.md
 ┃ ┣ 📜.keep
 ┃ ┗ 📜pull_request_template.md
 ┣ 📂model
 ┃ ┣ 📜cnn.py
 ┃ ┣ 📜mlp.py
 ┃ ┣ 📜model_selection.py
 ┃ ┣ 📜resnet18.py
 ┃ ┣ 📜timm.py
 ┃ ┗ 📜torchvision_model.py
 ┣ 📂util
 ┃ ┣ 📜augmentation.py
 ┃ ┣ 📜checkpoints.py
 ┃ ┣ 📜data.py
 ┃ ┣ 📜losses.py
 ┃ ┣ 📜metrics.py
 ┃ ┣ 📜optimizers.py
 ┃ ┗ 📜schedulers.py
 ┣ 📜.gitignore
 ┣ 📜README.md
 ┣ 📜args.py
 ┣ 📜eda.ipynb
 ┣ 📜eda.py
 ┣ 📜erase_dot_files.py
 ┣ 📜gradcam.py
 ┣ 📜image_augmentation.py
 ┣ 📜separate.py
 ┣ 📜test.py
 ┣ 📜test.sh
 ┣ 📜train.ipynb
 ┣ 📜train.py
 ┣ 📜train.sh
 ┗ 📜trainer.py
```

<br/>
<br/>
