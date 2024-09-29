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

train.sh: train.py 파일을 실행시키면서 학습에 필요한 인자를 입력하는 쉘 스크립트 파일. 학습 재개 시 저장 시점과 동일한 하이퍼파라미터를 사용해야한다.<br/>
 --mode: train 모드, test 모드 있음. train.sh에선 train 고정<br/>
 --device: cpu, gpu 선택<br/>
 --data_root: data 디렉터리 고정<br/>
 --csv_path: train(+validation) 데이터셋 파일 경로 설정.<br/>
 --val_csv: 사용x<br/>
 --height, --width: 학습 데이터셋의 Resize 크기 결정<br/>
 --num_classes: class 개수 입력<br/>
 --auto_split: 사용x<br/>
 --split_seed: train_test_split의 random state seed 값 설정<br/>
 --stratify: train_test_split의 비율을 고정하는 기준이 될 column 결정<br/>
 --model: 사용할 모델명 기입. timm의 경우 timm-model_name 형태로 입력하면 timm 라이브러리의 모델을 불러온다.<br/>
 --lr: 학습률 설정<br/>
 --lr_scheduler: 스케줄러 선택<br/>
 --lr_scheduler_gamma: stepLR, RduceLROnPlateau의 learning rate decay 감소 비율을 지정하는 파라미터<br/>
 --lr_scheduler_epochs_per_decay: stepLR의 lr 감소 주기 설정<br/>
 --batch: 배치 사이즈<br/>
 --loss: loss function 선택<br/>
 --optim: 옵티마이저 선택<br/>
 --r_epochs: train set과 validation set의 크기를 바꾸기 시작하는 에포크 설정 (뒤에서 n번째부터 시작)<br/>
 --seed: random값의 기준 설정<br/>
 --transform: 사용할 augmentation 클래스(라이브러리 기준으로 나눔)를 선택<br/>
 --augmentations: 사용할 augmentation 기법을 설정. "_"로 split하여 string 분리<br/>
 --adjust_ratio: 이미지의 종횡비를 1:1로 맞춤<br/>
 --eraly_stopping: 개선이 있는지 감시할 epoch 수 설정. 이 epoch동안 validation accuracy의 개선이 없으면 학습 중단함<br/>
 --verbose: tqdm 사용 여부 결정. 주석 풀면 True, 아니면 False<br/>
 --resume, --checkpoint_path: 체크포인트에 저장된 모델 불러오기 여부, 체크포인트.pt 파일 경로. 세트로 사용한다.<br/>
<br/><br/>
train.py: trainer.py의 trainer 클래스를 불러와서 학습 시킴<br/><br/>

test.sh, test.py : test.sh에서 인자를 받아 test.py 파일을 실행해 test data의 예측 결과 저장. train.sh와 비슷함<br/><br/>

trainer.py: 학습 모듈<br/>
 -create_config_txt : train.sh 호출 당시 내용을 checkpoint 폴더에 함께 저장하여 어떤 하이퍼파라미터를 사용했는지 기록<br/>
 -save_checkpoint_tmp : 이전 fold(or epoch)와 비교하여 validation accuracy가 1% 이상 개선되면 checkpoint 저장<br/>
 -final_save_model : 이전 accuracy와 관계 없이 마지막 모델 저장<br/>
 -train_epoch : 모델학습 1 epoch 수행<br/>
 -validate : 모델 검증 수행<br/>
 -train : epoch만큼 학습하는 함수. train.sh를 통해 전달받은 resome 파라미터가 true이면 self.load_settings 함수로 checkpoint 모델을 불러온다.<br/>
 -k_fold_train : train 함수에 K-Fold Cross Validation을 적용함<br/>
 -load_settings 체크포인트 저장 시점의 모델과 optimizer, scheduler 등 학습에 필요한 정보를 불러온다.<br/><br/>

eda.py: 모든 데이터의 메타데이터를 추출하여 csv파일로 만드는 파일<br/><br/>

args.py: train.sh, test.sh에서 받아온 인자를 파이썬에서 사용할 수 있는 변수로 변환하는 모듈<br/><br/>

gradcam.py<br/><br/>

image_augmentation.py: offline augmentation하는 파일. 종횡비를 맞추기 위해 흰 배경 추가하는 코드와 flip을 적용하는 코드가 있다. 추가된 이미지를 포함한 ./data/train1.csv 파일을 생성한다.<br/><br/>

separate.py: 데이터셋을 물리적으로 분리하는 파일<br/><br/>

util 폴더<br/>
augmentation.py: augmentation 라이브러리를 관리하는 모듈. Albumentation을 사용함<br/>
 -AlbumentationsTransforms 클래스: train.sh에서 받는 augmentations 인자를 가지고 클래스의 생성자가 full_aug_list를 보고 aug_list에 추가하여 사용할 증강 기법을 선택한다.<br/>
 -TransformSelector: train.sh에서 받은 transform 인자로 어떤 증강 클래스를 사용할지 선택<br/><br/>

checkpoints.py: 체크포인트를 저장/불러오기 하는 모듈<br/><br/>

data.py: Dataset, DataLoader를 재정의하는 모듈<br/>
 -CustomDataset 클래스: 대회를 위해 제공받은 데이터셋에 맞게 데이터를 불러오게하는 Dataset<br/>
 -HoDataset, HoDataLoader 클래스: K-Fold cross validation을 위한 Dataset, DataLoader<br/><br/>

losses.py: loss function을 가지고 있음<br/><br/>

metrics.py: f1 score을 계산하는 모듈<br/><br/>

optimizers.py: train.sh의 optim 인자를 받아서 optimizer를 선택할 수 있게 매핑하는 모듈<br/><br/>

schedulers.py: train.sh의 lr_scheduler 인자를 받아서 learning rate scheduler를 선택할 수 있게 매핑하는 모듈<br/><br/>

model 폴더<br/>
 model_selection 파일은 다른 모델을 불러오는 파일. timm, torchvision_model은 라이브러리를 쉽게 불러오기 위한 모듈<br/>
