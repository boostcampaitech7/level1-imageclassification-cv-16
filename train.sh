# 실행은 Terminal에 sh exp.sh
# 학습 재개를 원한다면 인자로 --resume True와 --weights_path [학습재개하고자하는 가중치 파일]을 넘겨주세요.
# timm- 관련 모델을 사용하려고 한다면 timm-resnet50, timm-resnet18 처럼 작성해주면 됨
python train.py \
    --mode train \
    --device cuda \
    --data_root ./data \
    --train_csv ./data/train1.csv \
    --val_csv ./data/val.csv \
    --height 384 \
    --width 384 \
    --num_classes 500 \
    --auto_split True \
    --split_seed 42 \
    --stratify target \
    --model timm-timm-tf_efficientnetv2_xl.in21k_ft_in1k \
    --lr 0.001 \
    --lr_scheduler ReduceLROnPlateau \
    --lr_scheduler_gamma 0.1 \
    --lr_scheduler_epochs_per_decay 2 \
    --batch 8 \
    --loss CE \
    --optim adam \
    --epochs 100 \
    --r_epochs 2 \
    --seed 2024 \
    --transform albumentations \
    --augmentations vflip_rotate\
    --adjust_ratio \
    --early_stopping 30 \
#    --resume True \
#    --weights_path ./checkpoints/checkpoint_epoch_16.pth

