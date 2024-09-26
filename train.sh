# 실행은 Terminal에 sh exp.sh
# 학습 재개를 원한다면 인자로 --resume True와 --weights_path [학습재개하고자하는 가중치 파일]을 넘겨주세요.
# timm- 관련 모델을 사용하려고 한다면 timm-resnet50, timm-resnet18 처럼 작성해주면 됨
python -u train.py \
    --mode train \
    --device cuda \
    --data_root ./data \
    --csv_path ./data/train_remove.csv \
    --val_csv ./data/val.csv \
    --height 224 \
    --width 224 \
    --num_classes 500 \
    --auto_split True \
    --split_seed 42 \
    --stratify target \
    --model timm-resnext101_32x32d.fb_wsl_ig1b_ft_in1k \
    --lr 0.001 \
    --lr_scheduler ReduceLROnPlateau \
    --lr_scheduler_gamma 0.1 \
    --lr_scheduler_epochs_per_decay 2 \
    --batch 32 \
    --loss CE \
    --optim adamw \
    --epochs 200 \
    --r_epochs 2 \
    --seed 2024 \
    --transform albumentations \
    --augmentations vflip_hflip_rotate_dropout_noise \
    --adjust_ratio \
    --early_stopping 10 \
#   --verbose
#    --resume \
#    --checkpoint_path ./checkpoints/checkpoint_epoch_16.pth

