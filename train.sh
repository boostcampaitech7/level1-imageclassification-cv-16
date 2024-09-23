# 실행은 Terminal에 sh exp.sh
python train.py \
    --mode train \
    --device cuda \
    --data_root ./data \
    --train_csv ./data/train.csv \
    --val_csv ./data/val.csv \
    --num_classes 500 \
    --auto_split True \
    --split_seed 42 \
    --stratify True \
    --model timm-resnet18 \
    --lr 0.001 \
    --lr_scheduler stepLR \
    --lr_scheduler_gamma 0.1 \
    --lr_scheduler_epochs_per_decay 2 \
    --batch 64 \
    --loss CE \
    --optim adam \
    --epochs 20 \
    --r_epochs 2 \
    --seed 2024 \
    --transform albumentations \

