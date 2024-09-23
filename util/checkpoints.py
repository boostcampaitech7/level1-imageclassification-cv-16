import torch
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '/model'))
from util.optimizers import get_optimizer
from model.model_selection import ModelSelector

def save_checkpoint(model, optimizer, epoch, loss, acc, seed, filepath, model_type, model_name, optimizer_name, lr, scheduler):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'acc': acc,
        'seed': seed,
        'model_type': model_type,
        'model_name': model_name,
        'optimizer_name': optimizer_name,
        'lr': lr,
        'scheduler': scheduler
    }
    os.makedirs('./checkpoints', exist_ok=True)
    torch.save(checkpoint, filepath) ## 제대로 저장되고 불러와지는지 확인해볼 것 => 냅둬도 괜찮을 듯
    print(f"Checkpoint saved at {filepath}")

def load_checkpoint(filepath):
    '''model, optimizer, epoch, val_loss, val_acc, scheduler 순서대로 출력'''
    checkpoint = torch.load(filepath)
    model = ModelSelector(model_type=checkpoint['model_type'], num_classes=500, model_name=checkpoint['model_name'], pretrained=True).get_model()
    optimizer = get_optimizer(model, checkpoint['optimizer_name'], checkpoint['lr'])
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    acc = checkpoint['acc']
    scheduler = checkpoint['scheduler']
    print(f"Checkpoint loaded from {filepath}")
    return model, optimizer, epoch, loss, acc, scheduler

