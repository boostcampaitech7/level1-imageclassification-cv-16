import torch.optim as optim

def get_scheduler(
    lr_scheduler: str, 
    optimizer: optim.Optimizer, 
    scheduler_gamma: float, 
    epochs_per_decay: int, 
    steps_per_epoch: int=0) -> optim.lr_scheduler.LRScheduler:
    
    if lr_scheduler == 'stepLR':
        scheduler_gamma = scheduler_gamma # float 0.1
        
        epochs_per_lr_decay = epochs_per_decay
        scheduler_step_size = steps_per_epoch * epochs_per_lr_decay
        
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_step_size,
            gamma=scheduler_gamma
        )
    elif lr_scheduler == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=scheduler_gamma,
            patience=steps_per_epoch,
            verbose=True
        )
    elif lr_scheduler == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=100, 
            eta_min=0.001
        )
    elif lr_scheduler == 'CosineAnnealingWarmRestarts':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=50, 
            T_mult=2, 
            eta_min=0.001
        )
    else:
        raise ValueError(f"Unsupported scheduler: {lr_scheduler}")
    
    return scheduler