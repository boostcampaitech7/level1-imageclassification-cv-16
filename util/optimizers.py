import torch.optim as optim

def get_optimizer(model, optimizer_name, lr=1e-3, **kwargs) -> optim.Optimizer:
    if optimizer_name == 'adam':
        return optim.Adam(model.parameters(), lr=lr, **kwargs)
    elif optimizer_name == 'sgd':
        return optim.SGD(model.parameters(), lr=lr, momentum=kwargs.get('momentum', 0.9))
    elif optimizer_name == 'rmsprop':
        return optim.RMSprop(model.parameters(), lr=lr, alpha=kwargs.get('alpha', 0.99))
    elif optimizer_name == 'nadam':
        return optim.NAdam(model.parameters(), lr=lr, **kwargs)
    elif optimizer_name == 'radam':
        return optim.RAdam(model.parameters(), lr=lr, **kwargs)
    elif optimizer_name == 'adagrad':
        return optim.Adagrad(model.parameters(), lr=lr, **kwargs)
    elif optimizer_name == 'adadelta':
        return optim.Adadelta(model.parameters(), lr=lr, **kwargs)
    elif optimizer_name == 'adamw':
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01) # 가중치감소를 직접 적용 -> L2 정규화 강화
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
"""
optimizer_name = 'adam'  (adam, sgd, rmsprop)
learning_rate = 1e-3
optimizer = get_optimizer(model, optimizer_name, lr=learning_rate)
"""