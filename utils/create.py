import torch
from torch.optim import lr_scheduler

def create_optimizer(optimizer_name, model_parameters, lr, weight_decay, extra_params=None):
    """
    지정된 이름과 매개변수를 사용하여 옵티마이저를 생성한다.

    Args:
        optimizer_name (str): 생성할 옵티마이저의 이름 (예: 'Adam', 'RMSprop', 'AdamW', 'sgd').
        model_parameters (iterable): 옵티마이저에 전달할 모델 파라미터.
        lr (float): 학습률.
        weight_decay (float): 가중치 감소(정규화) 매개변수.
        extra_params (dict, optional): 옵티마이저에 추가로 전달할 매개변수.

    Returns:
        torch.optim.Optimizer: 생성된 옵티마이저.
    """
    params = [p for p in model_parameters if p.requires_grad]
    if optimizer_name == 'Adam':
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay, amsgrad=True)
    elif optimizer_name == "RMSprop":
        return torch.optim.RMSprop(params, lr=lr, weight_decay=weight_decay, alpha=0.9, momentum=0.9, eps=1e-08, centered=False)
    elif optimizer_name == 'AdamW':
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, amsgrad=True)
    elif optimizer_name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
def create_scheduler(scheduler_name, optimizer, max_epochs, step_size=2, gamma=0.1):
    """
    지정된 이름과 매개변수를 사용하여 학습률 스케줄러를 생성한다.

    Args:
        scheduler_name (str): 생성할 스케줄러의 이름 (예: 'cosine', 'step', 'exponential').
        optimizer (torch.optim.Optimizer): 스케줄러에 연결할 옵티마이저.
        max_epochs (int): 최대 에폭 수.

    Returns:
        torch.optim.lr_scheduler._LRScheduler: 생성된 스케줄러.
    """
    if scheduler_name == "cosine":
        return lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    elif scheduler_name == "step":
        return lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.99)
    elif scheduler_name == "exponential":
        return lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    elif scheduler_name == "MultiStepLR":
        return lr_scheduler.MultiStepLR(optimizer, milestones=[max_epochs // 2], gamma=0.1)
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")