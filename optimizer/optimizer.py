import torch.optim as optim
import torch

# a customized scheduler that linear interpolates lr between specified  epochs
class CustomLinearScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, lr_epochs, lrs, last_epoch=-1, verbose=False):
        self.optimizer = optimizer
        # calculate learning rates
        self.learning_rates = []
        for epoch_start, epoch_end, lr_start, lr_end in zip(lr_epochs[:-1], lr_epochs[1:], lrs[:-1], lrs[1:]):
            self.learning_rates.extend(torch.linspace(start=lr_start, end=lr_end, steps=(epoch_end-epoch_start)))
            
        super(CustomLinearScheduler, self).__init__(optimizer, last_epoch, verbose)
        
        
    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.")
        lr = self.learning_rates[self.last_epoch] if len(self.learning_rates)>self.last_epoch else self.learning_rates[-1]
        return [lr]

def fetch_optimizer(cfg, model):
    weight_decay_l2 = cfg.optimizer.get("weight_decay_l2", 0)
    total_params = [p for p in model.parameters() if p.requires_grad]
    if cfg.optimizer.name == "adam":
        optimizer = optim.Adam(total_params, lr=cfg.optimizer.lr, weight_decay=weight_decay_l2)
    elif cfg.optimizer.name == "adamw":
        optimizer = optim.AdamW(total_params, lr=cfg.optimizer.lr, weight_decay=weight_decay_l2)
    elif cfg.optimizer.name == "sgd":
        optimizer = optim.SGD(total_params, lr=cfg.optimizer.lr, weight_decay=weight_decay_l2)
    else:
        raise NotImplementedError("Unknown optimizer type: {}.".format(cfg.optimizer.name))

    if cfg.scheduler.name == "exp":
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.scheduler.gamma)
    elif cfg.scheduler.name == "step":
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.scheduler.milestones, gamma=cfg.scheduler.gamma)
    elif cfg.scheduler.name == "custom_linear":
        scheduler = CustomLinearScheduler(optimizer, lr_epochs=cfg.scheduler.lr_epochs, lrs=cfg.scheduler.lrs)
    else:
        raise NotImplementedError("Unknown scheduler type: {}.".format(cfg.scheduler.name))
    return optimizer, scheduler



def fetch_optimizer_with_params(cfg, model, param_dicts):
    weight_decay_l2 = cfg.optimizer.get("weight_decay_l2", 0)
    if cfg.optimizer.name == "adam":
        optimizer = optim.Adam(param_dicts, weight_decay=weight_decay_l2)
    elif cfg.optimizer.name == "adamw":
        optimizer = optim.AdamW(param_dicts, weight_decay=weight_decay_l2)
    elif cfg.optimizer.name == "sgd":
        optimizer = optim.SGD(param_dicts, weight_decay=weight_decay_l2)
    else:
        raise NotImplementedError("Unknown optimizer type: {}.".format(cfg.optimizer.name))

    if cfg.scheduler.name == "exp":
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.scheduler.gamma)
    elif cfg.scheduler.name == "step":
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.scheduler.milestones, gamma=cfg.scheduler.gamma)
    else:
        raise NotImplementedError("Unknown scheduler type: {}.".format(cfg.scheduler.name))
    return optimizer, scheduler


def fetch_optimizer_with_params_same_lr(cfg, model, param_dicts):
    weight_decay_l2 = cfg.optimizer.get("weight_decay_l2", 0)
    if cfg.optimizer.name == "adam":
        optimizer = optim.Adam(param_dicts, lr=cfg.optimizer.lr, weight_decay=weight_decay_l2)
    elif cfg.optimizer.name == "adamw":
        optimizer = optim.AdamW(param_dicts, lr=cfg.optimizer.lr, weight_decay=weight_decay_l2)
    elif cfg.optimizer.name == "sgd":
        optimizer = optim.SGD(param_dicts, lr=cfg.optimizer.lr, weight_decay=weight_decay_l2)
    else:
        raise NotImplementedError("Unknown optimizer type: {}.".format(cfg.optimizer.name))

    if cfg.scheduler.name == "exp":
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.scheduler.gamma)
    elif cfg.scheduler.name == "step":
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.scheduler.milestones, gamma=cfg.scheduler.gamma)
    else:
        raise NotImplementedError("Unknown scheduler type: {}.".format(cfg.scheduler.name))
    return optimizer, scheduler

def fetch_optimizer_with_params_only(cfg, model, param_dicts):
    if cfg.optimizer.name == "adam":
        optimizer = optim.Adam(param_dicts)
    elif cfg.optimizer.name == "adamw":
        optimizer = optim.AdamW(param_dicts)
    elif cfg.optimizer.name == "sgd":
        optimizer = optim.SGD(param_dicts)
    else:
        raise NotImplementedError("Unknown optimizer type: {}.".format(cfg.optimizer.name))

    if cfg.scheduler.name == "exp":
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.scheduler.gamma)
    elif cfg.scheduler.name == "step":
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.scheduler.milestones, gamma=cfg.scheduler.gamma)
    else:
        raise NotImplementedError("Unknown scheduler type: {}.".format(cfg.scheduler.name))
    return optimizer, scheduler