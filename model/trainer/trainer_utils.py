import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class Get_Scalar:
    def __init__(self, value):
        self.value = value

    def get_value(self, iter):
        return self.value

    def __call__(self, iter):
        return self.value


class DRWLoss:
    def __init__(self, args, class_num_list, warm_iter=100000, beta=0.9999):
        self.args = args
        self.beta = beta
        self.warm_iter = warm_iter
        self.uniform_weights = self.get_per_class_weights(0, class_num_list).cuda(args.gpu)
        self.per_class_weights = self.get_per_class_weights(self.beta, class_num_list).cuda(args.gpu)

    def get_per_class_weights(self, beta, class_num_list):
        effective_num = 1.0 - np.power(beta, class_num_list)
        per_class_weights = (1.0 - beta) / np.array(effective_num)
        per_class_weights = per_class_weights / np.sum(per_class_weights) * len(class_num_list)
        return torch.FloatTensor(per_class_weights)

    def update_dist(self, class_num_list):
        self.per_class_weights = self.get_per_class_weights(self.beta, class_num_list).cuda(self.args.gpu)

    def __call__(self, it, logits, targets, use_hard_labels=True, reduction="none"):
        weight = self.uniform_weights if it < self.warm_iter else self.per_class_weights
        if use_hard_labels:
            log_pred = F.log_softmax(logits, dim=-1)
            return F.nll_loss(log_pred, targets, weight=weight, reduction=reduction)
        else:
            assert logits.shape == targets.shape
            log_pred = F.log_softmax(logits, dim=-1)
            nll_loss = torch.sum(-targets * log_pred * weight, dim=1)
            return nll_loss


def mixup_one_target(x, y, gpu, alpha=1.0, is_bias=False):
    """Returns mixed inputs, mixed targets, and lambda"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    if is_bias:
        lam = max(lam, 1 - lam)

    index = torch.randperm(x.size(0)).cuda(gpu)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    mixed_y = lam * y + (1 - lam) * y[index]
    return mixed_x, mixed_y, lam


def to_one_hot(targets, nClass, gpu):
    logits = torch.zeros(targets.size(0), nClass).cuda(gpu)
    return logits.scatter_(1, targets.unsqueeze(1), 1)


def ce_loss(logits, targets, use_hard_labels=True, reduction="none"):
    """
    wrapper for cross entropy loss in pytorch.

    Args
        logits: logit values, shape=[Batch size, # of classes]
        targets: integer or vector, shape=[Batch size] or [Batch size, # of classes]
        use_hard_labels: If True, targets have [Batch size] shape with int values. If False, the target is vector (default True)
    """
    if use_hard_labels:
        log_pred = F.log_softmax(logits, dim=-1)
        return F.nll_loss(log_pred, targets, reduction=reduction)
        # return F.cross_entropy(logits, targets, reduction=reduction) this is unstable
    else:
        assert logits.shape == targets.shape
        log_pred = F.log_softmax(logits, dim=-1)
        nll_loss = torch.sum(-targets * log_pred, dim=1)
        return nll_loss


class EMA:
    """
    Implementation from https://fyubang.com/2019/06/01/ema/
    """

    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def load(self, ema_model):
        for name, param in ema_model.named_parameters():
            self.shadow[name] = param.data.clone()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class Bn_Controller:
    def __init__(self):
        """
        freeze_bn and unfreeze_bn must appear in pairs
        """
        self.backup = {}

    def freeze_bn(self, model):
        assert self.backup == {}
        for name, m in model.named_modules():
            if isinstance(m, nn.SyncBatchNorm) or isinstance(m, nn.BatchNorm2d):
                self.backup[name + ".running_mean"] = m.running_mean.data.clone()
                self.backup[name + ".running_var"] = m.running_var.data.clone()
                self.backup[name + ".num_batches_tracked"] = m.num_batches_tracked.data.clone()

    def unfreeze_bn(self, model):
        for name, m in model.named_modules():
            if isinstance(m, nn.SyncBatchNorm) or isinstance(m, nn.BatchNorm2d):
                m.running_mean.data = self.backup[name + ".running_mean"]
                m.running_var.data = self.backup[name + ".running_var"]
                m.num_batches_tracked.data = self.backup[name + ".num_batches_tracked"]
        self.backup = {}
