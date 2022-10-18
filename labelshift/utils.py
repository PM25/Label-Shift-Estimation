import math
import yaml
import numpy as np

import torch
from torch.optim.lr_scheduler import LambdaLR


def softmax(logits):
    """
    Compute softmax values for each sets of scores in logits.

    Parameters:
        logits (numpy.ndarray): array containing m samples with n-dimensions (m,n)
    Returns:
        x_softmax (numpy.ndarray) softmaxed values for initial (m,n) array
    """
    e_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return e_logits / e_logits.sum(axis=-1, keepdims=True)


def inverse_softmax(softmax_preds):
    return np.log(softmax_preds) - np.mean(np.log(softmax_preds), axis=1)[:, None]


def to_one_hot(labels: np.array) -> np.array:
    if labels.ndim == 1:  # skip if labels are already one hot encoded
        n_labels = np.max(labels) + 1
        labels = np.eye(n_labels)[labels]
    return labels


def to_hard_labels(soft_labels, one_hot=True):
    hard_labels = np.argmax(soft_labels, axis=1)
    if one_hot:
        hard_labels = to_one_hot(hard_labels)
    return hard_labels


def get_confusion_matrix(labels, logits, normalize=False):
    assert logits.shape[0] == labels.shape[0]
    if labels.ndim != 1:  # check if labels are one hot encoded
        labels = np.argmax(labels, axis=1)

    num_class = np.unique(labels).shape[0]
    assert num_class == np.max(labels) + 1
    cf_mat = np.zeros((num_class, num_class))

    for logit, label in zip(logits, labels):
        cf_mat[:, label] += logit

    if normalize:
        # for c in range(num_class):
        #     cf_mat[:, c] /= cf_mat[:, c].sum()
        cf_mat /= cf_mat.sum()

    return cf_mat


def labels_to_dist(targets, num_classes, to_prob=True):
    ulb_dist = np.zeros(num_classes, dtype=int)
    for y in targets:
        ulb_dist[y] += 1
    if to_prob:
        ulb_dist = ulb_dist / ulb_dist.sum()
    return ulb_dist


def over_write_args_from_dict(args, dic):
    for k, v in dic.items():
        setattr(args, k, v)
    return args


def over_write_args_from_file(args, yml):
    if yml == "":
        return
    with open(yml, "r", encoding="utf-8") as f:
        dic = yaml.load(f.read(), Loader=yaml.Loader)
        for k in dic:
            setattr(args, k, dic[k])


def setattr_cls_from_kwargs(cls, kwargs, verbose=True):
    # if default values are in the cls,
    # overlap the value by kwargs
    for key in kwargs.keys():
        if hasattr(cls, key) and verbose:
            print(f"{key} in {cls} is overlapped by kwargs: {getattr(cls, key)} -> {kwargs[key]}")
        setattr(cls, key, kwargs[key])


def test_setattr_cls_from_kwargs():
    class _test_cls:
        def __init__(self):
            self.a = 1
            self.b = "hello"

    test_cls = _test_cls()
    config = {"a": 3, "b": "change_hello", "c": 5}
    setattr_cls_from_kwargs(test_cls, config)
    for key in config.keys():
        print(f"{key}:\t {getattr(test_cls, key)}")


def net_builder(net_name, from_name: bool, net_conf=None, is_remix=False, verbose=True):
    """
    return **class** of backbone network (not instance).
    Args
        net_name: 'WideResNet' or network names in torchvision.models
        from_name: If True, net_buidler takes models in torch.vision models. Then, net_conf is ignored.
        net_conf: When from_name is False, net_conf is the configuration of backbone network (now, only WRN is supported).
    """
    if from_name:
        import torchvision.models as models

        model_name_list = sorted(
            name for name in models.__dict__ if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
        )

        if net_name not in model_name_list:
            assert Exception(
                f"[!] Networks' Name is wrong, check net config, \
                               expected: {model_name_list}  \
                               received: {net_name}"
            )
        else:
            return models.__dict__[net_name]

    else:
        if net_name == "WideResNet":
            import labelshift.model.nets.wrn as net

            builder = getattr(net, "build_WideResNet")()
        elif net_name == "ResNet50":
            import labelshift.model.nets.resnet50 as net

            builder = getattr(net, "build_ResNet50")(is_remix)
        else:
            assert Exception("Not Implemented Error")

        if net_name != "ResNet50":
            setattr_cls_from_kwargs(builder, net_conf, verbose=verbose)
        
        return builder.build


def test_net_builder(net_name, from_name, net_conf=None):
    builder = net_builder(net_name, from_name, net_conf)
    print(f"net_name: {net_name}, from_name: {from_name}, net_conf: {net_conf}")
    print(builder)


def get_optimizer(net, optim_name="SGD", lr=0.1, momentum=0.9, weight_decay=0.0, nesterov=True, bn_wd_skip=True):
    """
    return optimizer (name) in torch.optim.
    If bn_wd_skip, the optimizer does not apply
    weight decay regularization on parameters in batch normalization.
    """

    decay = []
    no_decay = []
    for name, param in net.named_parameters():
        if ("bn" in name or "bias" in name) and bn_wd_skip:
            no_decay.append(param)
        else:
            decay.append(param)

    per_param_args = [{"params": decay}, {"params": no_decay, "weight_decay": 0.0}]

    if optim_name == "SGD":
        optimizer = torch.optim.SGD(per_param_args, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)
    elif optim_name == "AdamW":
        optimizer = torch.optim.AdamW(per_param_args, lr=lr, weight_decay=weight_decay)
    elif optim_name == "Adam":
        optimizer = torch.optim.Adam(per_param_args, lr=lr)
    else:
        raise RuntimeError(f"optimizer: {optim_name} is not provided.")
    
    return optimizer


def get_cosine_schedule_with_warmup(optimizer, num_training_steps, num_cycles=7.0 / 16.0, num_warmup_steps=0, last_epoch=-1):
    """
    Get cosine scheduler (LambdaLR).
    if warmup is needed, set num_warmup_steps (int) > 0.
    """

    def _lr_lambda(current_step):
        """
        _lr_lambda returns a multiplicative factor given an interger parameter epochs.
        Decaying criteria: last_epoch
        """

        if current_step < num_warmup_steps:
            _lr = float(current_step) / float(max(1, num_warmup_steps))
        else:
            num_cos_steps = float(current_step - num_warmup_steps)
            num_cos_steps = num_cos_steps / float(max(1, num_training_steps - num_warmup_steps))
            _lr = max(0.0, math.cos(math.pi * num_cycles * num_cos_steps))
        return _lr

    return LambdaLR(optimizer, _lr_lambda, last_epoch)
