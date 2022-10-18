import numpy as np
from numpy.random import dirichlet

import torch
from torch.utils.data import sampler, DataLoader
from torch.utils.data.sampler import BatchSampler


def sample_data(data, target, class_num_list, replace=False):
    """
    FIXME: add description
    samples for labeled data
    (sampling with imbalanced ratio over classes)
    """
    lb_data, lbs, lb_idx = [], [], []
    for c in range(len(class_num_list)):
        idx = np.where(target == c)[0]
        assert replace or len(idx) >= class_num_list[c], "insufficient unlabeled data to select"
        idx = np.random.choice(idx, class_num_list[c], replace=replace)
        lb_idx.extend(idx)

        lb_data.extend(data[idx])
        lbs.extend(target[idx])

    return np.array(lb_data), np.array(lbs), np.array(lb_idx)


def gen_imb_list(max_num_cls_lb, imb_ratio, imb_type, num_classes):
    # FIXME: add description
    imb_ratio_lt1 = imb_ratio < 1
    if imb_ratio_lt1:
        imb_ratio = 1 / imb_ratio

    if imb_type == "long":
        mu = np.power(1 / imb_ratio, 1 / (num_classes - 1))
        class_num_list = []
        for i in range(num_classes):
            if i == (num_classes - 1):
                class_num_list.append(int(max_num_cls_lb / imb_ratio))
            else:
                class_num_list.append(int(max_num_cls_lb * np.power(mu, i)))

    if imb_type == "step":
        class_num_list = []
        for i in range(num_classes):
            if i < int(num_classes / 2):
                class_num_list.append(int(max_num_cls_lb))
            else:
                class_num_list.append(int(max_num_cls_lb / imb_ratio))

    if imb_ratio_lt1:
        class_num_list.reverse()

    class_num_list = np.array(class_num_list).astype(int)

    return class_num_list


def gen_dirichlet_list(num_data, alpha, num_classes):
    # FIXME: add description
    if alpha == "uniform":
        dist = np.ones(num_classes) / num_classes
    else:
        dist = dirichlet([alpha] * num_classes)

    class_num_list = dist * num_data
    class_num_list = class_num_list.round().astype(int)

    return class_num_list


def get_sampler_by_name(name):
    """
    get sampler in torch.utils.data.sampler by name
    """
    sampler_name_list = sorted(
        name for name in torch.utils.data.sampler.__dict__ if not name.startswith("_") and callable(sampler.__dict__[name])
    )
    try:
        return getattr(torch.utils.data.sampler, name)
    
    except Exception as e:
        print(repr(e))
        print("[!] select sampler in:\t", sampler_name_list)


def get_data_loader(
    dset,
    batch_size=None,
    shuffle=False,
    num_workers=4,
    pin_memory=False,
    data_sampler=None,
    replacement=True,
    num_epochs=None,
    num_iters=None,
    generator=None,
    drop_last=True,
):
    """
    get_data_loader returns torch.utils.data.DataLoader for a Dataset.
    All arguments are comparable with those of pytorch DataLoader.
    However, if distributed, DistributedProxySampler, which is a wrapper of data_sampler, is used.

    Args
        num_epochs: total batch -> (# of batches in dset) * num_epochs
        num_iters: total batch -> num_iters
    """

    assert batch_size is not None

    if data_sampler is None:
        return DataLoader(dset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)

    else:
        if isinstance(data_sampler, str):
            data_sampler = get_sampler_by_name(data_sampler)

        num_replicas = 1

        if (num_epochs is not None) and (num_iters is None):
            num_samples = len(dset) * num_epochs
        elif (num_epochs is None) and (num_iters is not None):
            num_samples = batch_size * num_iters * num_replicas
        else:
            num_samples = len(dset)

        if data_sampler.__name__ == "RandomSampler":
            data_sampler = data_sampler(dset, replacement, num_samples, generator)
        else:
            raise RuntimeError(f"{data_sampler.__name__} is not implemented.")

        batch_sampler = BatchSampler(data_sampler, batch_size, drop_last)
        return DataLoader(dset, batch_sampler=batch_sampler, num_workers=num_workers, pin_memory=pin_memory)


def get_onehot(num_classes, idx):
    onehot = np.zeros([num_classes], dtype=np.float32)
    onehot[idx] += 1.0
    return onehot
