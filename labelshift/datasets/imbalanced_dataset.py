import numpy as np

import torchvision
from torchvision import transforms

from .dataset import BasicDataset, ResampleDataset, ResampleDataset
from .data_utils import gen_imb_list, sample_data


mean, std = {}, {}
mean["cifar10"] = [x / 255 for x in [125.3, 123.0, 113.9]]
mean["cifar100"] = [x / 255 for x in [129.3, 124.1, 112.4]]

std["cifar10"] = [x / 255 for x in [63.0, 62.1, 66.7]]
std["cifar100"] = [x / 255 for x in [68.2, 65.4, 70.4]]


def get_transform(mean, std, crop_size, train=True):
    if train:
        return transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(crop_size, padding=4, padding_mode="reflect"),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    else:
        return transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])


class Imbalanced_Dataset:
    """
    Imbalanced_Dataset class gets dataset from torchvision.datasets,
    separates labeled and unlabeled data,
    and return BasicDataset: torch.utils.data.Dataset (see datasets.dataset.py)
    """

    def __init__(self, name="cifar10", num_classes=10, data_dir="./data"):
        """
        Args
            name: name of dataset in torchvision.datasets (cifar10, cifar100, svhn, stl10)
            train: True means the dataset is training dataset (default=True)
            num_classes: number of label classes
            data_dir: path of directory, where data is downloaed or stored.
        """
        self.name = name
        self.train = True
        self.num_classes = num_classes
        self.data_dir = data_dir
        self.crop_size = 32
        self.train_transform = get_transform(mean[name], std[name], self.crop_size, True)
        self.test_transform = get_transform(mean[name], std[name], self.crop_size, False)

    def get_data(self):
        """
        get_data returns data (images) and targets (labels)
        shape of data: B, H, W, C
        shape of labels: B,
        """
        dset = getattr(torchvision.datasets, self.name.upper())
        if "CIFAR" in self.name.upper():
            dset = dset(self.data_dir, train=self.train, download=True)
            data, targets = dset.data, dset.targets
            return data, targets

    def get_lb_ulb_dset(
        self,
        max_labeled_per_class,
        max_unlabeled_per_class,
        lb_imb_ratio,
        ulb_imb_ratio,
        imb_type,
        onehot=False,
        seed=0,
    ):
        """
        get_lb_ulb_dset split training samples into labeled and unlabeled samples.
        The labeled and unlabeled data might be imbalanced over classes.

        Args:
            num_labels: number of labeled data.
            lb_img_ratio: imbalance ratio of labeled data.
            ulb_imb_ratio: imbalance ratio of unlabeled data.
            imb_type: type of imbalance data.
            onehot: If True, the target is converted into onehot vector.
            seed: Get deterministic results of labeled and unlabeld data.

        Returns:
            ResampleDataset (for labeled data), BasicDataset (for unlabeld data)
        """
        data, targets = self.get_data()

        state = np.random.get_state()
        np.random.seed(seed)

        data, target = np.array(data), np.array(targets)
        labeled_class_num_list = gen_imb_list(max_labeled_per_class, lb_imb_ratio, imb_type, self.num_classes)
        lb_data, lb_targets, lb_idx = sample_data(data, target, labeled_class_num_list, replace=False)

        rst_idx = np.array(sorted(list(set(range(len(data))) - set(lb_idx))))  # unlabeled_data index of data

        unlabeled_class_num_list = gen_imb_list(max_unlabeled_per_class, ulb_imb_ratio, imb_type, self.num_classes)
        ulb_data, ulb_targets, ulb_idx = sample_data(data[rst_idx], target[rst_idx], unlabeled_class_num_list, replace=False)
        ulb_idx = rst_idx[ulb_idx]  # correct the ulb_idx

        all_idx = np.concatenate([lb_idx, ulb_idx])
        assert np.unique(all_idx).shape == all_idx.shape  # check no duplicate value

        print(f"#labeled  : {labeled_class_num_list.sum()}, {labeled_class_num_list}")
        print(f"#unlabeled: {unlabeled_class_num_list.sum()}, {unlabeled_class_num_list}")

        np.random.set_state(state)

        lb_dset = ResampleDataset(lb_data, lb_targets, self.num_classes, self.train_transform, self.test_transform, onehot=onehot)
        ulb_dset = BasicDataset(ulb_data, ulb_targets, self.num_classes, self.test_transform, is_ulb=True, onehot=onehot)

        return lb_dset, ulb_dset
