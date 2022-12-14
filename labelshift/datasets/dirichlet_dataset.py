import numpy as np

import torchvision
from torchvision import transforms

from .dataset import BasicDataset, ResampleDataset, ResampleDataset
from .data_utils import gen_dirichlet_list, sample_data


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


class Dirichlet_Dataset:
    """
    Dirichlet_Dataset class gets dataset from torchvision.datasets,
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
            train_dset = dset(self.data_dir, train=True, download=True)
            train_data, train_targets = train_dset.data, train_dset.targets
            test_dset = dset(self.data_dir, train=False, download=True)
            test_data, test_targets = test_dset.data, test_dset.targets
            return train_data, train_targets, test_data, test_targets

    def get_lb_ulb_dset(
        self,
        num_train_labels,
        num_val_labels,
        num_unlabeled,
        lb_alpha,
        ulb_alpha,
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
        train_data, train_targets, test_data, test_targets = self.get_data()

        state = np.random.get_state()
        np.random.seed(seed)

        train_data, train_targets = np.array(train_data), np.array(train_targets)
        train_class_num_list = gen_dirichlet_list(num_train_labels, lb_alpha, self.num_classes)
        val_class_num_list = gen_dirichlet_list(num_val_labels, "uniform", self.num_classes)
        labeled_class_num_list = train_class_num_list + val_class_num_list
        lb_data, lb_targets, _ = sample_data(train_data, train_targets, labeled_class_num_list, replace=False)

        test_data, test_targets = np.array(test_data), np.array(test_targets)
        unlabeled_class_num_list = gen_dirichlet_list(num_unlabeled, ulb_alpha, self.num_classes)
        ulb_data, ulb_targets, _ = sample_data(test_data, test_targets, unlabeled_class_num_list, replace=True)

        print(f"#train     : {train_class_num_list.sum()}, {train_class_num_list}")
        print(f"#validation: {val_class_num_list.sum()}, {val_class_num_list}")
        print(f"#unlabeled : {unlabeled_class_num_list.sum()}, {unlabeled_class_num_list}")

        np.random.set_state(state)

        lb_dset = ResampleDataset(lb_data, lb_targets, self.num_classes, self.train_transform, self.test_transform, onehot=onehot)
        ulb_dset = BasicDataset(ulb_data, ulb_targets, self.num_classes, self.test_transform, is_ulb=True, onehot=onehot)

        return lb_dset, ulb_dset
