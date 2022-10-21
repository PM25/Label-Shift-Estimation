import numpy as np

import torchvision

from .dataset import BasicDataset, ResampleDataset, ResampleDataset
from .data_utils import gen_imb_list, sample_data
from .transform import get_transform_by_name


class Imbalanced_Dataset:
    """
    Imbalanced_Dataset class gets dataset from torchvision.datasets,
    separates labeled and unlabeled data,
    and return BasicDataset: torch.utils.data.Dataset (see datasets.dataset.py)
    """

    def __init__(self, name="cifar10", train=True, num_classes=10, data_dir="./data"):
        """
        Args
            name: name of dataset in torchvision.datasets (cifar10, cifar100, svhn, stl10)
            train: True means the dataset is training dataset (default=True)
            num_classes: number of label classes
            data_dir: path of directory, where data is downloaed or stored.
        """
        self.name = name
        self.train = train
        self.num_classes = num_classes
        self.data_dir = data_dir
        self.train_transform = get_transform_by_name(name, train=True)
        self.test_transform = get_transform_by_name(name, train=False)

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
            seed: Get deterministic results of labeled and unlabeld data.

        Returns:
            labeled (data, targets), unlabeled (data, targets)
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

        return (lb_data, lb_targets), (ulb_data, ulb_targets)
