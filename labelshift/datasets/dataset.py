import numpy as np
from PIL import Image
from numpy.random import choice

from torch.utils.data import Dataset

from .data_utils import get_onehot


class BasicDataset(Dataset):
    """
    BasicDataset returns a pair of image and labels (targets).
    If targets are not given, BasicDataset returns None as the label.
    """

    def __init__(self, data, targets=None, num_classes=None, transform=None, is_ulb=False, onehot=False, *args, **kwargs):
        """
        Args
            data: x_data
            targets: y_data (if not exist, None)
            num_classes: number of label classes
            transform: basic transformation of data
            onehot: If True, label is converted into onehot vector.
        """
        super(BasicDataset, self).__init__()
        self.data = data
        self.targets = targets
        self.num_classes = num_classes
        self.transform = transform
        self.is_ulb = is_ulb
        self.onehot = onehot

    def __getitem__(self, idx):
        # set idx-th target
        if self.targets is None:
            target = None
        else:
            target_ = self.targets[idx]
            target = target_ if not self.onehot else get_onehot(self.num_classes, target_)

        img = self.data[idx]

        if self.transform is not None:
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            img = self.transform(img)

        if self.is_ulb:
            # return img, target
            return img, -1
        else:
            return img, target

    def __len__(self):
        return len(self.data)


class ResampleDataset(BasicDataset):
    """
    Random resampling subsets of a dataset.
    """

    def __init__(self, data, targets=None, num_classes=None, train_transform=None, test_transform=None, onehot=False, seed=0):
        """
        Args
            data: x_data
            targets: y_data (if not exist, None)
            num_classes: number of label classes
            train_transform: transformation of training data
            test_transform: transformation of validation data
            onehot: If True, label is converted into onehot vector.
        """
        super().__init__(data, targets, num_classes, None, onehot)
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.seed = seed
        assert self.seed is not None

    def resample(self, num_val_per_class, num_splits, replace=False):
        state = np.random.get_state()
        np.random.seed(self.seed)

        n_idxes = []
        for _ in range(num_splits):
            val_idxes, train_idxes = [], []
            for c in range(self.num_classes):
                idxes = np.where(self.targets == c)[0]
                _val_idxes = choice(idxes, num_val_per_class, replace=replace)
                val_idxes.append(_val_idxes)
                _train_idxes = list(set(idxes) - set(_val_idxes))
                train_idxes.append(_train_idxes)
            val_idxes = np.hstack(val_idxes)
            train_idxes = np.hstack(train_idxes)
            assert not np.isin(val_idxes, train_idxes).any()  # check no overlap
            n_idxes.append((train_idxes, val_idxes))

        np.random.set_state(state)

        # split to training and validation dataset
        for idx, (train_idxes, val_idxes) in enumerate(n_idxes):
            train_lb_dset = Subset(self, train_idxes, self.train_transform)
            val_lb_dset = Subset(self, val_idxes, self.test_transform)
            yield idx, train_lb_dset, val_lb_dset


class Subset(Dataset):
    """
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """

    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __getitem__(self, idx):
        img, target = self.dataset[self.indices[idx]]
        if self.transform is not None:
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.indices)
