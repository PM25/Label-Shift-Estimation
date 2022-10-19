import numpy as np

from .utils import same_dset

from labelshift.datasets import ResampleDataset

# same data with the same seed should output the same result
def test_resample():
    ntrain = 10000  # number of training data
    w, h, c = 32, 32, 3  # width, height, channels
    num_classes = 10

    fake_train_data = np.random.randint(0, 256, ntrain * h * w * c, dtype=np.uint8).reshape((ntrain, h, w, c))
    fake_train_labels = np.random.choice(range(num_classes), ntrain, replace=True)

    dset = ResampleDataset(fake_train_data, fake_train_labels, num_classes, seed=0)

    dsets1 = [(idx, train_set, val_set) for idx, train_set, val_set in dset.resample(5, 10)]
    dsets2 = [(idx, train_set, val_set) for idx, train_set, val_set in dset.resample(5, 10)]
    assert len(dsets1) == len(dsets2)

    for (idx1, train_set1, val_set1), (idx2, train_set2, val_set2) in zip(dsets1, dsets2):
        assert idx1 == idx2
        assert same_dset(train_set1, train_set2)
        assert same_dset(val_set1, val_set2)

    dset3 = ResampleDataset(fake_train_data, fake_train_labels, num_classes, seed=0)
    dsets3 = [(idx, train_set, val_set) for idx, train_set, val_set in dset3.resample(5, 10)]
    
    for (idx1, train_set1, val_set1), (idx3, train_set3, val_set3) in zip(dsets1, dsets3):
        assert idx1 == idx3
        assert same_dset(train_set1, train_set3)
        assert same_dset(val_set1, val_set3)

    dset4 = ResampleDataset(fake_train_data, fake_train_labels, num_classes, seed=1)
    dsets4 = [(idx, train_set, val_set) for idx, train_set, val_set in dset4.resample(5, 10)]

    for (idx1, train_set1, val_set1), (idx4, train_set4, val_set4) in zip(dsets1, dsets4):
        assert idx1 == idx4
        assert not same_dset(train_set1, train_set4)
        assert not same_dset(val_set1, val_set4)
