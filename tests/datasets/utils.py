import numpy as np


def same_dset(dset1, dset2):
    # check dset size are equal
    if len(dset1) != len(dset2):
        return False
    for i in range(len(dset1)):
        # check output size are equal
        if len(dset1[i]) != len(dset2[i]):
            return False
        # check output are equal
        for j in range(len(dset1[i])):
            if np.any(dset1[i][j] != dset2[i][j]):
                return False
    return True
