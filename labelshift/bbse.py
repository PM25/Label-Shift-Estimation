import numpy as np
from numpy.linalg import inv

from utils import to_one_hot, softmax, to_hard_labels, get_confusion_matrix

from .base import BaseEstimator


class BBSE(BaseEstimator):
    def __init__(self, hard_label=False, calibrator=None, verbose=False):
        self.hard_label = hard_label
        self.calibrator = calibrator
        self.verbose = verbose
        self.estim_source_dist = None  # estimate source class distribution
        self.estim_target_dist = None  # estimate target class distribution
        self.imp_weights = None  # importance weights

    def __str__(self):
        return "Black Box Shift Estimation (BBSE)"

    def __copy__(self):
        calirator_copy = self.calibrator.__copy__() if self.calibrator is not None else None
        bbse_copy = BBSE(self.hard_label, calirator_copy, self.verbose)
        bbse_copy.estim_source_dist = self.estim_source_dist
        bbse_copy.estim_target_dist = self.estim_target_dist
        bbse_copy.imp_weights = self.imp_weights
        return bbse_copy

    def fit(self, target_logits, src_logits, src_labels):
        src_labels = to_one_hot(src_labels)

        if self.calibrator is not None:
            assert src_labels is not None, "source_labels can't be None"
            self.calibrator.fit(src_logits, src_labels)
            src_logits = self.calibrator.calibrate(src_logits)
            target_logits = self.calibrator.calibrate(target_logits)
        else:
            src_logits = softmax(src_logits)
            target_logits = softmax(target_logits)

        if self.hard_label:
            target_logits = to_hard_labels(target_logits, one_hot=True)
            src_logits = to_hard_labels(src_logits, one_hot=True)

        self.estim_source_dist = np.mean(src_labels, axis=0)
        self.estim_source_dist /= np.sum(self.estim_source_dist)  # normalize

        mean_target_logits = np.mean(target_logits, axis=0)
        confusion_matrix = get_confusion_matrix(src_labels, src_logits, normalize=True)

        self.imp_weights = np.matmul(inv(confusion_matrix), mean_target_logits)
        self.imp_weights = self.imp_weights * (self.imp_weights >= 0)  # mask out negative weights

        self.estim_target_dist = self.imp_weights * self.estim_source_dist
        self.estim_target_dist = self.estim_target_dist / np.sum(self.estim_target_dist)
        self.imp_weights = self.estim_target_dist / self.estim_source_dist

        if self.verbose:
            with np.printoptions(precision=3, suppress=True, formatter={"float": "{: 0.3f}".format}):
                print(f"Est Source Class Distribution: {self.estim_source_dist}")
                print(f"Est Target Class Distribution: {self.estim_target_dist}")
                print(f"Est Importance Weights       : {self.imp_weights}")

        return self
