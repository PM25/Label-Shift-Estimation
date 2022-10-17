import cvxpy as cp
import numpy as np

from utils import to_one_hot, softmax, to_hard_labels, get_confusion_matrix

from .base import BaseEstimator


class RLLS(BaseEstimator):
    def __init__(self, hard_label=False, calibrator=None, alpha=0.01, verbose=False):
        self.hard_label = hard_label
        self.calibrator = calibrator
        self.alpha = alpha
        self.verbose = verbose
        self.estim_source_dist = None  # estimate source class distribution
        self.estim_target_dist = None  # estimate target class distribution
        self.imp_weights = None  # importance weights

    def __str__(self):
        return "Regularized Learning of Label Shift (RLLS)"

    def __copy__(self):
        calirator_copy = self.calibrator.__copy__() if self.calibrator is not None else None
        rlls_copy = RLLS(self.hard_label, calirator_copy, self.alpha, self.verbose)
        rlls_copy.estim_source_dist = self.estim_source_dist
        rlls_copy.estim_target_dist = self.estim_target_dist
        rlls_copy.imp_weights = self.imp_weights
        return rlls_copy

    # modified from: https://github.com/Angela0428/labelshift/blob/5bbe517938f4e3f5bd14c2c105de973dcc2e0917/label_shift.py
    def compute_3deltaC(self, n_class, n_train, delta):
        rho = 3 * (2 * np.log(2 * n_class / delta) / (3 * n_train) + np.sqrt(2 * np.log(2 * n_class / delta) / n_train))
        return rho

    # modified from: https://github.com/Angela0428/labelshift/blob/5bbe517938f4e3f5bd14c2c105de973dcc2e0917/label_shift.py
    def compute_w_opt(self, C_yy, mu_y, mu_train_y, rho):
        n = C_yy.shape[0]
        theta = cp.Variable(n)
        b = mu_y - mu_train_y
        objective = cp.Minimize(cp.pnorm(C_yy @ theta - b) + rho * cp.pnorm(theta))
        constraints = [-1 <= theta]
        prob = cp.Problem(objective, constraints)

        # The optimal objective value is returned by `prob.solve()`.
        result = prob.solve()
        # The optimal value for x is stored in `x.value`.
        # print(theta.value)
        w = 1 + theta.value
        return w

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
        mean_src_logits = np.mean(src_logits, axis=0)
        confusion_matrix = get_confusion_matrix(src_labels, src_logits, normalize=True)

        num_train = src_logits.shape[0]
        num_classes = src_logits.shape[1]
        rho = self.compute_3deltaC(num_classes, num_train, delta=0.05)
        self.imp_weights = self.compute_w_opt(
            C_yy=confusion_matrix,
            mu_y=mean_target_logits,
            mu_train_y=mean_src_logits,
            rho=self.alpha * rho,
        )
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
