import numpy as np


class EnsembleWrapper:
    def __init__(self, estimator):
        self.estimator = estimator
        self.estimators = []
        self.estim_source_dist = None
        self.estim_target_dist = None
        self.imp_weights = None

    def __str__(self):
        return self.estimator.__str__()

    def fit(self, target_logits, src_logits, src_labels=None):
        """
        Args:
            target_logits (list): list of target logits
            src_logits (list): list of source logits
            src_labels (list): list of source labels
        """
        if src_labels is None:
            src_labels = [None] * len(src_logits)
        assert len(target_logits) == len(src_logits) == len(src_labels), "length of target_logits, src_logits and src_labels mismatch"

        estim_source_dists = []
        estim_target_dists = []
        imp_weights = []
        for _target_logits, _src_logits, _src_labels in zip(target_logits, src_logits, src_labels):
            _estimator = self.estimator.__copy__()
            e = _estimator.fit(_target_logits, _src_logits, _src_labels)
            estim_source_dists.append(e.estim_source_dist)
            estim_target_dists.append(e.estim_target_dist)
            imp_weights.append(e.imp_weights)
            self.estimators.append(_estimator)

        self.estim_source_dist = np.mean(estim_source_dists, axis=0)
        self.estim_target_dist = np.mean(estim_target_dists, axis=0)
        self.imp_weights = np.mean(imp_weights, axis=0)

        return self

    def calibrate(self, logits):
        assert len(logits) == len(self.estimators)
        outputs = []
        for _logits, _estimator in zip(logits, self.estimators):
            output = _estimator.calibrate(_logits)
            outputs.append(output)
        return np.mean(outputs, axis=0)
