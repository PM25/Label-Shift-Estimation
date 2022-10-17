import numpy as np


class BaseEstimator:
    def __str__(self):
        raise NotImplementedError()
    
    def __copy__(self):
        raise NotImplementedError()

    def fit(self, target_logits, src_logits, src_labels):
        raise NotImplementedError()

    def calibrate(self, logits):
        if self.calibrator is not None:
            logits = self.calibrator.calibrate(logits)
        cal_logits = logits * self.imp_weights[None, :]
        cal_logits = cal_logits / np.sum(cal_logits, axis=-1)[:, None]
        return cal_logits
