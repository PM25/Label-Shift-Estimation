class BaseCalibrator(object):
    def __str__(self):
        raise NotImplementedError()
    
    def __copy__(self):
        raise NotImplementedError()

    def fit(self, logits, labels):
        raise NotImplementedError()

    def calibrate(self):
        raise NotImplementedError()
