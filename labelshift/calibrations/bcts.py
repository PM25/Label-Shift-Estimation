from scipy.optimize import minimize
from sklearn.metrics import log_loss

from labelshift.utils import softmax

from .base import BaseCalibrator


# Bias Corrected Temperature Scaling
class BCTemperatureScaling(BaseCalibrator):
    def __init__(self, temp=1, bias=0, maxiter=100, solver="L-BFGS-B"):
        self.temp = temp
        self.biases = bias
        self.maxiter = maxiter
        self.solver = solver

    def __str__(self):
        return "Bias-Corrected Temperature Scaling (BCTS)"

    def __copy__(self):
        return BCTemperatureScaling(self.temp, self.biases, self.maxiter, self.solver)

    def _loss_func(self, x, probs, labels):
        temp, biases = x[0], x[1:]
        scaled_probs = self.calibrate(probs, temp, biases)
        loss = log_loss(y_true=labels, y_pred=scaled_probs)
        return loss

    def calibrate(self, logits, temp=None, biases=None):
        if temp is not None and biases is not None:
            return softmax(logits / temp + biases)
        else:
            return softmax(logits / self.temp + self.biases)

    def fit(self, logits, labels):
        init_guess = [1.0] + [0.0 for _ in range(logits.shape[1])]

        result = minimize(
            fun=self._loss_func,
            args=(logits, labels),
            x0=init_guess,
            method=self.solver,
            tol=1e-07,
            options={"maxiter": self.maxiter},
        )

        assert result.success == True, f"BCTS Optimization Failed:\n{result}"
        self.temp, self.biases = result.x[0], result.x[1:]
        return self
