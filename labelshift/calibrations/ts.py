from scipy.optimize import minimize
from sklearn.metrics import log_loss

from labelshift.utils import softmax

from .base import BaseCalibrator


class TemperatureScaling(BaseCalibrator):
    def __init__(self, temp=1, maxiter=100, solver="BFGS"):
        self.temp = temp
        self.maxiter = maxiter
        self.solver = solver
    
    def __str__(self):
        return "Temperature Scaling (TS)"

    def __copy__(self):
        return TemperatureScaling(self.temp, self.maxiter, self.solver)

    def _loss_func(self, x, probs, labels):
        scaled_probs = self.calibrate(probs, x)
        loss = log_loss(y_true=labels, y_pred=scaled_probs)
        return loss

    def calibrate(self, logits, temp=None):
        if temp is not None:
            return softmax(logits / temp)
        else:
            return softmax(logits / self.temp)

    def fit(self, logits, labels):
        init_guess = 1.0

        result = minimize(
            fun=self._loss_func,
            args=(logits, labels),
            x0=init_guess,
            method=self.solver,
            tol=1e-07,
            options={"maxiter": self.maxiter},
        )

        assert result.success == True, f"TS Optimization Failed:\n{result}"
        self.temp = result.x[0]
        return self
