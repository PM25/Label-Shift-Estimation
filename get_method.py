from labelshift import *
from calibration import *


def get_lse_methods(lse_algs, calibrations, use_ensemble=True):
    names, estimators = [], []
    for lse_alg in lse_algs:
        for calibration in calibrations:
            names.append(f"{lse_alg}_{calibration}")
            estimator = get_lse_method(lse_alg, calibration, use_ensemble)
            estimators.append(estimator)

    return names, estimators


def get_lse_method(lse_alg, calibration=None, use_ensemble=True):
    # get calibration method
    if calibration is None or calibration.upper() == "NONE":
        calibrator = None
    elif calibration.upper() == "TS":
        calibrator = TemperatureScaling()
    elif calibration.upper() == "BCTS":
        calibrator = BCTemperatureScaling()
    else:
        raise RuntimeError(f"calibration method: {calibration} is not implemented.")

    # get label shift estimation method
    if lse_alg.upper() == "BBSE":
        Estimator = BBSE
    elif lse_alg.upper() == "RLLS":
        Estimator = RLLS
    elif lse_alg.upper() == "MLLS":
        Estimator = MLLS
    else:
        raise RuntimeError(f"label shift estimator: {lse_alg} is not implemented.")

    estimator = Estimator(calibrator=calibrator)

    if use_ensemble:
        estimator = EnsembleWrapper(estimator)

    return estimator
