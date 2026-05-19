import inspect
import logging

import numpy as np


def correct_residuals(residuals, res_const=1):
    if np.all(residuals > 0):
        return residuals
    residuals[residuals <= 0] = res_const
    return residuals


def cqr(q_lo, q_hi, y, alpha):
    n = len(y)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    scores = np.maximum(q_lo - y, y - q_hi)
    q = np.partition(scores, k - 1)[k - 1]
    return q, q


def cqr_r(q_lo, q_hi, y, alpha):
    n = len(y)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    residual = correct_residuals(q_hi - q_lo)
    scores = np.maximum(q_lo - y, y - q_hi)
    scores = scores / residual
    q = np.partition(scores, k - 1)[k - 1]
    return q, q


def cqr_m(q_lo, q_hi, q_med, y, alpha):
    n = len(y)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    residual_lo = correct_residuals(q_med - q_lo)
    residual_hi = correct_residuals(q_hi - q_med)
    scores = np.maximum((q_lo - y) / residual_lo, (y - q_hi) / residual_hi)
    q = np.partition(scores, k - 1)[k - 1]
    return q, q


def signed_err_cr(err, alpha):
    n = len(err)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    q = np.partition(err, k - 1)[k - 1]
    return q


def se_cqr(q_lo, q_hi, y, alpha):
    q_lo_se = signed_err_cr(q_lo - y, alpha / 2)
    q_hi_se = signed_err_cr(y - q_hi, alpha / 2)
    return q_lo_se, q_hi_se


def se_cqr_r(q_lo, q_hi, y, alpha):
    residual = correct_residuals(q_hi - q_lo)
    q_lo_se = signed_err_cr((q_lo - y) / residual, alpha / 2)
    q_hi_se = signed_err_cr((y - q_hi) / residual, alpha / 2)
    return q_lo_se, q_hi_se


def se_cqr_m(q_lo, q_hi, q_med, y, alpha):
    residual_lo = correct_residuals(q_med - q_lo)
    residual_hi = correct_residuals(q_hi - q_med)
    q_lo_se = signed_err_cr((q_lo - y) / residual_lo, alpha / 2)
    q_hi_se = signed_err_cr((y - q_hi) / residual_hi, alpha / 2)
    return q_lo_se, q_hi_se


class ConformalPredictor:
    supported_methods = [
        "CQR",
        "CQR-r",
        "CQR-m",
        "SE-CQR",
        "SE-CQR-r",
        "SE-CQR-m",
    ]
    methods_mapping = {
        "CQR": cqr,
        "CQR-r": cqr_r,
        "CQR-m": cqr_m,
        "SE-CQR": se_cqr,
        "SE-CQR-r": se_cqr_r,
        "SE-CQR-m": se_cqr_m,
    }

    def __init__(self, method_name, alpha, q_lo_cal, q_hi_cal, q_med_cal, y):
        self.alpha = alpha
        self.method_name = method_name
        try:
            self.cp_function = self.methods_mapping[method_name]
        except KeyError:
            raise ValueError(f"Unsupported CP method: {method_name}")

        coverage_init = np.mean((q_lo_cal <= y) & (y <= q_hi_cal))
        logging.debug(
            "Initial coverage: %.2f",
            coverage_init,
        )

        sig = inspect.signature(self.cp_function)
        arg_names = list(sig.parameters.keys())
        params_all = {
            "q_lo": q_lo_cal,
            "q_hi": q_hi_cal,
            "q_med": q_med_cal,
            "y": y,
            "alpha": alpha,
        }
        method_params = {
            param_name: params_all[param_name] for param_name in arg_names
        }
        q_lo, q_hi = self.cp_function(**method_params)
        self.q_lo, self.q_hi = float(q_lo), float(q_hi)

        coverage = (q_lo - q_lo <= y) & (y <= q_hi + q_hi)
        empirical_coverage = np.mean(coverage)
        width_increase = q_lo + q_hi
        avg_base_width = (q_hi - q_lo).mean()
        logging.debug(
            "Calibration coverage: %.2f, width increase from avg. %.2f by %.2f",
            empirical_coverage,
            avg_base_width,
            width_increase,
        )

    @classmethod
    def from_precomputed(cls, method_name, alpha, q_lo, q_hi):
        instance = cls.__new__(cls)
        try:
            instance.cp_function = instance.methods_mapping[method_name]
        except KeyError:
            raise ValueError(f"Unsupported CP method: {method_name}")
        instance.alpha = alpha
        instance.method_name = method_name
        instance.q_lo, instance.q_hi = q_lo, q_hi
        return instance

    def calibrate(self, q_lo, q_hi, q_med):
        if self.method_name in ["CQR", "SE-CQR"]:
            q_lo_corrected = q_lo - self.q_lo
            q_hi_corrected = q_hi + self.q_hi
        elif self.method_name in ["CQR-r", "SE-CQR-r"]:
            q_lo_corrected = q_lo - self.q_lo * correct_residuals(q_hi - q_lo)
            q_hi_corrected = q_hi + self.q_hi * correct_residuals(q_hi - q_lo)
        elif self.method_name in ["CQR-m", "SE-CQR-m"]:
            q_lo_corrected = q_lo - self.q_lo * correct_residuals(q_med - q_lo)
            q_hi_corrected = q_hi + self.q_hi * correct_residuals(q_hi - q_med)
        else:
            raise ValueError(f"Unknown CP method name: {self.method_name}")
        return q_lo_corrected, q_hi_corrected
