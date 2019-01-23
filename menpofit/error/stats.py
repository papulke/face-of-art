from __future__ import division
import numpy as np
from scipy.integrate import simps
from collections import Iterable


def compute_cumulative_error(errors, bins):
    r"""
    Computes the values of the Cumulative Error Distribution (CED).

    Parameters
    ----------
    errors : `list` of `float`
        The `list` of errors per image.
    bins : `list` of `float`
        The values of the error bins centers at which the CED is evaluated.

    Returns
    -------
    ced : `list` of `float`
        The computed CED.
    """
    n_errors = len(errors)
    return [np.count_nonzero([errors <= x]) / n_errors for x in bins]


def mad(errors):
    r"""
    Computes the Median Absolute Deviation of a set of errors.

    Parameters
    ----------
    errors : `list` of `float`
        The `list` of errors per image.

    Returns
    -------
    mad : `float`
        The median absolute deviation value.
    """
    med = np.median(errors)
    return np.median(np.abs(errors - med))


def area_under_curve_and_failure_rate(errors, step_error, max_error,
                                      min_error=0.):
    r"""
    Computes the Area Under the Curve (AUC) and Failure Rate (FR) of a given
    Cumulative Distribution Error (CED).

    Parameters
    ----------
    errors : `list` of `float`
        The `list` of errors per image.
    step_error : `float`
        The sampling step of the error bins of the CED.
    max_error : `float`
        The maximum error value of the CED.
    min_error : `float`
        The minimum error value of the CED.

    Returns
    -------
    auc : `float`
        The Area Under the Curve value.
    fr : `float`
        The Failure Rate value.
    """
    x_axis = list(np.arange(min_error, max_error + step_error, step_error))
    ced = np.array(compute_cumulative_error(errors, x_axis))
    return simps(ced, x=x_axis) / max_error, 1. - ced[-1]


def compute_statistical_measures(errors, step_error, max_error, min_error=0.):
    r"""
    Computes various statistics given a set of errors that correspond to
    multiple images. It can also deal with multiple sets of errors that
    correspond to different methods.

    Parameters
    ----------
    errors : `list` of `float` or `list` of `list` of `float`
        The `list` of errors per image. You can provide a `list` of `lists`
        for the errors of multiple methods.
    step_error : `float`
        The sampling step of the error bins of the CED for computing the Area
        Under the Curve and the Failure Rate.
    max_error : `float`
        The maximum error value of the CED for computing the Area Under the
        Curve and the Failure Rate.
    min_error : `float`
        The minimum error value of the CED for computing the Area Under the
        Curve and the Failure Rate.

    Returns
    -------
    mean : `float` or `list` of `float`
        The mean value.
    mean : `float` or `list` of `float`
        The standard deviation.
    median : `float` or `list` of `float`
        The median value.
    mad : `float` or `list` of `float`
        The mean absolute deviation value.
    max : `float` or `list` of `float`
        The maximum value.
    auc : `float` or `list` of `float`
        The area under the curve value.
    fr : `float` or `list` of `float`
        The failure rate value.
    """
    if isinstance(errors[0], Iterable):
        mean_val = []
        std_val = []
        median_val = []
        mad_val = []
        max_val = []
        auc_val = []
        fail_val = []
        for e in errors:
            mean_val.append(np.mean(e))
            std_val.append(np.std(e))
            median_val.append(np.median(e))
            mad_val.append(mad(e))
            max_val.append(np.max(e))
            auc_v, fail_v = area_under_curve_and_failure_rate(
                    e, step_error=step_error, max_error=max_error,
                    min_error=min_error)
            auc_val.append(auc_v)
            fail_val.append(fail_v)
    else:
        mean_val = np.mean(errors)
        std_val = np.std(errors)
        median_val = np.median(errors)
        mad_val = mad(errors)
        max_val = np.max(errors)
        auc_val, fail_val = area_under_curve_and_failure_rate(
                errors, step_error=step_error, max_error=max_error,
                min_error=min_error)
    return mean_val, std_val, median_val, mad_val, max_val, auc_val, fail_val
