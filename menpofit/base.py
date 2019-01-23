from __future__ import division
import itertools
import os
import numpy as np


def batch(iterable, n):
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


def build_grid(shape):
    r"""
    """
    shape = np.asarray(shape)
    half_shape = np.floor(shape / 2)
    half_shape = np.require(half_shape, dtype=int)
    start = -half_shape
    end = half_shape + shape % 2
    sampling_grid = np.mgrid[start[0]:end[0], start[1]:end[1]]
    return np.rollaxis(sampling_grid, 0, 3)


class MenpoFitCostsWarning(Warning):
    r"""
    A warning that the costs cannot be computed for the selected fitting
    algorithm.
    """
    pass


def menpofit_src_dir_path():
    r"""The path to the top of the menpofit Python package.

    Useful for locating where the data folder is stored.

    Returns
    -------
    path : ``pathlib.Path``
        The full path to the top of the menpofit package
    """
    from pathlib import Path  # to avoid cluttering the menpo.base namespace
    return Path(os.path.abspath(__file__)).parent
