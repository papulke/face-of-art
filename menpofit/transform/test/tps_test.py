import numpy as np
from numpy.testing import assert_allclose

from menpo.shape import PointCloud
from menpofit.transform import DifferentiableThinPlateSplines


def test_tps_d_dx():
    src = PointCloud(np.array([[-1.0, -1.0], [-1, 1], [1, -1], [1, 1]]))
    tgt = PointCloud(np.array([[-2.0, -2.0], [-2, 2], [2, -2], [2, 2]]))
    tps = DifferentiableThinPlateSplines(src, tgt)
    result = tps.d_dx(src.points)

    assert_allclose(result, np.array([[[2, 0], [0., 2]],
                                      [[2, 0], [0, 2]],
                                      [[2, 0], [0, 2]],
                                      [[2, 0], [0, 2]]]), atol=10 ** -6)


def test_tps_d_dl():
    src = PointCloud(np.array([[-1.0, -1.0], [-1, 1], [1, -1], [1, 1]]))
    tgt = PointCloud(np.array([[-2.0, -2.0], [-2, 2], [2, -2], [2, 2]]))
    pts = np.array([[-0.1, -1.0], [-0.5, 1.0], [2.1, -2.5]])
    tps = DifferentiableThinPlateSplines(src, tgt)
    result = tps.d_dl(pts)
    expected = np.array([[[0.55399517, 0.55399517], [-0.00399517, -0.00399517],
                          [0.44600483, 0.44600483], [0.00399517, 0.00399517]],
                         [[-0.01625165, -0.01625165], [0.76625165, 0.76625165],
                          [0.01625165, 0.01625165], [0.23374835, 0.23374835]],
                         [[0.01631597, 0.01631597], [-0.56631597, -0.56631597],
                          [1.73368403, 1.73368403],
                          [-0.18368403, -0.18368403]]])
    assert_allclose(result, expected, rtol=10 ** -6)
