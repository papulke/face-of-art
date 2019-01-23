import numpy as np

from menpo.shape import PointCloud
from menpo.landmark import (face_ibug_68_to_face_ibug_49,
                            face_ibug_68_to_face_ibug_68,
                            face_ibug_49_to_face_ibug_49)

from menpofit.error import euclidean_error
from menpofit.error.base import (distance_normalised_error,
                                 distance_indexed_normalised_error,
                                 bb_normalised_error)


def _convert_68_to_51(shape):
    return PointCloud(shape.points[17:])


def _convert_68_to_49(shape):
    sp = shape.points.copy()
    sp = np.delete(sp, 64, 0)
    sp = np.delete(sp, 60, 0)
    sp = sp[17:]
    return PointCloud(sp)


def _convert_66_to_49(shape):
    return PointCloud(shape.points[17:])


def _convert_51_to_49(shape):
    sp = shape.points.copy()
    sp = np.delete(sp, 47, 0)
    sp = np.delete(sp, 43, 0)
    return PointCloud(sp)


def mean_pupil_68_error(shape, gt_shape):
    r"""
    Computes the Euclidean error based on 68 points normalised with the
    distance between the mean eye points (pupils), i.e.

    .. math::
       \frac{\mathcal{F}(s,s^*)}{\mathcal{N}(s)}

    where

    .. math::
       \mathcal{F}(s,s^*) = \frac{1}{68}\sum_{i=1}^{68}\sqrt{(s_{i,x}-s^*_{i,x})^2 + (s_{i,y}-s^*_{i,y})^2}

    where :math:`s` and :math:`s^*` are the final and ground truth shapes,
    respectively. :math:`(s_{i,x}, s_{i,y})` are the `x` and `y` coordinates of
    the :math:`i`'th point of the final shape, :math:`(s^*_{i,x}, s^*_{i,y})`
    are the `x` and `y` coordinates of the :math:`i`'th point of the ground
    truth shape. Finally, :math:`\mathcal{N}(s)` is the distance between the
    mean eye points (pupils).

    Parameters
    ----------
    shape : `menpo.shape.PointCloud`
        The input shape (e.g. the final shape of a fitting procedure). It
        must have 68 points.
    gt_shape : `menpo.shape.PointCloud`
        The ground truth shape. It must have 68 points.

    Returns
    -------
    normalised_error : `float`
        The computed normalised Euclidean error.

    Raises
    ------
    ValueError
        Final shape must have 68 points
    ValueError
        Ground truth shape must have 68 points
    """
    if shape.n_points != 68:
        raise ValueError('Final shape must have 68 points')
    if gt_shape.n_points != 68:
        raise ValueError('Ground truth shape must have 68 points')

    def pupil_dist(_, s):
        _, mapping = face_ibug_68_to_face_ibug_68(s, include_mapping=True)
        return euclidean_error(np.mean(s[mapping['left_eye']], axis=0),
                               np.mean(s[mapping['right_eye']], axis=0))
    return distance_normalised_error(euclidean_error, pupil_dist, shape,
                                     gt_shape)


def mean_pupil_49_error(shape, gt_shape):
    r"""
    Computes the euclidean error based on 49 points normalised with the
    distance between the mean eye points (pupils), i.e.

    .. math::
       \frac{\mathcal{F}(s,s^*)}{\mathcal{N}(s)}

    where

    .. math::
       \mathcal{F}(s,s^*) = \frac{1}{49}\sum_{i=1}^{49}\sqrt{(s_{i,x}-s^*_{i,x})^2 + (s_{i,y}-s^*_{i,y})^2}

    where :math:`s` and :math:`s^*` are the final and ground truth shapes,
    respectively. :math:`(s_{i,x}, s_{i,y})` are the `x` and `y` coordinates of
    the :math:`i`'th point of the final shape, :math:`(s^*_{i,x}, s^*_{i,y})`
    are the `x` and `y` coordinates of the :math:`i`'th point of the ground
    truth shape. Finally, :math:`\mathcal{N}(s)` is the distance between the
    mean eye points (pupils).

    Parameters
    ----------
    shape : `menpo.shape.PointCloud`
        The input shape (e.g. the final shape of a fitting procedure). It
        must have either 68 or 66 or 51 or 49 points.
    gt_shape : `menpo.shape.PointCloud`
        The ground truth shape. It must have either 68 or 66 or 51 or 49 points.

    Returns
    -------
    normalised_error : `float`
        The computed normalised Euclidean error.

    Raises
    ------
    ValueError
        Final shape must have 68 or 66 or 51 or 49 points
    ValueError
        Ground truth shape must have 68 or 66 or 51 or 49 points
    """
    if shape.n_points not in [68, 66, 51, 49]:
        raise ValueError('Final shape must have 68 or 66 or 51 or 49 points')
    if gt_shape.n_points not in [68, 66, 51, 49]:
        raise ValueError('Ground truth shape must have 68 or 66 or 51 or 49 '
                         'points')

    def pupil_dist(_, s):
        _, mapping = face_ibug_49_to_face_ibug_49(s, include_mapping=True)
        return euclidean_error(np.mean(s[mapping['left_eye']], axis=0),
                               np.mean(s[mapping['right_eye']], axis=0))
    if shape.n_points == 68:
        shape = _convert_68_to_49(shape)
    elif shape.n_points == 66:
        shape = _convert_66_to_49(shape)
    elif shape.n_points == 51:
        shape = _convert_51_to_49(shape)
    if gt_shape.n_points == 68:
        gt_shape = _convert_68_to_49(gt_shape)
    elif gt_shape.n_points == 66:
        gt_shape = _convert_66_to_49(gt_shape)
    elif gt_shape.n_points == 51:
        gt_shape = _convert_51_to_49(gt_shape)
    return distance_normalised_error(euclidean_error, pupil_dist, shape,
                                     gt_shape)


def outer_eye_corner_68_euclidean_error(shape, gt_shape):
    r"""
    Computes the Euclidean error based on 68 points normalised with the
    distance between the mean eye points (pupils), i.e.

    .. math::
       \frac{\mathcal{F}(s,s^*)}{\mathcal{N}(s^*)}

    where

    .. math::
       \mathcal{F}(s,s^*) = \frac{1}{68}\sum_{i=1}^{68}\sqrt{(s_{i,x}-s^*_{i,x})^2 + (s_{i,y}-s^*_{i,y})^2}

    where :math:`s` and :math:`s^*` are the final and ground truth shapes,
    respectively. :math:`(s_{i,x}, s_{i,y})` are the `x` and `y` coordinates of
    the :math:`i`'th point of the final shape, :math:`(s^*_{i,x}, s^*_{i,y})`
    are the `x` and `y` coordinates of the :math:`i`'th point of the ground
    truth shape. Finally, :math:`\mathcal{N}(s^*)` is the distance between the
    ``36``-th and ``45``-th points.

    Parameters
    ----------
    shape : `menpo.shape.PointCloud`
        The input shape (e.g. the final shape of a fitting procedure). It
        must have 68 points.
    gt_shape : `menpo.shape.PointCloud`
        The ground truth shape. It must have 68 points.

    Returns
    -------
    normalised_error : `float`
        The computed normalised Euclidean error.

    Raises
    ------
    ValueError
        Final shape must have 68 points
    ValueError
        Ground truth shape must have 68 points
    """
    if shape.n_points != 68:
        raise ValueError('Final shape must have 68 points')
    if gt_shape.n_points != 68:
        raise ValueError('Ground truth shape must have 68 points')
    return distance_indexed_normalised_error(euclidean_error, 36, 45, shape,
                                             gt_shape)


def outer_eye_corner_51_euclidean_error(shape, gt_shape):
    r"""
    Computes the Euclidean error based on 51 points normalised with the
    distance between the mean eye points (pupils), i.e.

    .. math::
       \frac{\mathcal{F}(s,s^*)}{\mathcal{N}(s^*)}

    where

    .. math::
       \mathcal{F}(s,s^*) = \frac{1}{51}\sum_{i=1}^{51}\sqrt{(s_{i,x}-s^*_{i,x})^2 + (s_{i,y}-s^*_{i,y})^2}

    where :math:`s` and :math:`s^*` are the final and ground truth shapes,
    respectively. :math:`(s_{i,x}, s_{i,y})` are the `x` and `y` coordinates of
    the :math:`i`'th point of the final shape, :math:`(s^*_{i,x}, s^*_{i,y})`
    are the `x` and `y` coordinates of the :math:`i`'th point of the ground
    truth shape. Finally, :math:`\mathcal{N}(s^*)` is the distance between the
    ``19``-th and ``28``-th points.

    Parameters
    ----------
    shape : `menpo.shape.PointCloud`
        The input shape (e.g. the final shape of a fitting procedure). It
        must 68 or 51 points.
    gt_shape : `menpo.shape.PointCloud`
        The ground truth shape. It must have 68 or 51 points.

    Returns
    -------
    normalised_error : `float`
        The computed normalised Euclidean error.

    Raises
    ------
    ValueError
        Final shape must have 68 or 51 points
    ValueError
        Ground truth shape must have 68 or 51 points
    """
    if shape.n_points not in [68, 51]:
        raise ValueError('Final shape must have 68 or 51 points')
    if gt_shape.n_points not in [68, 51]:
        raise ValueError('Ground truth shape must have 68 or 51 points')
    if shape.n_points == 68:
        shape = _convert_68_to_51(shape)
    if gt_shape.n_points == 68:
        gt_shape = _convert_68_to_51(gt_shape)
    return distance_indexed_normalised_error(euclidean_error, 19, 28, shape,
                                             gt_shape)


def outer_eye_corner_49_euclidean_error(shape, gt_shape):
    r"""
    Computes the Euclidean error based on 49 points normalised with the
    distance between the mean eye points (pupils), i.e.

    .. math::
       \frac{\mathcal{F}(s,s^*)}{\mathcal{N}(s^*)}

    where

    .. math::
       \mathcal{F}(s,s^*) = \frac{1}{49}\sum_{i=1}^{49}\sqrt{(s_{i,x}-s^*_{i,x})^2 + (s_{i,y}-s^*_{i,y})^2}

    where :math:`s` and :math:`s^*` are the final and ground truth shapes,
    respectively. :math:`(s_{i,x}, s_{i,y})` are the `x` and `y` coordinates of
    the :math:`i`'th point of the final shape, :math:`(s^*_{i,x}, s^*_{i,y})`
    are the `x` and `y` coordinates of the :math:`i`'th point of the ground
    truth shape. Finally, :math:`\mathcal{N}(s^*)` is the distance between the
    ``19``-th and ``28``-th points.

    Parameters
    ----------
    shape : `menpo.shape.PointCloud`
        The input shape (e.g. the final shape of a fitting procedure). It
        must 68 or 66 or 51 or 49 points.
    gt_shape : `menpo.shape.PointCloud`
        The ground truth shape. It must have 68 or 66 or 51 or 49 points.

    Returns
    -------
    normalised_error : `float`
        The computed normalised Euclidean error.

    Raises
    ------
    ValueError
        Final shape must have 68 or 66 or 51 or 49 points
    ValueError
        Ground truth shape must have 68 or 66 or 51 or 49 points
    """
    if shape.n_points not in [68, 66, 51, 49]:
        raise ValueError('Final shape must have 68 or 66 or 51 or 49 points')
    if gt_shape.n_points not in [68, 66, 51, 49]:
        raise ValueError('Ground truth shape must have 68 or 66 or 51 or 49 '
                         'points')
    if shape.n_points == 68:
        shape = _convert_68_to_49(shape)
    elif shape.n_points == 66:
        shape = _convert_66_to_49(shape)
    elif shape.n_points == 51:
        shape = _convert_51_to_49(shape)
    if gt_shape.n_points == 68:
        gt_shape = _convert_68_to_49(gt_shape)
    elif gt_shape.n_points == 66:
        gt_shape = _convert_66_to_49(gt_shape)
    elif gt_shape.n_points == 51:
        gt_shape = _convert_51_to_49(gt_shape)
    return distance_indexed_normalised_error(euclidean_error, 19, 28, shape,
                                             gt_shape)


def bb_avg_edge_length_68_euclidean_error(shape, gt_shape):
    r"""
    Computes the Euclidean error based on 68 points normalised by the average
    edge length of the 68-point ground truth shape's bounding box, i.e.

    .. math::
       \frac{\mathcal{F}(s,s^*)}{\mathcal{N}(s^*)}

    where

    .. math::
       \mathcal{F}(s,s^*) = \frac{1}{68}\sum_{i=1}^{68}\sqrt{(s_{i,x}-s^*_{i,x})^2 + (s_{i,y}-s^*_{i,y})^2}

    where :math:`s` and :math:`s^*` are the final and ground truth shapes,
    respectively. :math:`(s_{i,x}, s_{i,y})` are the `x` and `y` coordinates of
    the :math:`i`'th point of the final shape, :math:`(s^*_{i,x}, s^*_{i,y})`
    are the `x` and `y` coordinates of the :math:`i`'th point of the ground
    truth shape. Finally, :math:`\mathcal{N}(s^*)` is a normalising function
    that returns the average edge length of the bounding box of the 68-point
    ground truth shape (:map:`bb_avg_edge_length`).

    Parameters
    ----------
    shape : `menpo.shape.PointCloud`
        The input shape (e.g. the final shape of a fitting procedure). It
        must have 68 points.
    gt_shape : `menpo.shape.PointCloud`
        The ground truth shape. It must have 68 points.

    Returns
    -------
    normalised_error : `float`
        The computed Euclidean normalised error.

    Raises
    ------
    ValueError
        Final shape must have 68 points
    ValueError
        Ground truth shape must have 68 points
    """
    if shape.n_points != 68:
        raise ValueError('Final shape must have 68 points')
    if gt_shape.n_points != 68:
        raise ValueError('Ground truth shape must have 68 points')
    return bb_normalised_error(euclidean_error, shape, gt_shape,
                               norm_type='avg_edge_length', norm_shape=gt_shape)


def bb_avg_edge_length_49_euclidean_error(shape, gt_shape):
    r"""
    Computes the Euclidean error based on 49 points normalised by the average
    edge length of the 68-point ground truth shape's bounding box, i.e.

    .. math::
       \frac{\mathcal{F}(s,s^*)}{\mathcal{N}(s^*)}

    where

    .. math::
       \mathcal{F}(s,s^*) = \frac{1}{49}\sum_{i=1}^{49}\sqrt{(s_{i,x}-s^*_{i,x})^2 + (s_{i,y}-s^*_{i,y})^2}

    where :math:`s` and :math:`s^*` are the final and ground truth shapes,
    respectively. :math:`(s_{i,x}, s_{i,y})` are the `x` and `y` coordinates of
    the :math:`i`'th point of the final shape, :math:`(s^*_{i,x}, s^*_{i,y})`
    are the `x` and `y` coordinates of the :math:`i`'th point of the ground
    truth shape. Finally, :math:`\mathcal{N}(s^*)` is a normalising function
    that returns the average edge length of the bounding box of the 68-point
    ground truth shape (:map:`bb_avg_edge_length`).

    Parameters
    ----------
    shape : `menpo.shape.PointCloud`
        The input shape (e.g. the final shape of a fitting procedure). It
        must have 68 or 66 or 51 or 49 points.
    gt_shape : `menpo.shape.PointCloud`
        The ground truth shape. It must have 68 points.

    Returns
    -------
    normalised_error : `float`
        The computed Euclidean normalised error.

    Raises
    ------
    ValueError
        Final shape must have 68 or 51 or 49 points
    ValueError
        Ground truth shape must have 68 points
    """
    if shape.n_points not in [68, 66, 51, 49]:
        raise ValueError('Final shape must have 68 or 66 or 51 or 49 points')
    if gt_shape.n_points != 68:
        raise ValueError('Ground truth shape must have 68 points')
    if shape.n_points == 68:
        shape = _convert_68_to_49(shape)
    elif shape.n_points == 66:
        shape = _convert_66_to_49(shape)
    elif shape.n_points == 51:
        shape = _convert_51_to_49(shape)
    gt_shape_68 = gt_shape.copy()
    gt_shape = _convert_68_to_49(gt_shape)
    return bb_normalised_error(euclidean_error, shape, gt_shape,
                               norm_type='avg_edge_length',
                               norm_shape=gt_shape_68)
