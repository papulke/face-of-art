from functools import wraps, partial
import numpy as np

from menpo.shape import PointCloud


def pointcloud_to_points(wrapped):
    @wraps(wrapped)
    def wrapper(*args, **kwargs):
        args = list(args)
        for index, arg in enumerate(args):
            if isinstance(arg, PointCloud):
                args[index] = arg.points
        for key in kwargs:
            if isinstance(kwargs[key], PointCloud):
                kwargs[key] = kwargs[key].points
        return wrapped(*args, **kwargs)
    return wrapper


# BOUNDING BOX NORMALISERS
def bb_area(shape):
    r"""
    Computes the area of the bounding box of the provided shape,
    i.e.

    .. math::
       h w

    where :math:`h` and :math:`w` are the height and width of the bounding box.

    Parameters
    ----------
    shape : `menpo.shape.PointCloud` or `subclass`
        The input shape.

    Returns
    -------
    bb_area : `float`
        The area of the bounding box.
    """
    # Area = w * h
    height, width = np.max(shape, axis=0) - np.min(shape, axis=0)
    return height * width


def bb_perimeter(shape):
    r"""
    Computes the perimeter of the bounding box of the provided shape, i.e.

    .. math::
       2(h + w)

    where :math:`h` and :math:`w` are the height and width of the bounding box.

    Parameters
    ----------
    shape : `menpo.shape.PointCloud` or `subclass`
        The input shape.

    Returns
    -------
    bb_perimeter : `float`
        The perimeter of the bounding box.
    """
    # Area = 2(w + h)
    height, width = np.max(shape, axis=0) - np.min(shape, axis=0)
    return 2 * (height + width)


def bb_avg_edge_length(shape):
    r"""
    Computes the average edge length of the bounding box of the provided shape,
    i.e.

    .. math::
       \frac{h + w}{2} = \frac{2h + 2w}{4}

    where :math:`h` and :math:`w` are the height and width of the bounding box.

    Parameters
    ----------
    shape : `menpo.shape.PointCloud` or `subclass`
        The input shape.

    Returns
    -------
    bb_avg_edge_length : `float`
        The average edge length of the bounding box.
    """
    # 0.5(w + h) = (2w + 2h) / 4
    height, width = np.max(shape, axis=0) - np.min(shape, axis=0)
    return 0.5 * (height + width)


def bb_diagonal(shape):
    r"""
    Computes the diagonal of the bounding box of the provided shape, i.e.

    .. math::
       \sqrt{h^2 + w^2}

    where :math:`h` and :math:`w` are the height and width of the bounding box.

    Parameters
    ----------
    shape : `menpo.shape.PointCloud` or `subclass`
        The input shape.

    Returns
    -------
    bb_diagonal : `float`
        The diagonal of the bounding box.
    """
    # sqrt(w**2 + h**2)
    height, width = np.max(shape, axis=0) - np.min(shape, axis=0)
    return np.sqrt(width ** 2 + height ** 2)


def bb_sqrt_edge_length(shape, gt_shape):
    r"""
    Computes the average edge length of the bounding box of the provided shape,
    i.e.

    .. math::
       \frac{h + w}{2} = \frac{2h + 2w}{4}

    where :math:`h` and :math:`w` are the height and width of the bounding box.

    Parameters
    ----------
    shape : `menpo.shape.PointCloud` or `subclass`
        The input shape.

    Returns
    -------
    bb_sqrt_edge_length : `float`
        The average edge length of the bounding box.
    """
    # 0.5(w + h) = (2w + 2h) / 4
    height, width = np.max(gt_shape, axis=0) - np.min(gt_shape, axis=0)
    return np.sqrt(height*width)


# innerPupil normalize
def inner_pupil(shape,gt_shape):
    # innerPupil normalize
    rPupil = (gt_shape[37]+gt_shape[38]+gt_shape[40]+gt_shape[41])/4
    lPupil = (gt_shape[43]+gt_shape[44]+gt_shape[46]+gt_shape[47])/4
    innerPupil_dis = euclidean_error(rPupil, lPupil)
    return innerPupil_dis


bb_norm_types = {
    'avg_edge_length': bb_avg_edge_length,
    'perimeter': bb_perimeter,
    'diagonal': bb_diagonal,
    'area': bb_area
}


# EUCLIDEAN AND ROOT MEAN SQUARE ERRORS
@pointcloud_to_points
def root_mean_square_error(shape, gt_shape):
    r"""
    Computes the root mean square error between two shapes, i.e.

    .. math::
       \sqrt{\frac{1}{N}\sum_{i=1}^N(s_i-s^*_i)^2}

    where :math:`s_i` and :math:`s^*_i` are the coordinates of the :math:`i`'th
    point of the final and ground truth shapes, and :math:`N` is the total
    number of points.

    Parameters
    ----------
    shape : `menpo.shape.PointCloud`
        The input shape (e.g. the final shape of a fitting procedure).
    gt_shape : `menpo.shape.PointCloud`
        The ground truth shape.

    Returns
    -------
    root_mean_square_error : `float`
        The root mean square error.
    """
    return np.sqrt(np.mean((shape.ravel() - gt_shape.ravel()) ** 2))


@pointcloud_to_points
def euclidean_error(shape, gt_shape):
    r"""
    Computes the Euclidean error between two shapes, i.e.

    .. math::
       \frac{1}{N}\sum_{i=1}^N\sqrt{(s_{i,x}-s^*_{i,x})^2 + (s_{i,y}-s^*_{i,y})^2}

    where :math:`(s_{i,x}, s_{i,y})` are the `x` and `y` coordinates of the
    :math:`i`'th point of the final shape, :math:`(s^*_{i,x}, s^*_{i,y})`
    are the `x` and `y` coordinates of the :math:`i`'th point of the ground
    truth shape and :math:`N` is the total number of points.

    Parameters
    ----------
    shape : `menpo.shape.PointCloud`
        The input shape (e.g. the final shape of a fitting procedure).
    gt_shape : `menpo.shape.PointCloud`
        The ground truth shape.

    Returns
    -------
    root_mean_square_error : `float`
        The Euclidean error.
    """
    return np.mean(np.sqrt(np.sum((shape - gt_shape) ** 2, axis=-1)))


# DISTANCE NORMALISER
def distance_two_indices(index1, index2, shape):
    r"""
    Computes the Euclidean distance between two points of a shape, i.e.

    .. math::
       \sqrt{(s_{i,x}-s_{j,x})^2 + (s_{i,y}-s_{j,y})^2}

    where :math:`s_{i,x}`, :math:`s_{i,y}` are the `x` and `y` coordinates of
    the :math:`i`'th point (`index1`) and :math:`s_{j,x}`, :math:`s_{j,y}` are
    the `x` and `y` coordinates of the :math:`j`'th point (`index2`).

    Parameters
    ----------
    index1 : `int`
        The index of the first point.
    index2 : `int`
        The index of the second point.
    shape : `menpo.shape.PointCloud`
        The input shape.

    Returns
    -------
    distance_two_indices : `float`
        The Euclidean distance between the points.
    """
    return euclidean_error(shape[index1], shape[index2])


# GENERIC NORMALISED ERROR FUNCTIONS
@pointcloud_to_points
def bb_normalised_error(shape_error_f, shape, gt_shape,
                        norm_shape=None, norm_type='avg_edge_length'):
    r"""
    Computes an error normalised by a measure based on the shape's bounding
    box, i.e.

    .. math::
       \frac{\mathcal{F}(s,s^*)}{\mathcal{N}(s^*)}

    where :math:`\mathcal{F}(s,s^*)` is an error metric function between the
    final shape :math:`s` and the ground truth shape :math:`s^*` and
    :math:`\mathcal{N}(s^*)` is a normalising function that returns a measure
    based on the ground truth shape's bounding box.

    Parameters
    ----------
    shape_error_f : `callable`
        The function to be used for computing the error.
    shape : `menpo.shape.PointCloud`
        The input shape (e.g. the final shape of a fitting procedure).
    gt_shape : `menpo.shape.PointCloud`
        The ground truth shape.
    norm_shape : `menpo.shape.PointCloud` or ``None``, optional
        The shape to be used to compute the normaliser. If ``None``, then the
        ground truth shape is used.
    norm_type : ``{'area', 'perimeter', 'avg_edge_length', 'diagonal'}``, optional
        The type of the normaliser. Possible options are:

        ========================= ==========================================
        Method                    Description
        ========================= ==========================================
        :map:`bb_area`            Area of `norm_shape`'s bounding box
        :map:`bb_perimeter`       Perimeter of `norm_shape`'s bounding box
        :map:`bb_avg_edge_length` Average edge length of `norm_shape`'s bbox
        :map:`bb_diagonal`        Diagonal of `norm_shape`'s bounding box
        ========================= ==========================================

    Returns
    -------
    normalised_error : `float`
        The computed normalised error.
    """
    if norm_type not in bb_norm_types:
        raise ValueError('norm_type must be one of '
                         '{avg_edge_length, perimeter, diagonal, area}.')
    if norm_shape is None:
        norm_shape = gt_shape
    return (shape_error_f(shape, gt_shape) /
            bb_norm_types[norm_type](norm_shape))


@pointcloud_to_points
def distance_normalised_error(shape_error_f, distance_norm_f, shape, gt_shape):
    r"""
    Computes an error normalised by a distance measure between two shapes, i.e.

    .. math::
       \frac{\mathcal{F}(s,s^*)}{\mathcal{N}(s,s^*)}

    where :math:`\mathcal{F}(s,s^*)` is an error metric function between the
    final shape :math:`s` and the ground truth shape :math:`s^*` and
    :math:`\mathcal{N}(s,s^*)` is a normalising function based on a distance
    metric between the two shapes.

    Parameters
    ----------
    shape_error_f : `callable`
        The function to be used for computing the error.
    distance_norm_f : `callable`
        The function to be used for computing the normalisation distance metric.
    shape : `menpo.shape.PointCloud`
        The input shape (e.g. the final shape of a fitting procedure).
    gt_shape : `menpo.shape.PointCloud`
        The ground truth shape.

    Returns
    -------
    normalised_error : `float`
        The computed normalised error.
    """
    return shape_error_f(shape, gt_shape) / distance_norm_f(shape, gt_shape)


@pointcloud_to_points
def distance_indexed_normalised_error(shape_error_f, index1, index2, shape,
                                      gt_shape):
    r"""
    Computes an error normalised by the distance measure between two points
    of the ground truth shape, i.e.

    .. math::
       \frac{\mathcal{F}(s,s^*)}{\mathcal{N}(s^*)}

    where :math:`\mathcal{F}(s,s^*)` is an error metric function between the
    final shape :math:`s` and the ground truth shape :math:`s^*` and
    :math:`\mathcal{N}(s^*)` is a normalising function that returns the
    distance between two points of the ground truth shape.

    Parameters
    ----------
    shape_error_f : `callable`
        The function to be used for computing the error.
    index1 : `int`
        The index of the first point.
    index2 : `int`
        The index of the second point.
    shape : `menpo.shape.PointCloud`
        The input shape (e.g. the final shape of a fitting procedure).
    gt_shape : `menpo.shape.PointCloud`
        The ground truth shape.

    Returns
    -------
    normalised_error : `float`
        The computed normalised error.
    """
    '''
    # innerPupil normalize
    rPupil = (gt_shape[37]+gt_shape[38]+gt_shape[40]+gt_shape[41])/4
    lPupil = (gt_shape[43]+gt_shape[44]+gt_shape[46]+gt_shape[47])/4
    innerPupil_dis = euclidean_error(rPupil, lPupil)
    return shape_error_f(shape, gt_shape) / innerPupil_dis
    '''
    return shape_error_f(shape, gt_shape) / distance_two_indices(index1, index2,
                                                             gt_shape)



# EUCLIDEAN AND ROOT MEAN SQUARE NORMALISED ERRORS
def root_mean_square_bb_normalised_error(shape, gt_shape, norm_shape=None,
                                         norm_type='avg_edge_length'):
    r"""
    Computes the root mean square error between two shapes normalised by a
    measure based on the ground truth shape's bounding box, i.e.

    .. math::
       \frac{\mathcal{F}(s,s^*)}{\mathcal{N}(s^*)}

    where

    .. math::
       \mathcal{F}(s,s^*) = \sqrt{\frac{1}{N}\sum_{i=1}^N(s_i-s^*_i)^2}

    where :math:`s` and :math:`s^*` are the final and ground truth shapes,
    respectively. :math:`s_i` and :math:`s^*_i` are the coordinates of the
    :math:`i`'th point of the final and ground truth shapes, and :math:`N` is
    the total number of points. Finally, :math:`\mathcal{N}(s^*)` is a
    normalising function that returns a measure based on the ground truth
    shape's bounding box.

    Parameters
    ----------
    shape : `menpo.shape.PointCloud`
        The input shape (e.g. the final shape of a fitting procedure).
    gt_shape : `menpo.shape.PointCloud`
        The ground truth shape.
    norm_shape : `menpo.shape.PointCloud` or ``None``, optional
        The shape to be used to compute the normaliser. If ``None``, then the
        ground truth shape is used.
    norm_type : ``{'area', 'perimeter', 'avg_edge_length', 'diagonal'}``, optional
        The type of the normaliser. Possible options are:

        ========================= ==========================================
        Method                    Description
        ========================= ==========================================
        :map:`bb_area`            Area of `norm_shape`'s bounding box
        :map:`bb_perimeter`       Perimeter of `norm_shape`'s bounding box
        :map:`bb_avg_edge_length` Average edge length of `norm_shape`'s bbox
        :map:`bb_diagonal`        Diagonal of `norm_shape`'s bounding box
        ========================= ==========================================

    Returns
    -------
    error : `float`
        The computed root mean square normalised error.
    """
    return bb_normalised_error(shape_error_f=root_mean_square_error,
                               shape=shape, gt_shape=gt_shape,
                               norm_shape=norm_shape, norm_type=norm_type)


def root_mean_square_distance_normalised_error(shape, gt_shape,
                                               distance_norm_f):
    r"""
    Computes the root mean square error between two shapes normalised by a
    distance measure between two shapes, i.e.

    .. math::
       \frac{\mathcal{F}(s,s^*)}{\mathcal{N}(s,s^*)}

    where

    .. math::
       \mathcal{F}(s,s^*) = \sqrt{\frac{1}{N}\sum_{i=1}^N(s_i-s^*_i)^2}

    where :math:`s` and :math:`s^*` are the final and ground truth shapes,
    respectively. :math:`s_i` and :math:`s^*_i` are the coordinates of the
    :math:`i`'th point of the final and ground truth shapes, and :math:`N` is
    the total number of points. Finally, :math:`\mathcal{N}(s,s^*)` is a
    normalising function based on a distance metric between the two shapes.

    Parameters
    ----------
    shape : `menpo.shape.PointCloud`
        The input shape (e.g. the final shape of a fitting procedure).
    gt_shape : `menpo.shape.PointCloud`
        The ground truth shape.
    distance_norm_f : `callable`
        The function to be used for computing the normalisation distance metric.

    Returns
    -------
    error : `float`
        The computed root mean square normalised error.
    """
    return distance_normalised_error(shape_error_f=root_mean_square_error,
                                     distance_norm_f=distance_norm_f,
                                     shape=shape, gt_shape=gt_shape)


def root_mean_square_distance_indexed_normalised_error(shape, gt_shape,
                                                       index1, index2):
    r"""
    Computes the root mean square error between two shapes normalised by the
    distance measure between two points of the ground truth shape, i.e.

    .. math::
       \frac{\mathcal{F}(s,s^*)}{\mathcal{N}(s^*)}

    where

    .. math::
       \mathcal{F}(s,s^*) = \sqrt{\frac{1}{N}\sum_{i=1}^N(s_i-s^*_i)^2}

    where :math:`s` and :math:`s^*` are the final and ground truth shapes,
    respectively. :math:`s_i` and :math:`s^*_i` are the coordinates of the
    :math:`i`'th point of the final and ground truth shapes, and :math:`N` is
    the total number of points. Finally, :math:`\mathcal{N}(s^*)` is a
    normalising function that returns the distance between two points of the
    ground truth shape.

    Parameters
    ----------
    shape : `menpo.shape.PointCloud`
        The input shape (e.g. the final shape of a fitting procedure).
    gt_shape : `menpo.shape.PointCloud`
        The ground truth shape.
    index1 : `int`
        The index of the first point.
    index2 : `int`
        The index of the second point.

    Returns
    -------
    error : `float`
        The computed root mean square normalised error.
    """
    return distance_indexed_normalised_error(
            shape_error_f=root_mean_square_error, index1=index1, index2=index2,
            shape=shape, gt_shape=gt_shape)


def euclidean_bb_normalised_error(shape, gt_shape, norm_shape=None,
                                  norm_type='avg_edge_length'):
    r"""
    Computes the Euclidean error between two shapes normalised by a measure
    based on the ground truth shape's bounding box, i.e.

    .. math::
       \frac{\mathcal{F}(s,s^*)}{\mathcal{N}(s^*)}

    where

    .. math::
       \mathcal{F}(s,s^*) = \frac{1}{N}\sum_{i=1}^N\sqrt{(s_{i,x}-s^*_{i,x})^2 + (s_{i,y}-s^*_{i,y})^2}

    where :math:`s` and :math:`s^*` are the final and ground truth shapes,
    respectively. :math:`(s_{i,x}, s_{i,y})` are the `x` and `y` coordinates of
    the :math:`i`'th point of the final shape, :math:`(s^*_{i,x}, s^*_{i,y})`
    are the `x` and `y` coordinates of the :math:`i`'th point of the ground
    truth shape and :math:`N` is the total number of points. Finally,
    :math:`\mathcal{N}(s^*)` is a normalising function that returns a measure
    based on the ground truth shape's bounding box.

    Parameters
    ----------
    shape : `menpo.shape.PointCloud`
        The input shape (e.g. the final shape of a fitting procedure).
    gt_shape : `menpo.shape.PointCloud`
        The ground truth shape.
    norm_shape : `menpo.shape.PointCloud` or ``None``, optional
        The shape to be used to compute the normaliser. If ``None``, then the
        ground truth shape is used.
    norm_type : ``{'area', 'perimeter', 'avg_edge_length', 'diagonal'}``, optional
        The type of the normaliser. Possible options are:

        ========================= ==========================================
        Method                    Description
        ========================= ==========================================
        :map:`bb_area`            Area of `norm_shape`'s bounding box
        :map:`bb_perimeter`       Perimeter of `norm_shape`'s bounding box
        :map:`bb_avg_edge_length` Average edge length of `norm_shape`'s bbox
        :map:`bb_diagonal`        Diagonal of `norm_shape`'s bounding box
        ========================= ==========================================

    Returns
    -------
    error : `float`
        The computed Euclidean normalised error.
    """
    return bb_normalised_error(shape_error_f=euclidean_error,
                               shape=shape, gt_shape=gt_shape,
                               norm_shape=norm_shape, norm_type=norm_type)


def euclidean_distance_normalised_error(shape, gt_shape, distance_norm_f):
    r"""
    Computes the Euclidean error between two shapes normalised by a distance
    measure between two shapes, i.e.

    .. math::
       \frac{\mathcal{F}(s,s^*)}{\mathcal{N}(s,s^*)}

    where

    .. math::
       \mathcal{F}(s,s^*) = \frac{1}{N}\sum_{i=1}^N\sqrt{(s_{i,x}-s^*_{i,x})^2 + (s_{i,y}-s^*_{i,y})^2}

    where :math:`s` and :math:`s^*` are the final and ground truth shapes,
    respectively. :math:`(s_{i,x}, s_{i,y})` are the `x` and `y` coordinates of
    the :math:`i`'th point of the final shape, :math:`(s^*_{i,x}, s^*_{i,y})`
    are the `x` and `y` coordinates of the :math:`i`'th point of the ground
    truth shape and :math:`N` is the total number of points. Finally,
    :math:`\mathcal{N}(s,s^*)` is a normalising function based on a distance
    metric between the two shapes.

    Parameters
    ----------
    shape : `menpo.shape.PointCloud`
        The input shape (e.g. the final shape of a fitting procedure).
    gt_shape : `menpo.shape.PointCloud`
        The ground truth shape.
    distance_norm_f : `callable`
        The function to be used for computing the normalisation distance metric.

    Returns
    -------
    error : `float`
        The computed Euclidean normalised error.
    """
    return distance_normalised_error(shape_error_f=euclidean_error,
                                     distance_norm_f=distance_norm_f,
                                     shape=shape, gt_shape=gt_shape)


def euclidean_distance_indexed_normalised_error(shape, gt_shape, index1,
                                                index2):
    r"""
    Computes the Euclidean error between two shapes normalised by the
    distance measure between two points of the ground truth shape, i.e.

    .. math::
       \frac{\mathcal{F}(s,s^*)}{\mathcal{N}(s^*)}

    where

    .. math::
       \mathcal{F}(s,s^*) = \frac{1}{N}\sum_{i=1}^N\sqrt{(s_{i,x}-s^*_{i,x})^2 + (s_{i,y}-s^*_{i,y})^2}

    where :math:`s` and :math:`s^*` are the final and ground truth shapes,
    respectively. :math:`(s_{i,x}, s_{i,y})` are the `x` and `y` coordinates of
    the :math:`i`'th point of the final shape, :math:`(s^*_{i,x}, s^*_{i,y})`
    are the `x` and `y` coordinates of the :math:`i`'th point of the ground
    truth shape and :math:`N` is the total number of points. Finally,
    :math:`\mathcal{N}(s^*)` is a normalising function that returns the
    distance between two points of the ground truth shape.

    Parameters
    ----------
    shape : `menpo.shape.PointCloud`
        The input shape (e.g. the final shape of a fitting procedure).
    gt_shape : `menpo.shape.PointCloud`
        The ground truth shape.
    index1 : `int`
        The index of the first point.
    index2 : `int`
        The index of the second point.

    Returns
    -------
    error : `float`
        The computed Euclidean normalised error.
    """
    return distance_indexed_normalised_error(
            shape_error_f=euclidean_error, index1=index1, index2=index2,
            shape=shape, gt_shape=gt_shape)
