import warnings
import collections
from functools import partial
import numpy as np

from menpo.base import name_of_callable
from menpo.shape import TriMesh
from menpo.transform import PiecewiseAffine


def check_diagonal(diagonal):
    r"""
    Checks that the diagonal length used to normalize the images' size is
    ``>= 20``.

    Parameters
    ----------
    diagonal : `int`
        The value to check.

    Returns
    -------
    diagonal : `int`
        The value if it's correct.

    Raises
    ------
    ValueError
        diagonal must be >= 20 or None
    """
    if diagonal is not None and diagonal < 20:
        raise ValueError("diagonal must be >= 20 or None")
    return diagonal


def check_landmark_trilist(image, transform, group=None):
    r"""
    Checks that the provided image has a triangulated shape (thus an isntance of
    `menpo.shape.TriMesh`) and the transform is `menpo.transform.PiecewiseAffine`

    Parameters
    ----------
    image : `menpo.image.Image` or subclass
        The input image.
    transform : `menpo.transform.PiecewiseAffine`
        The transform object.
    group : `str` or ``None``, optional
        The group of the shape to check.

    Raises
    ------
    Warning
        The given images do not have an explicit triangulation applied. A
        Delaunay Triangulation will be computed and used for warping. This may
        be suboptimal and cause warping artifacts.
    """
    shape = image.landmarks[group].lms
    check_trilist(shape, transform)


def check_trilist(shape, transform):
    r"""
    Checks that the provided shape is triangulated (thus an isntance of
    `menpo.shape.TriMesh`) and the transform is `menpo.transform.PiecewiseAffine`

    Parameters
    ----------
    shape : `menpo.shape.TriMesh`
        The input shape (usually the reference/mean shape of a model).
    transform : `menpo.transform.PiecewiseAffine`
        The transform object.

    Raises
    ------
    Warning
        The given images do not have an explicit triangulation applied. A
        Delaunay Triangulation will be computed and used for warping. This may
        be suboptimal and cause warping artifacts.
    """
    if not isinstance(shape, TriMesh) and isinstance(transform,
                                                     PiecewiseAffine):
        warnings.warn('The given images do not have an explicit triangulation '
                      'applied. A Delaunay Triangulation will be computed '
                      'and used for warping. This may be suboptimal and cause '
                      'warping artifacts.')


def check_scales(scales):
    r"""
    Checks that the provided `scales` argument is either `int` or `float` or an
    iterable of those. It makes sure that it returns a `list` of `scales`.

    Parameters
    ----------
    scales : `int` or `float` or `list/tuple` of those
        The value to check.

    Returns
    -------
    scales : `list` of `int` or `float`
        The scales in a list.

    Raises
    ------
    ValueError
        scales must be an int/float or a list/tuple of int/float
    """
    if isinstance(scales, (int, float)):
        return [scales]
    elif len(scales) == 1 and isinstance(scales[0], (int, float)):
        return list(scales)
    elif len(scales) > 1:
        return check_scales(scales[0]) + check_scales(scales[1:])
    else:
        raise ValueError("scales must be an int/float or a list/tuple of "
                         "int/float")


def check_multi_scale_param(n_scales, types, param_name, param):
    r"""
    General function for checking a parameter defined for multiple scales. It
    raises an error if the parameter is not an iterable with the correct size and
    correct types.

    Parameters
    ----------
    n_scales : `int`
        The number of scales.
    types : `tuple`
        The `tuple` of variable types that the parameter is allowed to have.
    param_name : `str`
        The name of the parameter.
    param : `types`
        The parameter value.

    Returns
    -------
    param : `list` of `types`
        The list of values per scale.

    Raises
    ------
    ValueError
        {param_name} must be in {types} or a list/tuple of {types} with the same
        length as the number of scales
    """
    error_msg = "{0} must be in {1} or a list/tuple of " \
                "{1} with the same length as the number " \
                "of scales".format(param_name, types)

    # Could be a single value - or we have an error
    if isinstance(param, types):
        return [param] * n_scales
    elif not isinstance(param, collections.Iterable):
        raise ValueError(error_msg)

    # Must be an iterable object
    len_param = len(param)
    isinstance_all_in_param = all(isinstance(p, types) for p in param)

    if len_param == 1 and isinstance_all_in_param:
        return list(param) * n_scales
    elif len_param == n_scales and isinstance_all_in_param:
        return list(param)
    else:
        raise ValueError(error_msg)


def check_callable(callables, n_scales):
    r"""
    Checks the callable type per level.

    Parameters
    ----------
    callables : `callable` or `list` of `callables`
        The callable to be used per scale.
    n_scales : `int`
        The number of scales.

    Returns
    -------
    callable_list : `list`
        A `list` of callables.

    Raises
    ------
    ValueError
        callables must be a callable or a list/tuple of callables with the same
        length as the number of scales
    """
    if callable(callables):
        return [callables] * n_scales
    elif len(callables) == 1 and np.alltrue([callable(f) for f in callables]):
        return list(callables) * n_scales
    elif len(callables) == n_scales and np.alltrue([callable(f)
                                                    for f in callables]):
        return list(callables)
    else:
        raise ValueError("callables must be a callable or a list/tuple of "
                         "callables with the same length as the number "
                         "of scales")


def check_patch_shape(patch_shape, n_scales):
    r"""
    Function for checking a multi-scale `patch_shape` parameter value.

    Parameters
    ----------
    patch_shape : `list/tuple` of `int/float` or `list` of those
        The patch shape per scale
    n_scales : `int`
        The number of scales.

    Returns
    -------
    patch_shape : `list` of `list/tuple` of `int/float`
        The list of patch shape per scale.

    Raises
    ------
    ValueError
        patch_shape must be a list/tuple of int or a list/tuple of lit/tuple of
        int/float with the same length as the number of scales
    """
    if len(patch_shape) == 2 and isinstance(patch_shape[0], int):
        return [patch_shape] * n_scales
    elif len(patch_shape) == 1:
        return check_patch_shape(patch_shape[0], 1)
    elif len(patch_shape) == n_scales:
        l1 = check_patch_shape(patch_shape[0], 1)
        l2 = check_patch_shape(patch_shape[1:], n_scales-1)
        return l1 + l2
    else:
        raise ValueError("patch_shape must be a list/tuple of int or a "
                         "list/tuple of lit/tuple of int/float with the "
                         "same length as the number of scales")


def check_max_components(max_components, n_scales, var_name):
    r"""
    Checks the maximum number of components per scale. It must be ``None`` or
    `int` or `float` or a `list` of those containing ``1`` or ``{n_scales}``
    elements.

    Parameters
    ----------
    max_components : ``None`` or `int` or `float` or a `list` of those
        The value to check.
    n_scales : `int`
        The number of scales.
    var_name : `str`
        The name of the variable.

    Returns
    -------
    max_components : `list` of ``None`` or `int` or `float`
        The list of max components per scale.

    Raises
    ------
    ValueError
        {var_name} must be None or an int > 0 or a 0 <= float <= 1 or a list of
        those containing 1 or {n_scales} elements
    """
    str_error = ("{} must be None or an int > 0 or a 0 <= float <= 1 or "
                 "a list of those containing 1 or {} elements").format(
        var_name, n_scales)
    if not isinstance(max_components, (list, tuple)):
        max_components_list = [max_components] * n_scales
    elif len(max_components) == 1:
        max_components_list = [max_components[0]] * n_scales
    elif len(max_components) == n_scales:
        max_components_list = max_components
    else:
        raise ValueError(str_error)
    for comp in max_components_list:
        if comp is not None:
            if not isinstance(comp, int):
                if not isinstance(comp, float):
                    raise ValueError(str_error)
    return max_components_list


def check_max_iters(max_iters, n_scales):
    r"""
    Function that checks the value of a `max_iters` parameter defined for
    multiple scales. It must be `int` or `list` of `int`.

    Parameters
    ----------
    max_iters : `int` or `list` of `int`
        The value to check.
    n_scales : `int`
        The number of scales.

    Returns
    -------
    max_iters : `list` of `int`
        The list of values per scale.

    Raises
    ------
    ValueError
        max_iters can be integer, integer list containing 1 or {n_scales}
        elements or None
    """
    if type(max_iters) is int:
        max_iters = [np.round(max_iters/n_scales)
                     for _ in range(n_scales)]
    elif len(max_iters) == 1 and n_scales > 1:
        max_iters = [np.round(max_iters[0]/n_scales)
                     for _ in range(n_scales)]
    elif len(max_iters) != n_scales:
        raise ValueError('max_iters can be integer, integer list '
                         'containing 1 or {} elements or '
                         'None'.format(n_scales))
    return np.require(max_iters, dtype=np.int)


def check_sampling(sampling, n_scales):
    r"""
    Function that checks the value of a `sampling` parameter defined for
    multiple scales. It must be `int` or `ndarray` or `list` of those.

    Parameters
    ----------
    sampling : `int` or `ndarray` or `list` of those
        The value to check.
    n_scales : `int`
        The number of scales.

    Returns
    -------
    sampling : `list` of `int` or `ndarray`
        The list of values per scale.

    Raises
    ------
    ValueError
        A sampling list can only contain 1 element or {n_scales} elements
    ValueError
        sampling can be an integer or ndarray, a integer or ndarray list
        containing 1 or {n_scales} elements or None
    """
    if (isinstance(sampling, (list, tuple)) and
        np.alltrue([isinstance(s, (np.ndarray, np.int)) or sampling is None
                    for s in sampling])):
        if len(sampling) == 1:
            return sampling * n_scales
        elif len(sampling) == n_scales:
            return sampling
        else:
            raise ValueError('A sampling list can only '
                             'contain 1 element or {} '
                             'elements'.format(n_scales))
    elif isinstance(sampling, (np.ndarray, np.int)) or sampling is None:
        return [sampling] * n_scales
    else:
        raise ValueError('sampling can be an integer or ndarray, '
                         'a integer or ndarray list '
                         'containing 1 or {} elements or '
                         'None'.format(n_scales))


def set_models_components(models, n_components):
    r"""
    Function that sets the number of active components to a list of models.

    Parameters
    ----------
    models : `list` or `class`
        The list of models per scale.
    n_components : `int` or `float` or ``None`` or `list` of those
        The number of components per model.

    Raises
    ------
    ValueError
        n_components can be an integer or a float or None or a list containing 1
        or {n_scales} of those
    """
    if n_components is not None:
        n_scales = len(models)
        if type(n_components) is int or type(n_components) is float:
            for am in models:
                am.n_active_components = n_components
        elif len(n_components) == 1 and n_scales > 1:
            for am in models:
                am.n_active_components = n_components[0]
        elif len(n_components) == n_scales:
            for am, n in zip(models, n_components):
                am.n_active_components = n
        else:
            raise ValueError('n_components can be an integer or a float '
                             'or None or a list containing 1 or {} of '
                             'those'.format(n_scales))


def check_model(model, cls):
    r"""
    Function that checks whether the provided `class` object is a subclass of
    the provided base `class`.

    Parameters
    ----------
    model : `class`
        The object.
    cls : `class`
        The required base class.

    Raises
    ------
    ValueError
        Model must be a {cls} instance.
    """
    if not isinstance(model, cls):
        raise ValueError('Model must be a {} instance.'.format(
                name_of_callable(cls)))


def check_algorithm_cls(algorithm_cls, n_scales, base_algorithm_cls):
    r"""
    Function that checks whether the `list` of `class` objects defined per scale
    are subclasses of the provided base `class`.

    Parameters
    ----------
    algorithm_cls : `class` or `list` of `class`
        The list of objects per scale.
    n_scales : `int`
        The number of scales.
    base_algorithm_cls : `class`
        The required base class.

    Raises
    ------
    ValueError
        algorithm_cls must be a subclass of {base_algorithm_cls} or a list/tuple
        of {base_algorithm_cls} subclasses with the same length as the number of
        scales {n_scales}
    """
    if (isinstance(algorithm_cls, partial) and
            base_algorithm_cls in algorithm_cls.func.mro()):
        return [algorithm_cls] * n_scales
    elif (isinstance(algorithm_cls, type) and
          base_algorithm_cls in algorithm_cls.mro()):
        return [algorithm_cls] * n_scales
    elif len(algorithm_cls) == 1:
        return check_algorithm_cls(algorithm_cls[0], n_scales,
                                   base_algorithm_cls)
    elif len(algorithm_cls) == n_scales:
        return [check_algorithm_cls(a, 1, base_algorithm_cls)[0]
                for a in algorithm_cls]
    else:
        raise ValueError("algorithm_cls must be a subclass of {} or a "
                         "list/tuple of {} subclasses with the same length "
                         "as the number of scales {}"
                         .format(base_algorithm_cls, base_algorithm_cls,
                                 n_scales))


def check_graph(graph, graph_types, param_name, n_scales):
    r"""
    Checks the provided graph per pyramidal level. The graph must be a
    subclass of `graph_types` or a `list` of those.

    Parameters
    ----------
    graph : `graph` or `list` of `graph` types
        The graph argument to check.
    graph_types : `graph` or `tuple` of `graphs`
        The `tuple` of allowed graph types.
    param_name : `str`
        The name of the graph parameter.
    n_scales : `int`
        The number of pyramidal levels.

    Returns
    -------
    graph : `list` of `graph` types
        The graph per scale in a `list`.

    Raises
    ------
    ValueError
        {param_name} must be a list of length equal to the number of scales.
    ValueError
        {param_name} must be a list of {graph_types_str}. {} given instead.
    """
    # check if the provided graph is a list
    if not isinstance(graph, list):
        graphs = [graph] * n_scales
    elif len(graph) == 1:
        graphs = graph * n_scales
    elif len(graph) == n_scales:
        graphs = graph
    else:
        raise ValueError('{} must be a list of length equal to the number of '
                         'scales.'.format(param_name))
    # check if the provided graph_types is a list
    if not isinstance(graph_types, list):
        graph_types = [graph_types]

    # check each member of the graphs list
    for g in graphs:
        if g is not None:
            if type(g) not in graph_types:
                graph_types_str = ' or '.join(gt.__name__ for gt in graph_types)
                raise ValueError('{} must be a list of {}. {} given '
                                 'instead.'.format(param_name, graph_types_str,
                                                   type(g).__name__))
    return graphs
