from menpofit.result import (ParametricIterativeResult,
                             MultiScaleParametricIterativeResult)


class LucasKanadeAlgorithmResult(ParametricIterativeResult):
    r"""
    Class for storing the iterative result of a Lucas-Kanade Image Alignment
    optimization algorithm.

    Parameters
    ----------
    shapes : `list` of `menpo.shape.PointCloud`
        The `list` of shapes per iteration. The first and last members
        correspond to the initial and final shapes, respectively.
    homogeneous_parameters : `list` of ``(n_parameters,)`` `ndarray`
        The `list` of parameters of the homogeneous transform per iteration.
        The first and last members correspond to the initial and final
        shapes, respectively.
    initial_shape : `menpo.shape.PointCloud` or ``None``, optional
        The initial shape from which the fitting process started. If
        ``None``, then no initial shape is assigned.
    image : `menpo.image.Image` or `subclass` or ``None``, optional
        The image on which the fitting process was applied. Note that a copy
        of the image will be assigned as an attribute. If ``None``, then no
        image is assigned.
    gt_shape : `menpo.shape.PointCloud` or ``None``, optional
        The ground truth shape associated with the image. If ``None``, then no
        ground truth shape is assigned.
    costs : `list` of `float` or ``None``, optional
        The `list` of cost per iteration. If ``None``, then it is assumed that
        the cost function cannot be computed for the specific algorithm.
    """
    def __init__(self, shapes, homogeneous_parameters, initial_shape=None,
                 image=None, gt_shape=None, costs=None):
        super(LucasKanadeAlgorithmResult, self).__init__(
            shapes=shapes, shape_parameters=homogeneous_parameters,
            initial_shape=initial_shape, image=image, gt_shape=gt_shape,
            costs=costs)
        self._homogeneous_parameters = homogeneous_parameters

    @property
    def homogeneous_parameters(self):
        r"""
        Returns the `list` of parameters of the homogeneous transform
        obtained at each iteration of the fitting process. The `list`
        includes the parameters of the `initial_shape` (if it exists) and
        `final_shape`.

        :type: `list` of ``(n_params,)`` `ndarray`
        """
        return self._shape_parameters


class LucasKanadeResult(MultiScaleParametricIterativeResult):
    r"""
    Class for storing the multi-scale iterative fitting result of an ATM. It
    holds the shapes, shape parameters and costs per iteration.

    Parameters
    ----------
    results : `list` of :map:`ATMAlgorithmResult`
        The `list` of optimization results per scale.
    scales : `list` or `tuple`
        The `list` of scale values per scale (low to high).
    affine_transforms : `list` of `menpo.transform.Affine`
        The list of affine transforms per scale that transform the shapes into
        the original image space.
    scale_transforms : `list` of `menpo.shape.Scale`
        The list of scaling transforms per scale.
    image : `menpo.image.Image` or `subclass` or ``None``, optional
        The image on which the fitting process was applied. Note that a copy
        of the image will be assigned as an attribute. If ``None``, then no
        image is assigned.
    gt_shape : `menpo.shape.PointCloud` or ``None``, optional
        The ground truth shape associated with the image. If ``None``, then no
        ground truth shape is assigned.
    """
    def __init__(self, results, scales, affine_transforms, scale_transforms,
                 image=None, gt_shape=None):
        super(LucasKanadeResult, self).__init__(
            results=results, scales=scales, affine_transforms=affine_transforms,
            scale_transforms=scale_transforms, image=image, gt_shape=gt_shape)
        # Create parameters list
        self._homogeneous_parameters = []
        for r in results:
            self._homogeneous_parameters += r.homogeneous_parameters
        # Correct n_iters
        self._n_iters -= len(scales)

    @property
    def homogeneous_parameters(self):
        r"""
        Returns the `list` of parameters of the homogeneous transform
        obtained at each iteration of the fitting process. The `list`
        includes the parameters of the `initial_shape` (if it exists) and
        `final_shape`.

        :type: `list` of ``(n_params,)`` `ndarray`
        """
        return self._homogeneous_parameters

    @property
    def shape_parameters(self):
        # Use homogeneous_parameters instead.
        raise AttributeError
