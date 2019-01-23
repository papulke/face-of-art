from menpofit.result import (ParametricIterativeResult,
                             MultiScaleParametricIterativeResult)


class AAMAlgorithmResult(ParametricIterativeResult):
    r"""
    Class for storing the iterative result of an AAM optimisation algorithm.

    .. note:: When using a method with a parametric shape model, the first step
              is to **reconstruct the initial shape** using the shape model. The
              generated reconstructed shape is then used as initialisation for
              the iterative optimisation. This step is not counted in the number
              of iterations.

    Parameters
    ----------
    shapes : `list` of `menpo.shape.PointCloud`
        The `list` of shapes per iteration. The first and last members
        correspond to the initial and final shapes, respectively.
    shape_parameters : `list` of ``(n_shape_parameters,)`` `ndarray`
        The `list` of shape parameters per iteration. The first and last members
        correspond to the initial and final shapes, respectively.
    appearance_parameters : `list` of ``(n_appearance_parameters,)`` `ndarray`
        The `list` of appearance parameters per iteration. The first and last
        members correspond to the initial and final shapes, respectively.
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
    def __init__(self, shapes, shape_parameters, appearance_parameters,
                 initial_shape=None, image=None, gt_shape=None, costs=None):
        super(AAMAlgorithmResult, self).__init__(
            shapes=shapes, shape_parameters=shape_parameters,
            initial_shape=initial_shape, image=image, gt_shape=gt_shape,
            costs=costs)
        self._appearance_parameters = appearance_parameters

    @property
    def appearance_parameters(self):
        r"""
        Returns the `list` of appearance parameters obtained at each iteration
        of the fitting process. The `list` includes the parameters of the
        `initial_shape` (if it exists) and `final_shape`.

        :type: `list` of ``(n_params,)`` `ndarray`
        """
        return self._appearance_parameters


class AAMResult(MultiScaleParametricIterativeResult):
    r"""
    Class for storing the multi-scale iterative fitting result of an AAM. It
    holds the shapes, shape parameters, appearance parameters and costs per
    iteration.

    .. note:: When using a method with a parametric shape model, the first step
              is to **reconstruct the initial shape** using the shape model. The
              generated reconstructed shape is then used as initialisation for
              the iterative optimisation. This step is not counted in the number
              of iterations.

    Parameters
    ----------
    results : `list` of :map:`AAMAlgorithmResult`
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
        super(AAMResult, self).__init__(
            results=results, scales=scales, affine_transforms=affine_transforms,
            scale_transforms=scale_transforms, image=image, gt_shape=gt_shape)
        # Create appearance parameters list
        self._appearance_parameters = None
        if results[0].appearance_parameters is not None:
            self._appearance_parameters = []
            for r in results:
                self._appearance_parameters += r.appearance_parameters

    @property
    def appearance_parameters(self):
        r"""
        Returns the `list` of appearance parameters obtained at each iteration
        of the fitting process. The `list` includes the parameters of the
        `initial_shape` (if it exists) and `final_shape`.

        :type: `list` of ``(n_params,)`` `ndarray`
        """
        return self._appearance_parameters
