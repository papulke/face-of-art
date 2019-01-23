from functools import partial

from menpo.feature import no_op

from menpofit.math import (IIRLRegression, IRLRegression, PCRRegression,
                           OptimalLinearRegression, OPPRegression)
from menpofit.modelinstance import OrthoPDM
from menpofit.error import euclidean_bb_normalised_error
from menpofit.result import MultiScaleParametricIterativeResult

from .base import (BaseSupervisedDescentAlgorithm,
                   compute_parametric_delta_x, features_per_image,
                   features_per_patch, update_parametric_estimates,
                   print_parametric_info, fit_parametric_shape)


class ParametricShapeSDAlgorithm(BaseSupervisedDescentAlgorithm):
    r"""
    Abstract class for training a cascaded-regression Supervised Descent
    algorithm that employs a parametric shape model.

    Parameters
    ----------
    shape_model_cls : `subclass` of :map:`PDM`, optional
        The class to be used for building the shape model. The most common
        choice is :map:`OrthoPDM`.
    """
    def __init__(self, shape_model_cls=OrthoPDM):
        super(ParametricShapeSDAlgorithm, self).__init__()
        self.regressors = []
        self.shape_model_cls = shape_model_cls
        self.shape_model = None

    @property
    def _multi_scale_fitter_result(self):
        # The result class to be used by a multi-scale fitter
        return MultiScaleParametricIterativeResult

    def _compute_delta_x(self, gt_shapes, current_shapes):
        # This is called first - so train shape model here
        if self.shape_model is None:
            self.shape_model = self.shape_model_cls(gt_shapes)

        return compute_parametric_delta_x(gt_shapes, current_shapes,
                                          self.shape_model)

    def _update_estimates(self, estimated_delta_x, delta_x, gt_x,
                          current_shapes):
        update_parametric_estimates(estimated_delta_x, delta_x, gt_x,
                                    current_shapes, self.shape_model)

    def _compute_training_features(self, images, gt_shapes, current_shapes,
                                   prefix='', verbose=False):
        # initialize sample counter
        return features_per_image(images, current_shapes, self.patch_shape,
                                  self.patch_features, prefix=prefix,
                                  verbose=verbose)

    def _compute_test_features(self, image, current_shape):
        return features_per_patch(image, current_shape,
                                  self.patch_shape, self.patch_features)

    def run(self, image, initial_shape, gt_shape=None, return_costs=False,
            **kwargs):
        r"""
        Run the algorithm to an image given an initial shape.

        Parameters
        ----------
        image : `menpo.image.Image` or subclass
            The image to be fitted.
        initial_shape : `menpo.shape.PointCloud`
            The initial shape from which the fitting procedure will start.
        gt_shape : `menpo.shape.PointCloud` or ``None``, optional
            The ground truth shape associated to the image.
        return_costs : `bool`, optional
            If ``True``, then the cost function values will be computed
            during the fitting procedure. Then these cost values will be
            assigned to the returned `fitting_result`. *Note that this
            argument currently has no effect and will raise a warning if set
            to ``True``. This is because it is not possible to evaluate the
            cost function of this algorithm.*

        Returns
        -------
        fitting_result: :map:`ParametricIterativeResult`
            The result of the fitting procedure.
        """
        return fit_parametric_shape(image, initial_shape, self,
                                    gt_shape=gt_shape,
                                    return_costs=return_costs)

    def _print_regression_info(self, _, gt_shapes, n_perturbations,
                               delta_x, estimated_delta_x, level_index,
                               prefix=''):
        print_parametric_info(self.shape_model, gt_shapes, n_perturbations,
                              delta_x, estimated_delta_x, level_index,
                              self._compute_error, prefix=prefix)


class ParametricShapeNewton(ParametricShapeSDAlgorithm):
    r"""
    Class for training a cascaded-regression algorithm that employs a
    parametric shape model using Incremental Regularized Linear Regression
    (:map:`IRLRegression`).

    Parameters
    ----------
    patch_features : `callable`, optional
        The features to be extracted from the patches of an image.
    patch_shape : `(int, int)`, optional
        The shape of the extracted patches.
    n_iterations : `int`, optional
        The number of iterations (cascades).
    shape_model_cls : `subclass` of :map:`PDM`, optional
        The class to be used for building the shape model. The most common
        choice is :map:`OrthoPDM`.
    compute_error : `callable`, optional
        The function to be used for computing the fitting error when training
        each cascade.
    alpha : `float`, optional
        The regularization parameter.
    bias : `bool`, optional
        Flag that controls whether to use a bias term.
    """
    def __init__(self, patch_features=no_op, patch_shape=(17, 17),
                 n_iterations=3, shape_model_cls=OrthoPDM,
                 compute_error=euclidean_bb_normalised_error,
                 alpha=0, bias=True):
        super(ParametricShapeNewton, self).__init__(
            shape_model_cls=shape_model_cls)

        self._regressor_cls = partial(IRLRegression, alpha=alpha, bias=bias)
        self.patch_shape = patch_shape
        self.patch_features = patch_features
        self.n_iterations = n_iterations
        self._compute_error = compute_error


class ParametricShapeGaussNewton(ParametricShapeSDAlgorithm):
    r"""
    Class for training a cascaded-regression algorithm that employs a
    parametric shape model using Indirect Incremental Regularized Linear
    Regression (:map:`IIRLRegression`).

    Parameters
    ----------
    patch_features : `callable`, optional
        The features to be extracted from the patches of an image.
    patch_shape : `(int, int)`, optional
        The shape of the extracted patches.
    n_iterations : `int`, optional
        The number of iterations (cascades).
    shape_model_cls : `subclass` of :map:`PDM`, optional
        The class to be used for building the shape model. The most common
        choice is :map:`OrthoPDM`.
    compute_error : `callable`, optional
        The function to be used for computing the fitting error when training
        each cascade.
    alpha : `float`, optional
        The regularization parameter.
    bias : `bool`, optional
        Flag that controls whether to use a bias term.
    alpha2 : `float`, optional
        The regularization parameter of the Hessian matrix.
    """
    def __init__(self, patch_features=no_op, patch_shape=(17, 17),
                 n_iterations=3, shape_model_cls=OrthoPDM,
                 compute_error=euclidean_bb_normalised_error,
                 alpha=0, bias=True, alpha2=0):
        super(ParametricShapeGaussNewton, self).__init__(
            shape_model_cls=shape_model_cls)

        self._regressor_cls = partial(IIRLRegression, alpha=alpha, bias=bias,
                                      alpha2=alpha2)
        self.patch_shape = patch_shape
        self.patch_features = patch_features
        self.n_iterations = n_iterations
        self._compute_error = compute_error


class ParametricShapeOptimalRegression(ParametricShapeSDAlgorithm):
    r"""
    Class for training a cascaded-regression algorithm that employs a parametric
    shape model using Multivariate Linear Regression with optimal
    reconstructions (:map:`OptimalLinearRegression`).

    Parameters
    ----------
    patch_features : `callable`, optional
        The features to be extracted from the patches of an image.
    patch_shape : `(int, int)`, optional
        The shape of the extracted patches.
    n_iterations : `int`, optional
        The number of iterations (cascades).
    shape_model_cls : `subclass` of :map:`PDM`, optional
        The class to be used for building the shape model. The most common
        choice is :map:`OrthoPDM`.
    compute_error : `callable`, optional
        The function to be used for computing the fitting error when training
        each cascade.
    variance : `float` or ``None``, optional
        The SVD variance.
    bias : `bool`, optional
        Flag that controls whether to use a bias term.
    """
    def __init__(self, patch_features=no_op, patch_shape=(17, 17),
                 n_iterations=3, shape_model_cls=OrthoPDM,
                 compute_error=euclidean_bb_normalised_error,
                 variance=None, bias=True):
        super(ParametricShapeOptimalRegression, self).__init__(
            shape_model_cls=shape_model_cls)

        self._regressor_cls = partial(OptimalLinearRegression,
                                      variance=variance, bias=bias)
        self.patch_shape = patch_shape
        self.patch_features = patch_features
        self.n_iterations = n_iterations
        self._compute_error = compute_error


class ParametricShapePCRRegression(ParametricShapeSDAlgorithm):
    r"""
    Class for training a cascaded-regression algorithm that employs a parametric
    shape model using Principal Component Regression (:map:`PCRRegression`).

    Parameters
    ----------
    patch_features : `callable`, optional
        The features to be extracted from the patches of an image.
    patch_shape : `(int, int)`, optional
        The shape of the extracted patches.
    n_iterations : `int`, optional
        The number of iterations (cascades).
    shape_model_cls : `subclass` of :map:`PDM`, optional
        The class to be used for building the shape model. The most common
        choice is :map:`OrthoPDM`.
    compute_error : `callable`, optional
        The function to be used for computing the fitting error when training
        each cascade.
    variance : `float` or ``None``, optional
        The SVD variance.
    bias : `bool`, optional
        Flag that controls whether to use a bias term.

    Raises
    ------
    ValueError
        variance must be set to a number between 0 and 1
    """
    def __init__(self, patch_features=no_op, patch_shape=(17, 17),
                 n_iterations=3, shape_model_cls=OrthoPDM,
                 compute_error=euclidean_bb_normalised_error,
                 variance=None, bias=True):
        super(ParametricShapePCRRegression, self).__init__(
            shape_model_cls=shape_model_cls)

        self._regressor_cls = partial(PCRRegression,
                                      variance=variance, bias=bias)
        self.patch_shape = patch_shape
        self.patch_features = patch_features
        self.n_iterations = n_iterations
        self._compute_error = compute_error


class ParametricShapeOPPRegression(ParametricShapeSDAlgorithm):
    r"""
    Class for training a cascaded-regression algorithm that employs a parametric
    shape model using Multivariate Linear Regression with Orthogonal Procrustes
    Problem reconstructions (:map:`OPPRegression`).

    Parameters
    ----------
    patch_features : `callable`, optional
        The features to be extracted from the patches of an image.
    patch_shape : `(int, int)`, optional
        The shape of the extracted patches.
    n_iterations : `int`, optional
        The number of iterations (cascades).
    shape_model_cls : `subclass` of :map:`PDM`, optional
        The class to be used for building the shape model. The most common
        choice is :map:`OrthoPDM`.
    compute_error : `callable`, optional
        The function to be used for computing the fitting error when training
        each cascade.
    whiten : `bool`, optional
        Whether to use a whitened PCA model.
    bias : `bool`, optional
        Flag that controls whether to use a bias term.
    """
    def __init__(self, patch_features=no_op, patch_shape=(17, 17),
                 n_iterations=3, shape_model_cls=OrthoPDM,
                 compute_error=euclidean_bb_normalised_error,
                 whiten=False, bias=True):
        super(ParametricShapeOPPRegression, self).__init__(
            shape_model_cls=shape_model_cls)

        self._regressor_cls = partial(OPPRegression,
                                      whiten=whiten, bias=bias)
        self.patch_shape = patch_shape
        self.patch_features = patch_features
        self.n_iterations = n_iterations
        self._compute_error = compute_error
