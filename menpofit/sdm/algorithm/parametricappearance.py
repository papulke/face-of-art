import numpy as np
from functools import partial

from menpo.feature import no_op
from menpo.model import PCAVectorModel

from menpofit.error import euclidean_bb_normalised_error
from menpofit.result import MultiScaleNonParametricIterativeResult
from menpofit.math import IIRLRegression, IRLRegression
from menpofit.visualize import print_progress

from .base import (BaseSupervisedDescentAlgorithm,
                   features_per_patch, update_non_parametric_estimates,
                   compute_non_parametric_delta_x, print_non_parametric_info,
                   build_appearance_model, fit_non_parametric_shape)


class ParametricAppearanceSDAlgorithm(BaseSupervisedDescentAlgorithm):
    r"""
    Abstract class for training a cascaded-regression Supervised Descent
    algorithm that employs a parametric appearance model.

    Parameters
    ----------
    appearance_model_cls : `menpo.model.PCAVectorModel` or `subclass`
        The class to be used for building the appearance model.
    """
    def __init__(self, appearance_model_cls=PCAVectorModel):
        super(ParametricAppearanceSDAlgorithm, self).__init__()
        self.regressors = []
        self.appearance_model_cls = appearance_model_cls
        self.appearance_model = None

    @property
    def _multi_scale_fitter_result(self):
        # The result class to be used by a multi-scale fitter
        return MultiScaleNonParametricIterativeResult

    def _compute_delta_x(self, gt_shapes, current_shapes):
        return compute_non_parametric_delta_x(gt_shapes, current_shapes)

    def _update_estimates(self, estimated_delta_x, delta_x, gt_x,
                          current_shapes):
        update_non_parametric_estimates(estimated_delta_x, delta_x, gt_x,
                                        current_shapes)

    def _compute_training_features(self, images, gt_shapes, current_shapes,
                                   prefix='', verbose=False):

        if self.appearance_model is None:
            self.appearance_model = build_appearance_model(
                images, gt_shapes, self.patch_shape, self.patch_features,
                self.appearance_model_cls, verbose=verbose, prefix=prefix)

        wrap = partial(print_progress,
                       prefix='{}Extracting patches'.format(prefix),
                       end_with_newline=not prefix, verbose=verbose)

        features = []
        for im, shapes in wrap(list(zip(images, current_shapes))):
            for s in shapes:
                param_feature = self._compute_test_features(im, s)
                features.append(param_feature)

        return np.vstack(features)

    def _compute_parametric_features(self, patch):
        raise NotImplementedError()

    def _compute_test_features(self, image, current_shape):
        patch_feature = features_per_patch(
            image, current_shape, self.patch_shape, self.patch_features)
        return self._compute_parametric_features(patch_feature)

    def run(self, image, initial_shape, gt_shape=None,
            return_costs=False, **kwargs):
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
        fitting_result: :map:`NonParametricIterativeResult`
            The result of the fitting procedure.
        """
        return fit_non_parametric_shape(image, initial_shape, self,
                                        gt_shape=gt_shape,
                                        return_costs=return_costs)

    def _print_regression_info(self, template_shape, gt_shapes, n_perturbations,
                               delta_x, estimated_delta_x, level_index,
                               prefix=''):
        print_non_parametric_info(template_shape, gt_shapes, n_perturbations,
                                  delta_x, estimated_delta_x, level_index,
                                  self._compute_error, prefix=prefix)


class ParametricAppearanceNewton(ParametricAppearanceSDAlgorithm):
    r"""
    Class for training a cascaded-regression algorithm that employs a
    parametric appearance model using Incremental Regularized Linear
    Regression (:map:`IRLRegression`).

    Parameters
    ----------
    patch_features : `callable`, optional
        The features to be extracted from the patches of an image.
    patch_shape : `(int, int)`, optional
        The shape of the extracted patches.
    n_iterations : `int`, optional
        The number of iterations (cascades).
    appearance_model_cls : `menpo.model.PCAVectorModel` or `subclass`
        The class to be used for building the appearance model.
    compute_error : `callable`, optional
        The function to be used for computing the fitting error when training
        each cascade.
    alpha : `float`, optional
        The regularization parameter.
    bias : `bool`, optional
        Flag that controls whether to use a bias term.
    """
    def __init__(self, patch_features=no_op, patch_shape=(17, 17),
                 n_iterations=3, appearance_model_cls=PCAVectorModel,
                 compute_error=euclidean_bb_normalised_error,
                 alpha=0, bias=True):
        super(ParametricAppearanceNewton, self).__init__(
            appearance_model_cls=appearance_model_cls)

        self._regressor_cls = partial(IRLRegression, alpha=alpha, bias=bias)
        self.patch_shape = patch_shape
        self.patch_features = patch_features
        self.n_iterations = n_iterations
        self._compute_error = compute_error


class ParametricAppearanceGaussNewton(ParametricAppearanceSDAlgorithm):
    r"""
    Class for training a cascaded-regression Gauss-Newton algorithm that employs
    a parametric appearance model using Indirect Incremental Regularized Linear
    Regression (:map:`IIRLRegression`).

    Parameters
    ----------
    patch_features : `callable`, optional
        The features to be extracted from the patches of an image.
    patch_shape : `(int, int)`, optional
        The shape of the extracted patches.
    n_iterations : `int`, optional
        The number of iterations (cascades).
    appearance_model_cls : `menpo.model.PCAVectorModel` or `subclass`
        The class to be used for building the appearance model.
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
                 n_iterations=3, appearance_model_cls=PCAVectorModel,
                 compute_error=euclidean_bb_normalised_error,
                 alpha=0, bias=True, alpha2=0):
        super(ParametricAppearanceGaussNewton, self).__init__(
            appearance_model_cls=appearance_model_cls)

        self._regressor_cls = partial(IIRLRegression, alpha=alpha, bias=bias,
                                      alpha2=alpha2)
        self.patch_shape = patch_shape
        self.patch_features = patch_features
        self.n_iterations = n_iterations
        self._compute_error = compute_error


class ParametricAppearanceProjectOutNewton(ParametricAppearanceNewton):
    r"""
    Class for training a cascaded-regression Newton algorithm that employs a
    parametric appearance model using Incremental Regularized Linear
    Regression (:map:`IRLRegression`). The algorithm uses the projected-out
    appearance vectors as features in the regression.
    """
    def _compute_parametric_features(self, patch):
        return self.appearance_model.project_out(patch.ravel())


class ParametricAppearanceMeanTemplateNewton(ParametricAppearanceNewton):
    r"""
    Class for training a cascaded-regression Newton algorithm that employs a
    parametric appearance model using Incremental Regularized Linear
    Regression (:map:`IRLRegression`). The algorithm uses the centered
    appearance vectors as features in the regression.
    """
    def _compute_parametric_features(self, patch):
        return patch.ravel() - self.appearance_model.mean().ravel()


class ParametricAppearanceWeightsNewton(ParametricAppearanceNewton):
    r"""
    Class for training a cascaded-regression Newton algorithm that employs a
    parametric appearance model using Incremental Regularized Linear
    Regression (:map:`IRLRegression`). The algorithm uses the projection
    weights of the appearance vectors as features in the regression.
    """
    def _compute_parametric_features(self, patch):
        return self.appearance_model.project(patch.ravel())


class ParametricAppearanceProjectOutGuassNewton(ParametricAppearanceGaussNewton):
    r"""
    Class for training a cascaded-regression Gauss-Newton algorithm that employs
    a parametric appearance model using Indirect Incremental Regularized Linear
    Regression (:map:`IIRLRegression`). The algorithm uses the projected-out
    appearance vectors as features in the regression.
    """
    def _compute_parametric_features(self, patch):
        return self.appearance_model.project_out(patch.ravel())


class ParametricAppearanceMeanTemplateGuassNewton(ParametricAppearanceGaussNewton):
    r"""
    Class for training a cascaded-regression Gauss-Newton algorithm that employs
    a parametric appearance model using Indirect Incremental Regularized Linear
    Regression (:map:`IIRLRegression`). The algorithm uses the centered
    appearance vectors as features in the regression.
    """
    def _compute_parametric_features(self, patch):
        return patch.ravel() - self.appearance_model.mean().ravel()


class ParametricAppearanceWeightsGuassNewton(ParametricAppearanceGaussNewton):
    r"""
    Class for training a cascaded-regression Gauss-Newton algorithm that employs
    a parametric appearance model using Indirect Incremental Regularized Linear
    Regression (:map:`IIRLRegression`). The algorithm uses the projection
    weights of the appearance vectors as features in the regression.
    """
    def _compute_parametric_features(self, patch):
        return self.appearance_model.project(patch.ravel())
