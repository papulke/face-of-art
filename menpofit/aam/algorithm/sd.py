from __future__ import division
from functools import partial
import numpy as np

from menpofit.fitter import raise_costs_warning
from menpofit.math import IRLRegression, IIRLRegression
from menpofit.result import euclidean_bb_normalised_error
from menpofit.sdm.algorithm.base import (BaseSupervisedDescentAlgorithm,
                                         compute_parametric_delta_x,
                                         update_parametric_estimates,
                                         print_parametric_info)
from menpofit.visualize import print_progress


class ParametricSupervisedDescentAlgorithm(BaseSupervisedDescentAlgorithm):
    r"""
    Base class for defining a cascaded-regression Supervised Descent Algorithm
    given a trained AAM model.

    Parameters
    ----------
    aam_interface : The AAM interface class from `menpofit.aam.algorithm.lk`.
        Existing interfaces include:

            ============================== =============================
            'LucasKanadeStandardInterface' Suitable for holistic AAMs
            'LucasKanadeLinearInterface'   Suitable for linear AAMs
            'LucasKanadePatchInterface'    Suitable for patch-based AAMs
            ============================== =============================

    n_iterations : `int`, optional
        The number of iterations (cascades).
    compute_error : `callable`, optional
        The function to be used for computing the fitting error when training
        each cascade.
    """
    def __init__(self, aam_interface, n_iterations=10,
                 compute_error=euclidean_bb_normalised_error):
        super(ParametricSupervisedDescentAlgorithm, self).__init__()

        self.interface = aam_interface
        self.n_iterations = n_iterations

        self._compute_error = compute_error
        self._precompute()

    @property
    def appearance_model(self):
        r"""
        Returns the appearance model of the AAM.

        :type: `menpo.model.PCAModel`
        """
        return self.interface.appearance_model

    @property
    def transform(self):
        r"""
        Returns the model driven differential transform object of the AAM, e.g.
        :map:`DifferentiablePiecewiseAffine` or
        :map:`DifferentiableThinPlateSplines`.

        :type: `subclass` of :map:`DL` and :map:`DX`
        """
        return self.interface.transform

    def _precompute(self):
        # Grab appearance model mean
        a_bar = self.appearance_model.mean()
        # Vectorise it and mask it
        self.a_bar_m = a_bar.as_vector()[self.interface.i_mask]

    def _compute_delta_x(self, gt_shapes, current_shapes):
        # This is called first - so train shape model here
        return compute_parametric_delta_x(gt_shapes, current_shapes,
                                          self.transform)

    def _update_estimates(self, estimated_delta_x, delta_x, gt_x,
                          current_shapes):
        update_parametric_estimates(estimated_delta_x, delta_x, gt_x,
                                    current_shapes, self.transform)

    def _compute_training_features(self, images, gt_shapes, current_shapes,
                                   prefix='', verbose=False):
        wrap = partial(print_progress,
                       prefix='{}Extracting patches'.format(prefix),
                       end_with_newline=not prefix, verbose=verbose)

        features = []
        for im, shapes in wrap(list(zip(images, current_shapes))):
            for s in shapes:
                param_feature = self._compute_test_features(im, s)
                features.append(param_feature)

        return np.vstack(features)

    def _compute_test_features(self, image, current_shape):
        # Make sure you call: self.transform.set_target(current_shape)
        # before calculating the warp
        raise NotImplementedError()

    def _print_regression_info(self, _, gt_shapes, n_perturbations,
                               delta_x, estimated_delta_x, level_index,
                               prefix=''):
        print_parametric_info(self.transform, gt_shapes, n_perturbations,
                              delta_x, estimated_delta_x, level_index,
                              self._compute_error, prefix=prefix)

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
        fitting_result : :map:`AAMAlgorithmResult`
            The parametric iterative fitting result.
        """
        # costs warning
        if return_costs:
            raise_costs_warning(self)

        # initialize transform
        self.transform.set_target(initial_shape)
        p_list = [self.transform.as_vector()]
        shapes = [self.transform.target]

        # Cascaded Regression loop
        for r in self.regressors:
            # Assumes that the transform is correctly set
            features = self._compute_test_features(image,
                                                   self.transform.target)

            # solve for increments on the shape parameters
            dx = r.predict(features)

            # We need to update the transform to set the state for the warping
            # of the image above.
            new_x = p_list[-1] + dx
            self.transform._from_vector_inplace(new_x)
            p_list.append(new_x)
            shapes.append(self.transform.target)

        # return algorithm result
        return self.interface.algorithm_result(
            image=image, shapes=shapes, shape_parameters=p_list,
            initial_shape=initial_shape, gt_shape=gt_shape)


class MeanTemplate(ParametricSupervisedDescentAlgorithm):
    r"""
    Base class for defining a cascaded-regression Supervised Descent Algorithm
    given a trained AAM model. The algorithm uses the centered appearance vectors
    as features in the regression.
    """
    def _compute_test_features(self, image, current_shape):
        self.transform.set_target(current_shape)
        i = self.interface.warp(image)
        i_m = i.as_vector()[self.interface.i_mask]
        return i_m - self.a_bar_m


class MeanTemplateNewton(MeanTemplate):
    r"""
    Class for training a cascaded-regression Newton algorithm using Incremental
    Regularized Linear Regression (:map:`IRLRegression`) given a trained AAM
    model. The algorithm uses the centered appearance vectors as features in
    the regression.

    Parameters
    ----------
    aam_interface : The AAM interface class from `menpofit.aam.algorithm.lk`.
        Existing interfaces include:

            ============================== =============================
            'LucasKanadeStandardInterface' Suitable for holistic AAMs
            'LucasKanadeLinearInterface'   Suitable for linear AAMs
            'LucasKanadePatchInterface'    Suitable for patch-based AAMs
            ============================== =============================

    n_iterations : `int`, optional
        The number of iterations (cascades).
    compute_error : `callable`, optional
        The function to be used for computing the fitting error when training
        each cascade.
    alpha : `float`, optional
        The regularization parameter.
    bias : `bool`, optional
        Flag that controls whether to use a bias term.
    """
    def __init__(self, aam_interface, n_iterations=3,
                 compute_error=euclidean_bb_normalised_error,
                 alpha=0, bias=True):
        super(MeanTemplateNewton, self).__init__(
            aam_interface, n_iterations=n_iterations,
            compute_error=compute_error)

        self._regressor_cls = partial(IRLRegression, alpha=alpha, bias=bias)


class MeanTemplateGaussNewton(MeanTemplate):
    r"""
    Class for training a cascaded-regression Gauss-Newton algorithm using
    Indirect Incremental Regularized Linear Regression (:map:`IIRLRegression`)
    given a trained AAM model. The algorithm uses the centered appearance
    vectors as features in the regression.

    Parameters
    ----------
    aam_interface : The AAM interface class from `menpofit.aam.algorithm.lk`.
        Existing interfaces include:

            ============================== =============================
            'LucasKanadeStandardInterface' Suitable for holistic AAMs
            'LucasKanadeLinearInterface'   Suitable for linear AAMs
            'LucasKanadePatchInterface'    Suitable for patch-based AAMs
            ============================== =============================

    n_iterations : `int`, optional
        The number of iterations (cascades).
    compute_error : `callable`, optional
        The function to be used for computing the fitting error when training
        each cascade.
    alpha : `float`, optional
        The regularization parameter.
    alpha2 : `float`, optional
        The regularization parameter of the Hessian matrix.
    bias : `bool`, optional
        Flag that controls whether to use a bias term.
    """
    def __init__(self, aam_interface, n_iterations=3,
                 compute_error=euclidean_bb_normalised_error,
                 alpha=0, alpha2=0, bias=True):
        super(MeanTemplateGaussNewton, self).__init__(
            aam_interface, n_iterations=n_iterations,
            compute_error=compute_error)

        self._regressor_cls = partial(IIRLRegression, alpha=alpha,
                                      alpha2=alpha2, bias=bias)


class ProjectOut(ParametricSupervisedDescentAlgorithm):
    r"""
    Base class for defining a cascaded-regression Supervised Descent Algorithm
    given a trained AAM model. The algorithm uses the projected-out appearance
    vectors as features in the regression.
    """
    def _precompute(self):
        super(ProjectOut, self)._precompute()
        A = self.appearance_model.components
        self.A_m = A.T[self.interface.i_mask, :]

        self.pinv_A_m = np.linalg.pinv(self.A_m)

    def project_out(self, J):
        r"""
        Projects-out the appearance subspace from a given vector or matrix.

        :type: `ndarray`
        """
        # Project-out appearance bases from a particular vector or matrix
        return J - self.A_m.dot(self.pinv_A_m.dot(J))

    def _compute_test_features(self, image, current_shape):
        self.transform.set_target(current_shape)
        i = self.interface.warp(image)
        i_m = i.as_vector()[self.interface.i_mask]
        # TODO: This project out could actually be cached at test time -
        # but we need to think about the best way to implement this and still
        # allow incrementing
        e_m = i_m - self.a_bar_m
        return self.project_out(e_m)


class ProjectOutNewton(ProjectOut):
    r"""
    Class for training a cascaded-regression Newton algorithm using Incremental
    Regularized Linear Regression (:map:`IRLRegression`) given a trained AAM
    model. The algorithm uses the projected-out appearance vectors as
    features in the regression.

    Parameters
    ----------
    aam_interface : The AAM interface class from `menpofit.aam.algorithm.lk`.
        Existing interfaces include:

            ============================== =============================
            Class                          AAM
            ============================== =============================
            'LucasKanadeStandardInterface' Suitable for holistic AAMs
            'LucasKanadeLinearInterface'   Suitable for linear AAMs
            'LucasKanadePatchInterface'    Suitable for patch-based AAMs
            ============================== =============================

    n_iterations : `int`, optional
        The number of iterations (cascades).
    compute_error : `callable`, optional
        The function to be used for computing the fitting error when training
        each cascade.
    alpha : `float`, optional
        The regularization parameter.
    bias : `bool`, optional
        Flag that controls whether to use a bias term.
    """
    def __init__(self, aam_interface, n_iterations=3,
                 compute_error=euclidean_bb_normalised_error,
                 alpha=0, bias=True):
        super(ProjectOutNewton, self).__init__(
            aam_interface, n_iterations=n_iterations,
            compute_error=compute_error)

        self._regressor_cls = partial(IRLRegression, alpha=alpha, bias=bias)


class ProjectOutGaussNewton(ProjectOut):
    r"""
    Class for training a cascaded-regression Gauss-Newton algorithm using
    Indirect Incremental Regularized Linear Regression (:map:`IIRLRegression`)
    given a trained AAM model. The algorithm uses the projected-out
    appearance vectors as features in the regression.

    Parameters
    ----------
    aam_interface : The AAM interface class from `menpofit.aam.algorithm.lk`.
        Existing interfaces include:

            ============================== =============================
            'LucasKanadeStandardInterface' Suitable for holistic AAMs
            'LucasKanadeLinearInterface'   Suitable for linear AAMs
            'LucasKanadePatchInterface'    Suitable for patch-based AAMs
            ============================== =============================

    n_iterations : `int`, optional
        The number of iterations (cascades).
    compute_error : `callable`, optional
        The function to be used for computing the fitting error when training
        each cascade.
    alpha : `float`, optional
        The regularization parameter.
    alpha2 : `float`, optional
        The regularization parameter of the Hessian matrix.
    bias : `bool`, optional
        Flag that controls whether to use a bias term.
    """
    def __init__(self, aam_interface, n_iterations=3,
                 compute_error=euclidean_bb_normalised_error,
                 alpha=0, alpha2=0, bias=True):
        super(ProjectOutGaussNewton, self).__init__(
            aam_interface, n_iterations=n_iterations,
            compute_error=compute_error)

        self._regressor_cls = partial(IIRLRegression, alpha=alpha,
                                      alpha2=alpha2, bias=bias)


class AppearanceWeights(ParametricSupervisedDescentAlgorithm):
    r"""
    Base class for defining a cascaded-regression Supervised Descent Algorithm
    given a trained AAM model. The algorithm uses the projection weights of the
    appearance vectors as features in the regression.
    """
    def _precompute(self):
        super(AppearanceWeights, self)._precompute()
        A = self.appearance_model.components
        A_m = A.T[self.interface.i_mask, :]

        self.pinv_A_m = np.linalg.pinv(A_m)

    def project(self, J):
        r"""
        Projects a given vector or matrix onto the appearance subspace.

        :type: `ndarray`
        """
        # Project a particular vector or matrix onto the appearance bases
        return self.pinv_A_m.dot(J - self.a_bar_m)

    def _compute_test_features(self, image, current_shape):
        self.transform.set_target(current_shape)
        i = self.interface.warp(image)
        i_m = i.as_vector()[self.interface.i_mask]
        # Project image onto the appearance model
        return self.project(i_m)


class AppearanceWeightsNewton(AppearanceWeights):
    r"""
    Class for training a cascaded-regression Newton algorithm using Incremental
    Regularized Linear Regression (:map:`IRLRegression`) given a trained AAM
    model. The algorithm uses the projection weights of the appearance
    vectors as features in the regression.

    Parameters
    ----------
    aam_interface : The AAM interface class from `menpofit.aam.algorithm.lk`.
        Existing interfaces include:

            ============================== =============================
            'LucasKanadeStandardInterface' Suitable for holistic AAMs
            'LucasKanadeLinearInterface'   Suitable for linear AAMs
            'LucasKanadePatchInterface'    Suitable for patch-based AAMs
            ============================== =============================

    n_iterations : `int`, optional
        The number of iterations (cascades).
    compute_error : `callable`, optional
        The function to be used for computing the fitting error when training
        each cascade.
    alpha : `float`, optional
        The regularization parameter.
    bias : `bool`, optional
        Flag that controls whether to use a bias term.
    """
    def __init__(self, aam_interface, n_iterations=3,
                 compute_error=euclidean_bb_normalised_error,
                 alpha=0, bias=True):
        super(AppearanceWeightsNewton, self).__init__(
            aam_interface, n_iterations=n_iterations,
            compute_error=compute_error)

        self._regressor_cls = partial(IRLRegression, alpha=alpha,
                                      bias=bias)


class AppearanceWeightsGaussNewton(AppearanceWeights):
    r"""
    Class for training a cascaded-regression Gauss-Newton algorithm using
    Indirect Incremental Regularized Linear Regression (:map:`IIRLRegression`)
    given a trained AAM model. The algorithm uses the projection weights of
    the appearance vectors as features in the regression.

    Parameters
    ----------
    aam_interface : The AAM interface class from `menpofit.aam.algorithm.lk`.
        Existing interfaces include:

            ============================== =============================
            'LucasKanadeStandardInterface' Suitable for holistic AAMs
            'LucasKanadeLinearInterface'   Suitable for linear AAMs
            'LucasKanadePatchInterface'    Suitable for patch-based AAMs
            ============================== =============================

    n_iterations : `int`, optional
        The number of iterations (cascades).
    compute_error : `callable`, optional
        The function to be used for computing the fitting error when training
        each cascade.
    alpha : `float`, optional
        The regularization parameter.
    alpha2 : `float`, optional
        The regularization parameter of the Hessian matrix.
    bias : `bool`, optional
        Flag that controls whether to use a bias term.
    """
    def __init__(self, aam_interface, n_iterations=3,
                 compute_error=euclidean_bb_normalised_error,
                 alpha=0, alpha2=0, bias=True):
        super(AppearanceWeightsGaussNewton, self).__init__(
            aam_interface, n_iterations=n_iterations,
            compute_error=compute_error)

        self._regressor_cls = partial(IIRLRegression, alpha=alpha,
                                      alpha2=alpha2, bias=bias)
