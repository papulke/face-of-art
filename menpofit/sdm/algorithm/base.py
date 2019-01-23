from __future__ import division
from functools import partial
import numpy as np

from menpo.visualize import print_dynamic

from menpofit.fitter import raise_costs_warning
from menpofit.visualize import print_progress
from menpofit.result import (NonParametricIterativeResult,
                             ParametricIterativeResult)


class BaseSupervisedDescentAlgorithm(object):
    r"""
    Abstract class for defining a Supervised Descent algorithm.
    """
    @property
    def _multi_scale_fitter_result(self):
        raise NotImplementedError()

    def train(self, images, gt_shapes, current_shapes, prefix='',
              verbose=False):
        r"""
        Method to train the model given a set of initial shapes.

        Parameters
        ----------
        images : `list` of `menpo.image.Image`
            The `list` of training images.
        gt_shapes : `list` of `menpo.shape.PointCloud`
            The `list` of ground truth shapes that correspond to the images.
        current_shapes : `list` of `menpo.shape.PointCloud`
            The `list` of current shapes that correspond to the images,
            which will be used as initial shapes.
        prefix : `str`, optional
            The prefix to use when printing information.
        verbose : `bool`, optional
            If ``True``, then information is printed during training.

        Returns
        -------
        current_shapes : `list` of `menpo.shape.PointCloud`
            The `list` of current shapes that correspond to the images.
        """
        return self._train(images, gt_shapes, current_shapes, increment=False,
                           prefix=prefix, verbose=verbose)

    def increment(self, images, gt_shapes, current_shapes, prefix='',
                  verbose=False):
        r"""
        Method to increment the model with the set of current shapes.

        Parameters
        ----------
        images : `list` of `menpo.image.Image`
            The `list` of training images.
        gt_shapes : `list` of `menpo.shape.PointCloud`
            The `list` of ground truth shapes that correspond to the images.
        current_shapes : `list` of `menpo.shape.PointCloud`
            The `list` of current shapes that correspond to the images.
        prefix : `str`, optional
            The prefix to use when printing information.
        verbose : `bool`, optional
            If ``True``, then information is printed during training.

        Returns
        -------
        current_shapes : `list` of `menpo.shape.PointCloud`
            The `list` of current shapes that correspond to the images.
        """
        return self._train(images, gt_shapes, current_shapes, increment=True,
                           prefix=prefix, verbose=verbose)

    def _train(self, images, gt_shapes, current_shapes, increment=False,
               prefix='', verbose=False):
        if not increment:
            # Reset the regressors
            self.regressors = []
        elif increment and not (hasattr(self, 'regressors') and self.regressors):
            raise ValueError('Algorithm must be trained before it can be '
                             'incremented.')

        n_perturbations = len(current_shapes[0])
        template_shape = gt_shapes[0]

        # obtain delta_x and gt_x
        delta_x, gt_x = self._compute_delta_x(gt_shapes, current_shapes)

        # Cascaded Regression loop
        for k in range(self.n_iterations):
            # generate regression data
            features_prefix = '{}(Iteration {}) - '.format(prefix, k)
            features = self._compute_training_features(images, gt_shapes,
                                                       current_shapes,
                                                       prefix=features_prefix,
                                                       verbose=verbose)

            if verbose:
                print_dynamic('{}(Iteration {}) - Performing regression'.format(
                    prefix, k))

            if not increment:
                r = self._regressor_cls()
                r.train(features, delta_x)
                self.regressors.append(r)
            else:
                self.regressors[k].increment(features, delta_x)

            # Estimate delta_points
            estimated_delta_x = self.regressors[k].predict(features)
            if verbose:
                self._print_regression_info(template_shape, gt_shapes,
                                            n_perturbations, delta_x,
                                            estimated_delta_x, k,
                                            prefix=prefix)

            self._update_estimates(estimated_delta_x, delta_x, gt_x,
                                   current_shapes)

        return current_shapes

    def _compute_delta_x(self, gt_shapes, current_shapes):
        raise NotImplementedError()

    def _update_estimates(self, estimated_delta_x, delta_x, gt_x,
                          current_shapes):
        raise NotImplementedError()

    def _compute_training_features(self, images, gt_shapes, current_shapes,
                                   prefix='', verbose=False):
        raise NotImplementedError()

    def _compute_test_features(self, image, current_shape):
        raise NotImplementedError()

    def run(self, image, initial_shape, gt_shape=None, return_costs=False,
            **kwargs):
        r"""
        Run the predictor to an image given an initial shape.

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
        """
        raise NotImplementedError()

    def _print_regression_info(self, template_shape, gt_shapes, n_perturbations,
                               delta_x, estimated_delta_x, level_index,
                               prefix=''):
        raise NotImplementedError()


def features_per_patch(image, shape, patch_shape, features_callable):
    r"""
    Method that first extracts patches and then features from these patches.

    Parameters
    ----------
    image : `menpo.image.Image` or subclass
        The input image.
    shape : `menpo.shape.PointCloud`
        The points from which to extract the patches.
    patch_shape : (`int`, `int`)
        The shape of the patch to be extracted.
    features_callable : `callable`
        The function to be used for extracting features.

    Returns
    -------
    features_per_patch : 1D `ndarray`
        The concatenated features.
    """
    patches = image.extract_patches(shape, patch_shape=patch_shape,
                                    as_single_array=True)
    patch_features = [features_callable(p[0]).ravel() for p in patches]
    return np.hstack(patch_features)


def features_per_shapes(image, shapes, patch_shape, features_callable):
    r"""
    Method that given multiple shapes for an image, it first extracts patches
    that correspond to the shapes and then features from these patches.

    Parameters
    ----------
    image : `menpo.image.Image` or subclass
        The input image.
    shapes : `list` of `menpo.shape.PointCloud`
        The list of shapes from which to extract the patches.
    patch_shape : (`int`, `int`)
        The shape of the patch to be extracted.
    features_callable : `callable`
        The function to be used for extracting features.

    Returns
    -------
    features_per_shapes : ``(n_shapes, n_features)`` `ndarray`
        The concatenated feature vector per shape.
    """
    patch_features = [features_per_patch(image, s, patch_shape,
                                         features_callable)
                      for s in shapes]
    return np.vstack(patch_features)


def features_per_image(images, shapes, patch_shape, features_callable,
                       prefix='', verbose=False):
    r"""
    Method that given multiple images with multiple shapes per image, it first
    extracts patches that correspond to the shapes and then features from
    these patches.

    Parameters
    ----------
    images : `list` of `menpo.image.Image` or subclass
        The input images.
    shapes : `list` of `list` of `menpo.shape.PointCloud`
        The list of list of shapes per image from which to extract the patches.
    patch_shape : (`int`, `int`)
        The shape of the patch to be extracted.
    features_callable : `callable`
        The function to be used for extracting features.
    prefix : `str`, optional
        The prefix of the printed information.
    verbose : `bool`, optional
        If ``True``, then progress information is printed.

    Returns
    -------
    features_per_image : ``(n_images * n_shapes, n_features)`` `ndarray`
        The concatenated feature vector per image and per shape.
    """
    wrap = partial(print_progress,
                   prefix='{}Extracting patches'.format(prefix),
                   end_with_newline=not prefix, verbose=verbose)
    patch_features = [features_per_shapes(i, shapes[j], patch_shape,
                                          features_callable)
                      for j, i in enumerate(wrap(images))]
    return np.vstack(patch_features)


def compute_non_parametric_delta_x(gt_shapes, current_shapes):
    r"""
    Method that computes the difference between a given set of current shapes
    and the corresponding ground truth shapes.

    Parameters
    ----------
    gt_shapes : `list` of `menpo.shape.PointCloud`
        The ground truth shapes.
    current_shapes : `list` of `list` of `menpo.shape.PointCloud`
        The list of list of current shapes that correspond to each ground truth
        shape.

    Returns
    -------
    delta_x : ``(n_gt_shapes * n_current_shapes, n_features)`` `ndarray`
        The concatenated difference vectors per ground truth shape.
    gt_x : ``(n_gt_shapes * n_current_shapes, n_features)`` `ndarray`
        The ground truth shape vectors.
    """
    n_x = gt_shapes[0].n_parameters
    n_gt_shapes = len(gt_shapes)
    n_current_shapes = len(current_shapes[0])

    # initialize current, ground truth and delta parameters
    gt_x = np.empty((n_gt_shapes * n_current_shapes, n_x))
    delta_x = np.empty((n_gt_shapes * n_current_shapes, n_x))

    # obtain ground truth points and compute delta points
    k = 0
    for gt_s, shapes in zip(gt_shapes, current_shapes):
        c_gt_s = gt_s.as_vector()
        for s in shapes:
            # compute ground truth shape vector
            gt_x[k] = c_gt_s
            # compute delta shape vector
            delta_x[k] = c_gt_s - s.as_vector()
            k += 1

    return delta_x, gt_x


def update_non_parametric_estimates(estimated_delta_x, delta_x, gt_x,
                                    current_shapes):
    r"""
    Method that updates the current shapes given the current estimation of the
    increments. Note that the `current_shapes` and `delta_x` get updated inplace.

    Parameters
    ----------
    estimated_delta_x : ``(n_gt_shapes * n_current_shapes, n_features)`` `ndarray`
        The estimated increments of the shapes.
    delta_x : ``(n_gt_shapes * n_current_shapes, n_features)`` `ndarray`
        The current shape increments.
    gt_x : ``(n_gt_shapes * n_current_shapes, n_features)`` `ndarray`
        The ground truth shape vectors.
    current_shapes : `list` of `list` of `menpo.shape.PointCloud`
        The list of list of current shapes that correspond to each ground truth
        shape.
    """
    j = 0
    for shapes in current_shapes:
        for s in shapes:
            # update current x
            current_x = s.as_vector() + estimated_delta_x[j]
            # update current shape inplace
            s._from_vector_inplace(current_x)
            # update delta_x
            delta_x[j] = gt_x[j] - current_x
            # increase index
            j += 1


def print_non_parametric_info(template_shape, gt_shapes, n_perturbations,
                              delta_x, estimated_delta_x, level_index,
                              compute_error_f, prefix=''):
    r"""
    Method that prints information when training a non-parametric cascaded
    regression algorithm.

    Parameters
    ----------
    template_shape : `menpo.shape.PointCloud`
        The template shape used to construct the shape of increments.
    gt_shapes : `list` of `menpo.shape.PointCloud`
        The ground truth shapes.
    n_perturbations : `int`
        The number of perturbations applied on the ground truth shapes.
    delta_x : `list` of ``(2 * n_points,)`` `ndarray`
        The `list` of vectors with the shape increments between the current
        shapes and the ground truth.
    estimated_delta_x : `list` of ``(2 * n_points,)`` `ndarray`
        The `list` of vectors with the estimated shape increments.
    level_index : `int`
        The scale index.
    compute_error_f : `callable`
        The function to be used for computing the error.
    prefix : `str`
        The prefix attached to the printed information.
    """
    print_dynamic('{}(Iteration {}) - Calculating errors'.format(
        prefix, level_index))
    errors = []
    for j, (dx, edx) in enumerate(zip(delta_x, estimated_delta_x)):
        s1 = template_shape.from_vector(dx)
        s2 = template_shape.from_vector(edx)
        gt_s = gt_shapes[np.floor_divide(j, n_perturbations)]
        errors.append(compute_error_f(s1, s2, gt_s))
    mean = np.mean(errors)
    std = np.std(errors)
    median = np.median(errors)
    print_dynamic('{}(Iteration {}) - Training error -> '
                  'mean: {:.4f}, std: {:.4f}, median: {:.4f}.\n'.
                  format(prefix, level_index, mean, std, median))


def print_parametric_info(model, gt_shapes, n_perturbations,
                          delta_x, estimated_delta_x, level_index,
                          compute_error_f, prefix=''):
    r"""
    Method that prints information when training a parametric cascaded regression
    algorithm.

    Parameters
    ----------
    model : `menpofit.modelinstance.OrthoPDM`
        The shape model.
    gt_shapes : `list` of `menpo.shape.PointCloud`
        The ground truth shapes.
    n_perturbations : `int`
        The number of perturbations applied on the ground truth shapes.
    delta_x : `list` of ``(2 * n_points,)`` `ndarray`
        The `list` of vectors with the shape increments between the current
        shapes and the ground truth.
    estimated_delta_x : `list` of ``(2 * n_points,)`` `ndarray`
        The `list` of vectors with the estimated shape increments.
    level_index : `int`
        The scale index.
    compute_error_f : `callable`
        The function to be used for computing the error.
    prefix : `str`
        The prefix attached to the printed information.
    """
    print_dynamic('{}(Iteration {}) - Calculating errors'.format(
        prefix, level_index))
    errors = []
    for j, (dx, edx) in enumerate(zip(delta_x, estimated_delta_x)):
        model._from_vector_inplace(dx)
        s1 = model.target
        model._from_vector_inplace(edx)
        s2 = model.target

        gt_s = gt_shapes[np.floor_divide(j, n_perturbations)]
        errors.append(compute_error_f(s1, s2, gt_s))
    mean = np.mean(errors)
    std = np.std(errors)
    median = np.median(errors)
    print_dynamic('{}(Iteration {}) - Training error -> '
                  'mean: {:.4f}, std: {:.4f}, median: {:.4f}.\n'.
                  format(prefix, level_index, mean, std, median))


def compute_parametric_delta_x(gt_shapes, current_shapes, model):
    r"""
    Method that, given a linear parametric model, computes the difference
    between a set of current shape parameters and the corresponding ground truth
    shape parameters.

    Parameters
    ----------
    gt_shapes : `list` of `menpo.shape.PointCloud`
        The ground truth shapes from which the parameters will be extracted.
    current_shapes : `list` of `list` of `menpo.shape.PointCloud`
        The list of list of current shapes that correspond to each ground truth
        shape.
    model : `menpofit.modelinstance.OrthoPDM`
        A parametric model used to get the parameters of the ground truth shapes
        and current shapes.

    Returns
    -------
    delta_params : ``(n_gt_shapes * n_current_shapes, n_parameters)`` `ndarray`
        The concatenated parameters difference vectors per ground truth shape.
    gt_params : ``(n_gt_shapes * n_current_shapes, n_parameters)`` `ndarray`
        The ground truth parameters vectors.
    """
    # initialize current and delta parameters arrays
    n_samples = len(gt_shapes) * len(current_shapes[0])
    gt_params = np.empty((n_samples, model.n_parameters))
    delta_params = np.empty_like(gt_params)

    k = 0
    for gt_s, c_s in zip(gt_shapes, current_shapes):
        # Compute and cache ground truth parameters
        model.set_target(gt_s)
        c_gt_params = model.as_vector()
        for s in c_s:
            gt_params[k] = c_gt_params

            model.set_target(s)
            current_params = model.as_vector()
            delta_params[k] = c_gt_params - current_params

            k += 1

    return delta_params, gt_params


def update_parametric_estimates(estimated_delta_x, delta_x, gt_x,
                                current_shapes, model):
    r"""
    Method that updates the current shape parameters given the current estimation
    of the parameters increments. Note that the `current_shapes` and `delta_x`
    get updated inplace.

    Parameters
    ----------
    estimated_delta_x : ``(n_gt_shapes * n_current_shapes, n_parameters)`` `ndarray`
        The estimated parameters increments of the shapes.
    delta_x : ``(n_gt_shapes * n_current_shapes, n_parameters)`` `ndarray`
        The current shape parameters increments.
    gt_x : ``(n_gt_shapes * n_current_shapes, n_features)`` `ndarray`
        The ground truth shape parameters.
    current_shapes : `list` of `list` of `menpo.shape.PointCloud`
        The list of list of current shapes that correspond to each ground truth
        shape.
    model : `menpofit.modelinstance.OrthoPDM`
        A parametric model used to get the parameters of the ground truth shapes
        and current shapes.
    """
    j = 0
    for shapes in current_shapes:
        for s in shapes:
            # Estimate parameters
            edx = estimated_delta_x[j]
            # Current parameters
            model.set_target(s)
            cx = model.as_vector() + edx
            model._from_vector_inplace(cx)

            # Update current shape inplace
            s._from_vector_inplace(model.target.as_vector().copy())

            delta_x[j] = gt_x[j] - cx
            j += 1


def build_appearance_model(images, gt_shapes, patch_shape, patch_features,
                           appearance_model_cls, verbose=False, prefix=''):
    r"""
    Method that builds a parametric patch-based appearance model.

    Parameters
    ----------
    images : `list` of `menpo.image.Image`
        The `list` of training images.
    gt_shapes : `list` of `menpo.shape.PointCloud`
        The `list` of ground truth shapes that correspond to the training shapes.
    patch_shape : (`int`, `int`)
        The shape of the extracted patches.
    patch_features : `callable`
        The function to extract features from the patches. Please refer to
        `menpo.feature` or `menpofit.feature`.
    appearance_model_cls : `menpo.model.PCAModel`
        The class that will be used to train the model, e.g.
        `menpo.model.PCAModel`.
    verbose : `bool`, optional
        If ``True``, then information about the training progress will be
        printed.
    prefix : `str`, optional
        The prefix used in the printed information.
    """
    wrap = partial(print_progress,
                   prefix='{}Extracting ground truth patches'.format(prefix),
                   end_with_newline=not prefix, verbose=verbose)
    n_images = len(images)
    # Extract patches from ground truth
    gt_patches = [features_per_patch(im, gt_s, patch_shape,
                                     patch_features)
                  for gt_s, im in wrap(list(zip(gt_shapes, images)))]
    # Calculate appearance model from extracted gt patches
    gt_patches = np.array(gt_patches).reshape([n_images, -1])
    if verbose:
        print_dynamic('{}Building Appearance Model'.format(prefix))
    return appearance_model_cls(gt_patches)


def fit_parametric_shape(image, initial_shape, parametric_algorithm,
                         gt_shape=None, return_costs=False):
    r"""
    Method that fits a parametric cascaded regression algorithm to an image.

    Parameters
    ----------
    image : `menpo.image.Image`
        The input image.
    initial_shape : `menpo.shape.PointCloud`
        The initial estimation of the shape.
    parametric_algorithm : `class`
        A cascaded regression algorithm that employs a parametric shape model.
        Please refer to `menpofit.sdm.algorithm`.
    gt_shape : `menpo.shape.PointCloud`
        The ground truth shape that corresponds to the image.
    return_costs : `bool`, optional
        If ``True``, then the cost function values will be computed during
        the fitting procedure. Then these cost values will be assigned to the
        returned `fitting_result`. *Note that this argument currently has no
        effect and will raise a warning if set to ``True``. This is because
        it is not possible to evaluate the cost function of this algorithm.*

    Returns
    -------
    fitting_result : :map:`ParametricIterativeResult`
        The final fitting result.
    """
    # costs warning
    if return_costs:
        raise_costs_warning(parametric_algorithm)

    # set current shape and initialize list of shapes
    parametric_algorithm.shape_model.set_target(initial_shape)
    current_shape = initial_shape.from_vector(
            parametric_algorithm.shape_model.target.as_vector().copy())
    shapes = []
    shape_parameters = [parametric_algorithm.shape_model.as_vector()]

    # Cascaded Regression loop
    for r in parametric_algorithm.regressors:
        # compute regression features
        features = parametric_algorithm._compute_test_features(image,
                                                               current_shape)

        # solve for increments on the shape vector
        dx = r.predict(features).ravel()

        # update current shape
        p = parametric_algorithm.shape_model.as_vector() + dx
        parametric_algorithm.shape_model._from_vector_inplace(p)
        current_shape = current_shape.from_vector(
                parametric_algorithm.shape_model.target.as_vector().copy())
        shapes.append(current_shape)
        shape_parameters.append(p)

    # return algorithm result
    return ParametricIterativeResult(
            shapes=shapes, shape_parameters=shape_parameters,
            initial_shape=initial_shape, image=image, gt_shape=gt_shape)


def fit_non_parametric_shape(image, initial_shape, non_parametric_algorithm,
                             gt_shape=None, return_costs=False):
    r"""
    Method that fits a non-parametric cascaded regression algorithm to an image.

    Parameters
    ----------
    image : `menpo.image.Image`
        The input image.
    initial_shape : `menpo.shape.PointCloud`
        The initial estimation of the shape.
    non_parametric_algorithm : `class`
        A cascaded regression algorithm that does not use a parametric shape
        model. Please refer to `menpofit.sdm.algorithm`.
    gt_shape : `menpo.shape.PointCloud`
        The ground truth shape that corresponds to the image.
    return_costs : `bool`, optional
        If ``True``, then the cost function values will be computed during
        the fitting procedure. Then these cost values will be assigned to the
        returned `fitting_result`. *Note that this argument currently has no
        effect and will raise a warning if set to ``True``. This is because
        it is not possible to evaluate the cost function of this algorithm.*

    Returns
    -------
    fitting_result : :map:`NonParametricIterativeResult`
        The final fitting result.
    """
    # costs warning
    if return_costs:
        raise_costs_warning(non_parametric_algorithm)

    # set current shape and initialize list of shapes
    current_shape = initial_shape
    shapes = []

    # Cascaded Regression loop
    for r in non_parametric_algorithm.regressors:
        # compute regression features
        features = non_parametric_algorithm._compute_test_features(image,
                                                                   current_shape)

        # solve for increments on the shape vector
        dx = r.predict(features)

        # update current shape
        current_shape = current_shape.from_vector(
            current_shape.as_vector() + dx)
        shapes.append(current_shape)

    # return algorithm result
    return NonParametricIterativeResult(
            shapes=shapes, initial_shape=initial_shape, image=image,
            gt_shape=gt_shape)
