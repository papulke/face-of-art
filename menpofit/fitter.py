from __future__ import division
from functools import partial
import numpy as np
import scipy
import warnings

from menpo.base import name_of_callable
from menpo.shape import PointCloud
from menpo.transform import (scale_about_centre, rotate_ccw_about_centre,
                             Translation, Scale, AlignmentAffine,
                             AlignmentSimilarity)

from menpofit.base import MenpoFitCostsWarning
import menpofit.checks as checks
from menpofit.visualize import print_progress
from menpofit.result import (MultiScaleNonParametricIterativeResult,
                             MultiScaleParametricIterativeResult)


def raise_costs_warning(cls):
    r"""
    Method for raising a warning in case the costs for a selected
    optimisation class cannot be computed.

    Parameters
    ----------
    cls : `class`
        The optimisation (fitting) class.
    """
    cls_name = name_of_callable(cls)
    warnings.warn("costs cannot be computed for {}".format(cls_name),
                  MenpoFitCostsWarning)


def noisy_alignment_similarity_transform(source, target, noise_type='uniform',
                                         noise_percentage=0.1,
                                         allow_alignment_rotation=False):
    r"""
    Constructs and perturbs the optimal similarity transform between the source
    and target shapes by adding noise to its parameters.

    Parameters
    ----------
    source : `menpo.shape.PointCloud`
        The source pointcloud instance used in the alignment
    target : `menpo.shape.PointCloud`
        The target pointcloud instance used in the alignment
    noise_type : ``{'uniform', 'gaussian'}``, optional
        The type of noise to be added.
    noise_percentage : `float` in ``(0, 1)`` or `list` of `len` `3`, optional
        The standard percentage of noise to be added. If `float`, then the same
        amount of noise is applied to the scale, rotation and translation
        parameters of the optimal similarity transform. If `list` of
        `float` it must have length 3, where the first, second and third elements
        denote the amount of noise to be applied to the scale, rotation and
        translation parameters, respectively.
    allow_alignment_rotation : `bool`, optional
        If ``False``, then the rotation is not considered when computing the
        optimal similarity transform between source and target.

    Returns
    -------
    noisy_alignment_similarity_transform : `menpo.transform.Similarity`
        The noisy Similarity Transform between source and target.
    """
    if isinstance(noise_percentage, float):
        noise_percentage = [noise_percentage] * 3
    elif len(noise_percentage) == 1:
        noise_percentage *= 3

    similarity = AlignmentSimilarity(source, target,
                                     rotation=allow_alignment_rotation)

    if noise_type is 'gaussian':
        s = noise_percentage[0] * (0.5 / 3) * np.asscalar(np.random.randn(1))
        r = noise_percentage[1] * (180 / 3) * np.asscalar(np.random.randn(1))
        t = noise_percentage[2] * (target.range() / 3) * np.random.randn(2)

        s = scale_about_centre(target, 1 + s)
        r = rotate_ccw_about_centre(target, r)
        t = Translation(t, source.n_dims)
    elif noise_type is 'uniform':
        s = noise_percentage[0] * 0.5 * (2 * np.asscalar(np.random.randn(1)) - 1)
        r = noise_percentage[1] * 180 * (2 * np.asscalar(np.random.rand(1)) - 1)
        t = noise_percentage[2] * target.range() * (2 * np.random.rand(2) - 1)

        s = scale_about_centre(target, 1. + s)
        r = rotate_ccw_about_centre(target, r)
        t = Translation(t, source.n_dims)
    else:
        raise ValueError('Unexpected noise type. '
                         'Supported values are {gaussian, uniform}')

    return similarity.compose_after(t.compose_after(s.compose_after(r)))


def noisy_target_alignment_transform(source, target,
                                     alignment_transform_cls=AlignmentAffine,
                                     noise_std=0.1, **kwargs):
    r"""
    Constructs the optimal alignment transform between the source and a noisy
    version of the target obtained by adding white noise to each of its points.

    Parameters
    ----------
    source : `menpo.shape.PointCloud`
        The source pointcloud instance used in the alignment
    target : `menpo.shape.PointCloud`
        The target pointcloud instance used in the alignment
    alignment_transform_cls : `menpo.transform.Alignment`, optional
        The alignment transform class used to perform the alignment.
    noise_std : `float` or `list` of `float`, optional
        The standard deviation of the white noise to be added to each one of
        the target points. If `float`, then the same standard deviation is used
        for all points. If `list`, then it must define a value per point.

    Returns
    -------
    noisy_transform : `menpo.transform.Alignment`
        The noisy Similarity Transform
    """
    noise = noise_std * target.range() * np.random.randn(target.n_points,
                                                         target.n_dims)
    noisy_target = PointCloud(target.points + noise)
    return alignment_transform_cls(source, noisy_target, **kwargs)


def noisy_shape_from_bounding_box(shape, bounding_box, noise_type='uniform',
                                  noise_percentage=0.05,
                                  allow_alignment_rotation=False):
    r"""
    Constructs and perturbs the optimal similarity transform between the bounding
    box of the source shape and the target bounding box, by adding noise to its
    parameters. It returns the noisy version of the provided shape.

    Parameters
    ----------
    shape : `menpo.shape.PointCloud`
        The source pointcloud instance used in the alignment. Note that the
        bounding box of the shape will be used.
    bounding_box : `menpo.shape.PointDirectedGraph`
        The target bounding box instance used in the alignment
    noise_type : ``{'uniform', 'gaussian'}``, optional
        The type of noise to be added.
    noise_percentage : `float` in ``(0, 1)`` or `list` of `len` `3`, optional
        The standard percentage of noise to be added. If `float`, then the same
        amount of noise is applied to the scale, rotation and translation
        parameters of the optimal similarity transform. If `list` of
        `float` it must have length 3, where the first, second and third elements
        denote the amount of noise to be applied to the scale, rotation and
        translation parameters, respectively.
    allow_alignment_rotation : `bool`, optional
        If ``False``, then the rotation is not considered when computing the
        optimal similarity transform between source and target.

    Returns
    -------
    noisy_shape : `menpo.shape.PointCloud`
        The noisy shape.
    """
    transform = noisy_alignment_similarity_transform(
            shape.bounding_box(), bounding_box, noise_type=noise_type,
            noise_percentage=noise_percentage,
            allow_alignment_rotation=allow_alignment_rotation)
    return transform.apply(shape)


def noisy_shape_from_shape(reference_shape, shape, noise_type='uniform',
                           noise_percentage=0.05,
                           allow_alignment_rotation=False):
    r"""
    Constructs and perturbs the optimal similarity transform between the
    provided reference shape and the target shape, by adding noise to its
    parameters. It returns the noisy version of the reference shape.

    Parameters
    ----------
    reference_shape : `menpo.shape.PointCloud`
        The source reference shape instance used in the alignment.
    shape : `menpo.shape.PointDirectedGraph`
        The target shape instance used in the alignment
    noise_type : ``{'uniform', 'gaussian'}``, optional
        The type of noise to be added.
    noise_percentage : `float` in ``(0, 1)`` or `list` of `len` `3`, optional
        The standard percentage of noise to be added. If `float`, then the same
        amount of noise is applied to the scale, rotation and translation
        parameters of the optimal similarity transform. If `list` of
        `float` it must have length 3, where the first, second and third elements
        denote the amount of noise to be applied to the scale, rotation and
        translation parameters, respectively.
    allow_alignment_rotation : `bool`, optional
        If ``False``, then the rotation is not considered when computing the
        optimal similarity transform between source and target.

    Returns
    -------
    noisy_reference_shape : `menpo.shape.PointCloud`
        The noisy reference shape.
    """
    transform = noisy_alignment_similarity_transform(
            reference_shape, shape, noise_type=noise_type,
            noise_percentage=noise_percentage,
            allow_alignment_rotation=allow_alignment_rotation)
    return transform.apply(reference_shape)


def align_shape_with_bounding_box(shape, bounding_box,
                                  alignment_transform_cls=AlignmentSimilarity,
                                  **kwargs):
    r"""
    Aligns the provided shape with the bounding box using a particular alignment
    transform.

    Parameters
    ----------
    shape : `menpo.shape.PointCloud`
        The shape instance used in the alignment.
    bounding_box : `menpo.shape.PointDirectedGraph`
        The bounding box instance used in the alignment.
    alignment_transform_cls : `menpo.transform.Alignment`, optional
        The class of the alignment transform used to perform the alignment.

    Returns
    -------
    noisy_shape : `menpo.shape.PointCloud`
        The noisy shape
    """
    shape_bb = shape.bounding_box()
    transform = alignment_transform_cls(shape_bb, bounding_box, **kwargs)
    return transform.apply(shape)


class MultiScaleNonParametricFitter(object):
    r"""
    Class for defining a multi-scale fitter for a non-parametric fitting method,
    i.e. a method that does not optimise over a parametric shape model.

    Parameters
    ----------
    scales : `list` of `int` or `float`
        The scale value of each scale. They must provided in ascending order,
        i.e. from lowest to highest scale.
    reference_shape : `menpo.shape.PointCloud`
        The reference shape that will be used to normalise the size of an input
        image so that the scale of its initial fitting shape matches the scale of
        the reference shape.
    holistic_features : `list` of `closure`
        The features that will be extracted from the input image at each scale.
        They must provided in ascending order, i.e. from lowest to highest scale.
    algorithms : `list` of `class`
        The list of algorithm objects that will perform the fitting per scale.
    """
    def __init__(self, scales, reference_shape, holistic_features, algorithms):
        self._scales = scales
        self._reference_shape = reference_shape
        self._holistic_features = holistic_features
        self.algorithms = algorithms

    @property
    def scales(self):
        r"""
        The scale value of each scale in ascending order, i.e. from lowest to
        highest scale.

        :type: `list` of `int` or `float`
        """
        return self._scales

    @property
    def n_scales(self):
        r"""
        Returns the number of scales.

        :type: `int`
        """
        return len(self.scales)

    @property
    def reference_shape(self):
        r"""
        The reference shape that is used to normalise the size of an input image
        so that the scale of its initial fitting shape matches the scale of this
        reference shape.

        :type: `menpo.shape.PointCloud`
        """
        return self._reference_shape

    @property
    def holistic_features(self):
        r"""
        The features that are extracted from the input image at each scale in
        ascending order, i.e. from lowest to highest scale.

        :type: `list` of `closure`
        """
        return self._holistic_features

    def _prepare_image(self, image, initial_shape, gt_shape=None):
        r"""
        Function the performs pre-processing on the image to be fitted. This
        involves the following steps:

            1. Rescale image wrt the scale factor between the reference_shape
               and the initial_shape.
            2. For each scale:
                  3. Compute features
                  4. Estimate the affine transform introduced by the rescale to
                     reference shape and features extraction
                  5. Rescale image
                  6. Save affine transform, scale transform and final image

        Parameters
        ----------
        image : `menpo.image.Image` or subclass
            The image to be fitted.
        initial_shape : `menpo.shape.PointCloud`
            The initial shape estimate from which the fitting procedure
            will start.
        gt_shape : `menpo.shape.PointCloud`, optional
            The ground truth shape associated to the image.

        Returns
        -------
        images : `list` of `menpo.image.Image`
            The list of images per scale.
        initial_shapes : `list` of `menpo.shape.PointCloud`
            The list of initial shapes per scale.
        gt_shapes : `list` of `menpo.shape.PointCloud`
            The list of ground truth shapes per scale.
        affine_transforms : `list` of `menpo.transform.Affine`
            The list of affine transforms per scale that are the inverses of the
            transformations introduced by the rescale wrt the reference shape as
            well as the feature extraction.
        scale_transforms : `list` of `menpo.shape.Scale`
            The list of inverse scaling transforms per scale.
        """
        # Attach landmarks to the image, in order to make transforms easier
        image.landmarks['__initial_shape'] = initial_shape
        if gt_shape:
            image.landmarks['__gt_shape'] = gt_shape

        # Rescale image wrt the scale factor between reference_shape and
        # initial_shape
        #tmp_image = image.rescale_to_pointcloud(self.reference_shape,
        #                                      group='__initial_shape')

        tmp_image = image.rescale_to_pointcloud(gt_shape,
                                                group='__gt_shape')
        #tmp_image = image

        # For each scale:
        #     1. Compute features
        #     2. Estimate the affine transform introduced by the rescale to
        #        reference shape and features extraction
        #     2. Rescale image
        #     3. Save affine transform, scale transform and final image
        images = []
        affine_transforms = []
        scale_transforms = []
        for i in range(self.n_scales):
            # Extract features
            if (i == 0 or
                    self.holistic_features[i] != self.holistic_features[i - 1]):
                # Compute features only if this is the first pass through
                # the loop or the features at this scale are different from
                # the features at the previous scale
                feature_image = self.holistic_features[i](tmp_image)

                # Until now, we have introduced an affine transform that
                # consists of the image rescale to the reference shape,
                # as well as potential rescale (down-sampling) caused by
                # features. We need to store this transform (estimated by
                # AlignmentAffine) in order to be able to revert it at the
                # final fitting result.
                affine_transforms.append(AlignmentAffine(
                    feature_image.landmarks['__initial_shape'].lms,
                    initial_shape))
            else:
                # If features are not extracted, then the affine transform
                # should be identical with the one of the first (lowest) level.
                affine_transforms.append(affine_transforms[0])

            # Rescale images according to scales
            if self.scales[i] != 1:
                # Scale feature images only if scale is different than 1
                scaled_image, scale_transform = feature_image.rescale(
                    self.scales[i], return_transform=True)
            else:
                # Otherwise the image remains the same and the transform is the
                # identity matrix.
                scaled_image = feature_image
                scale_transform = Scale(1., initial_shape.n_dims)

            # rewrite response map data
            # scaled_image.rspmap_data = scipy.ndimage.zoom(image.rspmap_data, zoom=[1, 1, float(scaled_image.height) / image.rspmap_data.shape[-2],
            #                                                               float(scaled_image.width) / image.rspmap_data.shape[-1]], order=1)  # mode = 'nearest'

            scaled_image.rspmap_data = image.rspmap_data
            # Add scale transform to list
            scale_transforms.append(scale_transform)

            scaled_image.path = image.path
            # Add scaled image to list
            images.append(scaled_image)

        # Get initial shapes per level
        initial_shapes = [i.landmarks['__initial_shape'].lms for i in images]

        # Get ground truth shapes per level
        if gt_shape:
            gt_shapes = [i.landmarks['__gt_shape'].lms for i in images]
        else:
            gt_shapes = None

        # Detach added landmarks from image
        del image.landmarks['__initial_shape']
        if gt_shape:
            del image.landmarks['__gt_shape']

        return (images, initial_shapes, gt_shapes, affine_transforms,
                scale_transforms)

    def _fit(self, images, initial_shape, affine_transforms, scale_transforms,
             gt_shapes=None, max_iters=20, return_costs=False, **kwargs):
        r"""
        Function the applies the multi-scale fitting procedure on an image, given
        the initial shape.

        Parameters
        ----------
        images : `list` of `menpo.image.Image`
            The list of images per scale.
        initial_shape : `menpo.shape.PointCloud`
            The initial shape estimate from which the fitting procedure
            will start.
        affine_transforms : `list` of `menpo.transform.Affine`
            The list of affine transforms per scale that are the inverses of the
            transformations introduced by the rescale wrt the reference shape as
            well as the feature extraction.
        scale_transforms : `list` of `menpo.shape.Scale`
            The list of inverse scaling transforms per scale.
        gt_shapes : `list` of `menpo.shape.PointCloud`
            The list of ground truth shapes per scale.
        max_iters : `int` or `list` of `int`, optional
            The maximum number of iterations. If `int`, then it specifies the
            maximum number of iterations over all scales. If `list` of `int`,
            then specifies the maximum number of iterations per scale.
        return_costs : `bool`, optional
            If ``True``, then the cost function values will be computed
            during the fitting procedure. Then these cost values will be
            assigned to the returned `fitting_result`. *Note that the costs
            computation increases the computational cost of the fitting. The
            additional computation cost depends on the fitting method. Only
            use this option for research purposes.*
        kwargs : `dict`, optional
            Additional keyword arguments that can be passed to specific
            implementations.

        Returns
        -------
        algorithm_results : `list` of :map:`NonParametricIterativeResult` or subclass
            The list of fitting result per scale.
        """
        # Check max iters
        max_iters = checks.check_max_iters(max_iters, self.n_scales)

        # Set initial and ground truth shapes
        shape = initial_shape
        gt_shape = None

        # Initialize list of algorithm results
        algorithm_results = []
        for i in range(self.n_scales):
            # Handle ground truth shape
            if gt_shapes is not None:
                gt_shape = gt_shapes[i]

            # Run algorithm
            algorithm_result = self.algorithms[i].run(images[i], shape,
                                                      gt_shape=gt_shape,
                                                      max_iters=max_iters[i],
                                                      return_costs=return_costs,
                                                      **kwargs)
            # Add algorithm result to the list
            algorithm_results.append(algorithm_result)

            # Prepare this scale's final shape for the next scale
            if i < self.n_scales - 1:
                # This should not be done for the last scale.
                shape = algorithm_result.final_shape
                if self.holistic_features[i + 1] != self.holistic_features[i]:
                    # If the features function of the current scale is different
                    # than the one of the next scale, this means that the affine
                    # transform is different as well. Thus we need to do the
                    # following composition:
                    #
                    #    S_{i+1} \circ A_{i+1} \circ inv(A_i) \circ inv(S_i)
                    #
                    # where:
                    #    S_i : scaling transform of current scale
                    #    S_{i+1} : scaling transform of next scale
                    #    A_i : affine transform of current scale
                    #    A_{i+1} : affine transform of next scale
                    t1 = scale_transforms[i].compose_after(affine_transforms[i])
                    t2 = affine_transforms[i + 1].pseudoinverse().compose_after(t1)
                    transform = scale_transforms[i + 1].pseudoinverse().compose_after(t2)
                    shape = transform.apply(shape)
                elif (self.holistic_features[i + 1] == self.holistic_features[i] and
                      self.scales[i] != self.scales[i + 1]):
                    # If the features function of the current scale is the same
                    # as the one of the next scale, this means that the affine
                    # transform is the same as well, and thus can be omitted.
                    # Given that the scale factors are different, we need to do
                    # the # following composition:
                    #
                    #    S_{i+1} \circ inv(S_i)
                    #
                    # where:
                    #    S_i : scaling transform of current scale
                    #    S_{i+1} : scaling transform of next scale
                    transform = scale_transforms[i + 1].pseudoinverse().compose_after(scale_transforms[i])
                    shape = transform.apply(shape)

        # Return list of algorithm results
        return algorithm_results

    def _fitter_result(self, image, algorithm_results, affine_transforms,
                       scale_transforms, gt_shape=None):
        r"""
        Function the creates the multi-scale fitting result object.

        Parameters
        ----------
        image : `menpo.image.Image` or subclass
            The image that was fitted.
        algorithm_results : `list` of :map:`NonParametricIterativeResult` or subclass
            The list of fitting result per scale.
        affine_transforms : `list` of `menpo.transform.Affine`
            The list of affine transforms per scale that are the inverses of the
            transformations introduced by the rescale wrt the reference shape as
            well as the feature extraction.
        scale_transforms : `list` of `menpo.shape.Scale`
            The list of inverse scaling transforms per scale.
        gt_shape : `menpo.shape.PointCloud`, optional
            The ground truth shape associated to the image.

        Returns
        -------
        fitting_result : :map:`MultiScaleNonParametricIterativeResult` or subclass
            The multi-scale fitting result containing the result of the fitting
            procedure.
        """
        return MultiScaleNonParametricIterativeResult(
            results=algorithm_results, scales=self.scales,
            affine_transforms=affine_transforms,
            scale_transforms=scale_transforms, image=image, gt_shape=gt_shape)

    def fit_from_shape(self, image, initial_shape, max_iters=20, gt_shape=None,
                       return_costs=False, **kwargs):
        r"""
        Fits the multi-scale fitter to an image given an initial shape.

        Parameters
        ----------
        image : `menpo.image.Image` or subclass
            The image to be fitted.
        initial_shape : `menpo.shape.PointCloud`
            The initial shape estimate from which the fitting procedure
            will start.
        max_iters : `int` or `list` of `int`, optional
            The maximum number of iterations. If `int`, then it specifies the
            maximum number of iterations over all scales. If `list` of `int`,
            then specifies the maximum number of iterations per scale.
        gt_shape : `menpo.shape.PointCloud`, optional
            The ground truth shape associated to the image.
        return_costs : `bool`, optional
            If ``True``, then the cost function values will be computed
            during the fitting procedure. Then these cost values will be
            assigned to the returned `fitting_result`. *Note that the costs
            computation increases the computational cost of the fitting. The
            additional computation cost depends on the fitting method. Only
            use this option for research purposes.*
        kwargs : `dict`, optional
            Additional keyword arguments that can be passed to specific
            implementations.

        Returns
        -------
        fitting_result : :map:`MultiScaleNonParametricIterativeResult` or subclass
            The multi-scale fitting result containing the result of the fitting
            procedure.
        """
        # Generate the list of images to be fitted, as well as the correctly
        # scaled initial and ground truth shapes per level. The function also
        # returns the lists of affine and scale transforms per level that are
        # required in order to transform the shapes at the original image
        # space in the fitting result. The affine transforms refer to the
        # transform introduced by the rescaling to the reference shape as well
        # as potential affine transform from the features. The scale
        # transforms are the Scale objects that correspond to each level's
        # scale.
        (images, initial_shapes, gt_shapes, affine_transforms,
         scale_transforms) = self._prepare_image(image, initial_shape,
                                                 gt_shape=gt_shape)

        # Execute multi-scale fitting
        algorithm_results = self._fit(images=images,
                                      initial_shape=initial_shapes[0],
                                      affine_transforms=affine_transforms,
                                      scale_transforms=scale_transforms,
                                      max_iters=max_iters, gt_shapes=gt_shapes,
                                      return_costs=return_costs, **kwargs)

        # Return multi-scale fitting result
        return self._fitter_result(image=image,
                                   algorithm_results=algorithm_results,
                                   affine_transforms=affine_transforms,
                                   scale_transforms=scale_transforms,
                                   gt_shape=gt_shape)

    def fit_from_bb(self, image, bounding_box, max_iters=20, gt_shape=None,
                    return_costs=False, **kwargs):
        r"""
        Fits the multi-scale fitter to an image given an initial bounding box.

        Parameters
        ----------
        image : `menpo.image.Image` or subclass
            The image to be fitted.
        bounding_box : `menpo.shape.PointDirectedGraph`
            The initial bounding box from which the fitting procedure will
            start. Note that the bounding box is used in order to align the
            model's reference shape.
        max_iters : `int` or `list` of `int`, optional
            The maximum number of iterations. If `int`, then it specifies the
            maximum number of iterations over all scales. If `list` of `int`,
            then specifies the maximum number of iterations per scale.
        gt_shape : `menpo.shape.PointCloud`, optional
            The ground truth shape associated to the image.
        return_costs : `bool`, optional
            If ``True``, then the cost function values will be computed
            during the fitting procedure. Then these cost values will be
            assigned to the returned `fitting_result`. *Note that the costs
            computation increases the computational cost of the fitting. The
            additional computation cost depends on the fitting method. Only
            use this option for research purposes.*
        kwargs : `dict`, optional
            Additional keyword arguments that can be passed to specific
            implementations.

        Returns
        -------
        fitting_result : :map:`MultiScaleNonParametricIterativeResult` or subclass
            The multi-scale fitting result containing the result of the fitting
            procedure.
        """
        initial_shape = align_shape_with_bounding_box(self.reference_shape,
                                                      bounding_box)
        return self.fit_from_shape(image=image, initial_shape=initial_shape,
                                   max_iters=max_iters, gt_shape=gt_shape,
                                   return_costs=return_costs, **kwargs)


class MultiScaleParametricFitter(MultiScaleNonParametricFitter):
    r"""
    Class for defining a multi-scale fitter for a parametric fitting method, i.e.
    a method that optimises over the parameters of a statistical shape model.

    .. note:: When using a method with a parametric shape model, the first step
              is to **reconstruct the initial shape** using the shape model. The
              generated reconstructed shape is then used as initialisation for
              the iterative optimisation. This step takes place at each scale
              and it is not considered as an iteration, thus it is not counted
              for the provided `max_iters`.

    Parameters
    ----------
    scales : `list` of `int` or `float`
        The scale value of each scale. They must provided in ascending order,
        i.e. from lowest to highest scale.
    reference_shape : `menpo.shape.PointCloud`
        The reference shape that will be used to normalise the size of an input
        image so that the scale of its initial fitting shape matches the scale of
        the reference shape.
    holistic_features : `list` of `closure`
        The features that will be extracted from the input image at each scale.
        They must provided in ascending order, i.e. from lowest to highest scale.
    algorithms : `list` of `class`
        The list of algorithm objects that will perform the fitting per scale.
    """
    def __init__(self, scales, reference_shape, holistic_features, algorithms):
        super(MultiScaleParametricFitter, self).__init__(
            scales=scales, reference_shape=reference_shape,
            holistic_features=holistic_features, algorithms=algorithms)

    def _fitter_result(self, image, algorithm_results, affine_transforms,
                       scale_transforms, gt_shape=None):
        r"""
        Function the creates the multi-scale fitting result object.

        Parameters
        ----------
        image : `menpo.image.Image` or subclass
            The image that was fitted.
        algorithm_results : `list` of :map:`ParametricIterativeResult` or subclass
            The list of fitting result per scale.
        affine_transforms : `list` of `menpo.transform.Affine`
            The list of affine transforms per scale that are the inverses of the
            transformations introduced by the rescale wrt the reference shape as
            well as the feature extraction.
        scale_transforms : `list` of `menpo.shape.Scale`
            The list of inverse scaling transforms per scale.
        gt_shape : `menpo.shape.PointCloud`, optional
            The ground truth shape associated to the image.

        Returns
        -------
        fitting_result : :map:`MultiScaleParametricIterativeResult` or subclass
            The multi-scale fitting result containing the result of the fitting
            procedure.
        """
        return MultiScaleParametricIterativeResult(
            results=algorithm_results, scales=self.scales,
            affine_transforms=affine_transforms,
            scale_transforms=scale_transforms, image=image, gt_shape=gt_shape)


def generate_perturbations_from_gt(images, n_perturbations, perturb_func,
                                   gt_group=None, bb_group_glob=None,
                                   verbose=False):
    """
    Function that returns a callable that generates perturbations of the bounding
    boxes of the provided images.

    Parameters
    ----------
    images : `list` of `menpo.image.Image`
        The list of images.
    n_perturbations : `int`
        The number of perturbed shapes to be generated per image.
    perturb_func : `callable`
        The function that will be used for generating the perturbations.
    gt_group : `str`
        The group of the ground truth shapes attached to the images.
    bb_group_glob : `str`
        The group of the bounding boxes attached to the images.
    verbose : `bool`, optional
        If ``True``, then progress information is printed.

    Returns
    -------
    generated_bb_func : `callable`
        The function that generates the perturbations.
    """
    if bb_group_glob is None:
        bb_generator = lambda im: [im.landmarks[gt_group].lms.bounding_box()]
        n_bbs = 1
    else:
        def bb_glob(im):
            return [v.lms.bounding_box()
                    for _, v in im.landmarks.items_matching(bb_group_glob)]
        bb_generator = bb_glob
        n_bbs = len(bb_glob(images[0]))

    if n_bbs == 0:
        raise ValueError('Must provide a valid bounding box glob - no bounding '
                         'boxes matched the following '
                         'glob: {}'.format(bb_group_glob))

    # If we have multiple boxes - we didn't just throw them away, we re-add them
    # to the end
    if bb_group_glob is not None:
        msg = '- Generating {0} ({1} perturbations * {2} provided boxes) new ' \
              'initial bounding boxes + {2} provided boxes per image'.format(
            n_perturbations * n_bbs, n_perturbations, n_bbs)
    else:
        msg = '- Generating {} new bounding boxes directly from the ' \
              'ground truth shape'.format(n_perturbations)

    wrap = partial(print_progress, prefix=msg, verbose=verbose)
    for im in wrap(images):
        gt_s = im.landmarks[gt_group].lms.bounding_box()

        k = 0
        im_bounds = im.bounds()
        for bb in bb_generator(im):
            for _ in range(n_perturbations):
                p_s = perturb_func(gt_s, bb).bounding_box()
                perturb_bbox_group = '__generated_bb_{}'.format(k)
                im.landmarks[perturb_bbox_group] = p_s.constrain_to_bounds(im_bounds)
                k += 1

            if bb_group_glob is not None:
                perturb_bbox_group = '__generated_bb_{}'.format(k)
                im.landmarks[perturb_bbox_group] = bb.constrain_to_bounds(im_bounds)
                k += 1

    generated_bb_func = lambda x: [v.lms for k, v in x.landmarks.items_matching(
        '__generated_bb_*')]
    return generated_bb_func
