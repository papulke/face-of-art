from __future__ import division
import numpy as np
from functools import partial
import warnings

from menpo.feature import no_op
from menpo.base import name_of_callable

from menpofit.visualize import print_progress
from menpofit.base import batch
from menpofit.builder import (scale_images, rescale_images_to_reference_shape,
                              compute_reference_shape, MenpoFitBuilderWarning,
                              compute_features)
from menpofit.fitter import (MultiScaleNonParametricFitter,
                             noisy_shape_from_bounding_box,
                             align_shape_with_bounding_box,
                             generate_perturbations_from_gt)
import menpofit.checks as checks

from .algorithm import NonParametricNewton


class SupervisedDescentFitter(MultiScaleNonParametricFitter):
    r"""
    Class for training a multi-scale Supervised Descent model.

    Parameters
    ----------
    images : `list` of `menpo.image.Image`
        The `list` of training images.
    group : `str` or ``None``, optional
        The landmark group that corresponds to the ground truth shape of each
        image. If ``None`` and the images only have a single landmark group,
        then that is the one that will be used. Note that all the training
        images need to have the specified landmark group.
    bounding_box_group_glob : `glob` or ``None``, optional
        Glob that defines the bounding boxes to be used for training. If
        ``None``, then the bounding boxes of the ground truth shapes are used.
    sd_algorithm_cls : `class`, optional
        The Supervised Descent algorithm to be used. The possible algorithms
        are are separated in the following four categories:

        **Non-parametric:**

        ===================================== ==============================
        Class                                 Regression
        ===================================== ==============================
        :map:`NonParametricNewton`            :map:`IRLRegression`
        :map:`NonParametricGaussNewton`       :map:`IIRLRegression`
        :map:`NonParametricPCRRegression`     :map:`PCRRegression`
        :map:`NonParametricOptimalRegression` :map:`OptimalLinearRegression`
        :map:`NonParametricOPPRegression`     :map:`OPPRegression`
        ===================================== ==============================

        **Parametric shape:**

        ======================================= ===================================
        Class                                   Regression
        ======================================= ===================================
        :map:`ParametricShapeNewton`            :map:`IRLRegression`
        :map:`ParametricShapeGaussNewton`       :map:`IIRLRegression`
        :map:`ParametricShapePCRRegression`     :map:`PCRRegression`
        :map:`ParametricShapeOptimalRegression` :map:`OptimalLinearRegression`
        :map:`ParametricShapeOPPRegression`     :map:`ParametricShapeOPPRegression`
        ======================================= ===================================

        **Parametric appearance:**

        ================================================== =====================
        Class                                              Regression
        ================================================== =====================
        :map:`ParametricAppearanceProjectOutNewton`        :map:`IRLRegression`
        :map:`ParametricAppearanceProjectOutGuassNewton`   :map:`IIRLRegression`
        :map:`ParametricAppearanceMeanTemplateNewton`      :map:`IRLRegression`
        :map:`ParametricAppearanceMeanTemplateGuassNewton` :map:`IIRLRegression`
        :map:`ParametricAppearanceWeightsNewton`           :map:`IRLRegression`
        :map:`ParametricAppearanceWeightsGuassNewton`      :map:`IIRLRegression`
        ================================================== =====================

        **Parametric shape and appearance:**

        =========================================== =====================
        Class                                       Regression
        =========================================== =====================
        :map:`FullyParametricProjectOutNewton`      :map:`IRLRegression`
        :map:`FullyParametricProjectOutGaussNewton` :map:`IIRLRegression`
        :map:`FullyParametricMeanTemplateNewton`    :map:`IRLRegression`
        :map:`FullyParametricWeightsNewton`         :map:`IRLRegression`
        :map:`FullyParametricProjectOutOPP`         :map:`OPPRegression`
        =========================================== =====================
    reference_shape : `menpo.shape.PointCloud` or ``None``, optional
        The reference shape that will be used for normalising the size of the
        training images. The normalization is performed by rescaling all the
        training images so that the scale of their ground truth shapes
        matches the scale of the reference shape. Note that the reference
        shape is rescaled with respect to the `diagonal` before performing
        the normalisation. If ``None``, then the mean shape will be used.
    diagonal : `int` or ``None``, optional
        This parameter is used to rescale the reference shape so that the
        diagonal of its bounding box matches the provided value. In other
        words, this parameter controls the size of the model at the highest
        scale. If ``None``, then the reference shape does not get rescaled.
    holistic_features : `closure` or `list` of `closure`, optional
        The features that will be extracted from the training images. Note
        that the features are extracted before warping the images to the
        reference shape. If `list`, then it must define a feature function per
        scale. Please refer to `menpo.feature` for a list of potential features.
    patch_features : `closure` or `list` of `closure`, optional
        The features that will be extracted from the patches of the training
        images. Note that, as opposed to `holistic_features`, these features
        are extracted after extracting the patches. If `list`, then it must
        define a feature function per scale. Please refer to `menpo.feature`
        and `menpofit.feature` for a list of potential features.
    patch_shape : (`int`, `int`) or `list` of (`int`, `int`), optional
        The shape of the patches to be extracted. If a `list` is provided,
        then it defines a patch shape per scale.
    scales : `float` or `tuple` of `float`, optional
        The scale value of each scale. They must provided in ascending order,
        i.e. from lowest to highest scale. If `float`, then a single scale is
        assumed.
    n_iterations : `int` or `list` of `int`, optional
        The number of iterations (cascades) of each level. If `list`, it must
        specify a value per scale. If `int`, then it defines the total number of
        iterations (cascades) over all scales.
    n_perturbations : `int`, optional
        The number of perturbations to be generated from each of the bounding
        boxes using `perturb_from_gt_bounding_box`.
    perturb_from_gt_bounding_box : `callable`, optional
        The function that will be used to generate the perturbations from each
        of the bounding boxes.
    batch_size : `int` or ``None``, optional
        If an `int` is provided, then the training is performed in an
        incremental fashion on image batches of size equal to the provided
        value. If ``None``, then the training is performed directly on the
        all the images.
    verbose : `bool`, optional
        If ``True``, then the progress of the training will be printed.

    References
    ----------
    .. [1] X. Xiong, and F. De la Torre. "Supervised Descent Method and its
        applications to face alignment", Proceedings of the IEEE Conference on
        Computer Vision and Pattern Recognition (CVPR), 2013.
    .. [2] P. N. Belhumeur, D. W. Jacobs, D. J. Kriegman, and N. Kumar.
        "Localizing parts of faces using a consensus of exemplars", Proceedings
        of the IEEE Conference on Computer Vision and Pattern Recognition
        (CVPR), 2011.
    """
    def __init__(self, images, group=None, bounding_box_group_glob=None,
                 sd_algorithm_cls=None, reference_shape=None, diagonal=None,
                 holistic_features=no_op, patch_features=no_op,
                 patch_shape=(17, 17), scales=(0.5, 1.0), n_iterations=3,
                 n_perturbations=30,
                 perturb_from_gt_bounding_box=noisy_shape_from_bounding_box,
                 batch_size=None, verbose=False):
        if batch_size is not None:
            raise NotImplementedError('Training an SDM with a batch size '
                                      '(incrementally) is not implemented yet.')
        # Check parameters
        checks.check_diagonal(diagonal)
        scales = checks.check_scales(scales)
        n_scales = len(scales)
        patch_features = checks.check_callable(patch_features, n_scales)
        sd_algorithm_cls = checks.check_callable(sd_algorithm_cls, n_scales)
        holistic_features = checks.check_callable(holistic_features, n_scales)
        patch_shape = checks.check_patch_shape(patch_shape, n_scales)

        # Call superclass
        super(SupervisedDescentFitter, self).__init__(
            scales=scales, reference_shape=reference_shape,
            holistic_features=holistic_features, algorithms=[])

        # Set parameters
        self._sd_algorithm_cls = sd_algorithm_cls
        self.patch_features = patch_features
        self.patch_shape = patch_shape
        self.diagonal = diagonal
        self.n_perturbations = n_perturbations
        self.n_iterations = checks.check_max_iters(n_iterations, n_scales)
        self._perturb_from_gt_bounding_box = perturb_from_gt_bounding_box

        # Set up algorithms
        self._setup_algorithms()

        # Now, train the model!
        self._train(images, increment=False,  group=group,
                    bounding_box_group_glob=bounding_box_group_glob,
                    verbose=verbose, batch_size=batch_size)

    def _setup_algorithms(self):
        self.algorithms = [self._sd_algorithm_cls[j](
            patch_features=self.patch_features[j],
            patch_shape=self.patch_shape[j], n_iterations=self.n_iterations[j])
                           for j in range(self.n_scales)]

    def _train(self, images, increment=False, group=None,
               bounding_box_group_glob=None, verbose=False, batch_size=None):
        # If batch_size is not None, then we may have a generator, else we
        # assume we have a list.
        if batch_size is not None:
            # Create a generator of fixed sized batches. Will still work even
            # on an infinite list.
            image_batches = batch(images, batch_size)
        else:
            image_batches = [list(images)]

        for k, image_batch in enumerate(image_batches):
            if k == 0:
                if self.reference_shape is None:
                    # If no reference shape was given, use the mean of the first
                    # batch
                    if batch_size is not None:
                        warnings.warn('No reference shape was provided. The '
                                      'mean of the first batch will be the '
                                      'reference shape. If the batch mean is '
                                      'not representative of the true mean, '
                                      'this may cause issues.',
                                      MenpoFitBuilderWarning)
                    self._reference_shape = compute_reference_shape(
                        [i.landmarks[group].lms for i in image_batch],
                        self.diagonal, verbose=verbose)
            # We set landmarks on the images to archive the perturbations, so
            # when the default 'None' is used, we need to grab the actual
            # label to sort out the ambiguity
            if group is None:
                group = image_batch[0].landmarks.group_labels[0]

            # After the first batch, we are incrementing the model
            if k > 0:
                increment = True

            if verbose:
                print('Computing batch {}'.format(k))

            # Train each batch
            self._train_batch(
                image_batch, increment=increment, group=group,
                bounding_box_group_glob=bounding_box_group_glob,
                verbose=verbose)

    def _train_batch(self, image_batch, increment=False, group=None,
                     bounding_box_group_glob=None, verbose=False):
        # Rescale images wrt the scale factor between the existing
        # reference_shape and their ground truth (group) shapes
        image_batch = rescale_images_to_reference_shape(
            image_batch, group, self.reference_shape,
            verbose=verbose)

        # Create a callable that generates perturbations of the bounding boxes
        # of the provided images.
        generated_bb_func = generate_perturbations_from_gt(
            image_batch, self.n_perturbations,
            self._perturb_from_gt_bounding_box, gt_group=group,
            bb_group_glob=bounding_box_group_glob, verbose=verbose)

        # For each scale (low --> high)
        for j in range(self.n_scales):
            # Print progress if asked
            if verbose:
                if len(self.scales) > 1:
                    scale_prefix = '  - Scale {}: '.format(j)
                else:
                    scale_prefix = '  - '
            else:
                scale_prefix = None

            # Extract features. Features are extracted only if we are at the
            # first scale or if the features of the current scale are different
            # than the ones extracted at the previous scale.
            if j == 0 and self.holistic_features[j] == no_op:
                # Saves a lot of memory
                feature_images = image_batch
            elif (j == 0 or
                  self.holistic_features[j] != self.holistic_features[j - 1]):
                # Compute features only if this is the first pass through
                # the loop or the features at this scale are different from
                # the features at the previous scale
                feature_images = compute_features(image_batch,
                                                  self.holistic_features[j],
                                                  prefix=scale_prefix,
                                                  verbose=verbose)

            # Rescale images according to scales. Note that scale_images is smart
            # enough in order not to rescale the images if the current scale
            # factor equals to 1.
            scaled_images, scale_transforms = scale_images(
                feature_images, self.scales[j], prefix=scale_prefix,
                return_transforms=True, verbose=verbose)

            # Extract scaled ground truth shapes for current scale
            scaled_shapes = [i.landmarks[group].lms for i in scaled_images]

            # Get shape estimations of current scale. If we are at the first
            # scale, this is done by aligning the reference shape with the
            # perturbed bounding boxes. If we are at the rest of the scales,
            # then the current shapes are attached on the scaled_images with
            # key '__sdm_current_shape_{}'.
            current_shapes = []
            if j == 0:
                # At the first scale, the current shapes are created by aligning
                # the reference shape to the perturbed bounding boxes.
                msg = '{}Aligning reference shape with bounding boxes.'.format(
                    scale_prefix)
                wrap = partial(print_progress, prefix=msg,
                               end_with_newline=False, verbose=verbose)
                # Extract perturbations at the very bottom level
                for ii in wrap(scaled_images):
                    c_shapes = []
                    for bbox in generated_bb_func(ii):
                        c_s = align_shape_with_bounding_box(
                            self.reference_shape, bbox)
                        c_shapes.append(c_s)
                    current_shapes.append(c_shapes)
            else:
                # At the rest of the scales, extract the current shapes that
                # were attached to the images
                msg = '{}Extracting shape estimations from previous ' \
                      'scale.'.format(scale_prefix)
                wrap = partial(print_progress, prefix=msg,
                               end_with_newline=False, verbose=verbose)
                for ii in wrap(scaled_images):
                    c_shapes = []
                    for k in list(range(self.n_perturbations)):
                        c_key = '__sdm_current_shape_{}'.format(k)
                        c_shapes.append(ii.landmarks[c_key].lms)
                    current_shapes.append(c_shapes)

            # Train supervised descent algorithm. This returns the shape
            # estimations for the next scale.
            if not increment:
                current_shapes = self.algorithms[j].train(
                    scaled_images, scaled_shapes, current_shapes,
                    prefix=scale_prefix, verbose=verbose)
            else:
                current_shapes = self.algorithms[j].increment(
                    scaled_images, scaled_shapes, current_shapes,
                    prefix=scale_prefix, verbose=verbose)

            # Scale the current shape estimations for the next level. This
            # doesn't have to be done for the last scale. The only thing we need
            # to do at the last scale is to remove any attached landmarks from
            # the training images.
            if j < (self.n_scales - 1):
                if self.holistic_features[j + 1] != self.holistic_features[j]:
                    # Features will be extracted, thus attach current_shapes on
                    # the training images (image_batch)
                    for jj, image_shapes in enumerate(current_shapes):
                        for k, shape in enumerate(image_shapes):
                            c_key = '__sdm_current_shape_{}'.format(k)
                            image_batch[jj].landmarks[c_key] = \
                                scale_transforms[jj].apply(shape)
                else:
                    # Features won't be extracted;. the same feature_images will
                    # be used for the next scale, thus attach current_shapes on
                    # them.
                    for jj, image_shapes in enumerate(current_shapes):
                        for k, shape in enumerate(image_shapes):
                            c_key = '__sdm_current_shape_{}'.format(k)
                            feature_images[jj].landmarks[c_key] = \
                                scale_transforms[jj].apply(shape)
            else:
                # Check if original training image (image_batch) got some current
                # shape estimations attached. If yes, delete them.
                if '__sdm_current_shape_0' in image_batch[0].landmarks:
                    for image in image_batch:
                        for k in list(range(self.n_perturbations)):
                            c_key = '__sdm_current_shape_{}'.format(k)
                            del image.landmarks[c_key]

    def increment(self, images, group=None, bounding_box_group_glob=None,
                  verbose=False, batch_size=None):
        r"""
        Method to increment the trained SDM with a new set of training images.

        Parameters
        ----------
        images : `list` of `menpo.image.Image`
            The `list` of training images.
        group : `str` or ``None``, optional
            The landmark group that corresponds to the ground truth shape of
            each image. If ``None`` and the images only have a single
            landmark group, then that is the one that will be used. Note that
            all the training images need to have the specified landmark group.
        bounding_box_group_glob : `glob` or ``None``, optional
            Glob that defines the bounding boxes to be used for training. If
            ``None``, then the bounding boxes of the ground truth shapes are
            used.
        verbose : `bool`, optional
            If ``True``, then the progress of training will be printed.
        batch_size : `int` or ``None``, optional
            If an `int` is provided, then the training is performed in an
            incremental fashion on image batches of size equal to the provided
            value. If ``None``, then the training is performed directly on the
            all the images.
        """
        raise NotImplementedError('Incrementing SDM methods is not yet '
                                  'implemented as careful attention must '
                                  'be taken when considering the relationships '
                                  'between cascade levels.')

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
        return self.algorithms[0]._multi_scale_fitter_result(
            results=algorithm_results, scales=self.scales,
            affine_transforms=affine_transforms,
            scale_transforms=scale_transforms, image=image, gt_shape=gt_shape)

    def __str__(self):
        if self.diagonal is not None:
            diagonal = self.diagonal
        else:
            y, x = self.reference_shape.range()
            diagonal = np.sqrt(x ** 2 + y ** 2)
        is_custom_perturb_func = (self._perturb_from_gt_bounding_box !=
                                  noisy_shape_from_bounding_box)
        if is_custom_perturb_func:
            is_custom_perturb_func = name_of_callable(
                    self._perturb_from_gt_bounding_box)
        regressor_cls = self.algorithms[0]._regressor_cls

        # Compute scale info strings
        scales_info = []
        lvl_str_tmplt = r"""   - Scale {}
     - {} iterations
     - Patch shape: {}
     - Holistic feature: {}
     - Patch feature: {}"""
        for k, s in enumerate(self.scales):
            scales_info.append(lvl_str_tmplt.format(
                s, self.n_iterations[k], self.patch_shape[k],
                name_of_callable(self.holistic_features[k]),
                name_of_callable(self.patch_features[k])))
        scales_info = '\n'.join(scales_info)

        cls_str = r"""Supervised Descent Method
 - Regression performed using the {reg_alg} algorithm
   - Regression class: {reg_cls}
 - Perturbations generated per shape: {n_perturbations}
 - Images scaled to diagonal: {diagonal:.2f}
 - Custom perturbation scheme used: {is_custom_perturb_func}
 - Scales: {scales}
{scales_info}
""".format(
            reg_alg=name_of_callable(self._sd_algorithm_cls[0]),
            reg_cls=name_of_callable(regressor_cls),
            n_perturbations=self.n_perturbations,
            diagonal=diagonal,
            is_custom_perturb_func=is_custom_perturb_func,
            scales=self.scales,
            scales_info=scales_info)
        return cls_str


# *
# ************************* Non-Parametric Fitters *****************************
# *
# Aliases for common combinations of supervised descent fitting
class SDM(SupervisedDescentFitter):
    r"""
    Class for training a non-parametric multi-scale Supervised Descent model
    using :map:`NonParametricNewton`.

    Parameters
    ----------
    images : `list` of `menpo.image.Image`
        The `list` of training images.
    group : `str` or ``None``, optional
        The landmark group that corresponds to the ground truth shape of each
        image. If ``None`` and the images only have a single landmark group,
        then that is the one that will be used. Note that all the training
        images need to have the specified landmark group.
    bounding_box_group_glob : `glob` or ``None``, optional
        Glob that defines the bounding boxes to be used for training. If
        ``None``, then the bounding boxes of the ground truth shapes are used.
    reference_shape : `menpo.shape.PointCloud` or ``None``, optional
        The reference shape that will be used for normalising the size of the
        training images. The normalization is performed by rescaling all the
        training images so that the scale of their ground truth shapes
        matches the scale of the reference shape. Note that the reference
        shape is rescaled with respect to the `diagonal` before performing
        the normalisation. If ``None``, then the mean shape will be used.
    diagonal : `int` or ``None``, optional
        This parameter is used to rescale the reference shape so that the
        diagonal of its bounding box matches the provided value. In other
        words, this parameter controls the size of the model at the highest
        scale. If ``None``, then the reference shape does not get rescaled.
    holistic_features : `closure` or `list` of `closure`, optional
        The features that will be extracted from the training images. Note
        that the features are extracted before warping the images to the
        reference shape. If `list`, then it must define a feature function per
        scale. Please refer to `menpo.feature` for a list of potential features.
    patch_features : `closure` or `list` of `closure`, optional
        The features that will be extracted from the patches of the training
        images. Note that, as opposed to `holistic_features`, these features
        are extracted after extracting the patches. If `list`, then it must
        define a feature function per scale. Please refer to `menpo.feature`
        and `menpofit.feature` for a list of potential features.
    patch_shape : (`int`, `int`) or `list` of (`int`, `int`), optional
        The shape of the patches to be extracted. If a `list` is provided,
        then it defines a patch shape per scale.
    scales : `float` or `tuple` of `float`, optional
        The scale value of each scale. They must provided in ascending order,
        i.e. from lowest to highest scale. If `float`, then a single scale is
        assumed.
    n_iterations : `int` or `list` of `int`, optional
        The number of iterations (cascades) of each level. If `list`, it must
        specify a value per scale. If `int`, then it defines the total number of
        iterations (cascades) over all scales.
    n_perturbations : `int`, optional
        The number of perturbations to be generated from each of the bounding
        boxes using `perturb_from_gt_bounding_box`.
    perturb_from_gt_bounding_box : `callable`, optional
        The function that will be used to generate the perturbations from each
        of the bounding boxes.
    batch_size : `int` or ``None``, optional
        If an `int` is provided, then the training is performed in an
        incremental fashion on image batches of size equal to the provided
        value. If ``None``, then the training is performed directly on the
        all the images.
    verbose : `bool`, optional
        If ``True``, then the progress of the training will be printed.

    References
    ----------
    .. [1] X. Xiong, and F. De la Torre. "Supervised Descent Method and its
        applications to face alignment", Proceedings of the IEEE Conference on
        Computer Vision and Pattern Recognition (CVPR), 2013.
    """
    def __init__(self, images, group=None, bounding_box_group_glob=None,
                 reference_shape=None,  diagonal=None, holistic_features=no_op,
                 patch_features=no_op, patch_shape=(17, 17), scales=(0.5, 1.0),
                 n_iterations=3, n_perturbations=30,
                 perturb_from_gt_bounding_box=noisy_shape_from_bounding_box,
                 batch_size=None, verbose=False):
        super(SDM, self).__init__(
                images, group=group,
                bounding_box_group_glob=bounding_box_group_glob,
                reference_shape=reference_shape,
                sd_algorithm_cls=NonParametricNewton,
                holistic_features=holistic_features,
                patch_features=patch_features, patch_shape=patch_shape,
                diagonal=diagonal, scales=scales, n_iterations=n_iterations,
                n_perturbations=n_perturbations,
                perturb_from_gt_bounding_box=perturb_from_gt_bounding_box,
                batch_size=batch_size, verbose=verbose)


class RegularizedSDM(SupervisedDescentFitter):
    r"""
    Class for training a non-parametric multi-scale Supervised Descent model
    using :map:`NonParametricNewton` with regularization.

    Parameters
    ----------
    images : `list` of `menpo.image.Image`
        The `list` of training images.
    group : `str` or ``None``, optional
        The landmark group that corresponds to the ground truth shape of each
        image. If ``None`` and the images only have a single landmark group,
        then that is the one that will be used. Note that all the training
        images need to have the specified landmark group.
    bounding_box_group_glob : `glob` or ``None``, optional
        Glob that defines the bounding boxes to be used for training. If
        ``None``, then the bounding boxes of the ground truth shapes are used.
    alpha : `float`, optional
        The regression regularization parameter.
    reference_shape : `menpo.shape.PointCloud` or ``None``, optional
        The reference shape that will be used for normalising the size of the
        training images. The normalization is performed by rescaling all the
        training images so that the scale of their ground truth shapes
        matches the scale of the reference shape. Note that the reference
        shape is rescaled with respect to the `diagonal` before performing
        the normalisation. If ``None``, then the mean shape will be used.
    diagonal : `int` or ``None``, optional
        This parameter is used to rescale the reference shape so that the
        diagonal of its bounding box matches the provided value. In other
        words, this parameter controls the size of the model at the highest
        scale. If ``None``, then the reference shape does not get rescaled.
    holistic_features : `closure` or `list` of `closure`, optional
        The features that will be extracted from the training images. Note
        that the features are extracted before warping the images to the
        reference shape. If `list`, then it must define a feature function per
        scale. Please refer to `menpo.feature` for a list of potential features.
    patch_features : `closure` or `list` of `closure`, optional
        The features that will be extracted from the patches of the training
        images. Note that, as opposed to `holistic_features`, these features
        are extracted after extracting the patches. If `list`, then it must
        define a feature function per scale. Please refer to `menpo.feature`
        and `menpofit.feature` for a list of potential features.
    patch_shape : (`int`, `int`) or `list` of (`int`, `int`), optional
        The shape of the patches to be extracted. If a `list` is provided,
        then it defines a patch shape per scale.
    scales : `float` or `tuple` of `float`, optional
        The scale value of each scale. They must provided in ascending order,
        i.e. from lowest to highest scale. If `float`, then a single scale is
        assumed.
    n_iterations : `int` or `list` of `int`, optional
        The number of iterations (cascades) of each level. If `list`, it must
        specify a value per scale. If `int`, then it defines the total number of
        iterations (cascades) over all scales.
    n_perturbations : `int`, optional
        The number of perturbations to be generated from each of the bounding
        boxes using `perturb_from_gt_bounding_box`.
    perturb_from_gt_bounding_box : `callable`, optional
        The function that will be used to generate the perturbations from each
        of the bounding boxes.
    batch_size : `int` or ``None``, optional
        If an `int` is provided, then the training is performed in an
        incremental fashion on image batches of size equal to the provided
        value. If ``None``, then the training is performed directly on the
        all the images.
    verbose : `bool`, optional
        If ``True``, then the progress of the training will be printed.

    References
    ----------
    .. [1] X. Xiong, and F. De la Torre. "Supervised Descent Method and its
        applications to face alignment", Proceedings of the IEEE Conference on
        Computer Vision and Pattern Recognition (CVPR), 2013.
    """
    def __init__(self, images, group=None, bounding_box_group_glob=None,
                 alpha=0.0001, reference_shape=None, diagonal=None,
                 holistic_features=no_op, patch_features=no_op,
                 patch_shape=(17, 17), scales=(0.5, 1.0), n_iterations=6,
                 n_perturbations=30,
                 perturb_from_gt_bounding_box=noisy_shape_from_bounding_box,
                 batch_size=None, verbose=False):
        super(RegularizedSDM, self).__init__(
            images, group=group,
            bounding_box_group_glob=bounding_box_group_glob,
            reference_shape=reference_shape,
            sd_algorithm_cls=partial(NonParametricNewton, alpha=alpha),
            holistic_features=holistic_features, patch_features=patch_features,
            patch_shape=patch_shape, diagonal=diagonal, scales=scales,
            n_iterations=n_iterations, n_perturbations=n_perturbations,
            perturb_from_gt_bounding_box=perturb_from_gt_bounding_box,
            batch_size=batch_size, verbose=verbose)
