from __future__ import division
import warnings
import numpy as np

from menpo.feature import no_op
from menpo.visualize import print_dynamic
from menpo.transform import Scale
from menpo.shape import mean_pointcloud
from menpo.base import name_of_callable

from menpofit import checks
from menpofit.modelinstance import OrthoPDM
from menpofit.transform import (DifferentiableThinPlateSplines,
                                DifferentiablePiecewiseAffine, OrthoMDTransform,
                                LinearOrthoMDTransform)
from menpofit.base import batch
from menpofit.builder import (
    build_reference_frame, build_patch_reference_frame,
    compute_features, scale_images, warp_images,
    align_shapes, densify_shapes,
    extract_patches, MenpoFitBuilderWarning, compute_reference_shape)

from .algorithm import (ATMLucasKanadeStandardInterface, ATMLucasKanadeLinearInterface,
                        ATMLucasKanadePatchInterface)


class ATM(object):
    r"""
    Class for training a multi-scale holistic Active Template Model.

    Parameters
    ----------
    template : `menpo.image.Image`
        The template image.
    shapes : `list` of `menpo.shape.PointCloud`
        The `list` of training shapes.
    group : `str` or ``None``, optional
        The landmark group of the `template` that will be used to train the ATM.
        If ``None`` and the `template` only has a single landmark group, then
        that is the one that will be used.
    holistic_features : `closure` or `list` of `closure`, optional
        The features that will be extracted from the training images. Note
        that the features are extracted before warping the images to the
        reference shape. If `list`, then it must define a feature function per
        scale. Please refer to `menpo.feature` for a list of potential features.
    reference_shape : `menpo.shape.PointCloud` or ``None``, optional
        The reference shape that will be used for building the ATM. The purpose
        of the reference shape is to normalise the size of the training images.
        The normalization is performed by rescaling all the training images
        so that the scale of their ground truth shapes matches the scale of
        the reference shape. Note that the reference shape is rescaled with
        respect to the `diagonal` before performing the normalisation. If
        ``None``, then the mean shape will be used.
    diagonal : `int` or ``None``, optional
        This parameter is used to rescale the reference shape so that the
        diagonal of its bounding box matches the provided value. In other
        words, this parameter controls the size of the model at the highest
        scale. If ``None``, then the reference shape does not get rescaled.
    scales : `float` or `tuple` of `float`, optional
        The scale value of each scale. They must provided in ascending order,
        i.e. from lowest to highest scale. If `float`, then a single scale is
        assumed.
    transform : `subclass` of :map:`DL` and :map:`DX`, optional
        A differential warp transform object, e.g.
        :map:`DifferentiablePiecewiseAffine` or
        :map:`DifferentiableThinPlateSplines`.
    shape_model_cls : `subclass` of :map:`PDM`, optional
        The class to be used for building the shape model. The most common
        choice is :map:`OrthoPDM`.
    max_shape_components : `int`, `float`, `list` of those or ``None``, optional
        The number of shape components to keep. If `int`, then it sets the exact
        number of components. If `float`, then it defines the variance
        percentage that will be kept. If `list`, then it should
        define a value per scale. If a single number, then this will be
        applied to all scales. If ``None``, then all the components are kept.
        Note that the unused components will be permanently trimmed.
    verbose : `bool`, optional
        If ``True``, then the progress of building the ATM will be printed.
    batch_size : `int` or ``None``, optional
        If an `int` is provided, then the training is performed in an
        incremental fashion on image batches of size equal to the provided
        value. If ``None``, then the training is performed directly on the
        all the images.

    References
    ----------
    .. [1] S. Baker, and I. Matthews. "Lucas-Kanade 20 years on: A unifying
        framework", International Journal of Computer Vision, 56(3): 221-255,
        2004.
    """
    def __init__(self, template, shapes, group=None, holistic_features=no_op,
                 reference_shape=None, diagonal=None, scales=(0.5, 1.0),
                 transform=DifferentiablePiecewiseAffine,
                 shape_model_cls=OrthoPDM, max_shape_components=None,
                 verbose=False, batch_size=None):
        # Check arguments
        checks.check_diagonal(diagonal)
        n_scales = len(scales)
        scales = checks.check_scales(scales)
        holistic_features = checks.check_callable(holistic_features, n_scales)
        max_shape_components = checks.check_max_components(
            max_shape_components, n_scales, 'max_shape_components')
        shape_model_cls = checks.check_callable(shape_model_cls, n_scales)
        # Assign attributes
        self.holistic_features = holistic_features
        self.transform = transform
        self.diagonal = diagonal
        self.scales = scales
        self.max_shape_components = max_shape_components
        self.reference_shape = reference_shape
        self.shape_models = []
        self.warped_templates = []
        self._shape_model_cls = shape_model_cls
        # Train ATM
        self._train(template, shapes, increment=False, group=group,
                    verbose=verbose, batch_size=batch_size)

    def _train(self, template, shapes, increment=False, group=None,
               shape_forgetting_factor=1.0, verbose=False, batch_size=None):
        # If batch_size is not None, then we may have a generator, else we
        # assume we have a list.
        if batch_size is not None:
            # Create a generator of fixed sized batches. Will still work even
            # on an infinite list.
            shape_batches = batch(shapes, batch_size)
        else:
            shape_batches = [list(shapes)]

        for k, shape_batch in enumerate(shape_batches):
            if k == 0:
                # Rescale the template the reference shape
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
                    checks.check_trilist(shape_batch[0], self.transform)
                    self.reference_shape = compute_reference_shape(
                        shape_batch, self.diagonal, verbose=verbose)

                # Rescale the template the reference shape
                template = template.rescale_to_pointcloud(
                    self.reference_shape, group=group)

            # After the first batch, we are incrementing the model
            if k > 0:
                increment = True

            if verbose:
                print('Computing batch {}'.format(k))

            # Train each batch
            self._train_batch(template, shape_batch, increment=increment,
                              group=group,
                              shape_forgetting_factor=shape_forgetting_factor,
                              verbose=verbose)

    def _train_batch(self, template, shape_batch, increment=False, group=None,
                     shape_forgetting_factor=1.0, verbose=False):
        # build models at each scale
        if verbose:
            print_dynamic('- Building models\n')

        feature_images = []
        # for each scale (low --> high)
        for j in range(self.n_scales):
            if verbose:
                if len(self.scales) > 1:
                    scale_prefix = '  - Scale {}: '.format(j)
                else:
                    scale_prefix = '  - '
            else:
                scale_prefix = None

            # Handle features
            if j == 0 or self.holistic_features[j] is not self.holistic_features[j - 1]:
                # Compute features only if this is the first pass through
                # the loop or the features at this scale are different from
                # the features at the previous scale
                feature_images = compute_features([template],
                                                  self.holistic_features[j],
                                                  prefix=scale_prefix,
                                                  verbose=verbose)
            # handle scales
            if self.scales[j] != 1:
                # Scale feature images only if scale is different than 1
                scaled_images = scale_images(feature_images, self.scales[j],
                                             prefix=scale_prefix,
                                             verbose=verbose)
                # Extract potentially rescaled shapes
                scale_transform = Scale(scale_factor=self.scales[j],
                                        n_dims=2)
                scale_shapes = [scale_transform.apply(s)
                                for s in shape_batch]
            else:
                scaled_images = feature_images
                scale_shapes = shape_batch

            # Build the shape model
            if verbose:
                print_dynamic('{}Building shape model'.format(scale_prefix))

            if not increment:
                shape_model = self._build_shape_model(scale_shapes, j)
                self.shape_models.append(shape_model)
            else:
                self._increment_shape_model(
                    scale_shapes, j, forgetting_factor=shape_forgetting_factor)

            # Obtain warped images - we use a scaled version of the
            # reference shape, computed here. This is because the mean
            # moves when we are incrementing, and we need a consistent
            # reference frame.
            scaled_reference_shape = Scale(self.scales[j], n_dims=2).apply(
                self.reference_shape)
            warped_template = self._warp_template(scaled_images[0], group,
                                                  scaled_reference_shape,
                                                  j, scale_prefix, verbose)
            self.warped_templates.append(warped_template[0])

            if verbose:
                print_dynamic('{}Done\n'.format(scale_prefix))

    def increment(self, template, shapes, group=None,
                  shape_forgetting_factor=1.0, verbose=False, batch_size=None):
        r"""
        Method to increment the trained ATM with a new set of training shapes
        and a new template.

        Parameters
        ----------
        template : `menpo.image.Image`
            The template image.
        shapes : `list` of `menpo.shape.PointCloud`
            The `list` of training shapes.
        group : `str` or ``None``, optional
            The landmark group of the `template` that will be used to train the
            ATM. If ``None`` and the `template` only has a single landmark group,
            then that is the one that will be used.
        shape_forgetting_factor : ``[0.0, 1.0]`` `float`, optional
            Forgetting factor that weights the relative contribution of new
            samples vs old samples for the shape model. If ``1.0``, all samples
            are weighted equally and, hence, the result is the exact same as
            performing batch PCA on the concatenated list of old and new
            simples. If ``<1.0``, more emphasis is put on the new samples.
        verbose : `bool`, optional
            If ``True``, then the progress of building the ATM will be printed.
        batch_size : `int` or ``None``, optional
            If an `int` is provided, then the training is performed in an
            incremental fashion on image batches of size equal to the provided
            value. If ``None``, then the training is performed directly on the
            all the images.
        """
        return self._train(template, shapes, group=group,
                           verbose=verbose,
                           shape_forgetting_factor=shape_forgetting_factor,
                           increment=True, batch_size=batch_size)

    def _build_shape_model(self, shapes, scale_index):
        return self._shape_model_cls[scale_index](
            shapes, max_n_components=self.max_shape_components[scale_index])

    def _increment_shape_model(self, shapes, scale_index,
                               forgetting_factor=None):
        self.shape_models[scale_index].increment(
            shapes, forgetting_factor=forgetting_factor,
            max_n_components=self.max_shape_components[scale_index])

    def _warp_template(self, template, group, reference_shape, scale_index,
                       prefix, verbose):
        reference_frame = build_reference_frame(reference_shape)
        shape = template.landmarks[group].lms
        return warp_images([template], [shape], reference_frame, self.transform,
                           prefix=prefix, verbose=verbose)

    @property
    def n_scales(self):
        """
        Returns the number of scales.

        :type: `int`
        """
        return len(self.scales)

    @property
    def _str_title(self):
        return 'Holistic Active Template Model'

    def instance(self, shape_weights=None, scale_index=-1):
        r"""
        Generates a novel ATM instance given a set of shape weights. If no
        weights are provided, the mean ATM instance is returned.

        Parameters
        ----------
        shape_weights : ``(n_weights,)`` `ndarray` or `list` or ``None``, optional
            The weights of the shape model that will be used to create a novel
            shape instance. If ``None``, the weights are assumed to be zero,
            thus the mean shape is used.
        scale_index : `int`, optional
            The scale to be used.

        Returns
        -------
        image : `menpo.image.Image`
            The ATM instance.
        """
        if shape_weights is None:
            shape_weights = [0]

        sm = self.shape_models[scale_index].model
        template = self.warped_templates[scale_index]

        shape_instance = sm.instance(shape_weights, normalized_weights=True)
        return self._instance(shape_instance, template)

    def random_instance(self, scale_index=-1):
        r"""
        Generates a random instance of the ATM.

        Parameters
        ----------
        scale_index : `int`, optional
            The scale to be used.

        Returns
        -------
        image : `menpo.image.Image`
            The ATM instance.
        """
        sm = self.shape_models[scale_index].model
        template = self.warped_templates[scale_index]

        # TODO: this bit of logic should to be transferred down to PCAModel
        shape_weights = np.random.randn(sm.n_active_components)
        shape_instance = sm.instance(shape_weights, normalized_weights=True)

        return self._instance(shape_instance, template)

    def _instance(self, shape_instance, template):
        landmarks = template.landmarks['source'].lms

        reference_frame = build_reference_frame(shape_instance)

        transform = self.transform(
            reference_frame.landmarks['source'].lms, landmarks)

        return template.as_unmasked(copy=False).warp_to_mask(
            reference_frame.mask, transform, warp_landmarks=True)

    def view_shape_models_widget(self, n_parameters=5,
                                 parameters_bounds=(-3.0, 3.0),
                                 mode='multiple', figure_size=(10, 8)):
        r"""
        Visualizes the shape models of the ATM object using an interactive
        widget.

        Parameters
        ----------
        n_parameters : `int` or `list` of `int` or ``None``, optional
            The number of shape principal components to be used for the
            parameters sliders. If `int`, then the number of sliders per
            scale is the minimum between `n_parameters` and the number of
            active components per scale. If `list` of `int`, then a number of
            sliders is defined per scale. If ``None``, all the active
            components per scale will have a slider.
        parameters_bounds : ``(float, float)``, optional
            The minimum and maximum bounds, in std units, for the sliders.
        mode : {``single``, ``multiple``}, optional
            If ``'single'``, only a single slider is constructed along with a
            drop down menu. If ``'multiple'``, a slider is constructed for
            each parameter.
        figure_size : (`int`, `int`), optional
            The size of the rendered figure.
        """
        try:
            from menpowidgets import visualize_shape_model
            visualize_shape_model(
                [sm.model for sm in self.shape_models],
                n_parameters=n_parameters, parameters_bounds=parameters_bounds,
                figure_size=figure_size, mode=mode)
        except:
            from menpo.visualize.base import MenpowidgetsMissingError
            raise MenpowidgetsMissingError()

    def view_atm_widget(self, n_shape_parameters=5,
                        parameters_bounds=(-3.0, 3.0), mode='multiple',
                        figure_size=(10, 8)):
        r"""
        Visualizes the ATM using an interactive widget.

        Parameters
        ----------
        n_shape_parameters : `int` or `list` of `int` or ``None``, optional
            The number of shape principal components to be used for the
            parameters sliders. If `int`, then the number of sliders per
            scale is the minimum between `n_parameters` and the number of
            active components per scale. If `list` of `int`, then a number of
            sliders is defined per scale. If ``None``, all the active
            components per scale will have a slider.
        parameters_bounds : ``(float, float)``, optional
            The minimum and maximum bounds, in std units, for the sliders.
        mode : {``single``, ``multiple``}, optional
            If ``'single'``, only a single slider is constructed along with a
            drop down menu. If ``'multiple'``, a slider is constructed for
            each parameter.
        figure_size : (`int`, `int`), optional
            The size of the rendered figure.
        """
        try:
            from menpowidgets import visualize_atm
            visualize_atm(self, n_shape_parameters=n_shape_parameters,
                          parameters_bounds=parameters_bounds,
                          figure_size=figure_size, mode=mode)
        except:
            from menpo.visualize.base import MenpowidgetsMissingError
            raise MenpowidgetsMissingError()

    def build_fitter_interfaces(self, sampling):
        r"""
        Method that builds the correct Lucas-Kanade fitting interface.

        Parameters
        ----------
        sampling : `list` of `int` or `ndarray` or ``None``
            It defines a sampling mask per scale. If `int`, then it
            defines the sub-sampling step of the sampling mask. If `ndarray`,
            then it explicitly defines the sampling mask. If ``None``, then no
            sub-sampling is applied.

        Returns
        -------
        fitter_interfaces : `list`
            The `list` of Lucas-Kanade interface per scale.
        """
        interfaces = []
        for wt, sm, s in zip(self.warped_templates, self.shape_models,
                             sampling):
            md_transform = OrthoMDTransform(
                sm, self.transform,
                source=wt.landmarks['source'].lms)
            interface = ATMLucasKanadeStandardInterface(
                md_transform, wt, sampling=s)
            interfaces.append(interface)
        return interfaces

    def __str__(self):
        return _atm_str(self)


class MaskedATM(ATM):
    r"""
    Class for training a multi-scale patch-based Masked Active Template Model.
    The appearance of this model is formulated by simply masking an image
    with a patch-based mask.

    Parameters
    ----------
    template : `menpo.image.Image`
        The template image.
    shapes : `list` of `menpo.shape.PointCloud`
        The `list` of training shapes.
    group : `str` or ``None``, optional
        The landmark group of the `template` that will be used to train the ATM.
        If ``None`` and the `template` only has a single landmark group, then
        that is the one that will be used.
    holistic_features : `closure` or `list` of `closure`, optional
        The features that will be extracted from the training images. Note
        that the features are extracted before warping the images to the
        reference shape. If `list`, then it must define a feature function per
        scale. Please refer to `menpo.feature` for a list of potential features.
    reference_shape : `menpo.shape.PointCloud` or ``None``, optional
        The reference shape that will be used for building the ATM. The purpose
        of the reference shape is to normalise the size of the training images.
        The normalization is performed by rescaling all the training images
        so that the scale of their ground truth shapes matches the scale of
        the reference shape. Note that the reference shape is rescaled with
        respect to the `diagonal` before performing the normalisation. If
        ``None``, then the mean shape will be used.
    diagonal : `int` or ``None``, optional
        This parameter is used to rescale the reference shape so that the
        diagonal of its bounding box matches the provided value. In other
        words, this parameter controls the size of the model at the highest
        scale. If ``None``, then the reference shape does not get rescaled.
    scales : `float` or `tuple` of `float`, optional
        The scale value of each scale. They must provided in ascending order,
        i.e. from lowest to highest scale. If `float`, then a single scale is
        assumed.
    patch_shape : (`int`, `int`), optional
        The size of the patches of the mask that is used to sample the
        appearance vectors.
    max_shape_components : `int`, `float`, `list` of those or ``None``, optional
        The number of shape components to keep. If `int`, then it sets the exact
        number of components. If `float`, then it defines the variance
        percentage that will be kept. If `list`, then it should
        define a value per scale. If a single number, then this will be
        applied to all scales. If ``None``, then all the components are kept.
        Note that the unused components will be permanently trimmed.
    verbose : `bool`, optional
        If ``True``, then the progress of building the ATM will be printed.
    batch_size : `int` or ``None``, optional
        If an `int` is provided, then the training is performed in an
        incremental fashion on image batches of size equal to the provided
        value. If ``None``, then the training is performed directly on the
        all the images.
    """
    def __init__(self, template, shapes, group=None, holistic_features=no_op,
                 reference_shape=None, diagonal=None, scales=(0.5, 1.0),
                 patch_shape=(17, 17), max_shape_components=None,
                 verbose=False, batch_size=None):
        # Check arguments
        self.patch_shape = checks.check_patch_shape(patch_shape, len(scales))
        # Call superclass
        super(MaskedATM, self).__init__(
                template, shapes, group=group,
                holistic_features=holistic_features,
                reference_shape=reference_shape, diagonal=diagonal,
                scales=scales, transform=DifferentiableThinPlateSplines,
                max_shape_components=max_shape_components, verbose=verbose,
                batch_size=batch_size)

    def _warp_template(self, template, group, reference_shape, scale_index,
                       prefix, verbose):
        reference_frame = build_patch_reference_frame(
            reference_shape, patch_shape=self.patch_shape[scale_index])
        shape = template.landmarks[group].lms
        return warp_images([template], [shape], reference_frame, self.transform,
                           prefix=prefix, verbose=verbose)

    @property
    def _str_title(self):
        return 'Masked Active Template Model'

    def _instance(self, shape_instance, template):
        landmarks = template.landmarks['source'].lms

        reference_frame = build_patch_reference_frame(
            shape_instance, patch_shape=self.patch_shape)

        transform = self.transform(
            reference_frame.landmarks['source'].lms, landmarks)

        return template.as_unmasked().warp_to_mask(
            reference_frame.mask, transform, warp_landmarks=True)

    def __str__(self):
        return _atm_str(self)


class LinearATM(ATM):
    r"""
    Class for training a multi-scale Linear Active Template Model.

    Parameters
    ----------
    template : `menpo.image.Image`
        The template image.
    shapes : `list` of `menpo.shape.PointCloud`
        The `list` of training shapes.
    group : `str` or ``None``, optional
        The landmark group of the `template` that will be used to train the ATM.
        If ``None`` and the `template` only has a single landmark group, then
        that is the one that will be used.
    holistic_features : `closure` or `list` of `closure`, optional
        The features that will be extracted from the training images. Note
        that the features are extracted before warping the images to the
        reference shape. If `list`, then it must define a feature function per
        scale. Please refer to `menpo.feature` for a list of potential features.
    reference_shape : `menpo.shape.PointCloud` or ``None``, optional
        The reference shape that will be used for building the ATM. The purpose
        of the reference shape is to normalise the size of the training images.
        The normalization is performed by rescaling all the training images
        so that the scale of their ground truth shapes matches the scale of
        the reference shape. Note that the reference shape is rescaled with
        respect to the `diagonal` before performing the normalisation. If
        ``None``, then the mean shape will be used.
    diagonal : `int` or ``None``, optional
        This parameter is used to rescale the reference shape so that the
        diagonal of its bounding box matches the provided value. In other
        words, this parameter controls the size of the model at the highest
        scale. If ``None``, then the reference shape does not get rescaled.
    scales : `float` or `tuple` of `float`, optional
        The scale value of each scale. They must provided in ascending order,
        i.e. from lowest to highest scale. If `float`, then a single scale is
        assumed.
    transform : `subclass` of :map:`DL` and :map:`DX`, optional
        A differential warp transform object, e.g.
        :map:`DifferentiablePiecewiseAffine` or
        :map:`DifferentiableThinPlateSplines`.
    max_shape_components : `int`, `float`, `list` of those or ``None``, optional
        The number of shape components to keep. If `int`, then it sets the exact
        number of components. If `float`, then it defines the variance
        percentage that will be kept. If `list`, then it should
        define a value per scale. If a single number, then this will be
        applied to all scales. If ``None``, then all the components are kept.
        Note that the unused components will be permanently trimmed.
    verbose : `bool`, optional
        If ``True``, then the progress of building the ATM will be printed.
    batch_size : `int` or ``None``, optional
        If an `int` is provided, then the training is performed in an
        incremental fashion on image batches of size equal to the provided
        value. If ``None``, then the training is performed directly on the
        all the images.
    """
    def __init__(self, template, shapes, group=None, holistic_features=no_op,
                 reference_shape=None, diagonal=None, scales=(0.5, 1.0),
                 transform=DifferentiableThinPlateSplines,
                 max_shape_components=None, verbose=False, batch_size=None):
        super(LinearATM, self).__init__(
                template, shapes, group=group,
                holistic_features=holistic_features,
                reference_shape=reference_shape, diagonal=diagonal,
                scales=scales, transform=transform,
                max_shape_components=max_shape_components, verbose=verbose,
                batch_size=batch_size)

    @property
    def _str_title(self):
        r"""
        Returns a string containing name of the model.
        :type: `string`
        """
        return 'Linear Active Template Model'

    def _build_shape_model(self, shapes, scale_index):
        mean_aligned_shape = mean_pointcloud(align_shapes(shapes))
        self.n_landmarks = mean_aligned_shape.n_points
        self.reference_frame = build_reference_frame(mean_aligned_shape)
        dense_shapes = densify_shapes(shapes, self.reference_frame,
                                      self.transform)

        # Build dense shape model
        max_sc = self.max_shape_components[scale_index]
        return self._shape_model_cls[scale_index](dense_shapes,
                                                  max_n_components=max_sc)

    def _increment_shape_model(self, shapes, scale_index,
                               forgetting_factor=1.0):
        aligned_shapes = align_shapes(shapes)
        dense_shapes = densify_shapes(aligned_shapes, self.reference_frame,
                                      self.transform)
        # Increment shape model
        self.shape_models[scale_index].increment(
            dense_shapes, forgetting_factor=forgetting_factor,
            max_n_components=self.max_shape_components[scale_index])

    def _warp_template(self, template, group, reference_shape, scale_index,
                       prefix, verbose):
        shape = template.landmarks[group].lms
        return warp_images([template], [shape], self.reference_frame,
                           self.transform, prefix=prefix,
                           verbose=verbose)

    # TODO: implement me!
    def _instance(self, shape_instance, template):
        raise NotImplementedError()

    # TODO: implement me!
    def view_atm_widget(self, n_shape_parameters=5,
                        parameters_bounds=(-3.0, 3.0), mode='multiple',
                        figure_size=(10, 8)):
        raise NotImplementedError()

    def build_fitter_interfaces(self, sampling):
        r"""
        Method that builds the correct Lucas-Kanade fitting interface.

        Parameters
        ----------
        sampling : `list` of `int` or `ndarray` or ``None``
            It defines a sampling mask per scale. If `int`, then it
            defines the sub-sampling step of the sampling mask. If `ndarray`,
            then it explicitly defines the sampling mask. If ``None``, then no
            sub-sampling is applied.

        Returns
        -------
        fitter_interfaces : `list`
            The `list` of Lucas-Kanade interface per scale.
        """
        interfaces = []
        for wt, sm, s in zip(self.warped_templates, self.shape_models,
                             sampling):
            # This is pretty hacky as we just steal the OrthoPDM's PCAModel
            md_transform = LinearOrthoMDTransform(
                sm.model, self.reference_shape)
            interface = ATMLucasKanadeLinearInterface(md_transform, wt,
                                                      sampling=s)
            interfaces.append(interface)
        return interfaces

    def __str__(self):
        return _atm_str(self)


class LinearMaskedATM(ATM):
    r"""
    Class for training a multi-scale Linear Masked Active Template Model.

    Parameters
    ----------
    template : `menpo.image.Image`
        The template image.
    shapes : `list` of `menpo.shape.PointCloud`
        The `list` of training shapes.
    group : `str` or ``None``, optional
        The landmark group of the `template` that will be used to train the ATM.
        If ``None`` and the `template` only has a single landmark group, then
        that is the one that will be used.
    holistic_features : `closure` or `list` of `closure`, optional
        The features that will be extracted from the training images. Note
        that the features are extracted before warping the images to the
        reference shape. If `list`, then it must define a feature function per
        scale. Please refer to `menpo.feature` for a list of potential features.
    reference_shape : `menpo.shape.PointCloud` or ``None``, optional
        The reference shape that will be used for building the ATM. The purpose
        of the reference shape is to normalise the size of the training images.
        The normalization is performed by rescaling all the training images
        so that the scale of their ground truth shapes matches the scale of
        the reference shape. Note that the reference shape is rescaled with
        respect to the `diagonal` before performing the normalisation. If
        ``None``, then the mean shape will be used.
    diagonal : `int` or ``None``, optional
        This parameter is used to rescale the reference shape so that the
        diagonal of its bounding box matches the provided value. In other
        words, this parameter controls the size of the model at the highest
        scale. If ``None``, then the reference shape does not get rescaled.
    scales : `float` or `tuple` of `float`, optional
        The scale value of each scale. They must provided in ascending order,
        i.e. from lowest to highest scale. If `float`, then a single scale is
        assumed.
    patch_shape : (`int`, `int`) or `list` of (`int`, `int`), optional
        The shape of the patches of the mask that is used to extract the
        appearance vectors. If a `list` is provided, then it defines a patch
        shape per scale.
    max_shape_components : `int`, `float`, `list` of those or ``None``, optional
        The number of shape components to keep. If `int`, then it sets the exact
        number of components. If `float`, then it defines the variance
        percentage that will be kept. If `list`, then it should
        define a value per scale. If a single number, then this will be
        applied to all scales. If ``None``, then all the components are kept.
        Note that the unused components will be permanently trimmed.
    verbose : `bool`, optional
        If ``True``, then the progress of building the ATM will be printed.
    batch_size : `int` or ``None``, optional
        If an `int` is provided, then the training is performed in an
        incremental fashion on image batches of size equal to the provided
        value. If ``None``, then the training is performed directly on the
        all the images.
    """
    def __init__(self, template, shapes, group=None, holistic_features=no_op,
                 reference_shape=None, diagonal=None, scales=(0.5, 1.0),
                 patch_shape=(17, 17), max_shape_components=None,
                 verbose=False, batch_size=None):
        # Check arguments
        self.patch_shape = checks.check_patch_shape(patch_shape, len(scales))
        # Call superclass
        super(LinearMaskedATM, self).__init__(
                template, shapes, group=group,
                holistic_features=holistic_features,
                reference_shape=reference_shape, diagonal=diagonal,
                scales=scales, transform=DifferentiableThinPlateSplines,
                max_shape_components=max_shape_components, verbose=verbose,
                batch_size=batch_size)

    @property
    def _str_title(self):
        r"""
        Returns a string containing name of the model.
        :type: `string`
        """
        return 'Linear Masked Active Template Model'

    def _build_shape_model(self, shapes, scale_index):
        mean_aligned_shape = mean_pointcloud(align_shapes(shapes))
        self.n_landmarks = mean_aligned_shape.n_points
        self.reference_frame = build_patch_reference_frame(
            mean_aligned_shape, patch_shape=self.patch_shape[scale_index])
        dense_shapes = densify_shapes(shapes, self.reference_frame,
                                      self.transform)
        # Build dense shape model
        max_sc = self.max_shape_components[scale_index]
        return self._shape_model_cls[scale_index](dense_shapes,
                                                  max_n_components=max_sc)

    def _increment_shape_model(self, shapes, scale_index,
                               forgetting_factor=1.0):
        aligned_shapes = align_shapes(shapes)
        dense_shapes = densify_shapes(aligned_shapes, self.reference_frame,
                                      self.transform)
        # Increment shape model
        self.shape_models[scale_index].increment(
            dense_shapes, forgetting_factor=forgetting_factor,
            max_n_components=self.max_shape_components[scale_index])

    def _warp_template(self, template, group, reference_shape, scale_index,
                       prefix, verbose):
        shape = template.landmarks[group].lms
        return warp_images([template], [shape], self.reference_frame,
                           self.transform, prefix=prefix,
                           verbose=verbose)

    # TODO: implement me!
    def _instance(self, shape_instance, template):
        raise NotImplementedError()

    # TODO: implement me!
    def view_atm_widget(self, n_shape_parameters=5,
                        parameters_bounds=(-3.0, 3.0), mode='multiple',
                        figure_size=(10, 8)):
        raise NotImplementedError()

    def build_fitter_interfaces(self, sampling):
        r"""
        Method that builds the correct Lucas-Kanade fitting interface.

        Parameters
        ----------
        sampling : `list` of `int` or `ndarray` or ``None``
            It defines a sampling mask per scale. If `int`, then it
            defines the sub-sampling step of the sampling mask. If `ndarray`,
            then it explicitly defines the sampling mask. If ``None``, then no
            sub-sampling is applied.

        Returns
        -------
        fitter_interfaces : `list`
            The `list` of Lucas-Kanade interface per scale.
        """
        interfaces = []
        for wt, sm, s in zip(self.warped_templates, self.shape_models,
                             sampling):
            # This is pretty hacky as we just steal the OrthoPDM's PCAModel
            md_transform = LinearOrthoMDTransform(
                sm.model, self.reference_shape)
            interface = ATMLucasKanadeLinearInterface(md_transform, wt,
                                                      sampling=s)
            interfaces.append(interface)
        return interfaces

    def __str__(self):
        return _atm_str(self)


# TODO: implement offsets support?
class PatchATM(ATM):
    r"""
    Class for training a multi-scale Patch-Based Active Template Model.

    Parameters
    ----------
    template : `menpo.image.Image`
        The template image.
    shapes : `list` of `menpo.shape.PointCloud`
        The `list` of training shapes.
    group : `str` or ``None``, optional
        The landmark group of the `template` that will be used to train the ATM.
        If ``None`` and the `template` only has a single landmark group, then
        that is the one that will be used.
    holistic_features : `closure` or `list` of `closure`, optional
        The features that will be extracted from the training images. Note
        that the features are extracted before warping the images to the
        reference shape. If `list`, then it must define a feature function per
        scale. Please refer to `menpo.feature` for a list of potential features.
    reference_shape : `menpo.shape.PointCloud` or ``None``, optional
        The reference shape that will be used for building the ATM. The purpose
        of the reference shape is to normalise the size of the training images.
        The normalization is performed by rescaling all the training images
        so that the scale of their ground truth shapes matches the scale of
        the reference shape. Note that the reference shape is rescaled with
        respect to the `diagonal` before performing the normalisation. If
        ``None``, then the mean shape will be used.
    diagonal : `int` or ``None``, optional
        This parameter is used to rescale the reference shape so that the
        diagonal of its bounding box matches the provided value. In other
        words, this parameter controls the size of the model at the highest
        scale. If ``None``, then the reference shape does not get rescaled.
    scales : `float` or `tuple` of `float`, optional
        The scale value of each scale. They must provided in ascending order,
        i.e. from lowest to highest scale. If `float`, then a single scale is
        assumed.
    patch_shape : (`int`, `int`) or `list` of (`int`, `int`), optional
        The shape of the patches to be extracted. If a `list` is provided,
        then it defines a patch shape per scale.
    max_shape_components : `int`, `float`, `list` of those or ``None``, optional
        The number of shape components to keep. If `int`, then it sets the exact
        number of components. If `float`, then it defines the variance
        percentage that will be kept. If `list`, then it should
        define a value per scale. If a single number, then this will be
        applied to all scales. If ``None``, then all the components are kept.
        Note that the unused components will be permanently trimmed.
    verbose : `bool`, optional
        If ``True``, then the progress of building the ATM will be printed.
    batch_size : `int` or ``None``, optional
        If an `int` is provided, then the training is performed in an
        incremental fashion on image batches of size equal to the provided
        value. If ``None``, then the training is performed directly on the
        all the images.
    """
    def __init__(self, template, shapes, group=None, holistic_features=no_op,
                 reference_shape=None, diagonal=None, scales=(0.5, 1.0),
                 patch_shape=(17, 17), patch_normalisation=no_op,
                 max_shape_components=None, verbose=False, batch_size=None):
        # Check arguments
        self.patch_shape = checks.check_patch_shape(patch_shape, len(scales))
        self.patch_normalisation = patch_normalisation
        # Call superclass
        super(PatchATM, self).__init__(
                template, shapes, group=group,
                holistic_features=holistic_features,
                reference_shape=reference_shape, diagonal=diagonal,
                scales=scales, transform=DifferentiableThinPlateSplines,
                max_shape_components=max_shape_components, verbose=verbose,
                batch_size=batch_size)

    @property
    def _str_title(self):
        r"""
        Returns a string containing name of the model.
        :type: `string`
        """
        return 'Patch-based Active Template Model'

    def _warp_template(self, template, group, reference_shape, scale_index,
                       prefix, verbose):
        shape = template.landmarks[group].lms
        return extract_patches([template], [shape],
                               self.patch_shape[scale_index],
                               normalise_function=self.patch_normalisation,
                               prefix=prefix, verbose=verbose)

    def _instance(self, shape_instance, template):
        return shape_instance, template

    def view_atm_widget(self, n_shape_parameters=5,
                        parameters_bounds=(-3.0, 3.0), mode='multiple',
                        figure_size=(10, 8)):
        r"""
        Visualizes the ATM using an interactive widget.

        Parameters
        ----------
        n_shape_parameters : `int` or `list` of `int` or ``None``, optional
            The number of shape principal components to be used for the
            parameters sliders. If `int`, then the number of sliders per
            scale is the minimum between `n_parameters` and the number of
            active components per scale. If `list` of `int`, then a number of
            sliders is defined per scale. If ``None``, all the active
            components per scale will have a slider.
        parameters_bounds : ``(float, float)``, optional
            The minimum and maximum bounds, in std units, for the sliders.
        mode : {``single``, ``multiple``}, optional
            If ``'single'``, only a single slider is constructed along with a
            drop down menu. If ``'multiple'``, a slider is constructed for
            each parameter.
        figure_size : (`int`, `int`), optional
            The size of the rendered figure.
        """
        try:
            from menpowidgets import visualize_patch_atm
            visualize_patch_atm(self, n_shape_parameters=n_shape_parameters,
                                parameters_bounds=parameters_bounds,
                                figure_size=figure_size, mode=mode)
        except:
            from menpo.visualize.base import MenpowidgetsMissingError
            raise MenpowidgetsMissingError()

    def build_fitter_interfaces(self, sampling):
        r"""
        Method that builds the correct Lucas-Kanade fitting interface.

        Parameters
        ----------
        sampling : `list` of `int` or `ndarray` or ``None``
            It defines a sampling mask per scale. If `int`, then it
            defines the sub-sampling step of the sampling mask. If `ndarray`,
            then it explicitly defines the sampling mask. If ``None``, then no
            sub-sampling is applied.

        Returns
        -------
        fitter_interfaces : `list`
            The `list` of Lucas-Kanade interface per scale.
        """
        interfaces = []
        for j, (wt, sm, s) in enumerate(zip(self.warped_templates,
                                            self.shape_models,
                                            sampling)):
            interface = ATMLucasKanadePatchInterface(
                sm, wt, sampling=s,
                patch_shape=self.patch_shape[j],
                patch_normalisation=self.patch_normalisation)
            interfaces.append(interface)
        return interfaces

    def __str__(self):
        return _atm_str(self)


def _atm_str(atm):
    if atm.diagonal is not None:
        diagonal = atm.diagonal
    else:
        y, x = atm.reference_shape.range()
        diagonal = np.sqrt(x ** 2 + y ** 2)

    # Compute scale info strings
    scales_info = []
    lvl_str_tmplt = r"""   - Scale {}
     - Holistic feature: {}
     - Template shape: {}
     - Shape model class: {}
       - {} shape components
       - {} similarity transform parameters"""
    for k, s in enumerate(atm.scales):
        scales_info.append(lvl_str_tmplt.format(
            s, name_of_callable(atm.holistic_features[k]),
            atm.warped_templates[k].shape,
            name_of_callable(atm.shape_models[k]),
            atm.shape_models[k].model.n_components,
            atm.shape_models[k].n_global_parameters))
    # Patch based ATM
    if hasattr(atm, 'patch_shape'):
        for k in range(len(scales_info)):
            scales_info[k] += '\n     - Patch shape: {}'.format(
                atm.patch_shape[k])
    scales_info = '\n'.join(scales_info)

    cls_str = r"""{class_title}
 - Images warped with {transform} transform
 - Images scaled to diagonal: {diagonal:.2f}
 - Scales: {scales}
{scales_info}
""".format(class_title=atm._str_title,
           transform=name_of_callable(atm.transform),
           diagonal=diagonal,
           scales=atm.scales,
           scales_info=scales_info)
    return cls_str

HolisticATM = ATM
