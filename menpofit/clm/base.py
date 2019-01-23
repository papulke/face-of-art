from __future__ import division
import warnings
import numpy as np

from menpo.base import name_of_callable
from menpo.feature import no_op
from menpo.visualize import print_dynamic

from menpofit import checks
from menpofit.base import batch
from menpofit.builder import (compute_features, scale_images,
                              MenpoFitBuilderWarning, compute_reference_shape,
                              rescale_images_to_reference_shape)
from menpofit.modelinstance import OrthoPDM

from .expert import CorrelationFilterExpertEnsemble
from .expert.ensemble import ConvolutionBasedExpertEnsemble, FcnFilterExpertEnsemble


class CLM(object):
    r"""
    Class for training a multi-scale holistic Constrained Local Model. Please
    see the references for a basic list of relevant papers.

    Parameters
    ----------
    images : `list` of `menpo.image.Image`
        The `list` of training images.
    group : `str` or ``None``, optional
        The landmark group that will be used to train the CLM. If ``None`` and
        the images only have a single landmark group, then that is the one
        that will be used. Note that all the training images need to have the
        specified landmark group.
    holistic_features : `closure` or `list` of `closure`, optional
        The features that will be extracted from the training images. If `list`,
        then it must define a feature function per scale. Please refer to
        `menpo.feature` for a list of potential features.
    reference_shape : `menpo.shape.PointCloud` or ``None``, optional
        The reference shape that will be used for building the CLM. The purpose
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
    patch_normalisation : `callable`, optional
        The normalisation function to be applied on the extracted patches.
    context_shape : (`int`, `int`) or `list` of (`int`, `int`), optional
        The context shape for the convolution. If a `list` is provided,
        then it defines a context shape per scale.
    cosine_mask : `bool`, optional
        If ``True``, then a cosine mask (Hanning function) will be applied on
        the extracted patches.
    sample_offsets : ``(n_offsets, n_dims)`` `ndarray` or ``None``, optional
        The offsets to sample from within a patch. So ``(0, 0)`` is the centre
        of the patch (no offset) and ``(1, 0)`` would be sampling the patch
        from 1 pixel up the first axis away from the centre. If ``None``,
        then no offsets are applied.
    shape_model_cls : `subclass` of :map:`PDM`, optional
        The class to be used for building the shape model. The most common
        choice is :map:`OrthoPDM`.
    expert_ensemble_cls : `subclass` of :map:`ExpertEnsemble`, optional
        The class to be used for training the ensemble of experts. The most
        common choice is :map:`CorrelationFilterExpertEnsemble`.
    max_shape_components : `int`, `float`, `list` of those or ``None``, optional
        The number of shape components to keep. If `int`, then it sets the exact
        number of components. If `float`, then it defines the variance
        percentage that will be kept. If `list`, then it should
        define a value per scale. If a single number, then this will be
        applied to all scales. If ``None``, then all the components are kept.
        Note that the unused components will be permanently trimmed.
    verbose : `bool`, optional
        If ``True``, then the progress of building the CLM will be printed.
    batch_size : `int` or ``None``, optional
        If an `int` is provided, then the training is performed in an
        incremental fashion on image batches of size equal to the provided
        value. If ``None``, then the training is performed directly on the
        all the images.

    References
    ----------
    .. [1] D. Cristinacce, and T. F. Cootes. "Feature Detection and Tracking
        with Constrained Local Models", British Machine Vision Conference (BMVC),
        2006.
    .. [2] J.M. Saragih, S. Lucey, and J. F. Cohn. "Deformable model fitting by
        regularized landmark mean-shift", International Journal of Computer
        Vision (IJCV), 91(2): 200-215, 2011.
    .. [3] T. F. Cootes, C. J. Taylor, D. H. Cooper, and J. Graham. "Active
        Shape Models - their training and application", Computer Vision and Image
        Understanding (CVIU), 61(1): 38-59, 1995.
    """
    def __init__(self, images, group=None, holistic_features=no_op,
                 reference_shape=None, diagonal=None, scales=(1, ),                  # scales=(0.5, 1)
                 patch_shape=(8, 8), patch_normalisation=no_op,
                 context_shape=(8, 8), cosine_mask=True, sample_offsets=None,
                 shape_model_cls=OrthoPDM,
                 expert_ensemble_cls=CorrelationFilterExpertEnsemble,
                 max_shape_components=None, verbose=False, batch_size=None):
        self.scales = checks.check_scales(scales)
        n_scales = len(scales)
        self.diagonal = checks.check_diagonal(diagonal)
        self.holistic_features = checks.check_callable(holistic_features,
                                                       n_scales)
        self.expert_ensemble_cls = checks.check_callable(expert_ensemble_cls,
                                                         n_scales)
        self._shape_model_cls = checks.check_callable(shape_model_cls,
                                                      n_scales)
        self.max_shape_components = checks.check_max_components(
            max_shape_components, n_scales, 'max_shape_components')
        self.reference_shape = reference_shape
        self.patch_shape = checks.check_patch_shape(patch_shape, n_scales)
        self.patch_normalisation = patch_normalisation
        self.context_shape = checks.check_patch_shape(context_shape, n_scales)
        self.cosine_mask = cosine_mask
        self.sample_offsets = sample_offsets
        self.shape_models = []
        self.expert_ensembles = []

        # Train CLM
        self._train(images, increment=False, group=group, verbose=verbose,
                        batch_size=batch_size)

    @property
    def _str_title(self):
        return "Constrained Local Model"

    @property
    def n_scales(self):
        """
        Returns the number of scales.

        :type: `int`
        """
        return len(self.scales)

    def _train(self, images, increment=False, group=None, verbose=False,
               shape_forgetting_factor=1.0, batch_size=None):
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
                    self.reference_shape = compute_reference_shape(
                        [i.landmarks[group].lms for i in image_batch],
                        self.diagonal, verbose=verbose)

            # After the first batch, we are incrementing the model
            if k > 0:
                increment = True

            if verbose:
                print('Computing batch {}'.format(k))

            # Train each batch
            self._train_batch(image_batch, increment=increment, group=group,
                              shape_forgetting_factor=shape_forgetting_factor,
                              verbose=verbose)

    def _train_batch(self, image_batch, increment=False, group=None,
                     shape_forgetting_factor=1.0, verbose=False):
        # normalize images
        image_batch = rescale_images_to_reference_shape(
            image_batch, group, self.reference_shape, verbose=verbose)

        # build models at each scale
        if verbose:
            print_dynamic('- Training models\n')

        # for each level (low --> high)
        for i in range(self.n_scales):
            if verbose:
                if self.n_scales > 1:
                    prefix = '  - Scale {}: '.format(i)
                else:
                    prefix = '  - '
            else:
                prefix = None

            # Handle holistic features
            if i == 0 and self.holistic_features[i] == no_op:
                # Saves a lot of memory
                feature_images = image_batch
            elif i == 0 or self.holistic_features[i] is not self.holistic_features[i - 1]:
                # compute features only if this is the first pass through
                # the loop or the features at this scale are different from
                # the features at the previous scale
                feature_images = compute_features(image_batch,
                                                  self.holistic_features[i],
                                                  prefix=prefix,
                                                  verbose=verbose)
            # handle scales
            if self.scales[i] != 1:
                # scale feature images only if scale is different than 1
                scaled_images = scale_images(feature_images,
                                             self.scales[i],
                                             prefix=prefix,
                                             verbose=verbose)
            else:
                scaled_images = feature_images

            # extract scaled shapes
            scaled_shapes = [image.landmarks[group].lms
                             for image in scaled_images]

            # train shape model
            if verbose:
                print_dynamic('{}Training shape model'.format(prefix))

            if not increment:
                shape_model = self._build_shape_model(scaled_shapes, i)
                self.shape_models.append(shape_model)
            else:
                self._increment_shape_model(
                    scaled_shapes, i, forgetting_factor=shape_forgetting_factor)

            # train expert ensemble
            if verbose:
                print_dynamic('{}Training expert ensemble'.format(prefix))

            if increment:
                self.expert_ensembles[i].increment(scaled_images,
                                                   scaled_shapes,
                                                   prefix=prefix,
                                                   verbose=verbose)
            else:
                expert_ensemble = self.expert_ensemble_cls[i](
                        images=scaled_images, shapes=scaled_shapes,
                        patch_shape=self.patch_shape[i],
                        patch_normalisation=self.patch_normalisation,
                        cosine_mask=self.cosine_mask,
                        context_shape=self.context_shape[i],
                        sample_offsets=self.sample_offsets,
                        prefix=prefix, verbose=verbose)
                self.expert_ensembles.append(expert_ensemble)

            if verbose:
                print_dynamic('{}Done\n'.format(prefix))

    def _build_shape_model(self, shapes, scale_index):
        return self._shape_model_cls[scale_index](
            shapes, max_n_components=self.max_shape_components[scale_index])

    def _increment_shape_model(self, shapes, scale_index,
                               forgetting_factor=None):
        self.shape_models[scale_index].increment(
            shapes, forgetting_factor=forgetting_factor,
            max_n_components=self.max_shape_components[scale_index])

    def increment(self, images, group=None, shape_forgetting_factor=1.0,
                  verbose=False, batch_size=None):
        r"""
        Method to increment the trained CLM with a new set of training images.

        Parameters
        ----------
        images : `list` of `menpo.image.Image`
            The `list` of training images.
        group : `str` or ``None``, optional
            The landmark group that will be used to train the CLM. If ``None``
            and the images only have a single landmark group, then that is the
            one that will be used. Note that all the training images need to
            have the specified landmark group.
        shape_forgetting_factor : ``[0.0, 1.0]`` `float`, optional
            Forgetting factor that weights the relative contribution of new
            samples vs old samples for the shape model. If ``1.0``, all samples
            are weighted equally and, hence, the result is the exact same as
            performing batch PCA on the concatenated list of old and new
            simples. If ``<1.0``, more emphasis is put on the new samples.
        verbose : `bool`, optional
            If ``True``, then the progress of building the CLM will be printed.
        batch_size : `int` or ``None``, optional
            If an `int` is provided, then the training is performed in an
            incremental fashion on image batches of size equal to the provided
            value. If ``None``, then the training is performed directly on the
            all the images.
        """
        return self._train(images, increment=True, group=group, verbose=verbose,
                           shape_forgetting_factor=shape_forgetting_factor,
                           batch_size=batch_size)

    def shape_instance(self, shape_weights=None, scale_index=-1):
        r"""
        Generates a novel shape instance given a set of shape weights. If no
        weights are provided, the mean shape is returned.

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
        instance : `menpo.shape.PointCloud`
            The shape instance.
        """
        if shape_weights is None:
            shape_weights = [0]
        sm = self.shape_models[scale_index].model
        return sm.instance(shape_weights, normalized_weights=True)

    def view_shape_models_widget(self, n_parameters=5,
                                 parameters_bounds=(-3.0, 3.0),
                                 mode='multiple', figure_size=(10, 8)):
        r"""
        Visualizes the shape models of the CLM object using an interactive
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

    def view_expert_ensemble_widget(self, figure_size=(10, 8)):
        r"""
        Visualizes the ensemble of experts of the CLM object using an
        interactive widget.

        Parameters
        ----------
        figure_size : (`int`, `int`), optional
            The size of the plotted figures.

        Raises
        ------
        ValueError
            Only convolution-based expert ensembles can be visualized.
        """
        if not isinstance(self.expert_ensembles[0],
                          ConvolutionBasedExpertEnsemble):
            raise ValueError('Only convolution-based expert ensembles can be '
                             'visualized.')
        try:
            from menpowidgets import visualize_expert_ensemble
            centers = [sp.model.mean() for sp in self.shape_models]
            visualize_expert_ensemble(self.expert_ensembles, centers,
                                      figure_size=figure_size)
        except:
            from menpo.visualize.base import MenpowidgetsMissingError
            raise MenpowidgetsMissingError()

    def view_clm_widget(self, n_shape_parameters=5,
                        parameters_bounds=(-3.0, 3.0), mode='multiple',
                        figure_size=(10, 8)):
        r"""
        Visualizes the CLM object using an interactive widget.

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

        Raises
        ------
        ValueError
            Only convolution-based expert ensembles can be visualized.
        """
        if not isinstance(self.expert_ensembles[0],
                          ConvolutionBasedExpertEnsemble):
            raise ValueError('Only convolution-based expert ensembles can be '
                             'visualized.')
        try:
            from menpowidgets import visualize_clm
            visualize_clm(self, n_shape_parameters=n_shape_parameters,
                          parameters_bounds=parameters_bounds,
                          figure_size=figure_size, mode=mode)
        except:
            from menpo.visualize.base import MenpowidgetsMissingError
            raise MenpowidgetsMissingError()

    def __str__(self):
        if self.diagonal is not None:
            diagonal = self.diagonal
        else:
            y, x = self.reference_shape.range()
            diagonal = np.sqrt(x ** 2 + y ** 2)

        # Compute scale info strings
        scales_info = []
        lvl_str_tmplt = r"""   - Scale {}
     - Holistic feature: {}
     - Ensemble of experts class: {}
       - {} experts
       - {} class
       - Patch shape: {} x {}
       - Patch normalisation: {}
       - Context shape: {} x {}
       - Cosine mask: {}
     - Shape model class: {}
       - {} shape components
       - {} similarity transform parameters"""
        for k, s in enumerate(self.scales):
            scales_info.append(lvl_str_tmplt.format(
                    s, name_of_callable(self.holistic_features[k]),
                    name_of_callable(self.expert_ensemble_cls[k]),
                    self.expert_ensembles[k].n_experts,
                    name_of_callable(self.expert_ensembles[k]._icf),
                    self.expert_ensembles[k].patch_shape[0],
                    self.expert_ensembles[k].patch_shape[1],
                    name_of_callable(self.expert_ensembles[k].patch_normalisation),
                    self.expert_ensembles[k].context_shape[0],
                    self.expert_ensembles[k].context_shape[1],
                    self.expert_ensembles[k].cosine_mask,
                    name_of_callable(self.shape_models[k]),
                    self.shape_models[k].model.n_components,
                    self.shape_models[k].n_global_parameters))
        scales_info = '\n'.join(scales_info)

        cls_str = r"""{class_title}
 - Images scaled to diagonal: {diagonal:.2f}
 - Scales: {scales}
{scales_info}
""".format(class_title=self._str_title,
           diagonal=diagonal,
           scales=self.scales,
           scales_info=scales_info)
        return cls_str
