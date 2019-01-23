from __future__ import division
from functools import partial
import numpy as np
from scipy.stats import multivariate_normal

from menpo.feature import normalize_norm
from menpo.shape import PointCloud
from menpo.image import Image
from menpo.base import name_of_callable

from menpofit.base import build_grid
from menpofit.math.fft_utils import (fft2, ifft2, fftshift, pad, crop,
                                     fft_convolve2d_sum)
from menpofit.visualize import print_progress

from .base import IncrementalCorrelationFilterThinWrapper, probability_map


channel_normalize_norm = partial(normalize_norm,  mode='per_channel',
                                 error_on_divide_by_zero=False)


class ExpertEnsemble(object):
    r"""
    Abstract class for defining an ensemble of patch experts that correspond
    to landmark points.
    """
    @property
    def n_experts(self):
        r"""
        Returns the number of experts.

        :type: `int`
        """
        pass

    @property
    def search_shape(self):
        r"""
        Returns the search shape (`patch_shape`).

        :type: (`int`, `int`)
        """
        pass

    def predict_response(self, image, shape):
        r"""
        Method for predicting the response of the experts on a given image.

        Parameters
        ----------
        image : `menpo.image.Image` or `subclass`
            The test image.
        shape : `menpo.shape.PointCloud`
            The shape that corresponds to the image from which the patches
            will be extracted.

        Returns
        -------
        response : ``(n_experts, 1, height, width)`` `ndarray`
            The response of each expert.
        """
        pass

    def predict_probability(self, image, shape):
        r"""
        Method for predicting the probability map of the response experts on a
        given image. Note that the provided shape must have the same number of
        points as the number of experts.

        Parameters
        ----------
        image : `menpo.image.Image` or `subclass`
            The test image.
        shape : `menpo.shape.PointCloud`
            The shape that corresponds to the image from which the patches
            will be extracted.

        Returns
        -------
        probability_map : ``(n_experts, 1, height, width)`` `ndarray`
            The probability map of the response of each expert.
        """
        # Predict responses
        responses = self.predict_response(image, shape)
        # Turn them into proper probability maps
        return probability_map(responses)


class FcnFilterExpertEnsemble(ExpertEnsemble):
    r"""
    class for generate response map when given patches.
    """
    def __init__(self, images, shapes,
                 icf_cls=None,
                 patch_shape=(30, 30), context_shape=(30, 30),
                 response_covariance=3,
                 patch_normalisation=channel_normalize_norm,
                 cosine_mask=True, sample_offsets=None, prefix='',
                 verbose=False):
        # TODO: check parameters?
        # Set parameters
        self._icf = icf_cls
        self.patch_shape = patch_shape
        self.context_shape = context_shape
        self.response_covariance = response_covariance
        self.patch_normalisation = patch_normalisation
        self.cosine_mask = cosine_mask
        self.sample_offsets = sample_offsets

    @property
    def n_experts(self):
        r"""
        Returns the number of experts.

        :type: `int`
        """
        return 0

    @property
    def search_shape(self):
        r"""
        Returns the search shape (`patch_shape`).

        :type: (`int`, `int`)
        """
        return self.patch_shape

    def predict_response(self, image, shape, grid):
        r"""
        Method for predicting the response of the experts on a given image.

        Parameters
        ----------
        image : `menpo.image.Image` or `subclass`
            The test image.
        shape : `menpo.shape.PointCloud`
            The shape that corresponds to the image from which the patches
            will be extracted.

        Returns
        -------
        response : ``(n_experts, 1, height, width)`` `ndarray`
            The response of each expert.
        """
        # Predict responses
        '''
        rOffset = self.patch_shape[0]/2
        lOffset = self.patch_shape[0] - rOffset
        ratioW = image.shape[-1]/image.rspmap_data.shape[-1]
        ratioH = image.shape[-2]/image.rspmap_data.shape[-2]
        '''
        '''
        rspList = [image.rspmap_data[0, i, y-rOffset:y+lOffset, x-rOffset:x+lOffset]  for i in range(shape.n_points)
                                          for y in [np.around(shape.points[i][0]/ratioH+0.5)]
                                          for x in [np.around(shape.points[i][1]/ratioW+0.5)]]
        '''
        rOffset = np.floor(grid[0]/2).astype(int)
        lOffset = grid[0] - rOffset
        padH = int(image.shape[0]/2)
        padW = int(image.shape[1]/2)
        rspList = [image.rspmap_data[0, i, y-rOffset:y+lOffset, x-rOffset:x+lOffset]  for i in range(shape.n_points)
                                          for y in [np.around(shape.points[i][0]+1+padH).astype(int)]
                                          for x in [np.around(shape.points[i][1]+1+padW).astype(int)]]

        # debug mode:   ValueError: resize only works on single-segment arrays
        '''
        responses = []
        for i in rspList:
            #i_resize = np.array(i)
            #i_resize.resize((15, 15))
            i = np.array(i)
            i.resize(self.patch_shape)
            responses.append(i)
        '''
        responses = np.array(rspList)[:, None, :, :]

        return responses


    def predict_probability(self, image, shape, grid):
        r"""
        Method for predicting the probability map of the response experts on a
        given image. Note that the provided shape must have the same number of
        points as the number of experts.

        Parameters
        ----------
        image : `menpo.image.Image` or `subclass`
            The test image.
        shape : `menpo.shape.PointCloud`
            The shape that corresponds to the image from which the patches
            will be extracted.

        Returns
        -------
        probability_map : ``(n_experts, 1, height, width)`` `ndarray`
            The probability map of the response of each expert.
        """

        return self.predict_response(image, shape, grid)
        # Turn them into proper probability maps
        #return probability_map(responses)



# TODO: Should convolutional experts of ensembles support patch features?
class ConvolutionBasedExpertEnsemble(ExpertEnsemble):
    r"""
    Base class for defining an ensemble of convolution-based patch experts.
    """
    @property
    def n_experts(self):
        r"""
        Returns the number of experts.

        :type: `int`
        """
        return self.fft_padded_filters.shape[0]

    @property
    def n_sample_offsets(self):
        r"""
        Returns the number of offsets that are sampled within a patch.

        :type: `int`
        """
        if self.sample_offsets:
            return self.sample_offsets.shape[0]
        else:
            return 1

    @property
    def padded_size(self):
        r"""
        Returns the convolution pad size, i.e. ``floor(1.5 * patch_shape - 1)``.

        :type: (`int`, `int`)
        """
        pad_size = np.floor(1.5 * np.asarray(self.patch_shape) - 1).astype(int)
        return tuple(pad_size)

    @property
    def search_shape(self):
        r"""
        Returns the search shape (`patch_shape`).

        :type: (`int`, `int`)
        """
        return self.patch_shape

    def increment(self, images, shapes, prefix='', verbose=False):
        r"""
        Increments the learned ensemble of convolution-based experts given a new
        set of training data.

        Parameters
        ----------
        images : `list` of `menpo.image.Image`
            The list of training images.
        shapes : `list` of `menpo.shape.PointCloud`
            The list of training shapes that correspond to the images.
        prefix : `str`, optional
            The prefix of the printed training progress.
        verbose : `bool`, optional
            If ``True``, then information about the training progress will be
            printed.
        """
        self._train(images, shapes, prefix=prefix, verbose=verbose,
                    increment=True)

    @property
    def spatial_filter_images(self):
        r"""
        Returns a `list` of `n_experts` filter images on the spatial domain.

        :type: `list` of `menpo.image.Image`
        """
        filter_images = []
        for fft_padded_filter in self.fft_padded_filters:
            spatial_filter = np.real(ifft2(fft_padded_filter))
            spatial_filter = crop(spatial_filter,
                                  self.patch_shape)[:, ::-1, ::-1]
            filter_images.append(Image(spatial_filter))
        return filter_images

    @property
    def frequency_filter_images(self):
        r"""
        Returns a `list` of `n_experts` filter images on the frequency domain.

        :type: `list` of `menpo.image.Image`
        """
        filter_images = []
        for fft_padded_filter in self.fft_padded_filters:
            spatial_filter = np.real(ifft2(fft_padded_filter))
            spatial_filter = crop(spatial_filter,
                                  self.patch_shape)[:, ::-1, ::-1]
            frequency_filter = np.abs(fftshift(fft2(spatial_filter)))
            filter_images.append(Image(frequency_filter))
        return filter_images

    def _extract_patch(self, image, landmark):
        # Extract patch from image
        patch = image.extract_patches(
            landmark, patch_shape=self.patch_shape,
            sample_offsets=self.sample_offsets, as_single_array=True)
        # Reshape patch
        # patch: (offsets x ch) x h x w
        patch = patch.reshape((-1,) + patch.shape[-2:])
        # Normalise patch
        return self.patch_normalisation(patch)

    def _extract_patches(self, image, shape):
        # Obtain patch ensemble, the whole shape is used to extract patches
        # from all landmarks at once
        patches = image.extract_patches(shape, patch_shape=self.patch_shape,
                                        sample_offsets=self.sample_offsets,
                                        as_single_array=True)
        # Reshape patches
        # patches: n_patches x (n_offsets x n_channels) x height x width
        patches = patches.reshape((patches.shape[0], -1) + patches.shape[-2:])
        # Normalise patches
        return self.patch_normalisation(patches)

    def predict_response(self, image, shape):
        r"""
        Method for predicting the response of the experts on a given image. Note
        that the provided shape must have the same number of points as the
        number of experts.

        Parameters
        ----------
        image : `menpo.image.Image` or `subclass`
            The test image.
        shape : `menpo.shape.PointCloud`
            The shape that corresponds to the image from which the patches
            will be extracted.

        Returns
        -------
        response : ``(n_experts, 1, height, width)`` `ndarray`
            The response of each expert.
        """
        # Extract patches
        patches = self._extract_patches(image, shape)
        # Predict responses
        return fft_convolve2d_sum(patches, self.fft_padded_filters,
                                  fft_filter=True, axis=1)

    def view_spatial_filter_images_widget(self, figure_size=(10, 8),
                                          style='coloured',
                                          browser_style='buttons'):
        r"""
        Visualizes the filters on the spatial domain using an interactive widget.

        Parameters
        ----------
        figure_size : (`int`, `int`), optional
            The initial size of the rendered figure.
        style : {``'coloured'``, ``'minimal'``}, optional
            If ``'coloured'``, then the style of the widget will be coloured. If
            ``minimal``, then the style is simple using black and white colours.
        browser_style : {``'buttons'``, ``'slider'``}, optional
            It defines whether the selector of the objects will have the form of
            plus/minus buttons or a slider.
        """
        try:
            from menpowidgets import visualize_images
            visualize_images(self.spatial_filter_images,
                             figure_size=figure_size, style=style,
                             browser_style=browser_style)
        except ImportError:
            from menpo.visualize.base import MenpowidgetsMissingError
            raise MenpowidgetsMissingError()

    def view_frequency_filter_images_widget(self, figure_size=(10, 8),
                                            style='coloured',
                                            browser_style='buttons'):
        r"""
        Visualizes the filters on the frequency domain using an interactive
        widget.

        Parameters
        ----------
        figure_size : (`int`, `int`), optional
            The initial size of the rendered figure.
        style : {``'coloured'``, ``'minimal'``}, optional
            If ``'coloured'``, then the style of the widget will be coloured. If
            ``minimal``, then the style is simple using black and white colours.
        browser_style : {``'buttons'``, ``'slider'``}, optional
            It defines whether the selector of the objects will have the form of
            plus/minus buttons or a slider.
        """
        try:
            from menpowidgets import visualize_images
            visualize_images(self.frequency_filter_images,
                             figure_size=figure_size, style=style,
                             browser_style=browser_style)
        except ImportError:
            from menpo.visualize.base import MenpowidgetsMissingError
            raise MenpowidgetsMissingError()


class CorrelationFilterExpertEnsemble(ConvolutionBasedExpertEnsemble):
    r"""
    Class for defining an ensemble of correlation filter experts.

    Parameters
    ----------
    images : `list` of `menpo.image.Image`
        The `list` of training images.
    shapes : `list` of `menpo.shape.PointCloud`
        The `list` of training shapes that correspond to the images.
    icf_cls : `class`, optional
        The incremental correlation filter class. For example
        :map:`IncrementalCorrelationFilterThinWrapper`.
    patch_shape : (`int`, `int`), optional
        The shape of the patches that will be extracted around the landmarks.
        Those patches are used to train the experts.
    context_shape : (`int`, `int`), optional
        The context shape for the convolution.
    response_covariance : `int`, optional
        The covariance of the generated Gaussian response.
    patch_normalisation : `callable`, optional
        A normalisation function that will be applied on the extracted patches.
    cosine_mask : `bool`, optional
        If ``True``, then a cosine mask (Hanning function) will be applied on
        the extracted patches.
    sample_offsets : ``(n_offsets, n_dims)`` `ndarray` or ``None``, optional
        The offsets to sample from within a patch. So ``(0, 0)`` is the centre
        of the patch (no offset) and ``(1, 0)`` would be sampling the patch
        from 1 pixel up the first axis away from the centre. If ``None``,
        then no offsets are applied.
    prefix : `str`, optional
        The prefix of the printed progress information.
    verbose : `bool`, optional
        If ``True``, then information will be printed regarding the training
        progress.
    """
    def __init__(self, images, shapes,
                 icf_cls=IncrementalCorrelationFilterThinWrapper,
                 patch_shape=(17, 17), context_shape=(34, 34),
                 response_covariance=3,
                 patch_normalisation=channel_normalize_norm,
                 cosine_mask=True, sample_offsets=None, prefix='',
                 verbose=False):
        # TODO: check parameters?
        # Set parameters
        self._icf = icf_cls()
        self.patch_shape = patch_shape
        self.context_shape = context_shape
        self.response_covariance = response_covariance
        self.patch_normalisation = patch_normalisation
        self.cosine_mask = cosine_mask
        self.sample_offsets = sample_offsets

        # Generate cosine mask
        self._cosine_mask = generate_cosine_mask(self.context_shape)

        # Generate desired response, i.e. a Gaussian response with the
        # specified covariance centred at the middle of the patch
        self.response = generate_gaussian_response(
            self.patch_shape, self.response_covariance)[None, ...]

        # Train ensemble of correlation filter experts
        self._train(images, shapes, verbose=verbose, prefix=prefix)

    def _extract_patch(self, image, landmark):
        # Extract patch from image
        patch = image.extract_patches(
            landmark, patch_shape=self.context_shape,
            sample_offsets=self.sample_offsets, as_single_array=True)
        # Reshape patch
        # patch: (offsets x ch) x h x w
        patch = patch.reshape((-1,) + patch.shape[-2:])
        # Normalise patch
        patch = self.patch_normalisation(patch)
        if self.cosine_mask:
            # Apply cosine mask if required
            patch = self._cosine_mask * patch
        return patch

    def _train(self, images, shapes, prefix='', verbose=False,
               increment=False):
        # Define print_progress partial
        wrap = partial(print_progress,
                       prefix='{}Training experts'
                              .format(prefix),
                       end_with_newline=not prefix,
                       verbose=verbose)

        # If increment is False, we need to initialise/reset the ensemble of
        # experts
        if not increment:
            self.fft_padded_filters = []
            self.auto_correlations = []
            self.cross_correlations = []
            # Set number of images
            self.n_images = len(images)
        else:
            # Update number of images
            self.n_images += len(images)

        # Obtain total number of experts
        n_experts = shapes[0].n_points

        # Train ensemble of correlation filter experts
        fft_padded_filters = []
        auto_correlations = []
        cross_correlations = []
        for i in wrap(range(n_experts)):
            patches = []
            for image, shape in zip(images, shapes):
                # Select the appropriate landmark
                landmark = PointCloud([shape.points[i]])
                # Extract patch
                patch = self._extract_patch(image, landmark)
                # Add patch to the list
                patches.append(patch)

            if increment:
                # Increment correlation filter
                correlation_filter, auto_correlation, cross_correlation = (
                    self._icf.increment(self.auto_correlations[i],
                                        self.cross_correlations[i],
                                        self.n_images,
                                        patches,
                                        self.response))
            else:
                # Train correlation filter
                correlation_filter, auto_correlation, cross_correlation = (
                    self._icf.train(patches, self.response))

            # Pad filter with zeros
            padded_filter = pad(correlation_filter, self.padded_size)
            # Compute fft of padded filter
            fft_padded_filter = fft2(padded_filter)
            # Add fft padded filter to list
            fft_padded_filters.append(fft_padded_filter)
            auto_correlations.append(auto_correlation)
            cross_correlations.append(cross_correlation)

        # Turn list into ndarray
        self.fft_padded_filters = np.asarray(fft_padded_filters)
        self.auto_correlations = np.asarray(auto_correlations)
        self.cross_correlations = np.asarray(cross_correlations)

    def __str__(self):
        cls_str = r"""Ensemble of Correlation Filter Experts
 - {n_experts} experts
 - {icf_cls} class
 - Patch shape: {patch_height} x {patch_width}
 - Patch normalisation: {patch_norm}
 - Context shape: {context_height} x {context_width}
 - Cosine mask: {cosine_mask}""".format(
                n_experts=self.n_experts,
                icf_cls=name_of_callable(self._icf),
                patch_height=self.patch_shape[0],
                patch_width=self.patch_shape[1],
                patch_norm=name_of_callable(self.patch_normalisation),
                context_height=self.context_shape[0],
                context_width=self.context_shape[1],
                cosine_mask=self.cosine_mask)
        return cls_str


def generate_gaussian_response(patch_shape, response_covariance):
    r"""
    Method that generates a Gaussian response (probability density function)
    given the desired shape and a covariance value.

    Parameters
    ----------
    patch_shape : (`int`, `int`), optional
        The shape of the response.
    response_covariance : `int`, optional
        The covariance of the generated Gaussian response.

    Returns
    -------
    pdf : ``(patch_height, patch_width)`` `ndarray`
        The generated response.
    """
    grid = build_grid(patch_shape)
    mvn = multivariate_normal(mean=np.zeros(2), cov=response_covariance)
    return mvn.pdf(grid)


def generate_cosine_mask(patch_shape):
    r"""
    Function that generates a cosine mask (Hanning window).

    Parameters
    ----------
    patch_shape : (`int`, `int`), optional
        The shape of the mask.

    Returns
    -------
    mask : ``(patch_height, patch_width)`` `ndarray`
        The generated Hanning window.
    """
    cy = np.hanning(patch_shape[0])
    cx = np.hanning(patch_shape[1])
    return cy[..., None].dot(cx[None, ...])


