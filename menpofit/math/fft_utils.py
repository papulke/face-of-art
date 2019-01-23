from __future__ import division
import warnings
import numpy as np
from functools import wraps

from menpo.feature.base import rebuild_feature_image

try:
    # try importing pyfftw
    from pyfftw.interfaces.numpy_fft import fft2, ifft2, fftshift, ifftshift

    try:
        # try calling fft2 on a 4-dimensional array (this is known to have
        # problem in some linux distributions)
        fft2(np.zeros((1, 1, 1, 1)))
    except RuntimeError:
        warnings.warn("pyfftw is known to be buggy on your system, numpy.fft "
                      "will be used instead. Consequently, all algorithms "
                      "using ffts will be running at a slower speed.",
                      RuntimeWarning)
        from numpy.fft import fft2, ifft2, fftshift, ifftshift
except ImportError:
    warnings.warn("pyfftw is not installed on your system, numpy.fft will be "
                  "used instead. Consequently, all algorithms using ffts "
                  "will be running at a slower speed. Consider installing "
                  "pyfftw (pip install pyfftw) to speed up your ffts.",
                  ImportWarning)
    from numpy.fft import fft2, ifft2, fftshift, ifftshift


# TODO: Document me!
def pad(pixels, ext_shape, boundary='constant'):
    r"""
    """
    h, w = pixels.shape[-2:]

    h_margin = (ext_shape[0] - h) // 2
    w_margin = (ext_shape[1] - w) // 2

    h_margin2 = h_margin
    if h + 2 * h_margin < ext_shape[0]:
        h_margin += 1

    w_margin2 = w_margin
    if w + 2 * w_margin < ext_shape[1]:
        w_margin += 1

    pad_width = []
    for _ in pixels.shape[:-2]:
        pad_width.append((0, 0))
    pad_width += [(h_margin, h_margin2), (w_margin, w_margin2)]
    pad_width = tuple(pad_width)

    return np.lib.pad(pixels, pad_width, mode=boundary)


# TODO: Document me!
def crop(pixels, shape):
    r"""
    """
    h, w = pixels.shape[-2:]

    h_margin = (h - shape[0]) // 2
    w_margin = (w - shape[1]) // 2

    h_corrector = 1 if np.remainder(h - shape[0], 2) != 0 else 0
    w_corrector = 1 if np.remainder(w - shape[1], 2) != 0 else 0

    return pixels[...,
                  h_margin + h_corrector:-h_margin,
                  w_margin + w_corrector:-w_margin]


# TODO: Document me!
def ndconvolution(wrapped):
    r"""
    """
    @wraps(wrapped)
    def wrapper(image, filter, *args, **kwargs):
        if not isinstance(image, np.ndarray) and not isinstance(filter, np.ndarray):
            # Both image and filter are menpo images
            feature = wrapped(image.pixels, filter.pixels, *args, **kwargs)
            return rebuild_feature_image(image, feature)
        elif not isinstance(image, np.ndarray):
            # Image is menpo image
            feature = wrapped(image.pixels, filter, *args, **kwargs)
            return rebuild_feature_image(image, feature)
        elif not isinstance(filter, np.ndarray):
            # filter is menpo image
            return wrapped(image, filter, *args, **kwargs)
        else:
            return wrapped(image, filter, *args, **kwargs)
    return wrapper


# TODO: Document me!
@ndconvolution
def fft_convolve2d(x, f, mode='same', boundary='constant', fft_filter=False):
    r"""
    Performs fast 2d convolution in the frequency domain convolving each image
    channel with its corresponding filter channel.

    Parameters
    ----------
    x : ``(channels, height, width)`` `ndarray`
        Image.
    f : ``(channels, height, width)`` `ndarray`
        Filter.
    mode : str {`full`, `same`, `valid`}, optional
        Determines the shape of the resulting convolution.
    boundary: str {`constant`, `symmetric`}, optional
        Determines how the image is padded.
    fft_filter: `bool`, optional
        If `True`, the filter is assumed to be defined on the frequency
        domain. If `False` the filter is assumed to be defined on the
        spatial domain.

    Returns
    -------
    c: ``(channels, height, width)`` `ndarray`
        Result of convolving each image channel with its corresponding
        filter channel.
    """
    if fft_filter:
        # extended shape is filter shape
        ext_shape = np.asarray(f.shape[-2:])

        # extend image and filter
        ext_x = pad(x, ext_shape, boundary=boundary)

        # compute ffts of extended image
        fft_ext_x = fft2(ext_x)
        fft_ext_f = f
    else:
        # extended shape
        x_shape = np.asarray(x.shape[-2:])
        f_shape = np.asarray(f.shape[-2:])
        f_half_shape = (f_shape / 2).astype(int)
        ext_shape = x_shape + f_half_shape - 1

        # extend image and filter
        ext_x = pad(x, ext_shape, boundary=boundary)
        ext_f = pad(f, ext_shape)

        # compute ffts of extended image and extended filter
        fft_ext_x = fft2(ext_x)
        fft_ext_f = fft2(ext_f)

    # compute extended convolution in Fourier domain
    fft_ext_c = fft_ext_f * fft_ext_x

    # compute ifft of extended convolution
    ext_c = np.real(ifftshift(ifft2(fft_ext_c), axes=(-2, -1)))

    if mode is 'full':
        return ext_c
    elif mode is 'same':
        return crop(ext_c, x_shape)
    elif mode is 'valid':
        return crop(ext_c, x_shape - f_half_shape + 1)
    else:
        raise ValueError(
            "mode={}, is not supported. The only supported "
            "modes are: 'full', 'same' and 'valid'.".format(mode))


# TODO: Document me!
@ndconvolution
def fft_convolve2d_sum(x, f, mode='same', boundary='constant',
                       fft_filter=False, axis=0, keepdims=True):
    r"""
    Performs fast 2d convolution in the frequency domain convolving each image
    channel with its corresponding filter channel and summing across the
    channel axis.

    Parameters
    ----------
    x : ``(channels, height, width)`` `ndarray`
        Image.
    f : ``(channels, height, width)`` `ndarray`
        Filter.
    mode : str {`full`, `same`, `valid`}, optional
        Determines the shape of the resulting convolution.
    boundary: str {`constant`, `symmetric`}, optional
        Determines how the image is padded.
    fft_filter: `bool`, optional
        If `True`, the filter is assumed to be defined on the frequency
        domain. If `False` the filter is assumed to be defined on the
        spatial domain.
    axis : `int`, optional
        The axis across to which the summation is performed.
    keepdims: `boolean`, optional
        If `True` the number of dimensions of the result is the same as the
        number of dimensions of the filter. If `False` the channel dimension
        is lost in the result.
    Returns
    -------
    c: ``(1, height, width)`` `ndarray`
        Result of convolving each image channel with its corresponding
        filter channel and summing across the channel axis.
    """
    if fft_filter:
        fft_ext_f = f

        # extended shape is fft_ext_filter shape
        x_shape = np.asarray(x.shape[-2:])
        f_shape = ((np.asarray(fft_ext_f.shape[-2:]) + 1) / 1.5).astype(int)
        f_half_shape = (f_shape / 2).astype(int)
        ext_shape = np.asarray(f.shape[-2:])

        # extend image and filter
        ext_x = pad(x, ext_shape, boundary=boundary)

        # compute ffts of extended image
        fft_ext_x = fft2(ext_x)
    else:
        # extended shape
        x_shape = np.asarray(x.shape[-2:])
        f_shape = np.asarray(f.shape[-2:])
        f_half_shape = (f_shape / 2).astype(int)
        ext_shape = x_shape + f_half_shape - 1

        # extend image and filter
        ext_x = pad(x, ext_shape, boundary=boundary)
        ext_f = pad(f, ext_shape)

        # compute ffts of extended image and extended filter
        fft_ext_x = fft2(ext_x)
        fft_ext_f = fft2(ext_f)

    # compute extended convolution in Fourier domain
    fft_ext_c = np.sum(fft_ext_f * fft_ext_x, axis=axis, keepdims=keepdims)

    # compute ifft of extended convolution
    ext_c = np.real(ifftshift(ifft2(fft_ext_c), axes=(-2, -1)))

    if mode is 'full':
        return ext_c
    elif mode is 'same':
        return crop(ext_c, x_shape)
    elif mode is 'valid':
        return crop(ext_c, x_shape - f_half_shape + 1)
    else:
        raise ValueError(
            "mode={}, is not supported. The only supported "
            "modes are: 'full', 'same' and 'valid'.".format(mode))
