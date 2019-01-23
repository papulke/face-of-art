from __future__ import division
from functools import partial
import warnings
import numpy as np

from menpo.shape import mean_pointcloud, PointCloud, TriMesh
from menpo.image import Image, MaskedImage
from menpo.feature import no_op
from menpo.transform import Scale, Translation, GeneralizedProcrustesAnalysis
from menpo.visualize import print_dynamic

from menpofit.visualize import print_progress


class MenpoFitModelBuilderWarning(Warning):
    r"""
    A warning that the parameters chosen to build a given model may cause
    unexpected behaviour.
    """
    pass


def compute_reference_shape(shapes, diagonal, verbose=False):
    r"""
    Function that computes the reference shape as the mean shape of the provided
    shapes.

    Parameters
    ----------
    shapes : `list` of `menpo.shape.PointCloud`
        The set of shapes from which to build the reference shape.
    diagonal : `int` or ``None``
        If `int`, it ensures that the mean shape is scaled so that the diagonal
        of the bounding box containing it matches the provided value.
        If ``None``, then the mean shape is not rescaled.
    verbose : `bool`, optional
        If ``True``, then progress information is printed.

    Returns
    -------
    reference_shape : `menpo.shape.PointCloud`
        The reference shape.
    """
    # the reference_shape is the mean shape of the images' landmarks
    if verbose:
        print_dynamic('- Computing reference shape')
    reference_shape = mean_pointcloud(shapes)

    # fix the reference_shape's diagonal length if asked
    if diagonal:
        x, y = reference_shape.range()
        scale = diagonal / np.sqrt(x ** 2 + y ** 2)
        reference_shape = Scale(scale, reference_shape.n_dims).apply(
            reference_shape)

    return reference_shape


def rescale_images_to_reference_shape(images, group, reference_shape,
                                      verbose=False):
    r"""
    Function that normalizes the images' sizes with respect to the size of the
    provided reference shape. In other words, the function rescales the provided
    images so that the size of the bounding box of their attached shape is the
    same as the size of the bounding box of the provided reference shape.

    Parameters
    ----------
    images : `list` of `menpo.image.Image`
        The set of images that will be rescaled.
    group : `str` or ``None``
        If `str`, then it specifies the group of the images's shapes. If
        ``None``, then the images must have only one landmark group.
    reference_shape : `menpo.shape.PointCloud`
        The reference shape.
    verbose : `bool`, optional
        If ``True``, then progress information is printed.

    Returns
    -------
    normalized_images : `list` of `menpo.image.Image`
        The rescaled images.
    """
    wrap = partial(print_progress, prefix='- Normalizing images size',
                   end_with_newline=False, verbose=verbose)

    # Normalize the scaling of all images wrt the reference_shape size
    normalized_images = [i.rescale_to_pointcloud(reference_shape, group=group)
                         for i in wrap(images)]
    return normalized_images


def normalization_wrt_reference_shape(images, group, diagonal, verbose=False):
    r"""
    Function that normalizes the images' sizes with respect to the size of the
    mean shape. This step is essential before building a deformable model.

    The normalization includes:
    1) Computation of the reference shape as the mean shape of the images'
    landmarks.
    2) Scaling of the reference shape using the diagonal.
    3) Rescaling of all the images so that their shape's scale is in
    correspondence with the reference shape's scale.

    Parameters
    ----------
    images : `list` of `menpo.image.Image`
        The set of images to normalize.
    group : `str`
        If `str`, then it specifies the group of the images's shapes. If
        ``None``, then the images must have only one landmark group.
    diagonal : `int` or ``None``
        If `int`, it ensures that the mean shape is scaled so that the diagonal
        of the bounding box containing it matches the provided value.
        If ``None``, then the mean shape is not rescaled.
    verbose : `bool`, Optional
        Flag that controls information and progress printing.

    Returns
    -------
    reference_shape : `menpo.shape.PointCloud`
        The reference shape that was used to resize all training images to
        a consistent object size.
    normalized_images : `list` of `menpo.image.Image`
        The images with normalized size.
    """
    # get shapes
    shapes = [i.landmarks[group].lms for i in images]

    # compute the reference shape and fix its diagonal length
    reference_shape = compute_reference_shape(shapes, diagonal, verbose=verbose)

    # normalize the scaling of all images wrt the reference_shape size
    normalized_images = rescale_images_to_reference_shape(
        images, group, reference_shape, verbose=verbose)
    return reference_shape, normalized_images


def compute_features(images, features, prefix='', verbose=False):
    r"""
    Function that extracts features from a list of images.

    Parameters
    ----------
    images : `list` of `menpo.image.Image`
        The set of images.
    features : `callable`
        The features extraction function. Please refer to `menpo.feature` and
        `menpofit.feature`.
    prefix : `str`
        The prefix of the printed information.
    verbose : `bool`, Optional
        Flag that controls information and progress printing.

    Returns
    -------
    feature_images : `list` of `menpo.image.Image`
        The list of feature images.
    """
    wrap = partial(print_progress,
                   prefix='{}Computing feature space'.format(prefix),
                   end_with_newline=not prefix, verbose=verbose)

    return [features(i) for i in wrap(images)]


def scale_images(images, scale, prefix='', return_transforms=False,
                 verbose=False):
    r"""
    Function that rescales a list of images and optionally returns the scale
    transforms.

    Parameters
    ----------
    images : `list` of `menpo.image.Image`
        The set of images to scale.
    scale : `float` or `tuple` of `floats`
        The scale factor. If a tuple, the scale to apply to each dimension.
        If a single `float`, the scale will be applied uniformly across
        each dimension.
    prefix : `str`, optional
        The prefix of the printed information.
    return_transforms : `bool`, optional
        If ``True``, then a `list` with the `menpo.transform.Scale` objects that
        were used to perform the rescale for each image  is also returned.
    verbose : `bool`, optional
        Flag that controls information and progress printing.

    Returns
    -------
    scaled_images : `list` of `menpo.image.Image`
        The list of rescaled images.
    scale_transforms : `list` of `menpo.transform.Scale`
        The list of scale transforms that were used. It is returned only if
        `return_transforms` is ``True``.
    """
    wrap = partial(print_progress,
                   prefix='{}Scaling images'.format(prefix),
                   end_with_newline=not prefix, verbose=verbose)
    if not np.allclose(scale, 1):
        # initialise scaled images and transforms lists
        scaled_images = []
        scale_transforms = []
        # for each image
        for i in wrap(images):
            if return_transforms:
                # store scaled image and transform, if asked
                sc_image, tr = i.rescale(scale, return_transform=True)
                scaled_images.append(sc_image)
                scale_transforms.append(tr)
            else:
                # store only scaled image
                scaled_images.append(i.rescale(scale))
        if return_transforms:
            return scaled_images, scale_transforms
        else:
            return scaled_images
    else:
        if return_transforms:
            scale_transforms = [Scale(1., images[0].n_dims)] * len(images)
            return images, scale_transforms
        else:
            return images


def warp_images(images, shapes, reference_frame, transform, prefix='',
                verbose=None):
    r"""
    Function that warps a list of images into the provided reference frame.

    Parameters
    ----------
    images : `list` of `menpo.image.Image`
        The set of images to warp.
    shapes : `list` of `menpo.shape.PointCloud`
        The set of shapes that correspond to the images.
    reference_frame : `menpo.image.BooleanImage`
        The reference frame to warp to.
    transform : `menpo.transform.Transform`
        Transform **from the reference frame back to the image**.
        Defines, for each pixel location on the reference frame, which pixel
        location should be sampled from on the image.
    prefix : `str`
        The prefix of the printed information.
    verbose : `bool`, Optional
        Flag that controls information and progress printing.

    Returns
    -------
    warped_images : `list` of `menpo.image.MaskedImage`
        The list of warped images.
    """
    wrap = partial(print_progress,
                   prefix='{}Warping images'.format(prefix),
                   end_with_newline=not prefix, verbose=verbose)

    warped_images = []
    # Build a dummy transform, use set_target for efficiency
    warp_transform = transform(reference_frame.landmarks['source'].lms,
                               reference_frame.landmarks['source'].lms)
    for i, s in wrap(list(zip(images, shapes))):
        # Update Transform Target
        warp_transform.set_target(s)
        # warp images
        warped_i = i.warp_to_mask(reference_frame.mask, warp_transform,
                                  warp_landmarks=False)
        # attach reference frame landmarks to images
        warped_i.landmarks['source'] = reference_frame.landmarks['source']
        warped_images.append(warped_i)
    return warped_images


def extract_patches(images, shapes, patch_shape, normalise_function=no_op,
                    prefix='', verbose=False):
    r"""
    Function that extracts patches around the landmarks of the provided images.

    Parameters
    ----------
    images : `list` of `menpo.image.Image`
        The set of images to warp.
    shapes : `list` of `menpo.shape.PointCloud`
        The set of shapes that correspond to the images.
    patch_shape : (`int`, `int`)
        The shape of the patches.
    normalise_function : `callable`
        A normalisation function to apply on the values of the patches.
    prefix : `str`
        The prefix of the printed information.
    verbose : `bool`, Optional
        Flag that controls information and progress printing.

    Returns
    -------
    patch_images : `list` of `menpo.image.Image`
        The list of images with the patches per image. Each output image has
        shape ``(n_center, n_offset, n_channels, patch_shape)``.
    """
    wrap = partial(print_progress,
                   prefix='{}Extracting patches'.format(prefix),
                   end_with_newline=not prefix, verbose=verbose)

    parts_images = []
    for i, s in wrap(list(zip(images, shapes))):
        parts = i.extract_patches(s, patch_shape=patch_shape,
                                  as_single_array=True)
        parts = normalise_function(parts)
        parts_images.append(Image(parts, copy=False))
    return parts_images


def build_reference_frame(landmarks, boundary=3, group='source'):
    r"""
    Builds a reference frame from a particular set of landmarks.

    Parameters
    ----------
    landmarks : `menpo.shape.PointCloud`
        The landmarks that will be used to build the reference frame.
    boundary : `int`, optional
        The number of pixels to be left as a safe margin on the boundaries
        of the reference frame (has potential effects on the gradient
        computation).
    group : `str`, optional
        Group that will be assigned to the provided set of landmarks on the
        reference frame.

    Returns
    -------
    reference_frame : `manpo.image.MaskedImage`
        The reference frame.
    """
    if not isinstance(landmarks, TriMesh):
        warnings.warn('The reference shape passed is not a TriMesh or '
                      'subclass and therefore the reference frame (mask) will '
                      'be calculated via a Delaunay triangulation. This may '
                      'cause small triangles and thus suboptimal warps.',
                      MenpoFitModelBuilderWarning)
    return MaskedImage.init_from_pointcloud(landmarks, boundary=boundary,
                                            group=group, constrain_mask=True)


def build_patch_reference_frame(landmarks, boundary=3, group='source',
                                patch_shape=(17, 17)):
    r"""
    Builds a patch-based reference frame from a particular set of landmarks.

    Parameters
    ----------
    landmarks : `menpo.shape.PointCloud`
        The landmarks that will be used to build the reference frame.
    boundary : `int`, optional
        The number of pixels to be left as a safe margin on the boundaries
        of the reference frame (has potential effects on the gradient
        computation).
    group : `str`, optional
        Group that will be assigned to the provided set of landmarks on the
        reference frame.
    patch_shape : (`int`, `int`), optional
        The shape of the patches.

    Returns
    -------
    patch_based_reference_frame : `menpo.image.MaskedImage`
        The patch-based reference frame.
    """
    boundary = np.max(patch_shape) + boundary
    reference_frame = MaskedImage.init_from_pointcloud(
        landmarks, group=group, boundary=boundary, constrain_mask=False)

    # mask reference frame
    return reference_frame.constrain_mask_to_patches_around_landmarks(
        patch_shape, group=group)


def densify_shapes(shapes, reference_frame, transform):
    r"""
    Function that densifies a set of sparse shapes given a reference frame.

    Parameters
    ----------
    shapes : `list` of `menpo.shape.PointCloud`
        The input shapes.
    reference_frame : `menpo.image.BooleanImage`
        The reference frame, the mask of which will be used.
    transform : `menpo.transform.Transform`
        The transform to use for mapping the dense points.

    Returns
    -------
    dense_shapes : `list` of `menpo.shape.PointCloud`
        The list of dense shapes.
    """
    # compute non-linear transforms
    transforms = [transform(reference_frame.landmarks['source'].lms, s)
                  for s in shapes]
    # build dense shapes
    dense_shapes = []
    for (t, s) in zip(transforms, shapes):
        warped_points = t.apply(reference_frame.mask.true_indices())
        dense_shape = PointCloud(np.vstack((s.points, warped_points)))
        dense_shapes.append(dense_shape)

    return dense_shapes


def align_shapes(shapes):
    r"""
    Function that aligns a set of shapes by applying Generalized Procrustes
    Analysis.

    Parameters
    ----------
    shapes : `list` of `menpo.shape.PointCloud`
        The input shapes.

    Returns
    -------
    aligned_shapes : `list` of `menpo.shape.PointCloud`
        The list of aligned shapes.
    """
    # centralize shapes
    centered_shapes = [Translation(-s.centre()).apply(s) for s in shapes]
    # align centralized shape using Procrustes Analysis
    gpa = GeneralizedProcrustesAnalysis(centered_shapes)
    return [s.aligned_source() for s in gpa.transforms]


class MenpoFitBuilderWarning(Warning):
    r"""
    A warning that some part of building the model may cause issues.
    """
    pass
