from __future__ import division
import numpy as np

from menpo.image import Image
from menpo.feature import gradient as fast_gradient, no_op

from ..result import AAMAlgorithmResult


def _solve_all_map(H, J, e, Ja_prior, c, Js_prior, p, m, n):
    if n is not H.shape[0] - m:
        # Bidirectional Compositional case
        Js_prior = np.hstack((Js_prior, Js_prior))
        p = np.hstack((p, p))
        # compute and return MAP solution
    J_prior = np.hstack((Ja_prior, Js_prior))
    H += np.diag(J_prior)
    Je = J_prior * np.hstack((c, p)) + J.T.dot(e)
    dq = - np.linalg.solve(H, Je)
    return dq[:m], dq[m:]


def _solve_all_ml(H, J, e, m):
    # compute ML solution
    dq = - np.linalg.solve(H, J.T.dot(e))
    return dq[:m], dq[m:]


# ----------- INTERFACES -----------
class LucasKanadeBaseInterface(object):
    r"""
    Base interface for Lucas-Kanade optimization of AAMs.

    Parameters
    ----------
    transform : `subclass` of :map:`DL` and :map:`DX`, optional
        A differential warp transform object, e.g.
        :map:`DifferentiablePiecewiseAffine` or
        :map:`DifferentiableThinPlateSplines`.
    template : `menpo.image.Image` or subclass
        The image template (usually the mean of the AAM's appearance model).
    sampling : `list` of `int` or `ndarray` or ``None``
        It defines a sampling mask per scale. If `int`, then it defines the
        sub-sampling step of the sampling mask. If `ndarray`, then it explicitly
        defines the sampling mask. If ``None``, then no sub-sampling is applied.
    """
    def __init__(self, transform, template, sampling=None):
        self.transform = transform
        self.template = template
        self._build_sampling_mask(sampling)

    def _build_sampling_mask(self, sampling):
        n_true_pixels = self.template.n_true_pixels()
        n_channels = self.template.n_channels
        n_parameters = self.transform.n_parameters

        sampling_mask = np.zeros(n_true_pixels, dtype=np.bool)

        if sampling is None:
            sampling = range(0, n_true_pixels, 1)
        elif isinstance(sampling, np.int):
            sampling = range(0, n_true_pixels, sampling)

        sampling_mask[sampling] = 1

        self.i_mask = np.nonzero(np.tile(
            sampling_mask[None, ...], (n_channels, 1)).flatten())[0]
        self.dW_dp_mask = np.nonzero(np.tile(
            sampling_mask[None, ..., None], (2, 1, n_parameters)))
        self.nabla_mask = np.nonzero(np.tile(
            sampling_mask[None, None, ...], (2, n_channels, 1)))
        self.nabla2_mask = np.nonzero(np.tile(
            sampling_mask[None, None, None, ...], (2, 2, n_channels, 1)))

    @property
    def shape_model(self):
        r"""
        Returns the shape model that exists within the model driven transform.

        :type: `menpo.model.PCAModel`
        """
        return self.transform.pdm.model

    @property
    def n(self):
        r"""
        Returns the number of components of the model driven transform.

        :type: `int`
        """
        return self.transform.n_parameters

    @property
    def true_indices(self):
        r"""
        Returns the number pixels within the template's mask.

        :type: `int`
        """
        return self.template.mask.true_indices()

    def warp_jacobian(self):
        r"""
        Computes the ward jacobian.

        :type: ``(n_dims, n_pixels, n_params)`` `ndarray`
        """
        dW_dp = np.rollaxis(self.transform.d_dp(self.true_indices), -1)
        return dW_dp[self.dW_dp_mask].reshape((dW_dp.shape[0], -1,
                                               dW_dp.shape[2]))

    def warp(self, image):
        r"""
        Warps an image into the template's mask.

        Parameters
        ----------
        image : `menpo.image.Image` or subclass
            The input image to be warped.

        Returns
        -------
        warped_image : `menpo.image.Image` or subclass
            The warped image.
        """
        return image.warp_to_mask(self.template.mask, self.transform,
                                  warp_landmarks=False)

    def warped_images(self, image, shapes):
        r"""
        Given an input test image and a list of shapes, it warps the image
        into the shapes. This is useful for generating the warped images of a
        fitting procedure.

        Parameters
        ----------
        image : `menpo.image.Image` or subclass
            The input image to be warped.
        shapes : `list` of `menpo.shape.PointCloud`
            The list of shapes in which the image will be warped. The shapes
            are obtained during the iterations of a fitting procedure.

        Returns
        -------
        warped_images : `list` of `menpo.image.MaskedImage`
            The warped images.
        """
        warped_images = []
        for s in shapes:
            self.transform.set_target(s)
            warped_images.append(self.warp(image))
        return warped_images

    def gradient(self, image):
        r"""
        Computes the gradient of an image and vectorizes it.

        Parameters
        ----------
        image : `menpo.image.Image` or subclass
            The input image.

        Returns
        -------
        gradient : ``(2, n_channels, n_pixels)`` `ndarray`
            The vectorized gradients of the image.
        """
        nabla = fast_gradient(image)
        nabla = nabla.set_boundary_pixels()
        return nabla.as_vector().reshape((2, image.n_channels, -1))

    def steepest_descent_images(self, nabla, dW_dp):
        r"""
        Computes the steepest descent images, i.e. the product of the gradient
        and the warp jacobian.

        Parameters
        ----------
        nabla : ``(2, n_channels, n_pixels)`` `ndarray`
            The image gradient in vectorized form.
        dW_dp : ``(n_dims, n_pixels, n_params)`` `ndarray`
            The warp jacobian.

        Returns
        -------
        steepest_descent_images : ``(n_channels * n_pixels, n_params)`` `ndarray`
            The computed steepest descent images.
        """
        # reshape gradient
        # nabla: n_dims x n_channels x n_pixels
        nabla = nabla[self.nabla_mask].reshape(nabla.shape[:2] + (-1,))
        # compute steepest descent images
        # nabla: n_dims x n_channels x n_pixels
        # warp_jacobian: n_dims x            x n_pixels x n_params
        # sdi:            n_channels x n_pixels x n_params
        sdi = 0
        a = nabla[..., None] * dW_dp[:, None, ...]
        for d in a:
            sdi += d
        # reshape steepest descent images
        # sdi: (n_channels x n_pixels) x n_params
        return sdi.reshape((-1, sdi.shape[2]))

    @classmethod
    def solve_shape_map(cls, H, J, e, J_prior, p):
        r"""
        Computes and returns the MAP solution.

        Parameters
        ----------
        H : ``(n_params, n_params)`` `ndarray`
            The Hessian matrix.
        J : ``(n_channels * n_pixels, n_params)`` `ndarray`
            The jacobian matrix (i.e. steepest descent images).
        e : ``(n_channels * n_pixels, )`` `ndarray`
            The residual (i.e. error image).
        J_prior : ``(n_params, n_params)`` `ndarray`
            The prior on the shape model.
        p : ``(n_params, )`` `ndarray`
            The current estimation of the shape parameters.

        Returns
        -------
        params : ``(n_params, )`` `ndarray`
            The MAP solution.
        """
        if p.shape[0] is not H.shape[0]:
            # Bidirectional Compositional case
            J_prior = np.hstack((J_prior, J_prior))
            p = np.hstack((p, p))
        # compute and return MAP solution
        H += np.diag(J_prior)
        Je = J_prior * p + J.T.dot(e)
        return - np.linalg.solve(H, Je)

    @classmethod
    def solve_shape_ml(cls, H, J, e):
        r"""
        Computes and returns the ML solution.

        Parameters
        ----------
        H : ``(n_params, n_params)`` `ndarray`
            The Hessian matrix.
        J : ``(n_channels * n_pixels, n_params)`` `ndarray`
            The jacobian matrix (i.e. steepest descent images).
        e : ``(n_channels * n_pixels, )`` `ndarray`
            The residual (i.e. error image).

        Returns
        -------
        params : ``(n_params, )`` `ndarray`
            The ML solution.
        """
        # compute and return ML solution
        return -np.linalg.solve(H, J.T.dot(e))

    def algorithm_result(self, image, shapes, shape_parameters,
                         appearance_parameters=None, initial_shape=None,
                         gt_shape=None, costs=None):
        r"""
        Returns an AAM iterative optimization result object.

        Parameters
        ----------
        image : `menpo.image.Image` or subclass
            The image on which the optimization is applied.
        shapes : `list` of `menpo.shape.PointCloud`
            The `list` of shapes per iteration.
        shape_parameters : `list` of `ndarray`
            The `list` of shape parameters per iteration.
        appearance_parameters : `list` of `ndarray` or ``None``, optional
            The `list` of appearance parameters per iteration. If ``None``,
            then it is assumed that the optimization did not solve for the
            appearance parameters.
        initial_shape : `menpo.shape.PointCloud` or ``None``, optional
            The initial shape from which the fitting process started. If
            ``None``, then no initial shape is assigned.
        gt_shape : `menpo.shape.PointCloud` or ``None``, optional
            The ground truth shape that corresponds to the test image.
        costs : `list` of `float` or ``None``, optional
            The `list` of costs per iteration. If ``None``, then it is
            assumed that the cost computation for that particular algorithm
            is not well defined.

        Returns
        -------
        result : :map:`AAMAlgorithmResult`
            The optimization result object.
        """
        return AAMAlgorithmResult(
            shapes=shapes, shape_parameters=shape_parameters,
            appearance_parameters=appearance_parameters,
            initial_shape=initial_shape, image=image, gt_shape=gt_shape,
            costs=costs)


class LucasKanadeStandardInterface(LucasKanadeBaseInterface):
    r"""
    Interface for Lucas-Kanade optimization of standard AAMs. Suitable for
    `menpofit.aam.HolisticAAM`.

    Parameters
    ----------
    appearance_model : `menpo.model.PCAModel` or subclass
        The appearance PCA model of the AAM.
    transform : `subclass` of :map:`DL` and :map:`DX`, optional
        A differential warp transform object, e.g.
        :map:`DifferentiablePiecewiseAffine` or
        :map:`DifferentiableThinPlateSplines`.
    template : `menpo.image.Image` or subclass
        The image template (usually the mean of the AAM's appearance model).
    sampling : `list` of `int` or `ndarray` or ``None``
        It defines a sampling mask per scale. If `int`, then it defines the
        sub-sampling step of the sampling mask. If `ndarray`, then it explicitly
        defines the sampling mask. If ``None``, then no sub-sampling is applied.
    """
    def __init__(self, appearance_model, transform, template, sampling=None):
        super(LucasKanadeStandardInterface, self).__init__(transform, template,
                                                           sampling=sampling)
        self.appearance_model = appearance_model

    @property
    def m(self):
        r"""
        Returns the number of active components of the appearance model.

        :type: `int`
        """
        return self.appearance_model.n_active_components

    def solve_all_map(self, H, J, e, Ja_prior, c, Js_prior, p):
        r"""
        Computes and returns the MAP solution.

        Parameters
        ----------
        H : ``(n_params, n_params)`` `ndarray`
            The Hessian matrix.
        J : ``(n_channels * n_pixels, n_params)`` `ndarray`
            The jacobian matrix (i.e. steepest descent images).
        e : ``(n_channels * n_pixels, )`` `ndarray`
            The residual (i.e. error image).
        Ja_prior : ``(n_app_params, n_app_params)`` `ndarray`
            The prior on the appearance model.
        c : ``(n_app_params, )`` `ndarray`
            The current estimation of the appearance parameters.
        Js_prior : ``(n_sha_params, n_sha_params)`` `ndarray`
            The prior on the shape model.
        p : ``(n_sha_params, )`` `ndarray`
            The current estimation of the shape parameters.

        Returns
        -------
        sha_params : ``(n_sha_params, )`` `ndarray`
            The MAP solution for the shape parameters.
        app_params : ``(n_app_params, )`` `ndarray`
            The MAP solution for the appearance parameters.
        """
        return _solve_all_map(H, J, e, Ja_prior, c, Js_prior, p,
                              self.m, self.n)

    def solve_all_ml(self, H, J, e):
        r"""
        Computes and returns the ML solution.

        Parameters
        ----------
        H : ``(n_params, n_params)`` `ndarray`
            The Hessian matrix.
        J : ``(n_channels * n_pixels, n_params)`` `ndarray`
            The jacobian matrix (i.e. steepest descent images).
        e : ``(n_channels * n_pixels, )`` `ndarray`
            The residual (i.e. error image).

        Returns
        -------
        sha_params : ``(n_sha_params, )`` `ndarray`
            The MAP solution for the shape parameters.
        app_params : ``(n_app_params, )`` `ndarray`
            The MAP solution for the appearance parameters.
        """
        return _solve_all_ml(H, J, e, self.m)


class LucasKanadeLinearInterface(LucasKanadeStandardInterface):
    r"""
    Interface for Lucas-Kanade optimization of linear AAMs. Suitable for
    `menpofit.aam.LinearAAM` and `menpofit.aam.LinearMaskedAAM`.
    """
    @property
    def shape_model(self):
        r"""
        Returns the shape model of the AAM.

        :type: `menpo.model.PCAModel`
        """
        return self.transform.model

    def algorithm_result(self, image, shapes, shape_parameters,
                         appearance_parameters=None, initial_shape=None,
                         gt_shape=None, costs=None):
        r"""
        Returns an AAM iterative optimization result object.

        Parameters
        ----------
        image : `menpo.image.Image` or subclass
            The image on which the optimization is applied.
        shapes : `list` of `menpo.shape.PointCloud`
            The `list` of sparse shapes per iteration.
        shape_parameters : `list` of `ndarray`
            The `list` of shape parameters per iteration.
        appearance_parameters : `list` of `ndarray` or ``None``, optional
            The `list` of appearance parameters per iteration. If ``None``,
            then it is assumed that the optimization did not solve for the
            appearance parameters.
        initial_shape : `menpo.shape.PointCloud` or ``None``, optional
            The initial shape from which the fitting process started. If
            ``None``, then no initial shape is assigned.
        gt_shape : `menpo.shape.PointCloud` or ``None``, optional
            The ground truth shape that corresponds to the test image.
        costs : `list` of `float` or ``None``, optional
            The `list` of costs per iteration. If ``None``, then it is
            assumed that the cost computation for that particular algorithm
            is not well defined.

        Returns
        -------
        result : :map:`AAMAlgorithmResult`
            The optimization result object.
        """
        # TODO: Separate result for linear AAMs that stores both the sparse
        #       and dense shapes per iteration (@patricksnape will fix this)
        # This means that the linear AAM will only store the sparse shapes
        shapes = [self.transform.from_vector(p).sparse_target
                  for p in shape_parameters]
        return AAMAlgorithmResult(
            shapes=shapes, shape_parameters=shape_parameters,
            appearance_parameters=appearance_parameters,
            initial_shape=initial_shape, image=image, gt_shape=gt_shape,
            costs=costs)


class LucasKanadePatchBaseInterface(LucasKanadeBaseInterface):
    r"""
    Base interface for Lucas-Kanade optimization of patch-based AAMs.

    Parameters
    ----------
    transform : `subclass` of :map:`DL` and :map:`DX`, optional
        A differential warp transform object, e.g.
        :map:`DifferentiablePiecewiseAffine` or
        :map:`DifferentiableThinPlateSplines`.
    template : `menpo.image.Image` or subclass
        The image template (usually the mean of the AAM's appearance model).
    sampling : `list` of `int` or `ndarray` or ``None``
        It defines a sampling mask per scale. If `int`, then it defines the
        sub-sampling step of the sampling mask. If `ndarray`, then it explicitly
        defines the sampling mask. If ``None``, then no sub-sampling is applied.
    patch_shape : (`int`, `int`), optional
        The patch shape.
    patch_normalisation : `closure`, optional
        A method for normalizing the values of the extracted patches.
    """
    def __init__(self, transform, template, sampling=None,
                 patch_shape=(17, 17), patch_normalisation=no_op):
        self.patch_shape = patch_shape
        self.patch_normalisation = patch_normalisation

        super(LucasKanadePatchBaseInterface, self).__init__(
            transform, template, sampling=sampling)

    def _build_sampling_mask(self, sampling):
        if sampling is None:
            sampling = np.ones(self.patch_shape, dtype=np.bool)

        image_shape = self.template.pixels.shape
        image_mask = np.tile(sampling[None, None, None, ...],
                             image_shape[:3] + (1, 1))
        self.i_mask = np.nonzero(image_mask.flatten())[0]
        self.gradient_mask = np.nonzero(np.tile(
            image_mask[None, ...], (2, 1, 1, 1, 1, 1)))
        self.gradient2_mask = np.nonzero(np.tile(
            image_mask[None, None, ...], (2, 2, 1, 1, 1, 1, 1)))

    @property
    def shape_model(self):
        r"""
        Returns the shape model that exists within the model driven transform.

        :type: `menpo.model.PCAModel`
        """
        return self.transform.model

    def warp_jacobian(self):
        r"""
        Computes the ward jacobian.

        :type: ``(n_dims, n_pixels, n_params)`` `ndarray`
        """
        return np.rollaxis(self.transform.d_dp(None), -1)

    def warp(self, image):
        r"""
        Extracts the patches from the given image. This is basically
        equivalent to warping an image within a Holistic AAM.

        Parameters
        ----------
        image : `menpo.image.Image` or subclass
            The input image.

        Returns
        -------
        patches_image : `menpo.image.Image`
            The image patches.
        """
        parts = image.extract_patches(self.transform.target,
                                      patch_shape=self.patch_shape,
                                      as_single_array=True)
        parts = self.patch_normalisation(parts)
        return Image(parts, copy=False)

    def warped_images(self, image, shapes):
        r"""
        Given an input test image and a list of shapes, it warps the image
        into the shapes. This is useful for generating the warped images of a
        fitting procedure.

        Parameters
        ----------
        image : `menpo.image.Image` or subclass
            The input image to be warped.
        shapes : `list` of `menpo.shape.PointCloud`
            The list of shapes in which the image will be warped. The shapes
            are obtained during the iterations of a fitting procedure.

        Returns
        -------
        warped_images : `list` of `ndarray`
            The warped images.
        """
        warped_images = []
        for s in shapes:
            self.transform.set_target(s)
            warped_images.append(self.warp(image).pixels)
        return warped_images

    def gradient(self, image):
        r"""
        Computes the gradient of a patch-based image and vectorizes it.

        Parameters
        ----------
        image : `menpo.image.Image` or subclass
            The input image.

        Returns
        -------
        gradient : ``(2, n_patches, 1, patch_shape, n_pixels)`` `ndarray`
            The vectorized gradients of the image.
        """
        pixels = image.pixels
        nabla = fast_gradient(pixels.reshape((-1,) + self.patch_shape))
        # remove 1st dimension gradient which corresponds to the gradient
        # between parts
        return nabla.reshape((2,) + pixels.shape)

    def steepest_descent_images(self, nabla, dw_dp):
        r"""
        Computes the steepest descent images, i.e. the product of the gradient
        and the warp jacobian.

        Parameters
        ----------
        nabla : ``(2, n_patches, 1, patch_shape)`` `ndarray`
            The image gradient in vectorized form.
        dW_dp : ``(2, n_patches, 1, patch_shape, n_params)`` `ndarray`
            The warp jacobian.

        Returns
        -------
        steepest_descent_images : ``(n_channels * n_patches, n_params)`` `ndarray`
            The computed steepest descent images.
        """
        # reshape nabla
        # nabla: dims x parts x off x ch x (h x w)
        nabla = nabla[self.gradient_mask].reshape(nabla.shape[:-2] + (-1,))
        # compute steepest descent images
        # nabla: dims x parts x off x ch x (h x w)
        # ds_dp:    dims x parts x                             x params
        # sdi:             parts x off x ch x (h x w) x params
        sdi = 0
        a = nabla[..., None] * dw_dp[..., None, None, None, :]
        for d in a:
            sdi += d

        # reshape steepest descent images
        # sdi: (parts x offsets x ch x w x h) x params
        return sdi.reshape((-1, sdi.shape[-1]))


class LucasKanadePatchInterface(LucasKanadePatchBaseInterface):
    r"""
    Interface for Lucas-Kanade optimization of patch-based AAMs. Suitable for
    `menpofit.aam.PatchAAM`.

    Parameters
    ----------
    appearance_model : `menpo.model.PCAModel` or subclass
        The appearance PCA model of the patch-based AAM.
    transform : `subclass` of :map:`DL` and :map:`DX`, optional
        A differential warp transform object, e.g.
        :map:`DifferentiablePiecewiseAffine` or
        :map:`DifferentiableThinPlateSplines`.
    template : `menpo.image.Image` or subclass
        The image template (usually the mean of the AAM's appearance model).
    sampling : `list` of `int` or `ndarray` or ``None``
        It defines a sampling mask per scale. If `int`, then it defines the
        sub-sampling step of the sampling mask. If `ndarray`, then it explicitly
        defines the sampling mask. If ``None``, then no sub-sampling is applied.
    patch_shape : (`int`, `int`), optional
        The patch shape.
    patch_normalisation : `closure`, optional
        A method for normalizing the values of the extracted patches.
    """
    def __init__(self, appearance_model, transform, template, sampling=None,
                 patch_shape=(17, 17), patch_normalisation=no_op):
        self.appearance_model = appearance_model
        super(LucasKanadePatchInterface, self).__init__(
            transform, template, patch_shape=patch_shape,
            patch_normalisation=patch_normalisation, sampling=sampling)

    @property
    def m(self):
        r"""
        Returns the number of active components of the appearance model.

        :type: `int`
        """
        return self.appearance_model.n_active_components

    def solve_all_map(self, H, J, e, Ja_prior, c, Js_prior, p):
        r"""
        Computes and returns the MAP solution.

        Parameters
        ----------
        H : ``(n_params, n_params)`` `ndarray`
            The Hessian matrix.
        J : ``(n_channels * n_pixels, n_params)`` `ndarray`
            The jacobian matrix (i.e. steepest descent images).
        e : ``(n_channels * n_pixels, )`` `ndarray`
            The residual (i.e. error image).
        Ja_prior : ``(n_app_params, n_app_params)`` `ndarray`
            The prior on the appearance model.
        c : ``(n_app_params, )`` `ndarray`
            The current estimation of the appearance parameters.
        Js_prior : ``(n_sha_params, n_sha_params)`` `ndarray`
            The prior on the shape model.
        p : ``(n_sha_params, )`` `ndarray`
            The current estimation of the shape parameters.

        Returns
        -------
        sha_params : ``(n_sha_params, )`` `ndarray`
            The MAP solution for the shape parameters.
        app_params : ``(n_app_params, )`` `ndarray`
            The MAP solution for the appearance parameters.
        """
        return _solve_all_map(H, J, e, Ja_prior, c, Js_prior, p,
                              self.m, self.n)

    def solve_all_ml(self, H, J, e):
        r"""
        Computes and returns the ML solution.

        Parameters
        ----------
        H : ``(n_params, n_params)`` `ndarray`
            The Hessian matrix.
        J : ``(n_channels * n_pixels, n_params)`` `ndarray`
            The jacobian matrix (i.e. steepest descent images).
        e : ``(n_channels * n_pixels, )`` `ndarray`
            The residual (i.e. error image).

        Returns
        -------
        sha_params : ``(n_sha_params, )`` `ndarray`
            The MAP solution for the shape parameters.
        app_params : ``(n_app_params, )`` `ndarray`
            The MAP solution for the appearance parameters.
        """
        return _solve_all_ml(H, J, e, self.m)


# ----------- ALGORITHMS -----------
class LucasKanade(object):
    r"""
    Abstract class for a Lucas-Kanade optimization algorithm.

    Parameters
    ----------
    aam_interface : The AAM interface class. Existing interfaces include:

        ============================== =============================
        'LucasKanadeStandardInterface' Suitable for holistic AAMs
        'LucasKanadeLinearInterface'   Suitable for linear AAMs
        'LucasKanadePatchInterface'    Suitable for patch-based AAMs
        ============================== =============================

    eps : `float`, optional
        Value for checking the convergence of the optimization.
    """
    def __init__(self, aam_interface, eps=10**-5):
        self.eps = eps
        self.interface = aam_interface
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

    @property
    def template(self):
        r"""
        Returns the template of the AAM (usually the mean of the appearance
        model).

        :type: `menpo.image.Image` or subclass
        """
        return self.interface.template

    def _precompute(self):
        # grab number of shape and appearance parameters
        self.n = self.transform.n_parameters
        self.m = self.appearance_model.n_active_components

        # grab appearance model components
        self.A = self.appearance_model.components
        # mask them
        self.A_m = self.A.T[self.interface.i_mask, :]
        # compute their pseudoinverse
        self.pinv_A_m = np.linalg.pinv(self.A_m)

        # grab appearance model mean
        self.a_bar = self.appearance_model.mean()
        # vectorize it and mask it
        self.a_bar_m = self.a_bar.as_vector()[self.interface.i_mask]

        # compute warp jacobian
        self.dW_dp = self.interface.warp_jacobian()

        # compute shape model prior
        # TODO: Is this correct? It's like modelling no noise at all
        sm_noise_variance = self.interface.shape_model.noise_variance() or 1
        s2 = self.appearance_model.noise_variance() / sm_noise_variance
        L = self.interface.shape_model.eigenvalues
        self.s2_inv_L = np.hstack((np.ones((4,)), s2 / L))
        # compute appearance model prior
        S = self.appearance_model.eigenvalues
        self.s2_inv_S = s2 / S


class ProjectOut(LucasKanade):
    r"""
    Abstract class for defining Project-out AAM optimization algorithms.
    """
    def project_out(self, J):
        r"""
        Projects-out the appearance subspace from a given vector or matrix.

        :type: `ndarray`
        """
        # project-out appearance bases from a particular vector or matrix
        return J - self.A_m.dot(self.pinv_A_m.dot(J))

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            return_costs=False, map_inference=False):
        r"""
        Execute the optimization algorithm.

        Parameters
        ----------
        image : `menpo.image.Image`
            The input test image.
        initial_shape : `menpo.shape.PointCloud`
            The initial shape from which the optimization will start.
        gt_shape : `menpo.shape.PointCloud` or ``None``, optional
            The ground truth shape of the image. It is only needed in order
            to get passed in the optimization result object, which has the
            ability to compute the fitting error.
        max_iters : `int`, optional
            The maximum number of iterations. Note that the algorithm may
            converge, and thus stop, earlier.
        return_costs : `bool`, optional
            If ``True``, then the cost function values will be computed
            during the fitting procedure. Then these cost values will be
            assigned to the returned `fitting_result`. *Note that the costs
            computation increases the computational cost of the fitting. The
            additional computation cost depends on the fitting method. Only
            use this option for research purposes.*
        map_inference : `bool`, optional
            If ``True``, then the solution will be given after performing MAP
            inference.

        Returns
        -------
        fitting_result : :map:`AAMAlgorithmResult`
            The parametric iterative fitting result.
        """
        # define cost function
        def cost_closure(x, f):
            return x.T.dot(f(x))

        # initialize transform
        self.transform.set_target(initial_shape)
        p_list = [self.transform.as_vector()]
        shapes = [self.transform.target]

        # initialize iteration counter and epsilon
        k = 0
        eps = np.Inf

        # Compositional Gauss-Newton loop -------------------------------------

        # warp image
        self.i = self.interface.warp(image)
        # vectorize it and mask it
        i_m = self.i.as_vector()[self.interface.i_mask]

        # compute masked error
        self.e_m = i_m - self.a_bar_m

        # update costs
        costs = None
        if return_costs:
            costs = [cost_closure(self.e_m, self.project_out)]

        while k < max_iters and eps > self.eps:
            # solve for increments on the shape parameters
            self.dp = self._solve(map_inference)

            # update warp
            s_k = self.transform.target.points
            self._update_warp()
            p_list.append(self.transform.as_vector())
            shapes.append(self.transform.target)

            # warp image
            self.i = self.interface.warp(image)
            # vectorize it and mask it
            i_m = self.i.as_vector()[self.interface.i_mask]

            # compute masked error
            self.e_m = i_m - self.a_bar_m

            # update costs
            if return_costs:
                costs.append(cost_closure(self.e_m, self.project_out))

            # test convergence
            eps = np.abs(np.linalg.norm(s_k - self.transform.target.points))

            # increase iteration counter
            k += 1

        # return algorithm result
        return self.interface.algorithm_result(
            image=image, shapes=shapes, shape_parameters=p_list,
            initial_shape=initial_shape, costs=costs, gt_shape=gt_shape)


class ProjectOutForwardCompositional(ProjectOut):
    r"""
    Project-out Forward Compositional (POFC) Gauss-Newton algorithm.
    """
    def _solve(self, map_inference):
        # compute warped image gradient
        nabla_i = self.interface.gradient(self.i)
        # compute masked forward Jacobian
        J_m = self.interface.steepest_descent_images(nabla_i, self.dW_dp)
        # project out appearance model from it
        QJ_m = self.project_out(J_m)
        # compute masked forward Hessian
        JQJ_m = QJ_m.T.dot(J_m)
        # solve for increments on the shape parameters
        if map_inference:
            return self.interface.solve_shape_map(
                JQJ_m, QJ_m, self.e_m,  self.s2_inv_L,
                self.transform.as_vector())
        else:
            return self.interface.solve_shape_ml(JQJ_m, QJ_m, self.e_m)

    def _update_warp(self):
        # update warp based on forward composition
        self.transform._from_vector_inplace(
            self.transform.as_vector() + self.dp)

    def __str__(self):
        return "Project-Out Forward Compositional Algorithm"


class ProjectOutInverseCompositional(ProjectOut):
    r"""
    Project-out Inverse Compositional (POFC) Gauss-Newton algorithm.
    """
    def _precompute(self):
        # call super method
        super(ProjectOutInverseCompositional, self)._precompute()
        # compute appearance model mean gradient
        nabla_a = self.interface.gradient(self.a_bar)
        # compute masked inverse Jacobian
        J_m = self.interface.steepest_descent_images(-nabla_a, self.dW_dp)
        # project out appearance model from it
        self.QJ_m = self.project_out(J_m)
        # compute masked inverse Hessian
        self.JQJ_m = self.QJ_m.T.dot(J_m)
        # compute masked Jacobian pseudo-inverse
        self.pinv_QJ_m = np.linalg.solve(self.JQJ_m, self.QJ_m.T)

    def _solve(self, map_inference):
        # solve for increments on the shape parameters
        if map_inference:
            return self.interface.solve_shape_map(
                self.JQJ_m, self.QJ_m, self.e_m, self.s2_inv_L,
                self.transform.as_vector())
        else:
            return -self.pinv_QJ_m.dot(self.e_m)

    def _update_warp(self):
        # update warp based on inverse composition
        self.transform._from_vector_inplace(
            self.transform.as_vector() - self.dp)

    def __str__(self):
        return "Project-Out Inverse Compositional Algorithm"


class Simultaneous(LucasKanade):
    r"""
    Abstract class for defining Simultaneous AAM optimization algorithms.
    """
    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            return_costs=False, map_inference=False):
        r"""
        Execute the optimization algorithm.

        Parameters
        ----------
        image : `menpo.image.Image`
            The input test image.
        initial_shape : `menpo.shape.PointCloud`
            The initial shape from which the optimization will start.
        gt_shape : `menpo.shape.PointCloud` or ``None``, optional
            The ground truth shape of the image. It is only needed in order
            to get passed in the optimization result object, which has the
            ability to compute the fitting error.
        max_iters : `int`, optional
            The maximum number of iterations. Note that the algorithm may
            converge, and thus stop, earlier.
        return_costs : `bool`, optional
            If ``True``, then the cost function values will be computed
            during the fitting procedure. Then these cost values will be
            assigned to the returned `fitting_result`. *Note that the costs
            computation increases the computational cost of the fitting. The
            additional computation cost depends on the fitting method. Only
            use this option for research purposes.*
        map_inference : `bool`, optional
            If ``True``, then the solution will be given after performing MAP
            inference.

        Returns
        -------
        fitting_result : :map:`AAMAlgorithmResult`
            The parametric iterative fitting result.
        """
        # define cost closure
        def cost_closure(x):
            return x.T.dot(x)

        # initialize transform
        self.transform.set_target(initial_shape)
        p_list = [self.transform.as_vector()]
        shapes = [self.transform.target]

        # initialize iteration counter and epsilon
        k = 0
        eps = np.Inf

        # Compositional Gauss-Newton loop -------------------------------------

        # warp image
        self.i = self.interface.warp(image)
        # mask warped image
        i_m = self.i.as_vector()[self.interface.i_mask]

        # initialize appearance parameters by projecting masked image
        # onto masked appearance model
        self.c = self.pinv_A_m.dot(i_m - self.a_bar_m)
        self.a = self.appearance_model.instance(self.c)
        a_m = self.a.as_vector()[self.interface.i_mask]
        c_list = [self.c]

        # compute masked error
        self.e_m = i_m - a_m

        # update costs
        costs = None
        if return_costs:
            costs = [cost_closure(self.e_m)]

        while k < max_iters and eps > self.eps:
            # solve for increments on the appearance and shape parameters
            # simultaneously
            dc, self.dp = self._solve(map_inference)

            # update appearance parameters
            self.c = self.c + dc
            self.a = self.appearance_model.instance(self.c)
            a_m = self.a.as_vector()[self.interface.i_mask]
            c_list.append(self.c)

            # update warp
            s_k = self.transform.target.points
            self._update_warp()
            p_list.append(self.transform.as_vector())
            shapes.append(self.transform.target)

            # warp image
            self.i = self.interface.warp(image)
            # mask warped image
            i_m = self.i.as_vector()[self.interface.i_mask]

            # compute masked error
            self.e_m = i_m - a_m

            # update costs
            if return_costs:
                costs.append(cost_closure(self.e_m))

            # test convergence
            eps = np.abs(np.linalg.norm(s_k - self.transform.target.points))

            # increase iteration counter
            k += 1

        # return algorithm result
        return self.interface.algorithm_result(
            image=image, shapes=shapes, shape_parameters=p_list,
            appearance_parameters=c_list, initial_shape=initial_shape,
            costs=costs, gt_shape=gt_shape)

    def _solve(self, map_inference):
        # compute masked Jacobian
        J_m = self._compute_jacobian()
        # assemble masked simultaneous Jacobian
        J_sim_m = np.hstack((-self.A_m, J_m))
        # compute masked Hessian
        H_sim_m = J_sim_m.T.dot(J_sim_m)
        # solve for increments on the appearance and shape parameters
        # simultaneously
        if map_inference:
            return self.interface.solve_all_map(
                H_sim_m, J_sim_m, self.e_m, self.s2_inv_S, self.c,
                self.s2_inv_L, self.transform.as_vector())
        else:
            return self.interface.solve_all_ml(H_sim_m, J_sim_m, self.e_m)


class SimultaneousForwardCompositional(Simultaneous):
    r"""
    Simultaneous Forward Compositional (SFC) Gauss-Newton algorithm.
    """
    def _compute_jacobian(self):
        # compute warped image gradient
        nabla_i = self.interface.gradient(self.i)
        # return forward Jacobian
        return self.interface.steepest_descent_images(nabla_i, self.dW_dp)

    def _update_warp(self):
        # update warp based on forward composition
        self.transform._from_vector_inplace(
            self.transform.as_vector() + self.dp)

    def __str__(self):
        return "Simultaneous Forward Compositional Algorithm"


class SimultaneousInverseCompositional(Simultaneous):
    r"""
    Simultaneous Inverse Compositional (SIC) Gauss-Newton algorithm.
    """
    def _compute_jacobian(self):
        # compute warped appearance model gradient
        nabla_a = self.interface.gradient(self.a)
        # return inverse Jacobian
        return self.interface.steepest_descent_images(-nabla_a, self.dW_dp)

    def _update_warp(self):
        # update warp based on inverse composition
        self.transform._from_vector_inplace(
            self.transform.as_vector() - self.dp)

    def __str__(self):
        return "Project-Out Inverse Compositional Algorithm"


class Alternating(LucasKanade):
    r"""
    Abstract class for defining Alternating AAM optimization algorithms.
    """
    def _precompute(self, **kwargs):
        # call super method
        super(Alternating, self)._precompute()
        # compute MAP appearance Hessian
        self.AA_m_map = self.A_m.T.dot(self.A_m) + np.diag(self.s2_inv_S)

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            return_costs=False, map_inference=False):
        r"""
        Execute the optimization algorithm.

        Parameters
        ----------
        image : `menpo.image.Image`
            The input test image.
        initial_shape : `menpo.shape.PointCloud`
            The initial shape from which the optimization will start.
        gt_shape : `menpo.shape.PointCloud` or ``None``, optional
            The ground truth shape of the image. It is only needed in order
            to get passed in the optimization result object, which has the
            ability to compute the fitting error.
        max_iters : `int`, optional
            The maximum number of iterations. Note that the algorithm may
            converge, and thus stop, earlier.
        return_costs : `bool`, optional
            If ``True``, then the cost function values will be computed
            during the fitting procedure. Then these cost values will be
            assigned to the returned `fitting_result`. *Note that the costs
            computation increases the computational cost of the fitting. The
            additional computation cost depends on the fitting method. Only
            use this option for research purposes.*
        map_inference : `bool`, optional
            If ``True``, then the solution will be given after performing MAP
            inference.

        Returns
        -------
        fitting_result : :map:`AAMAlgorithmResult`
            The parametric iterative fitting result.
        """
        # define cost closure
        def cost_closure(x):
            return x.T.dot(x)

        # initialize transform
        self.transform.set_target(initial_shape)
        p_list = [self.transform.as_vector()]
        shapes = [self.transform.target]

        # initialize iteration counter and epsilon
        k = 0
        eps = np.Inf

        # Compositional Gauss-Newton loop -------------------------------------

        # warp image
        self.i = self.interface.warp(image)
        # mask warped image
        i_m = self.i.as_vector()[self.interface.i_mask]

        # initialize appearance parameters by projecting masked image
        # onto masked appearance model
        c = self.pinv_A_m.dot(i_m - self.a_bar_m)
        self.a = self.appearance_model.instance(c)
        a_m = self.a.as_vector()[self.interface.i_mask]
        c_list = [c]
        Jdp = 0

        # compute masked error
        e_m = i_m - a_m

        # update costs
        costs = None
        if return_costs:
            costs = [cost_closure(e_m)]

        while k < max_iters and eps > self.eps:
            # solve for increment on the appearance parameters
            if map_inference:
                Ae_m_map = - self.s2_inv_S * c + self.A_m.dot(e_m + Jdp)
                dc = np.linalg.solve(self.AA_m_map, Ae_m_map)
            else:
                dc = self.pinv_A_m.dot(e_m + Jdp)

            # compute masked Jacobian
            J_m = self._compute_jacobian()
            # compute masked Hessian
            H_m = J_m.T.dot(J_m)
            # solve for increments on the shape parameters
            if map_inference:
                self.dp = self.interface.solve_shape_map(
                    H_m, J_m, e_m - self.A_m.T.dot(dc), self.s2_inv_L,
                    self.transform.as_vector())
            else:
                self.dp = self.interface.solve_shape_ml(H_m, J_m,
                                                        e_m - self.A_m.dot(dc))

            # update appearance parameters
            c = c + dc
            self.a = self.appearance_model.instance(c)
            a_m = self.a.as_vector()[self.interface.i_mask]
            c_list.append(c)

            # update warp
            s_k = self.transform.target.points
            self._update_warp()
            p_list.append(self.transform.as_vector())
            shapes.append(self.transform.target)

            # warp image
            self.i = self.interface.warp(image)
            # mask warped image
            i_m = self.i.as_vector()[self.interface.i_mask]

            # compute Jdp
            Jdp = J_m.dot(self.dp)

            # compute masked error
            e_m = i_m - a_m

            # update costs
            if return_costs:
                costs.append(cost_closure(e_m))

            # test convergence
            eps = np.abs(np.linalg.norm(s_k - self.transform.target.points))

            # increase iteration counter
            k += 1

        # return algorithm result
        return self.interface.algorithm_result(
            image=image, shapes=shapes, shape_parameters=p_list,
            appearance_parameters=c_list, initial_shape=initial_shape,
            costs=costs, gt_shape=gt_shape)


class AlternatingForwardCompositional(Alternating):
    r"""
    Alternating Forward Compositional (AFC) Gauss-Newton algorithm.
    """
    def _compute_jacobian(self):
        # compute warped image gradient
        nabla_i = self.interface.gradient(self.i)
        # return forward Jacobian
        return self.interface.steepest_descent_images(nabla_i, self.dW_dp)

    def _update_warp(self):
        # update warp based on forward composition
        self.transform._from_vector_inplace(
            self.transform.as_vector() + self.dp)

    def __str__(self):
        return "Alternating Forward Compositional Algorithm"


class AlternatingInverseCompositional(Alternating):
    r"""
    Alternating Inverse Compositional (AIC) Gauss-Newton algorithm.
    """
    def _compute_jacobian(self):
        # compute warped appearance model gradient
        nabla_a = self.interface.gradient(self.a)
        # return inverse Jacobian
        return self.interface.steepest_descent_images(-nabla_a, self.dW_dp)

    def _update_warp(self):
        # update warp based on inverse composition
        self.transform._from_vector_inplace(
            self.transform.as_vector() - self.dp)

    def __str__(self):
        return "Alternating Inverse Compositional Algorithm"


class ModifiedAlternating(Alternating):
    r"""
    Abstract class for defining Modified Alternating AAM optimization
    algorithms.
    """
    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            return_costs=False, map_inference=False):
        r"""
        Execute the optimization algorithm.

        Parameters
        ----------
        image : `menpo.image.Image`
            The input test image.
        initial_shape : `menpo.shape.PointCloud`
            The initial shape from which the optimization will start.
        gt_shape : `menpo.shape.PointCloud` or ``None``, optional
            The ground truth shape of the image. It is only needed in order
            to get passed in the optimization result object, which has the
            ability to compute the fitting error.
        max_iters : `int`, optional
            The maximum number of iterations. Note that the algorithm may
            converge, and thus stop, earlier.
        return_costs : `bool`, optional
            If ``True``, then the cost function values will be computed
            during the fitting procedure. Then these cost values will be
            assigned to the returned `fitting_result`. *Note that the costs
            computation increases the computational cost of the fitting. The
            additional computation cost depends on the fitting method. Only
            use this option for research purposes.*
        map_inference : `bool`, optional
            If ``True``, then the solution will be given after performing MAP
            inference.

        Returns
        -------
        fitting_result : :map:`AAMAlgorithmResult`
            The parametric iterative fitting result.
        """
        # define cost closure
        def cost_closure(x):
            return x.T.dot(x)

        # initialize transform
        self.transform.set_target(initial_shape)
        p_list = [self.transform.as_vector()]
        shapes = [self.transform.target]

        # initialize iteration counter and epsilon
        a_m = self.a_bar_m
        c_list = []
        k = 0
        eps = np.Inf

        # Compositional Gauss-Newton loop -------------------------------------

        # warp image
        self.i = self.interface.warp(image)
        # mask warped image
        i_m = self.i.as_vector()[self.interface.i_mask]

        # initialize appearance parameters by projecting masked image
        # onto masked appearance model
        c = self.pinv_A_m.dot(i_m - a_m)
        self.a = self.appearance_model.instance(c)
        a_m = self.a.as_vector()[self.interface.i_mask]
        c_list.append(c)

        # compute masked error
        e_m = i_m - a_m

        # update costs
        costs = None
        if return_costs:
            costs = [cost_closure(e_m)]

        while k < max_iters and eps > self.eps:
            # compute masked Jacobian
            J_m = self._compute_jacobian()
            # compute masked Hessian
            H_m = J_m.T.dot(J_m)
            # solve for increments on the shape parameters
            if map_inference:
                self.dp = self.interface.solve_shape_map(
                    H_m, J_m, e_m, self.s2_inv_L, self.transform.as_vector())
            else:
                self.dp = self.interface.solve_shape_ml(H_m, J_m, e_m)

            # update warp
            s_k = self.transform.target.points
            self._update_warp()
            p_list.append(self.transform.as_vector())
            shapes.append(self.transform.target)

            # warp image
            self.i = self.interface.warp(image)
            # mask warped image
            i_m = self.i.as_vector()[self.interface.i_mask]

            # update appearance parameters
            c = self.pinv_A_m.dot(i_m - self.a_bar_m)
            self.a = self.appearance_model.instance(c)
            a_m = self.a.as_vector()[self.interface.i_mask]
            c_list.append(c)

            # compute masked error
            e_m = i_m - a_m

            # update costs
            if return_costs:
                costs.append(cost_closure(e_m))

            # test convergence
            eps = np.abs(np.linalg.norm(s_k - self.transform.target.points))

            # increase iteration counter
            k += 1

        # return algorithm result
        return self.interface.algorithm_result(
            image=image, shapes=shapes, shape_parameters=p_list,
            appearance_parameters=c_list, initial_shape=initial_shape,
            costs=costs, gt_shape=gt_shape)


class ModifiedAlternatingForwardCompositional(ModifiedAlternating):
    r"""
    Modified Alternating Forward Compositional (MAFC) Gauss-Newton algorithm
    """
    def _compute_jacobian(self):
        # compute warped image gradient
        nabla_i = self.interface.gradient(self.i)
        # return forward Jacobian
        return self.interface.steepest_descent_images(nabla_i, self.dW_dp)

    def _update_warp(self):
        # update warp based on forward composition
        self.transform._from_vector_inplace(
            self.transform.as_vector() + self.dp)

    def __str__(self):
        return "Modified Alternating Forward Compositional Algorithm"


class ModifiedAlternatingInverseCompositional(ModifiedAlternating):
    r"""
    Modified Alternating Inverse Compositional (MAIC) Gauss-Newton algorithm
    """
    def _compute_jacobian(self):
        # compute warped appearance model gradient
        nabla_a = self.interface.gradient(self.a)
        # return inverse Jacobian
        return self.interface.steepest_descent_images(-nabla_a, self.dW_dp)

    def _update_warp(self):
        # update warp based on inverse composition
        self.transform._from_vector_inplace(
            self.transform.as_vector() - self.dp)

    def __str__(self):
        return "Modified Alternating Inverse Compositional Algorithm"


class Wiberg(LucasKanade):
    r"""
    Abstract class for defining Wiberg AAM optimization algorithms.
    """
    def project_out(self, J):
        # project-out appearance bases from a particular vector or matrix
        return J - self.A_m.dot(self.pinv_A_m.dot(J))

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            return_costs=False, map_inference=False):
        r"""
        Execute the optimization algorithm.

        Parameters
        ----------
        image : `menpo.image.Image`
            The input test image.
        initial_shape : `menpo.shape.PointCloud`
            The initial shape from which the optimization will start.
        gt_shape : `menpo.shape.PointCloud` or ``None``, optional
            The ground truth shape of the image. It is only needed in order
            to get passed in the optimization result object, which has the
            ability to compute the fitting error.
        max_iters : `int`, optional
            The maximum number of iterations. Note that the algorithm may
            converge, and thus stop, earlier.
        return_costs : `bool`, optional
            If ``True``, then the cost function values will be computed
            during the fitting procedure. Then these cost values will be
            assigned to the returned `fitting_result`. *Note that the costs
            computation increases the computational cost of the fitting. The
            additional computation cost depends on the fitting method. Only
            use this option for research purposes.*
        map_inference : `bool`, optional
            If ``True``, then the solution will be given after performing MAP
            inference.

        Returns
        -------
        fitting_result : :map:`AAMAlgorithmResult`
            The parametric iterative fitting result.
        """
        # define cost closure
        def cost_closure(x, f):
            return x.T.dot(f(x))

        # initialize transform
        self.transform.set_target(initial_shape)
        p_list = [self.transform.as_vector()]
        shapes = [self.transform.target]

        # initialize iteration counter and epsilon
        k = 0
        eps = np.Inf

        # Compositional Gauss-Newton loop -------------------------------------

        # warp image
        self.i = self.interface.warp(image)
        # mask warped image
        i_m = self.i.as_vector()[self.interface.i_mask]

        # initialize appearance parameters by projecting masked image
        # onto masked appearance model
        c = self.pinv_A_m.dot(i_m - self.a_bar_m)
        self.a = self.appearance_model.instance(c)
        a_m = self.a.as_vector()[self.interface.i_mask]
        c_list = [c]

        # compute masked error
        e_m = i_m - self.a_bar_m

        # update costs
        costs = None
        if return_costs:
            costs = [cost_closure(e_m, self.project_out)]

        while k < max_iters and eps > self.eps:
            # compute masked Jacobian
            J_m = self._compute_jacobian()
            # project out appearance models
            QJ_m = self.project_out(J_m)
            # compute masked Hessian
            JQJ_m = QJ_m.T.dot(J_m)
            # solve for increments on the shape parameters
            if map_inference:
                self.dp = self.interface.solve_shape_map(
                    JQJ_m, QJ_m, e_m, self.s2_inv_L,
                    self.transform.as_vector())
            else:
                self.dp = self.interface.solve_shape_ml(JQJ_m, QJ_m, e_m)

            # update warp
            s_k = self.transform.target.points
            self._update_warp()
            p_list.append(self.transform.as_vector())
            shapes.append(self.transform.target)

            # warp image
            self.i = self.interface.warp(image)
            # mask warped image
            i_m = self.i.as_vector()[self.interface.i_mask]

            # update appearance parameters
            dc = self.pinv_A_m.dot(i_m - a_m + J_m.dot(self.dp))
            c = c + dc
            self.a = self.appearance_model.instance(c)
            a_m = self.a.as_vector()[self.interface.i_mask]
            c_list.append(c)

            # compute masked error
            e_m = i_m - self.a_bar_m

            # update costs
            if return_costs:
                costs.append(cost_closure(e_m, self.project_out))

            # test convergence
            eps = np.abs(np.linalg.norm(s_k - self.transform.target.points))

            # increase iteration counter
            k += 1

        # return algorithm result
        return self.interface.algorithm_result(
            image=image, shapes=shapes, shape_parameters=p_list,
            appearance_parameters=c_list, initial_shape=initial_shape,
            costs=costs, gt_shape=gt_shape)


class WibergForwardCompositional(Wiberg):
    r"""
    Wiberg Forward Compositional (WFC) Gauss-Newton algorithm.
    """
    def _compute_jacobian(self):
        # compute warped image gradient
        nabla_i = self.interface.gradient(self.i)
        # return forward Jacobian
        return self.interface.steepest_descent_images(nabla_i, self.dW_dp)

    def _update_warp(self):
        # update warp based on forward composition
        self.transform._from_vector_inplace(
            self.transform.as_vector() + self.dp)

    def __str__(self):
        return "Wiberg Forward Compositional Algorithm"


class WibergInverseCompositional(Wiberg):
    r"""
    Wiberg Inverse Compositional (WIC) Gauss-Newton algorithm.
    """
    def _compute_jacobian(self):
        # compute warped appearance model gradient
        nabla_a = self.interface.gradient(self.a)
        # return inverse Jacobian
        return self.interface.steepest_descent_images(-nabla_a, self.dW_dp)

    def _update_warp(self):
        # update warp based on inverse composition
        self.transform._from_vector_inplace(
            self.transform.as_vector() - self.dp)

    def __str__(self):
        return "Wiberg Inverse Compositional Algorithm"
