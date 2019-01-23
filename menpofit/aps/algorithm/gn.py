from __future__ import division
import numpy as np

from menpo.feature import gradient as fast_gradient
from menpo.image import Image

from ..result import APSAlgorithmResult


# ----------- INTERFACE -----------
class GaussNewtonBaseInterface(object):
    r"""
    Base interface for Gauss-Newton optimization of APS.

    Parameters
    ----------
    appearance_model : `menpo.model.GMRFModel`
        The trained appearance GMRF model.
    deformation_model : `menpo.model.GMRFModel`
        The trained deformation GMRF model.
    transform : :map:`OrthoPDM`
        The motion (shape) model.
    weight : `float`
        The weight between the appearance cost and the deformation cost. The
        provided value gets multiplied with the deformation cost.
    use_procrustes : `bool`
        Whether the shapes were aligned before training the deformation model.
    template : `menpo.image.Image`
        The template (in this case it is the mean appearance).
    sampling : `list` of `int` or `ndarray` or ``None``
        It defines a sampling mask per scale. If `int`, then it defines the
        sub-sampling step of the sampling mask. If `ndarray`, then it explicitly
        defines the sampling mask. If ``None``, then no sub-sampling is applied.
    patch_shape : (`int`, `int`)
        The patch shape.
    patch_normalisation : `callable`
        The method for normalizing the patches.
    """
    def __init__(self, appearance_model, deformation_model, transform,
                 weight, use_procrustes, template, sampling, patch_shape,
                 patch_normalisation):
        self.appearance_model = appearance_model
        self.deformation_model = deformation_model
        self.weight = weight
        self.use_procrustes = use_procrustes
        self.patch_shape = patch_shape
        self.patch_normalisation = patch_normalisation
        self.transform = transform
        self.template = template

        # build the sampling mask
        self._build_sampling_mask(sampling)

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
        self.sampling = sampling

    def ds_dp(self):
        r"""
        Calculates the shape jacobian. That is

        .. math::

            \frac{d\mathcal{S}}{d\mathbf{p}} = \mathbf{J}_S = \mathbf{U}_S

        with size :math:`2 \times n \times n_S`.

        :type: `ndarray`
        """
        return np.rollaxis(self.transform.d_dp(None), -1)

    def ds_dp_vectorized(self):
        r"""
        Calculates the vectorized shape jacobian. That is

        .. math::

            \frac{d\mathcal{S}}{d\mathbf{p}} = \mathbf{J}_S = \mathbf{U}_S

        with size :math:`2n \times n_S`.

        :type: `ndarray`
        """
        n_params = self.ds_dp().shape[-1]
        return self.ds_dp().reshape([-1, n_params], order='F')

    def Q_d(self):
        r"""
        Returns the deformation precision matrix :math:`\mathbf{Q}_d` that
        has size :math:`2n \times 2n`.

        :type: `ndarray`
        """
        return self.deformation_model.precision

    def H_s(self):
        r"""
        Calculates the deformation Hessian matrix

        .. :math:

            \mathbf{H}_s = \mathbf{U}_S^T \mathbf{Q}_d \mathbf{U}_S

        that has size :math:`n_S \times n_S`.

        :type: `ndarray`
        """
        tmp = self.ds_dp_vectorized().T.dot(self.Q_d())
        return tmp.dot(self.ds_dp_vectorized()) * self.weight

    def warp(self, image):
        r"""
        Function that warps the input image, i.e. extracts the patches and
        normalizes them.

        Parameters
        ----------
        image : :map:`Image`
            The input image.

        Returns
        -------
        parts : :map:`Image`
            The part-based image.
        """
        parts = image.extract_patches(self.transform.target,
                                      patch_shape=self.patch_shape,
                                      as_single_array=True)
        parts = self.patch_normalisation(parts)
        return Image(parts, copy=False)

    def gradient(self, image):
        r"""
        Function that computes the gradient of the image.

        Parameters
        ----------
        image : :map:`Image`
            The input image.

        Returns
        -------
        gradient : `ndarray`
            The computed gradient.
        """
        pixels = image.pixels
        nabla = fast_gradient(pixels.reshape((-1,) + self.patch_shape))
        # remove 1st dimension gradient which corresponds to the gradient
        # between parts
        return nabla.reshape((2,) + pixels.shape)

    def steepest_descent_images(self, nabla, ds_dp):
        r"""
        Function that computes the steepest descent images, i.e.

        .. math::

            \mathbf{J}_{\mathbf{a}} = \nabla\mathbf{a} \frac{dS}{d\mathbf{p}}

        with size :math:`mn \times n_S`.

        Parameters
        ----------
        nabla : `ndarray`
            The image (or mean appearance) gradient.
        ds_dp : `ndarray`
            The shape jacobian.

        Returns
        -------
        steepest_descent_images : `ndarray`
            The computed steepest descent images.
        """
        # reshape nabla
        # nabla: dims x parts x off x ch x (h x w)
        nabla = nabla[self.gradient_mask].reshape(nabla.shape[:-2] + (-1,))
        # compute steepest descent images
        # nabla: dims x parts x off x ch x (h x w)
        # dS_dp: dims x parts x                             x params
        # sdi:          parts x off x ch x (h x w) x params
        sdi = 0
        a = nabla[..., None] * ds_dp[..., None, None, None, :]
        for d in a:
            sdi += d

        # reshape steepest descent images
        # sdi: (parts x offsets x ch x w x h) x params
        return sdi.reshape((-1, sdi.shape[-1]))

    def J_a_T_Q_a(self, J_a, Q_a):
        r"""
        Function that computes the dot product between the appearance
        jacobian and the precision matrix, i.e.

        .. math::

            \mathbf{J}_{\mathbf{a}}^T \mathbf{Q}_{a}

        with size :math:`n_S \times mn`.

        Parameters
        ----------
        J_a : `ndarray`
            The appearance jacobian (steepest descent images).
        Q_a : `scipy.sparse.bsr_matrix`
            The appearance precision matrix.

        Returns
        -------
        J_a_T_Q_a : `ndarray`
            The dot product.
        """
        # compute the dot product between the appearance jacobian (J_a^T) and
        # the precision matrix (Q_a)
        # J_a: (parts x offsets x ch x w x h) x params
        # Q_a: (parts x offsets x ch x w x h) x (parts x offsets x ch x w x h)
        return Q_a.dot(J_a).T

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

    def algorithm_result(self, image, shapes, shape_parameters,
                         initial_shape=None, gt_shape=None,
                         appearance_costs=None, deformation_costs=None,
                         costs=None):
        r"""
        Returns an APS iterative optimization result object.

        Parameters
        ----------
        image : `menpo.image.Image` or subclass
            The image on which the optimization is applied.
        shapes : `list` of `menpo.shape.PointCloud`
            The `list` of shapes per iteration.
        shape_parameters : `list` of `ndarray`
            The `list` of shape parameters per iteration.
        initial_shape : `menpo.shape.PointCloud` or ``None``, optional
            The initial shape from which the fitting process started. If
            ``None``, then no initial shape is assigned.
        gt_shape : `menpo.shape.PointCloud` or ``None``, optional
            The ground truth shape that corresponds to the test image.
        appearance_costs : `list` of `float` or ``None``, optional
            The `list` of the appearance cost per iteration. If ``None``, then
            it is assumed that the cost function cannot be computed for the
            specific algorithm.
        deformation_costs : `list` of `float` or ``None``, optional
            The `list` of the deformation cost per iteration. If ``None``, then
            it is assumed that the cost function cannot be computed for the
            specific algorithm.
        costs : `list` of `float` or ``None``, optional
            The `list` of the total cost per iteration. If ``None``, then it is
            assumed that the cost function cannot be computed for the specific
            algorithm.

        Returns
        -------
        result : :map:`APSAlgorithmResult`
            The optimization result object.
        """
        return APSAlgorithmResult(
            shapes=shapes, shape_parameters=shape_parameters,
            initial_shape=initial_shape, image=image, gt_shape=gt_shape,
            appearance_costs=appearance_costs,
            deformation_costs=deformation_costs, costs=costs)


# ----------- ALGORITHMS -----------
class GaussNewton(object):
    r"""
    Abstract class for a Gauss-Newton optimization of APS.

    Parameters
    ----------
    aps_interface : `GaussNewtonBaseInterface` or subclass
        The Gauss-Newton interface object.
    eps : `float`, optional
        Value for checking the convergence of the optimization.
    """
    def __init__(self, aps_interface, eps=10**-5):
        self.eps = eps
        self.interface = aps_interface
        self._precompute()

    @property
    def appearance_model(self):
        r"""
        Returns the appearance GMRF model.

        :type: `menpo.model.GMRFModel`
        """
        return self.interface.appearance_model

    @property
    def deformation_model(self):
        r"""
        Returns the deformation GMRF model.

        :type: `menpo.model.GMRFModel`
        """
        return self.interface.deformation_model

    @property
    def transform(self):
        r"""
        Returns the motion model.

        :type: :map:`OrthoPDM`
        """
        return self.interface.transform

    @property
    def template(self):
        r"""
        Returns the template (usually the mean appearance).

        :type: `menpo.image.Image`
        """
        return self.interface.template

    def _precompute(self):
        # grab number of shape parameters
        self.n = self.transform.n_parameters

        # grab appearance model precision
        self.Q_a = self.appearance_model.precision
        # mask it only if the sampling mask is not all True
        if not np.all(self.interface.sampling):
            x, y = np.meshgrid(self.interface.i_mask, self.interface.i_mask)
            tmp = self.Q_a.tolil()[x, y]
            self.Q_a = tmp.tobsr()

        # grab appearance model mean
        self.a_bar = self.appearance_model.mean()
        # vectorize it and mask it
        self.a_bar_m = self.a_bar.as_vector()[self.interface.i_mask]


class Inverse(GaussNewton):
    r"""
    Inverse Gauss-Newton algorithm for APS.
    """
    def _precompute(self):
        # call super method
        super(Inverse, self)._precompute()
        # compute shape jacobian
        ds_dp = self.interface.ds_dp()
        # compute model's gradient
        nabla_a = self.interface.gradient(self.template)
        # compute appearance jacobian
        J_a = self.interface.steepest_descent_images(nabla_a, ds_dp)
        # transposed appearance jacobian and precision dot product
        self._J_a_T_Q_a = self.interface.J_a_T_Q_a(J_a, self.Q_a)
        # compute hessian inverse
        self._H_s = self.interface.H_s()
        H = self._J_a_T_Q_a.dot(J_a) + self._H_s
        self._inv_H = np.linalg.inv(H)

    def _algorithm_str(self):
        return 'Inverse Gauss-Newton'

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            return_costs=False):
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

        Returns
        -------
        fitting_result : :map:`APSAlgorithmResult`
            The parametric iterative fitting result.
        """
        # define cost closures
        def appearance_cost_closure(x):
            return self.appearance_model._mahalanobis_distance(
                x[..., None].T, subtract_mean=False, square_root=False)

        def deformation_cost_closure(x):
            tmp_shape = x.from_vector(x.as_vector() -
                                      self.deformation_model.mean_vector)
            cost = self.deformation_model.mahalanobis_distance(
                tmp_shape, subtract_mean=False, square_root=False)
            return cost * self.interface.weight

            # initialize transform
        self.transform.set_target(initial_shape)
        p_list = [self.transform.as_vector()]
        shapes = [self.transform.target]

        # initialize iteration counter and epsilon
        k = 0
        eps = np.Inf

        # Inverse Gauss-Newton loop -------------------------------------

        # warp image
        self.i = self.interface.warp(image)
        # vectorize it and mask it
        i_m = self.i.as_vector()[self.interface.i_mask]

        # compute masked error
        self.e_m = i_m - self.a_bar_m

        # update costs
        appearance_costs = None
        deformation_costs = None
        costs = None
        if return_costs:
            appearance_costs = [appearance_cost_closure(self.e_m)]
            deformation_costs = [deformation_cost_closure(shapes[-1])]
            costs = [appearance_costs[-1] + deformation_costs[-1]]

        while k < max_iters and eps > self.eps:
            # compute gauss-newton parameter updates
            b = self._J_a_T_Q_a.dot(self.e_m)
            p = p_list[-1].copy()
            if self.interface.use_procrustes:
                p[0:4] = 0
            b += self._H_s.dot(p)
            dp = self._inv_H.dot(b)

            # update warp
            s_k = self.transform.target.points
            self.transform._from_vector_inplace(
                self.transform.as_vector() - dp)
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
                appearance_costs.append(appearance_cost_closure(self.e_m))
                deformation_costs.append(deformation_cost_closure(shapes[-1]))
                costs.append(appearance_costs[-1] + deformation_costs[-1])

            # test convergence
            eps = np.abs(np.linalg.norm(s_k - self.transform.target.points))

            # increase iteration counter
            k += 1

        # return algorithm result
        return self.interface.algorithm_result(
            image=image, shapes=shapes, shape_parameters=p_list,
            initial_shape=initial_shape, gt_shape=gt_shape,
            appearance_costs=appearance_costs,
            deformation_costs=deformation_costs, costs=costs)

    def __str__(self):
        return "Inverse Weighted Gauss-Newton Algorithm with fixed Jacobian " \
               "and Hessian"


class Forward(GaussNewton):
    r"""
    Forward Gauss-Newton algorithm for APS.

    .. note:: The Forward optimization is too slow. It is not recommended to be
              used for fitting an APS and is only included for comparison
              purposes. Use `Inverse` instead.
    """
    def _precompute(self):
        # call super method
        super(Forward, self)._precompute()
        # compute shape jacobian
        self._ds_dp = self.interface.ds_dp()
        # compute shape hessian
        self._H_s = self.interface.H_s()

    def _algorithm_str(self):
        return 'Forward Gauss-Newton'

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            return_costs=False):
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

        Returns
        -------
        fitting_result : :map:`APSAlgorithmResult`
            The parametric iterative fitting result.
        """
        # define cost closures
        def appearance_cost_closure(x):
            return self.appearance_model._mahalanobis_distance(
                x[..., None].T, subtract_mean=False, square_root=False)

        def deformation_cost_closure(x):
            tmp_shape = x.from_vector(x.as_vector() -
                                      self.deformation_model.mean_vector)
            cost = self.deformation_model.mahalanobis_distance(
                tmp_shape, subtract_mean=False, square_root=False)
            return cost * self.interface.weight

        # initialize transform
        self.transform.set_target(initial_shape)
        p_list = [self.transform.as_vector()]
        shapes = [self.transform.target]

        # initialize iteration counter and epsilon
        k = 0
        eps = np.Inf

        # Forward Gauss-Newton loop -------------------------------------

        # warp image
        i = self.interface.warp(image)
        # vectorize it and mask it
        i_m = i.as_vector()[self.interface.i_mask]

        # compute masked error
        self.e_m = i_m - self.a_bar_m

        # update costs
        appearance_costs = None
        deformation_costs = None
        costs = None
        if return_costs:
            appearance_costs = [appearance_cost_closure(self.e_m)]
            deformation_costs = [deformation_cost_closure(shapes[-1])]
            costs = [appearance_costs[-1] + deformation_costs[-1]]

        while k < max_iters and eps > self.eps:
            # compute image gradient
            nabla_i = self.interface.gradient(i)

            # compute appearance jacobian
            Ja = self.interface.steepest_descent_images(nabla_i, self._ds_dp)

            # transposed jacobian and precision dot product
            J_a_T_Q_a = self.interface.J_a_T_Q_a(Ja, self.Q_a)

            # compute hessian
            H = J_a_T_Q_a.dot(Ja) + self._H_s

            # compute gauss-newton parameter updates
            b = J_a_T_Q_a.dot(self.e_m)
            p = p_list[-1].copy()
            if self.interface.use_procrustes:
                p[0:4] = 0
            b += self._H_s.dot(p)
            dp = -np.linalg.solve(H, b)

            # update warp
            s_k = self.transform.target.points
            self.transform._from_vector_inplace(
                self.transform.as_vector() + dp)
            p_list.append(self.transform.as_vector())
            shapes.append(self.transform.target)

            # warp image
            i = self.interface.warp(image)
            # vectorize it and mask it
            i_m = i.as_vector()[self.interface.i_mask]

            # compute masked error
            self.e_m = i_m - self.a_bar_m

            # update costs
            if return_costs:
                appearance_costs.append(appearance_cost_closure(self.e_m))
                deformation_costs.append(deformation_cost_closure(shapes[-1]))
                costs.append(appearance_costs[-1] + deformation_costs[-1])

            # test convergence
            eps = np.abs(np.linalg.norm(s_k - self.transform.target.points))

            # increase iteration counter
            k += 1

        # return algorithm result
        return self.interface.algorithm_result(
            image=image, shapes=shapes, shape_parameters=p_list,
            initial_shape=initial_shape, gt_shape=gt_shape,
            appearance_costs=appearance_costs,
            deformation_costs=deformation_costs, costs=costs)

    def __str__(self):
        return "Forward Gauss-Newton Algorithm"
