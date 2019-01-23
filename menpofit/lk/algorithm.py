from scipy.linalg import norm
import numpy as np

from .result import LucasKanadeAlgorithmResult


# TODO: implement Inverse Additive Algorithm?
# TODO: implement sampling?
class LucasKanade(object):
    r"""
    Abstract class for a Lucas-Kanade optimization algorithm.

    Parameters
    ----------
    template : `menpo.image.Image` or subclass
        The image template.
    transform : `subclass` of :map:`DP` and :map:`DX`, optional
        A differential affine transform object, e.g.
        :map:`DifferentiableAlignmentAffine`.
    residual : `class` subclass, optional
        The residual that will get applied. All possible residuals are:

        ========================== ============================================
        Class                      Description
        ========================== ============================================
        :map:`SSD`                 Sum of Squared Differences
        :map:`FourierSSD`          Sum of Squared Differences on Fourier domain
        :map:`ECC`                 Enhanced Correlation Coefficient
        :map:`GradientImages`      Image Gradient
        :map:`GradientCorrelation` Gradient Correlation
        ========================== ============================================
    eps : `float`, optional
        Value for checking the convergence of the optimization.
    """
    def __init__(self, template, transform, residual, eps=10**-10):
        self.template = template
        self.transform = transform
        self.residual = residual
        self.eps = eps

    def warped_images(self, image, shapes):
        r"""
        Given an input test image and a list of shapes, it warps the image
        into the shapes. This is useful for generating the warped images of a
        fitting procedure stored within a :map:`LucasKanadeResult`.

        Parameters
        ----------
        image : `menpo.image.Image` or `subclass`
            The input image to be warped.
        shapes : `list` of `menpo.shape.PointCloud`
            The list of shapes in which the image will be warped. The shapes
            are obtained during the iterations of a fitting procedure.

        Returns
        -------
        warped_images : `list` of `menpo.image.MaskedImage` or `ndarray`
            The warped images.
        """
        warped_images = []
        for s in shapes:
            self.transform.set_target(s)
            warped_images.append(image.warp_to_mask(
                    self.template.mask, self.transform, warp_landmarks=False))
        return warped_images


class ForwardAdditive(LucasKanade):
    r"""
    Forward Additive (FA) Lucas-Kanade algorithm.
    """
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
        fitting_result : :map:`LucasKanadeAlgorithmResult`
            The parametric iterative fitting result.
        """
        # initialize transform
        self.transform.set_target(initial_shape)
        p_list = [self.transform.as_vector()]
        shapes = [self.transform.target]

        costs = None
        if return_costs:
            costs = []

        # initialize iteration counter and epsilon
        k = 0
        eps = np.Inf

        # Forward Compositional Algorithm
        while k < max_iters and eps > self.eps:
            # warp image
            IWxp = image.warp_to_mask(self.template.mask, self.transform,
                                      warp_landmarks=False)

            # compute warp jacobian
            dW_dp = np.rollaxis(
                self.transform.d_dp(self.template.indices()), -1)
            dW_dp = dW_dp.reshape(dW_dp.shape[:1] + self.template.shape +
                                  dW_dp.shape[-1:])

            # compute steepest descent images
            filtered_J, J = self.residual.steepest_descent_images(
                image, dW_dp, forward=(self.template, self.transform))

            # compute hessian
            H = self.residual.hessian(filtered_J, sdi2=J)

            # compute steepest descent parameter updates.
            sd_dp = self.residual.steepest_descent_update(
                filtered_J, IWxp, self.template)

            # compute gradient descent parameter updates
            dp = -np.real(np.linalg.solve(H, sd_dp))

            # Update warp weights
            self.transform.from_vector_inplace(self.transform.as_vector() + dp)
            p_list.append(self.transform.as_vector())
            shapes.append(self.transform.target)

            # update costs
            if return_costs:
                costs.append(self.residual.cost_closure())

            # test convergence
            eps = np.abs(norm(dp))

            # increase iteration counter
            k += 1

        # return algorithm result
        return LucasKanadeAlgorithmResult(
            shapes=shapes, homogeneous_parameters=p_list,
            initial_shape=initial_shape, image=image, gt_shape=gt_shape,
            costs=costs)

    def __str__(self):
        return "Forward Additive Algorithm"


class ForwardCompositional(LucasKanade):
    r"""
    Forward Compositional (FC) Lucas-Kanade algorithm

    Parameters
    ----------
    template : `menpo.image.Image` or subclass
        The image template.
    transform : `subclass` of :map:`DP` and :map:`DX`, optional
        A differential affine transform object, e.g.
        :map:`DifferentiableAlignmentAffine`.
    residual : `class` subclass, optional
        The residual that will get applied. All possible residuals are:

        ========================== ============================================
        Class                      Description
        ========================== ============================================
        :map:`SSD`                 Sum of Squared Differences
        :map:`FourierSSD`          Sum of Squared Differences on Fourier domain
        :map:`ECC`                 Enhanced Correlation Coefficient
        :map:`GradientImages`      Image Gradient
        :map:`GradientCorrelation` Gradient Correlation
        ========================== ============================================
    eps : `float`, optional
        Value for checking the convergence of the optimization.
    """
    def __init__(self, template, transform, residual, eps=10**-10):
        super(ForwardCompositional, self).__init__(
            template, transform, residual, eps=eps)
        self._precompute()

    def _precompute(self):
        # compute warp jacobian
        dW_dp = np.rollaxis(
            self.transform.d_dp(self.template.indices()), -1)
        self.dW_dp = dW_dp.reshape(dW_dp.shape[:1] + self.template.shape +
                                   dW_dp.shape[-1:])

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
        fitting_result : :map:`LucasKanadeAlgorithmResult`
            The parametric iterative fitting result.
        """
        # initialize transform
        self.transform.set_target(initial_shape)
        p_list = [self.transform.as_vector()]
        shapes = [self.transform.target]

        costs = None
        if return_costs:
            costs = []

        # initialize iteration counter and epsilon
        k = 0
        eps = np.Inf

        # Forward Compositional Algorithm
        while k < max_iters and eps > self.eps:
            # warp image
            IWxp = image.warp_to_mask(self.template.mask, self.transform,
                                      warp_landmarks=False)

            # compute steepest descent images
            filtered_J, J = self.residual.steepest_descent_images(
                IWxp, self.dW_dp)

            # compute hessian
            H = self.residual.hessian(filtered_J, sdi2=J)

            # compute steepest descent parameter updates.
            sd_dp = self.residual.steepest_descent_update(
                filtered_J, IWxp, self.template)

            # compute gradient descent parameter updates
            dp = -np.real(np.linalg.solve(H, sd_dp))

            # Update warp weights
            self.transform.compose_after_from_vector_inplace(dp)
            p_list.append(self.transform.as_vector())
            shapes.append(self.transform.target)

            # update cost
            if return_costs:
                costs.append(self.residual.cost_closure())

            # test convergence
            eps = np.abs(norm(dp))

            # increase iteration counter
            k += 1

        # return algorithm result
        return LucasKanadeAlgorithmResult(
            shapes=shapes, homogeneous_parameters=p_list,
            initial_shape=initial_shape, image=image, gt_shape=gt_shape,
            costs=costs)

    def __str__(self):
        return "Forward Compositional Algorithm"


class InverseCompositional(LucasKanade):
    r"""
    Inverse Compositional (IC) Lucas-Kanade algorithm

    Parameters
    ----------
    template : `menpo.image.Image` or subclass
        The image template.
    transform : `subclass` of :map:`DP` and :map:`DX`, optional
        A differential affine transform object, e.g.
        :map:`DifferentiableAlignmentAffine`.
    residual : `class` subclass, optional
        The residual that will get applied. All possible residuals are:

        ========================== ============================================
        Class                      Description
        ========================== ============================================
        :map:`SSD`                 Sum of Squared Differences
        :map:`FourierSSD`          Sum of Squared Differences on Fourier domain
        :map:`ECC`                 Enhanced Correlation Coefficient
        :map:`GradientImages`      Image Gradient
        :map:`GradientCorrelation` Gradient Correlation
        ========================== ============================================
    eps : `float`, optional
        Value for checking the convergence of the optimization.
    """
    def __init__(self, template, transform, residual, eps=10**-10):
        super(InverseCompositional, self).__init__(
            template, transform, residual, eps=eps)
        self._precompute()

    def _precompute(self):
        # compute warp jacobian
        dW_dp = np.rollaxis(self.transform.d_dp(self.template.indices()), -1)
        dW_dp = dW_dp.reshape(dW_dp.shape[:1] + self.template.shape +
                              dW_dp.shape[-1:])
        # compute steepest descent images
        self.filtered_J, J = self.residual.steepest_descent_images(
            self.template, dW_dp)
        # compute hessian
        self.H = self.residual.hessian(self.filtered_J, sdi2=J)

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
        fitting_result : :map:`LucasKanadeAlgorithmResult`
            The parametric iterative fitting result.
        """
        # initialize transform
        self.transform.set_target(initial_shape)
        p_list = [self.transform.as_vector()]
        shapes = [self.transform.target]

        costs = None
        if return_costs:
            costs = []

        # initialize iteration counter and epsilon
        k = 0
        eps = np.Inf

        # Baker-Matthews, Inverse Compositional Algorithm
        while k < max_iters and eps > self.eps:
            # warp image
            IWxp = image.warp_to_mask(self.template.mask, self.transform,
                                      warp_landmarks=False)

            # compute steepest descent parameter updates.
            sd_dp = self.residual.steepest_descent_update(
                self.filtered_J, IWxp, self.template)

            # compute gradient descent parameter updates
            dp = np.real(np.linalg.solve(self.H, sd_dp))

            # update warp
            inv_dp = self.transform.pseudoinverse_vector(dp)
            self.transform.compose_after_from_vector_inplace(inv_dp)
            p_list.append(self.transform.as_vector())
            shapes.append(self.transform.target)

            # update cost
            if return_costs:
                costs.append(self.residual.cost_closure())

            # test convergence
            eps = np.abs(norm(dp))

            # increase iteration counter
            k += 1

        # return algorithm result
        return LucasKanadeAlgorithmResult(
            shapes=shapes, homogeneous_parameters=p_list,
            initial_shape=initial_shape, image=image, gt_shape=gt_shape,
            costs=costs)

    def __str__(self):
        return "Inverse Compositional Algorithm"

