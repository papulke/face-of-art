from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
from time import time
import os

from menpofit.base import build_grid
from menpofit.fitter import raise_costs_warning
from menpofit.result import ParametricIterativeResult

import rspimage

multivariate_normal = None  # expensive, from scipy.stats

from scipy.stats import multivariate_normal

class GradientDescentCLMAlgorithm(object):
    r"""
    Abstract class for a Gradient-Descent optimization algorithm.

    Parameters
    ----------
    expert_ensemble : `subclass` of :map:`ExpertEnsemble`
        The ensemble of experts object, e.g.
        :map:`CorrelationFilterExpertEnsemble`.
    shape_model : `subclass` of :map:`PDM`, optional
        The shape model object, e.g. :map:`OrthoPDM`.
    eps : `float`, optional
        Value for checking the convergence of the optimization.
    """
    def __init__(self, expert_ensemble, shape_model, eps=10**-5):
        # Set parameters
        self.expert_ensemble = expert_ensemble
        self.transform = shape_model
        self.eps = eps
        # Perform pre-computations
        self._precompute()

    def _precompute(self):
        # Import multivariate normal distribution from scipy
        global multivariate_normal
        if multivariate_normal is None:
            from scipy.stats import multivariate_normal  # expensive

        # Build grid associated to size of the search space
        search_shape = self.expert_ensemble.search_shape
        self.search_grid = build_grid(search_shape)

        # set rho2
        self.rho2 = self.transform.model.noise_variance()

        # Compute shape model prior
        sim_prior = np.zeros((4,))
        pdm_prior = self.rho2 / self.transform.model.eigenvalues
        self.rho2_inv_L = np.hstack((sim_prior, pdm_prior))

        # Compute Jacobian
        J = np.rollaxis(self.transform.d_dp(None), -1, 1)
        self.J = J.reshape((-1, J.shape[-1]))
        # Compute inverse Hessian
        self.JJ = self.J.T.dot(self.J)
        # Compute Jacobian pseudo-inverse
        self.pinv_J = np.linalg.solve(self.JJ, self.J.T)
        self.inv_JJ_prior = np.linalg.inv(self.JJ + np.diag(self.rho2_inv_L))


class ActiveShapeModel(GradientDescentCLMAlgorithm):
    r"""
    Active Shape Model (ASM) algorithm.

    Parameters
    ----------
    expert_ensemble : `subclass` of :map:`ExpertEnsemble`
        The ensemble of experts object, e.g.
        :map:`CorrelationFilterExpertEnsemble`.
    shape_model : `subclass` of :map:`PDM`, optional
        The shape model object, e.g. :map:`OrthoPDM`.
    gaussian_covariance : `int` or `float`, optional
        The covariance of the Gaussian kernel.
    eps : `float`, optional
        Value for checking the convergence of the optimization.

    References
    ----------
    .. [1] T. F. Cootes, and C. J. Taylor. "Active shape models-'smart snakes'",
        British Machine Vision Conference, pp. 266-275, 1992.
    .. [2] T. F. Cootes, C. J. Taylor, D. H. Cooper, and J. Graham. "Active
        Shape Models - their training and application", Computer Vision and Image
        Understanding (CVIU), 61(1): 38-59, 1995.
    .. [3] A. Blake, and M. Isard. "Active Shape Models", Active Contours,
        Springer, pp. 25-37, 1998.
    """
    def __init__(self, expert_ensemble, shape_model, gaussian_covariance=10,
                 eps=10**-5):
        self.gaussian_covariance = gaussian_covariance
        super(ActiveShapeModel, self).__init__(expert_ensemble=expert_ensemble,
                                               shape_model=shape_model, eps=eps)

    def _precompute(self):
        # Call super method
        super(ActiveShapeModel, self)._precompute()

        # Build grid associated to size of the search space
        self.half_search_shape = np.round(
            np.asarray(self.expert_ensemble.search_shape) / 2).astype(np.int64)
        self.search_grid = self.search_grid[None, None]

        # Compute Gaussian-KDE grid
        self.mvn = multivariate_normal(mean=np.zeros(2),
                                       cov=self.gaussian_covariance)

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
            assigned to the returned `fitting_result`. *Note that this
            argument currently has no effect and will raise a warning if set
            to ``True``. This is because it is not possible to evaluate the
            cost function of this algorithm.*
        map_inference : `bool`, optional
            If ``True``, then the solution will be given after performing MAP
            inference.

        Returns
        -------
        fitting_result : :map:`ParametricIterativeResult`
            The parametric iterative fitting result.
        """
        # costs warning
        if return_costs:
            raise_costs_warning(self)

        # Initialize transform
        self.transform.set_target(initial_shape)
        p_list = [self.transform.as_vector()]
        shapes = [self.transform.target]

        # Initialize iteration counter and epsilon
        k = 0
        eps = np.Inf

        # Expectation-Maximisation loop
        while k < max_iters and eps > self.eps:

            target = self.transform.target
            # Obtain all landmark positions l_i = (x_i, y_i) being considered
            # ie all pixel positions in each landmark's search space
            candidate_landmarks = (target.points[:, None, None, None, :] +
                                   self.search_grid)

            # Compute responses
            responses = self.expert_ensemble.predict_probability(image, target)
            # Approximate responses using isotropic Gaussian
            max_indices = np.argmax(
                responses.reshape(responses.shape[:2] + (-1,)), axis=-1)
            max_indices = np.unravel_index(max_indices, responses.shape)[-2:]
            max_indices = np.hstack((max_indices[0], max_indices[1]))
            max_indices = max_indices[:, None, None, None, ...]
            max_indices -= self.half_search_shape
            gaussian_responses = self.mvn.pdf(max_indices + self.search_grid)
            # Normalise smoothed responses
            gaussian_responses /= np.sum(gaussian_responses,
                                         axis=(-2, -1))[..., None, None]

            # Compute new target
            new_target = np.sum(gaussian_responses[:, None, ..., None] *
                                candidate_landmarks, axis=(-3, -2))

            # Compute shape error term
            error = target.as_vector() - new_target.ravel()

            # Solve for increments on the shape parameters
            if map_inference:
                Je = (self.rho2_inv_L * self.transform.as_vector() -
                      self.J.T.dot(error))
                dp = -self.inv_JJ_prior.dot(Je)
            else:
                dp = self.pinv_J.dot(error)

            # Update pdm
            s_k = self.transform.target.points
            self.transform._from_vector_inplace(self.transform.as_vector() + dp)
            p_list.append(self.transform.as_vector())
            shapes.append(self.transform.target)

            # Test convergence
            eps = np.abs(np.linalg.norm(s_k - self.transform.target.points))

            # Increase iteration counter
            k += 1

        # Return algorithm result
        return ParametricIterativeResult(shapes=shapes, shape_parameters=p_list,
                                         initial_shape=initial_shape,
                                         image=image, gt_shape=gt_shape)

    def __str__(self):
        return "Active Shape Model Algorithm"


class RegularisedLandmarkMeanShift(GradientDescentCLMAlgorithm):
    r"""
    Regularized Landmark Mean-Shift (RLMS) algorithm.

    Parameters
    ----------
    expert_ensemble : `subclass` of :map:`ExpertEnsemble`
        The ensemble of experts object, e.g.
        :map:`CorrelationFilterExpertEnsemble`.
    shape_model : `subclass` of :map:`PDM`, optional
        The shape model object, e.g. :map:`OrthoPDM`.
    kernel_covariance : `int` or `float`, optional
        The covariance of the kernel.
    eps : `float`, optional
        Value for checking the convergence of the optimization.

    References
    ----------
    .. [1] J.M. Saragih, S. Lucey, and J. F. Cohn. "Deformable model fitting by
        regularized landmark mean-shift", International Journal of Computer
        Vision (IJCV), 91(2): 200-215, 2011.
    """
    def __init__(self, expert_ensemble, shape_model, kernel_covariance=10, kernel_idealmap=100,
                 confidence_gama=0.5, opt=None, eps=10**-5, imgSize=256):
        self.kernel_covariance = kernel_covariance
        self.kernel_idealmap = kernel_idealmap
        self.confidence_gama = confidence_gama
        self.opt = opt
        mvnideal = multivariate_normal(mean=np.zeros(2), cov=self.kernel_idealmap)
        self.ideal_response = mvnideal.pdf(build_grid((2 * 256, 2 * 256)))
        self.rps_zeros = np.zeros((1, shape_model.n_points, 2*imgSize, 2*imgSize))
        #self.ideal_response = mvnideal.pdf(build_grid((2 * image.shape[0], 2 * image.shape[1])))
        super(RegularisedLandmarkMeanShift, self).__init__(
                expert_ensemble=expert_ensemble, shape_model=shape_model,
                eps=eps)

    def _precompute(self):
        # Call super method
        super(RegularisedLandmarkMeanShift, self)._precompute()

        # Compute Gaussian-KDE grid
        mvn = multivariate_normal(mean=np.zeros(2), cov=self.kernel_covariance)
        self.kernel_grid = mvn.pdf(self.search_grid)[None, None]

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            return_costs=False, map_inference=True):
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
            assigned to the returned `fitting_result`. *Note that this
            argument currently has no effect and will raise a warning if set
            to ``True``. This is because it is not possible to evaluate the
            cost function of this algorithm.*
        map_inference : `bool`, optional
            If ``True``, then the solution will be given after performing MAP
            inference.

        Returns
        -------
        fitting_result : :map:`ParametricIterativeResult`
            The parametric iterative fitting result.
        """
        # costs warning
        if return_costs:
            raise_costs_warning(self)

        patch_size = self.opt['ratio1']
        patch_size2 = self.opt['ratio2']
        search_ratio = [patch_size]*3+[patch_size2]*(self.opt['numIter']-3)

        numLandmarks = image.rspmap_data.shape[1]
        # np.save('/Users/arik/Desktop/test/map/' + image.path.name.rsplit('.', 1)[0], image.rspmap_data)

        timeFitStart = time()
        # pad responsemaps with zeros
        padH = int(image.shape[0]/2)
        padW = int(image.shape[1]/2)
        # image.rspmap_data = np.pad(image.rspmap_data, ((0, 0), (0, 0), (padH, padH), (padW, padW)), 'constant')
        self.rps_zeros[0, :, padH:padH+image.shape[0], padW:padW+image.shape[1]] = image.rspmap_data
        image.rspmap_data = self.rps_zeros
        # cost = time() - timeFitStart
        timeFitStart = time()

        try:
            weighted_project = self.opt['ablation'][0]
            weighted_meanshift = self.opt['ablation'][1]
        except:
            weighted_project = True
            weighted_meanshift = True

        if weighted_project:
            # Correction step
            search_grid = build_grid(np.array(search_ratio[0]*np.array(image.shape),int))

            # Compute patch responses
            patch_responses = self.expert_ensemble.predict_probability(image,
                                                                       initial_shape, search_grid.shape)

            project_weight = rspimage.calculate_evidence(patch_responses, rate=self.confidence_gama, offset=self.kernel_idealmap).reshape((1,-1))

            # write project_weight
            inirho = self.opt['pdm_rho']
            # inirho = 0
            ini_rho2_inv_prior = np.hstack((np.zeros((4,)), inirho / self.transform.model.eigenvalues))

            initial_shape_mean = initial_shape.points.ravel() - self.transform.model._mean
            iniJe = - self.J.T.dot(initial_shape_mean * project_weight[0])
            iniJWJ = self.J.T.dot(np.diag(project_weight[0]).dot(self.J))
            inv_JJ = np.linalg.inv(iniJWJ + np.diag(ini_rho2_inv_prior))
            initial_p = -inv_JJ.dot(iniJe)  # self.inv_JJ_prior

            # Update pdm
            self.transform._from_vector_inplace(initial_p)
            # np.save('/Users/arik/Desktop/test/init_pdm/'+image.path.name.rsplit('.',1)[0], self.transform.target.points)
        else:
            self.transform.set_target(initial_shape)

        # Initialize transform
        p_list = [self.transform.as_vector()]

        #  check boundary
        if np.any(self.transform.target.points < 0) or np.any(self.transform.target.points >= image.shape[0]):
            self.transform.target.points[self.transform.target.points < 0] = 1
            self.transform.target.points[self.transform.target.points >= image.shape[0]] = image.shape[0] - 2
            self.transform.set_target(self.transform.target)
            self.transform.target.points[self.transform.target.points < 0] = 1
            self.transform.target.points[self.transform.target.points >= image.shape[0]] = image.shape[0] - 2
            # print image.path.name
            # print 'out of bound'

        shapes = [self.transform.target]

        # tuning step
        # Initialize iteration counter and epsilon
        k = 0
        eps = np.Inf

        errSum = [np.Inf]
        epsList = [np.Inf]
        searchR = [ ]
        try:
            max_iters = self.opt['numIter']
        except:
            max_iters = 10

        rho2 = self.opt['rho2']
        # timeFitStart = time()
        while k < max_iters and eps > self.eps:

            #target = self.transform.target
            target = shapes[k]

            search_grid = build_grid(np.array(search_ratio[k]*np.array(image.shape),int))

            # search_grid = self.search_grid *np.array(image.shape, dtype=np.float32)/np.array(image.rspmap_data.shape)[-2:-1]

            candidate_landmarks = (target.points[:, None, None, None, :] +
                                   search_grid)

            # Compute patch responses
            patch_responses = self.expert_ensemble.predict_probability(image,
                                                                       target, search_grid.shape)

            if weighted_meanshift:
                kernel_covariance = rspimage.calculate_evidence(patch_responses, rate=self.confidence_gama, offset=self.kernel_idealmap)
            else:
                kernel_covariance = np.ones(2*patch_responses.shape[0])

            try:
                smooth_rspmap = self.opt['smooth']
            except:
                smooth_rspmap = True

            if smooth_rspmap:
                try:
                    patch_responses /= np.sum(patch_responses,
                                       axis=(-2, -1))[..., None, None]
                except:
                    print image.path.name
                    print('Normalize fail.')

                mvnList = [multivariate_normal(mean=np.zeros(2), cov=2.0*rho2/(kernel_covariance[2*i] + kernel_covariance[2*i+1])) for i in range(patch_responses.shape[0])]
                kernel_grid = np.array([mvnList[i].pdf(search_grid)[None] for i in range(patch_responses.shape[0])])

                patch_kernels = patch_responses * kernel_grid
            else:
                patch_kernels = patch_responses

            # Normalise smoothed responses
            patch_kernels /= np.sum(patch_kernels,
                                    axis=(-2, -1))[..., None, None]

            # Compute mean shift target
            mean_shift_target = np.sum(patch_kernels[..., None] *
                                       candidate_landmarks, axis=(-3, -2))

            # Compute shape error term
            error = mean_shift_target.ravel() - target.as_vector()
            errSum.append(np.sum(np.abs(error)))
            # error = error

            # Solve for increments on the shape parameters
            if map_inference:
                '''
                Je = (self.rho2_inv_L * self.transform.as_vector() -
                      self.J.T.dot(error))
                # inv_JJ_prior = np.linalg.inv(self.JJ + np.diag(self.rho2_inv_L))
                dp = -self.inv_JJ_prior.dot(Je)     #self.inv_JJ_prior
                '''
                Je = (self.rho2_inv_L * self.transform.as_vector() -
                      self.J.T.dot(error*kernel_covariance))
                JWJ = self.J.T.dot(np.diag(kernel_covariance).dot(self.J))
                inv_JJ_prior = np.linalg.inv(JWJ + np.diag(self.rho2_inv_L))
                dp = -inv_JJ_prior.dot(Je)  # self.inv_JJ_prior
                '''
                sim_prior = np.zeros((4,))
                pdm_prior = rho2 / self.transform.model.eigenvalues
                rho2_inv_L = np.hstack((sim_prior, pdm_prior))
                Je = (rho2_inv_L * self.transform.as_vector() -
                      self.J.T.dot(error*kernel_covariance))
                JWJ = self.J.T.dot(np.diag(kernel_covariance).dot(self.J))
                inv_JJ_prior = np.linalg.inv(JWJ + np.diag(rho2_inv_L))
                dp = -inv_JJ_prior.dot(Je)  # self.inv_JJ_prior
                '''
            else:
                J = np.rollaxis(self.transform.d_dp(target.points), -1, 1)
                J = J.reshape((-1, J.shape[-1]))
                JJ = J.T.dot(np.diag(kernel_covariance**2).dot(J))
                # Compute Jacobian pseudo-inverse
                pinv_J = np.linalg.solve(JJ, J.T.dot(np.diag(kernel_covariance)))  #
                dp = pinv_J.dot(error)
                print('notMAP')

            # Update pdm
            s_k = self.transform.target.points
            self.transform._from_vector_inplace(self.transform.as_vector() + dp)
            if np.any(self.transform.target.points<0) or np.any(self.transform.target.points>=image.shape[0]):
                self.transform.target.points[self.transform.target.points<0] =1
                self.transform.target.points[self.transform.target.points>=image.shape[0]] = image.shape[0]-2
                self.transform.set_target(self.transform.target)
                self.transform.target.points[self.transform.target.points < 0] = 1
                self.transform.target.points[self.transform.target.points >= image.shape[0]] = image.shape[0] - 2

            p_list.append(self.transform.as_vector())
            shapes.append(self.transform.target)

            # Test convergence
            eps = np.abs(np.linalg.norm(s_k - self.transform.target.points))
            epsList.append(eps)
            # Increase iteration counter
            k += 1

        cost = time() - timeFitStart
        cost = [cost]

        visualize = self.opt['verbose']

        if visualize:
            # wrFilebase = self.opt['imgDir']
            # wrFilebase = '../'

            image.landmarks['initial'] = shapes[0]
            image.landmarks['final'] = shapes[-1]

            plt.clf()
            image._view_landmarks_2d(group='final', marker_size=5, marker_edge_colour='k', marker_face_colour='g')

            plt.show(block=False)
            plt.pause(0.05)

            # plt.savefig(wrFilebase + '/output/' + image.path.stem + '.png')
        # np.save('/Users/arik/Desktop/test/ect/' + image.path.name.rsplit('.', 1)[0], self.transform.target.points)

        return ParametricIterativeResult(shapes=shapes, shape_parameters=p_list,
                                         initial_shape=initial_shape,   #image.landmarks['__initial_shape'] initial_shape
                                         image=image, gt_shape=gt_shape, costs=cost)

    def __str__(self):
        return "Weighted Regularized Mean Shift"
