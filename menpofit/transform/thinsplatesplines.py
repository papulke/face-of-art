import numpy as np

from menpo.transform import ThinPlateSplines

from menpofit.differentiable import DL, DX

from .rbf import DifferentiableR2LogR2RBF


class DifferentiableThinPlateSplines(ThinPlateSplines, DL, DX):
    r"""
    The Thin Plate Splines (TPS) alignment between 2D `source` and `target`
    landmarks. The transform can compute its own derivative with respect to
    spatial changes, as well as anchor landmark changes.

    Parameters
    ----------
    source : ``(N, 2)`` `ndarray`
        The source points to apply the tps from
    target : ``(N, 2)`` `ndarray`
        The target points to apply the tps to
    kernel : `class` or ``None``, optional
        The differentiable kernel to apply. Possible options are
        :map:`DifferentiableR2LogRRBF` and :map:`DifferentiableR2LogR2RBF`. If
        ``None``, then :map:`DifferentiableR2LogR2RBF` is used.
    """
    def __init__(self, source, target, kernel=None):
        if kernel is None:
            kernel = DifferentiableR2LogR2RBF(source.points)
        ThinPlateSplines.__init__(self, source, target, kernel=kernel)

    def d_dl(self, points):
        """
        Calculates the Jacobian of the TPS warp wrt to the source landmarks
        assuming that he target is equal to the source. This is a special
        case of the Jacobian wrt to the source landmarks that is used in AAMs
        to weight the relative importance of each pixel in the reference
        frame wrt to each one of the source landmarks.

        dW_dl =      dOmega_dl         *  k(points)
              = T *     d_L**-1_dl     *  k(points)
              = T * -L**-1 dL_dl L**-1 *  k(points)

        # per point
        (c, d) = (d, c+3) (c+3, c+3) (c+3, c+3, c, d) (c+3, c+3) (c+3)
        (c, d) = (d,            c+3) (c+3, c+3, c, d) (c+3,)
        (c, d) = (d,               ) (          c, d)
        (c, d) = (                 ) (          c, d)

        Parameters
        ----------
        points : ``(n_points, n_dims)`` `ndarray`
            The spatial points at which the derivative should be evaluated.

        Returns
        -------
        dW/dl : (n_points, n_params, n_dims) ndarray
            The Jacobian of the transform wrt to the source landmarks evaluated
            at the previous points and assuming that the target is equal to
            the source.
        """
        n_centres = self.n_points
        n_points = points.shape[0]

        # TPS kernel (nonlinear + affine)

        # for each input, evaluate the rbf
        # (n_points, n_centres)
        k_points = self.kernel.apply(points)

        # k_points with (1, x, y) appended to each point
        # (n_points, n_centres+3)  - 3 is (1, x, y) for affine component
        k = np.hstack([k_points, np.ones([n_points, 1]), points])

        # (n_centres+3, n_centres+3)
        try:
            inv_L = np.linalg.inv(self.l)
        except np.linalg.LinAlgError:
            # If two points are coincident, or very close to being so, then the
            # matrix is rank deficient and thus not-invertible. Therefore,
            # only take the inverse on the full-rank set of indices.
            _u, _s, _v = np.linalg.svd(self.l)
            keep = _s.shape[0] - sum(_s < self.min_singular_val)
            inv_L = _u[:, :keep].dot(1.0 / _s[:keep, None] * _v[:keep, :])


        # Taking the derivative of L for changes in l must yield an x,y change
        # for each centre.
        # (n_centres+3, n_centres+3, n_centres, n_dims)
        dL_dl = np.zeros(self.l.shape + (n_centres, 2))

        # take the derivative of the kernel wrt centres at the centres
        # SHOULD be (n_centres, n_dims, n_centres, n_dims)
        # IS        (n_centres,         n_centres, n_dims
        dK_dl_at_tgt = self.kernel.d_dl(self.source.points)

        # we want to build a tensor where for each slice where
        # dK_dl[i, j, k, l] is the derivative wrt the l'th dimension of the
        # i'th centre for L[j, k] -> first axis is just looping over centres
        # and last looping over dims
        # (n_centres, n_centres, n_centres, n_dims)
        dK_dl = np.zeros((n_centres, ) + dK_dl_at_tgt.shape)

        # make a linear iterator over the centres
        iter = np.arange(n_centres)

        # efficiently build the repeated pattern for dK_dl
        # note that the repetition over centres happens over axis 0
        # and the dims axis is the last
        # so dK_dl[0, ..., 0] corresponds to dK/dx0 in Joan's paper
        #    dK_dl[3, ..., 1] corresponds to dK_dy3 in Joan's paper
        dK_dl[iter, iter] = dK_dl_at_tgt[iter]
        dK_dl[iter, :, iter] = dK_dl_at_tgt[:, iter]

        # prepare memory for the answer
        # SHOULD be (n_points, n_dims, n_centres, n_dims)
        # IS        (n_points,       , n_centres, n_dims)
        dW_dl = np.zeros((n_points, n_centres, 2))

        # pretend the target is equal to the source
        # (n_dims, n_centres+3)
        pseudo_target = np.hstack([self.source.points.T, np.zeros([2, 3])])

        for i in np.arange(n_centres):
            # dP_dli (n_centres, n_points, n_dims, n_dims)
            dP_dli = np.zeros(self.p.shape + (2,))
            dP_dli[i, 1, 0] = -1
            dP_dli[i, 2, 1] = -1

            dL_dl[:n_centres, :n_centres, i] = dK_dl[i]
            dL_dl[:n_centres, n_centres:, i] = dP_dli
            dL_dl[n_centres:, :n_centres, i] = np.swapaxes(dP_dli, 0, 1)

            omega_x = -inv_L.dot(dL_dl[..., i, 0].dot(inv_L))
            omega_y = -inv_L.dot(dL_dl[..., i, 1].dot(inv_L))
            dW_dl[:, i, 0] = k.dot(omega_x).dot(pseudo_target[0])
            dW_dl[:, i, 1] = k.dot(omega_y).dot(pseudo_target[1])

        return dW_dl

    def d_dx(self, points):
        r"""
        The first order derivative of this TPS warp wrt spatial changes
        evaluated at points.

        Parameters
        ----------
        points : ``(n_points, n_dims)`` `ndarray`
            The spatial points at which the derivative should be evaluated.

        Returns
        -------
        d_dx : ``(n_points, n_dims, n_dims)`` `ndarray`
            The Jacobian wrt spatial changes.

            ``d_dx[i, j, k]`` is the scalar differential change that the
            ``j``'th dimension of the ``i``'th point experiences due to a first
            order change in the ``k``'th dimension.

            It may be the case that the Jacobian is constant across space -
            in this case axis zero may have length ``1`` to allow for
            broadcasting.
        """
        dk_dx = np.zeros((points.shape[0] + 3,   # i
                          self.source.n_points,  # k
                          self.source.n_dims))   # l
        dk_dx[:-3, :] = self.kernel.d_dl(points)

        affine_derivative = np.array([[0, 0],
                                      [1, 0],
                                      [0, 1]])
        dk_dx[-3:, :] = affine_derivative[:, None]

        return np.einsum('ij, ikl -> klj', self.coefficients, dk_dx)
