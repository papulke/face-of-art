import numpy as np

from menpo.transform import (Affine, AlignmentAffine, Similarity,
                             AlignmentSimilarity)

from menpofit.differentiable import DP, DX


class DifferentiableAffine(Affine, DP, DX):
    r"""
    Base class for an affine transformation that can compute its own derivative
    with respect to spatial changes, as well as its parametrisation.
    """
    def d_dp(self, points):
        r"""
        The derivative with respect to the parametrisation changes evaluated at
        points.

        Parameters
        ----------
        points : ``(n_points, n_dims)`` `ndarray`
            The spatial points at which the derivative should be evaluated.

        Returns
        -------
        d_dp : ``(n_points, n_parameters, n_dims)`` `ndarray`
            The Jacobian with respect to the parametrisation.

            ``d_dp[i, j, k]`` is the scalar differential change that the
            ``k``'th dimension of the ``i``'th point experiences due to a first
            order change in the ``j``'th scalar in the parametrisation vector.
        """
        return affine_d_dp(self, points)

    def d_dx(self, points):
        r"""
        The first order derivative with respect to spatial changes evaluated at
        points.

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
        return affine_d_dx(self)


class DifferentiableAlignmentAffine(AlignmentAffine, DP, DX):
    r"""
    Base class that constructs an affine transformation that is the optimal
    transform to align the `source` to the `target`. It can compute its own
    derivative with respect to spatial changes, as well as its parametrisation.
    """
    def as_non_alignment(self):
        r"""
        Returns the non-alignment version of the transform.

        :type: :map:`DifferentiableAffine`
        """
        return DifferentiableAffine(self.h_matrix, skip_checks=True)

    def d_dp(self, points):
        r"""
        The derivative with respect to the parametrisation changes evaluated at
        points.

        Parameters
        ----------
        points : ``(n_points, n_dims)`` `ndarray`
            The spatial points at which the derivative should be evaluated.

        Returns
        -------
        d_dp : ``(n_points, n_parameters, n_dims)`` `ndarray`
            The Jacobian with respect to the parametrisation.

            ``d_dp[i, j, k]`` is the scalar differential change that the
            ``k``'th dimension of the ``i``'th point experiences due to a first
            order change in the ``j``'th scalar in the parametrisation vector.
        """
        return affine_d_dp(self, points)

    def d_dx(self, points):
        r"""
        The first order derivative with respect to spatial changes evaluated at
        points.

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
        return affine_d_dx(self)


class DifferentiableSimilarity(Similarity, DP, DX):
    r"""
    Base class for a similarity transformation that can compute its own
    derivative  with respect to spatial changes, as well as its parametrisation.
    """
    def d_dp(self, points):
        r"""
        The derivative with respect to the parametrisation changes evaluated at
        points.

        Parameters
        ----------
        points : ``(n_points, n_dims)`` `ndarray`
            The spatial points at which the derivative should be evaluated.

        Returns
        -------
        d_dp : ``(n_points, n_parameters, n_dims)`` `ndarray`
            The Jacobian with respect to the parametrisation.

            ``d_dp[i, j, k]`` is the scalar differential change that the
            ``k``'th dimension of the ``i``'th point experiences due to a first
            order change in the ``j``'th scalar in the parametrisation vector.
        """
        return similarity_d_dp(self, points)

    def d_dx(self, points):
        r"""
        The first order derivative with respect to spatial changes evaluated at
        points.

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
        return affine_d_dx(self)


class DifferentiableAlignmentSimilarity(AlignmentSimilarity, DP, DX):
    r"""
    Base class that constructs a similarity transformation that is the optimal
    transform to align the `source` to the `target`. It can compute its own
    derivative with respect to spatial changes, as well as its parametrisation.
    """
    def as_non_alignment(self):
        r"""
        Returns the non-alignment version of the transform.

        :type: :map:`DifferentiableSimilarity`
        """
        return DifferentiableSimilarity(self.h_matrix, skip_checks=True)

    def d_dp(self, points):
        r"""
        The derivative with respect to the parametrisation changes evaluated at
        points.

        Parameters
        ----------
        points : ``(n_points, n_dims)`` `ndarray`
            The spatial points at which the derivative should be evaluated.

        Returns
        -------
        d_dp : ``(n_points, n_parameters, n_dims)`` `ndarray`
            The Jacobian with respect to the parametrisation.

            ``d_dp[i, j, k]`` is the scalar differential change that the
            ``k``'th dimension of the ``i``'th point experiences due to a first
            order change in the ``j``'th scalar in the parametrisation vector.
        """
        return similarity_d_dp(self, points)

    def d_dx(self, points):
        r"""
        The first order derivative with respect to spatial changes evaluated at
        points.

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
        return affine_d_dx(self)


def affine_d_dx(affine):
    r"""
    The first order derivative of an Affine transform with respect to spatial
    changes evaluated at points.

    The Jacobian for a given point (for 2D) is of the form::

        Jx = [(1 + a),     -b  ]
        Jy = [   b,     (1 + a)]
        J =  [Jx, Jy] = [[(1 + a), -b], [b, (1 + a)]]

    where a and b come from:

        W(x;p) = [1 + a   -b      tx] [x]
                 [b       1 + a   ty] [y]
                                      [1]
    Hence it is simply the linear component of the transform.

    Returns
    -------
    d_dx : ``(1, n_dims, n_dims)`` `ndarray`
        The Jacobian with respect to spatial changes.

        ``d_dx[0, j, k]`` is the scalar differential change that the
        ``j``'th dimension of the ``i``'th point experiences due to a first order
        change in the ``k``'th dimension.

        Note that because the Jacobian is constant across space the first
        axis is length ``1`` to allow for broadcasting.
    """
    return affine.linear_component[None, ...]


def affine_d_dp(self, points):
    r"""
    The first order derivative of this Affine transform wrt parameter
    changes evaluated at points.

    The Jacobian generated (for 2D) is of the form::

        x 0 y 0 1 0
        0 x 0 y 0 1

    This maintains a parameter order of::

      W(x;p) = [1 + p1  p3      p5] [x]
               [p2      1 + p4  p6] [y]
                                    [1]

    Parameters
    ----------
    points : ``(n_points, n_dims)`` `ndarray`
        The spatial points at which the derivative should be evaluated.

    Returns
    -------
    d_dp : ``(n_points, n_parameters, n_dims)`` `ndarray`
        The Jacobian with respect to the parametrisation.

        ``d_dp[i, j, k]`` is the scalar differential change that the
        ``k``'th dimension of the ``i``'th point experiences due to a first
        order change in the ``j``'th scalar in the parametrisation vector.
    """
    n_points, points_n_dim = points.shape
    if points_n_dim != self.n_dims:
        raise ValueError(
            "Trying to sample jacobian in incorrect dimensions "
            "(transform is {0}D, sampling at {1}D)".format(
                self.n_dims, points_n_dim))
    # prealloc the jacobian
    jac = np.zeros((n_points, self.n_parameters, self.n_dims))
    # a mask that we can apply at each iteration
    dim_mask = np.eye(self.n_dims, dtype=np.bool)

    for i, s in enumerate(
            range(0, self.n_dims * self.n_dims, self.n_dims)):
        # i is current axis
        # s is slicing offset
        # make a mask for a single points jacobian
        full_mask = np.zeros((self.n_parameters, self.n_dims), dtype=bool)
        # fill the mask in for the ith axis
        full_mask[slice(s, s + self.n_dims)] = dim_mask
        # assign the ith axis points to this mask, broadcasting over all
        # points
        jac[:, full_mask] = points[:, i][..., None]
        # finally, just repeat the same but for the ones at the end
    full_mask = np.zeros((self.n_parameters, self.n_dims), dtype=bool)
    full_mask[slice(s + self.n_dims, s + 2 * self.n_dims)] = dim_mask
    jac[:, full_mask] = 1
    return jac


def similarity_d_dp(sim, points):
    r"""
    Computes the Jacobian of the transform w.r.t the parameters.

    The Jacobian generated (for 2D) is of the form::

        x -y 1 0
        y  x 0 1

    This maintains a parameter order of::

      W(x;p) = [1 + a  -b   ] [x] + tx
               [b      1 + a] [y] + ty

    Parameters
    ----------
    points : ``(n_points, n_dims)`` `ndarray`
        The spatial points at which the derivative should be evaluated.

    Returns
    -------
    d_dp : ``(n_points, n_parameters, n_dims)`` `ndarray`
        The Jacobian with respect to the parametrisation.

        ``d_dp[i, j, k]`` is the scalar differential change that the
        ``k``'th dimension of the ``i``'th point experiences due to a first
        order change in the ``j``'th scalar in the parametrisation vector.

    Raises
    ------
    DimensionalityError
        'points.n_dims != self.n_dim' or transform is not 2D
    """
    n_points, points_n_dim = points.shape
    if points_n_dim != sim.n_dims:
        raise ValueError('Trying to sample jacobian in incorrect '
                         'dimensions (transform is {0}D, sampling '
                         'at {1}D)'.format(sim.n_dims, points_n_dim))
    elif sim.n_dims != 2:
        # TODO: implement 3D Jacobian
        raise ValueError("Only the Jacobian of a 2D similarity "
                         "transform is currently supported.")

    # prealloc the jacobian
    jac = np.zeros((n_points, sim.n_parameters, sim.n_dims))
    ones = np.ones_like(points)

    # Build a mask and apply it to the points to build the jacobian
    # Do this for each parameter - [a, b, tx, ty] respectively
    _apply_jacobian_mask(sim, jac, np.array([1, 1]), 0, points)
    _apply_jacobian_mask(sim, jac, np.array([-1, 1]), 1, points[:, ::-1])
    _apply_jacobian_mask(sim, jac, np.array([1, 0]), 2, ones)
    _apply_jacobian_mask(sim, jac, np.array([0, 1]), 3, ones)

    return jac


def _apply_jacobian_mask(sim, jac, param_mask, row_index, points):
    # make a mask for a single points jacobian
    full_mask = np.zeros((sim.n_parameters, sim.n_dims), dtype=np.bool)
    # fill the mask in for the ith axis
    full_mask[row_index] = [True, True]
    # assign the ith axis points to this mask, broadcasting over all
    # points
    jac[:, full_mask] = points * param_mask
