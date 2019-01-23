import numpy as np
from menpo.transform import PiecewiseAffine
from menpofit.differentiable import DL, DX


class DifferentiablePiecewiseAffine(PiecewiseAffine, DL, DX):
    r"""
    A differentiable Piecewise Affine Transformation.

    This is composed of a number of triangles defined be a set of `source` and
    `target` vertices. These vertices are related by a common triangle `list`.
    No limitations on the nature of the triangle `list` are imposed. Points can
    then be mapped via barycentric coordinates from the `source` to the `target`
    space. Trying to map points that are not contained by any source triangle
    throws a `TriangleContainmentError`, which contains diagnostic information.

    The transform can compute its own derivative with respect to spatial changes,
    as well as anchor landmark changes.
    """

    def d_dl(self, points):
        r"""
        The derivative of the warp with respect to spatial changes in anchor
        landmark points or centres, evaluated at points.

        Parameters
        ----------
        points : ``(n_points, n_dims)`` `ndarray`
            The spatial points at which the derivative should be evaluated.

        Returns
        -------
        d_dl : ``(n_points, n_centres, n_dims)`` `ndarray`
            The Jacobian wrt landmark changes.

            ``d_dl[i, k, m]`` is the scalar differential change that the
            any dimension of the ``i``'th point experiences due to a first order
            change in the ``m``'th dimension of the ``k``'th landmark point.

            Note that at present this assumes that the change in every
            dimension is equal.
        """
        tri_index, alpha_i, beta_i = self.index_alpha_beta(points)
        # for the jacobian we only need
        # gamma = 1 - alpha - beta
        # for each vertex (i, j, & k)
        # gamma is the 'far edge' weighting wrt the vertex in question.
        # given gamma implicitly for the first vertex in our trilist,
        # we can permute around to get the others. (e.g. rotate CW around
        # the triangle to get the j'th vertex-as-prime variant,
        # and once again to the kth).
        #
        # alpha_j = 1 - alpha_i - beta_i
        # gamma_j = alpha_i
        # gamma_k = beta_i
        #
        # TODO this ordering is empirically correct but I don't know why..
        #
        # we stack all the gamma's together
        # so gamma_ijk.shape = (n_sample_points, 3)
        gamma_ijk = np.hstack(((1 - alpha_i - beta_i)[:, None],
                               alpha_i[:, None],
                               beta_i[:, None]))
        # the jacobian wrt source is of shape
        # (n_sample_points, n_source_points, 2)
        jac = np.zeros((points.shape[0], self.n_points, 2))
        # per sample point, find the source points for the ijk vertices of
        # the containing triangle - only these points will get a non 0
        # jacobian value
        ijk_per_point = self.trilist[tri_index]
        # to index into the jacobian, we just need a linear iterator for the
        # first axis - literally [0, 1, ... , n_sample_points]. The
        # reshape is needed to make it broadcastable with the other indexing
        # term, ijk_per_point.
        linear_iterator = np.arange(points.shape[0]).reshape((-1, 1))
        # in one line, we are done.
        jac[linear_iterator, ijk_per_point] = gamma_ijk[..., None]
        return jac

    def d_dx(self, points):
        r"""
        The first order derivative of the warp with respect to spatial changes
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

        Raises
        ------
        TriangleContainmentError:
            If any point is outside any triangle of this PWA.
        """
        # TODO check for position and return true d_dx (see docstring)
        # for the time being we assume the points are on the source landmarks
        return np.eye(2, 2)[None, ...]
