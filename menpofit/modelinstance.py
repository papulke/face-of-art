import numpy as np

from menpo.base import Targetable, Vectorizable
from menpo.model import MeanLinearModel, PCAModel
from menpo.model.vectorizable import VectorizableBackedModel
from menpo.shape import mean_pointcloud

from menpofit.builder import align_shapes
from menpofit.differentiable import DP


class _SimilarityModel(VectorizableBackedModel, MeanLinearModel):

    def __init__(self, components, mean):
        MeanLinearModel.__init__(self, components, mean.as_vector())
        VectorizableBackedModel.__init__(self, mean)

    def project_vector(self, instance_vector, project_weight=None):
        return MeanLinearModel.project(self, instance_vector)

    def reconstruct_vector(self, instance_vector):
        return MeanLinearModel.reconstruct(self, instance_vector)

    def instance_vector(self, weights):
        return MeanLinearModel.instance(self, weights)

    def component_vector(self, index):
        return MeanLinearModel.component(self, index)

    def project_out_vector(self, instance_vector):
        return MeanLinearModel.project_out(self, instance_vector)

    def __str__(self):
        str_out = 'Similarity Transform Model \n' \
                  ' - # features:           {}\n' \
                  ' - total # components:   {}\n' \
                  ' - components shape:     {}\n'.format(
            self.n_features, self.n_components, self.components.shape)
        return str_out


def similarity_2d_instance_model(shape):
    r"""
    Creates a `menpo.model.MeanLinearModel` that encodes the 2D similarity
    transforms that can be applied on a 2D shape that consists of `n_points`.

    Parameters
    ----------
    shape : `menpo.shape.PointCloud`
        The input 2D shape.

    Returns
    -------
    model : `subclass` of `menpo.model.MeanLinearModel`
        Linear model with four components, the linear combinations of which
        represent the original shape under a similarity transform. The model is
        exhaustive (that is, all possible similarity transforms can be expressed
        with the model).
    """
    shape_vector = shape.as_vector()
    components = np.zeros((4, shape_vector.shape[0]))
    components[0, :] = shape_vector  # Comp. 1 - just the shape
    rotated_ccw = shape.points[:, ::-1].copy()  # flip x,y -> y,x
    rotated_ccw[:, 0] = -rotated_ccw[:, 0]  # negate (old) y
    components[1, :] = rotated_ccw.flatten()  # C2 - the shape rotated 90 degs
    components[2, ::2] = 1  # Tx
    components[3, 1::2] = 1  # Ty
    return _SimilarityModel(components, shape)


class ModelInstance(Targetable, Vectorizable, DP):
    r"""
    Base class for creating a model that can produce a target
    `menpo.shape.PointCloud` and knows how to compute its own derivative with
    respect to its parametrisation.

    Parameters
    ----------
    model : `class`
        The trained model (e.g. `menpo.model.PCAModel`).
    """
    def __init__(self, model):
        self.model = model
        self._target = None
        # set all weights to 0 (yielding the mean, first call to
        # from_vector_inplace() or set_target() will update this)
        self._weights = np.zeros(self.model.n_active_components)

    @property
    def n_weights(self):
        r"""
        The number of parameters in the linear model.

        :type: `int`
        """
        return self.model.n_active_components

    @property
    def weights(self):
        r"""
        The weights of the model.

        :type: ``(n_weights,)`` `ndarray`
        """
        return self._weights

    @property
    def target(self):
        r"""
        The current `menpo.shape.PointCloud` that this object produces.

        :type: `menpo.shape.PointCloud`
        """
        return self._target

    def _target_setter(self, new_target):
        r"""
        Called by the Targetable framework when set_target() is called.
        This method **ONLY SETS THE NEW TARGET** it does no synchronisation
        logic (for that, see _sync_state_from_target())
        """
        self._target = new_target

    def _new_target_from_state(self):
        r"""
        Return the appropriate target for the parameters provided.
        Subclasses can override this.

        Returns
        -------
        new_target : model instance
        """
        return self.model.instance(self.weights)

    def _sync_state_from_target(self):
        # 1. Find the optimum parameters and set them
        self._weights = self._weights_for_target(self.target)
        # 2. Find the closest target the model can reproduce and trigger an
        # update of our transform
        self._target_setter(self._new_target_from_state())

    def _weights_for_target(self, target):
        r"""
        Return the appropriate model weights for target provided.
        Subclasses can override this.

        Parameters
        ----------

        target: model instance
            The target that the statistical model will try to reproduce

        Returns
        -------

        weights: (P,) ndarray
            Weights of the statistical model that generate the closest
            instance to the requested target
        """
        return self.model.project(target)

    def _as_vector(self):
        r"""
        Return the current parameters of this transform - this is the
        just the linear model's weights

        Returns
        -------
        params : (`n_parameters`,) ndarray
            The vector of parameters
        """
        return self.weights

    def _from_vector_inplace(self, vector):
        r"""
        Updates this `menpofit.modelinstance.ModelInstance` from it's
        vectorized form (in this case, simply the weights on the linear model)
        """
        self._weights = vector
        self._sync_target_from_state()


class GlobalSimilarityModel(Targetable, Vectorizable):
    r"""
    Class for creating a model that represents a global similarity transform
    (in-plane rotation, scaling, translation).

    Parameters
    ----------
    data : `list` of `menpo.shape.PointCloud`
        The `list` of shapes to use as training data.
    """
    def __init__(self, data, **kwargs):
        from menpofit.transform import DifferentiableAlignmentSimilarity

        aligned_shapes = align_shapes(data)
        self.mean = mean_pointcloud(aligned_shapes)
        # Default target is the mean
        self._target = self.mean
        self.transform = DifferentiableAlignmentSimilarity(self.target,
                                                           self.target)

    @property
    def n_weights(self):
        r"""
        The number of parameters in the linear model.

        :type: `int`
        """
        return 4

    @property
    def weights(self):
        r"""
        The weights of the model.

        :type: ``(n_weights,)`` `ndarray`
        """
        return self.transform.as_vector()

    @property
    def target(self):
        r"""
        The current `menpo.shape.PointCloud` that this object produces.

        :type: `menpo.shape.PointCloud`
        """
        return self._target

    def set_target(self, new_target):
        r"""
        Update this object so that it attempts to recreate the ``new_target``.

        Parameters
        ----------
        new_target : `menpo.shape.PointCloud`
            The new target that this object should try and regenerate.
        """
        self.transform.set_target(new_target)
        self._target = self.transform.apply(self.mean)
        return self

    def _as_vector(self):
        r"""
        Return the current parameters of this transform - this is the
        just the linear model's weights

        Returns
        -------
        params : (`n_parameters`,) ndarray
            The vector of parameters
        """
        return self.transform.as_vector()

    def _from_vector_inplace(self, vector):
        self.transform._from_vector_inplace(vector)
        self._target = self.transform.apply(self.mean)

    @property
    def n_dims(self):
        r"""
        The number of dimensions of the spatial instance of the model.

        :type: `int`
        """
        return self.mean.n_dims

    def d_dp(self, _):
        r"""
        Returns the Jacobian of the similarity model reshaped in order to have
        the standard Jacobian shape, i.e. ``(n_points, n_weights, n_dims)``
        which maps to ``(n_features, n_components, n_dims)`` on the linear model.

        Returns
        -------
        jacobian : ``(n_features, n_components, n_dims)`` `ndarray`
            The Jacobian of the model in the standard Jacobian shape.
        """
        # Always evaluated at the mean shape
        return self.transform.d_dp(self.mean.points)


class PDM(ModelInstance):
    r"""
    Class for building a Point Distribution Model. It is a specialised version
    of :map:`ModelInstance` for use with spatial data.

    Parameters
    ----------
    data : `list` of `menpo.shape.PointCloud` or `menpo.model.PCAModel` instance
        If a `list` of `menpo.shape.PointCloud`, then a `menpo.model.PCAModel`
        will be trained from those training shapes. Otherwise, a trained
        `menpo.model.PCAModel` instance can be provided.
    max_n_components : `int` or ``None``, optional
        The maximum number of components that the model will keep. If ``None``,
        then all the components will be kept.
    """
    def __init__(self, data, max_n_components=None):
        if isinstance(data, PCAModel):
            shape_model = data
        else:
            aligned_shapes = align_shapes(data)
            shape_model = PCAModel(aligned_shapes)

        if max_n_components is not None:
            shape_model.trim_components(max_n_components)
        super(PDM, self).__init__(shape_model)
        # Default target is the mean
        self._target = self.model.mean()

    @property
    def n_active_components(self):
        r"""
        The number of components currently in use on this model.

        :type: `int`
        """
        return self.model.n_active_components

    @n_active_components.setter
    def n_active_components(self, value):
        r"""
        Sets an updated number of active components on this model. The number
        of active components represents the number of principal components
        that will be used for generative purposes. Note that this therefore
        makes the model stateful. Also note that setting the number of
        components will not affect memory unless `trim_components` is called.

        Parameters
        ----------
        value : `int`
            The new number of active components.

        Raises
        ------
        ValueError
            Tried setting n_active_components to {value} - value needs to be a
            float 0.0 < n_components < self._total_kept_variance_ratio ({}) or
            an integer 1 < n_components < self.n_components ({})
        """
        self.model.n_active_components = value
        self._sync_state_from_target()

    @property
    def n_dims(self):
        r"""
        The number of dimensions of the spatial instance of the model

        :type: `int`
        """
        return self.model.template_instance.n_dims

    def d_dp(self, points):
        r"""
        Returns the Jacobian of the similarity model reshaped in order to have
        the standard Jacobian shape, i.e. ``(n_points, n_weights, n_dims)``
        which maps to ``(n_features, n_components, n_dims)`` on the linear model.

        Returns
        -------
        jacobian : ``(n_features, n_components, n_dims)`` `ndarray`
            The Jacobian of the model in the standard Jacobian shape.
        """
        d_dp = self.model.components.reshape(self.model.n_active_components,
                                             -1, self.n_dims)
        return d_dp.swapaxes(0, 1)

    def increment(self, shapes, n_shapes=None, forgetting_factor=1.0,
                  max_n_components=None, verbose=False):
        r"""
        Update the eigenvectors, eigenvalues and mean vector of this model
        by performing incremental PCA on the given samples.

        Parameters
        ----------
        shapes : `list` of `menpo.shape.PointCloud`
            List of new shapes to update the model from.
        n_shapes : `int` or ``None``, optional
            If `int`, then `shapes`  must be an iterator that yields `n_shapes`.
            If ``None``, then `shapes` has to be a list (so we know how large
            the data matrix needs to be).
        forgetting_factor : ``[0.0, 1.0]`` `float`, optional
            Forgetting factor that weights the relative contribution of new
            samples vs old samples. If 1.0, all samples are weighted equally
            and, hence, the results is the exact same as performing batch
            PCA on the concatenated list of old and new simples. If <1.0,
            more emphasis is put on the new samples. See [1] for details.
        max_n_components : `int` or ``None``, optional
            The maximum number of components that the model will keep.
            If ``None``, then all the components will be kept.
        verbose : `bool`, optional
            If ``True``, then information about the progress will be printed.

        References
        ----------
        .. [1] D. Ross, J. Lim, R.S. Lin, M.H. Yang. "Incremental Learning for
            Robust Visual Tracking". International Journal on Computer Vision,
            2007.
        """
        old_target = self.target
        aligned_shapes = align_shapes(shapes)
        self.model.increment(aligned_shapes, n_samples=n_shapes,
                             forgetting_factor=forgetting_factor,
                             verbose=verbose)
        if max_n_components is not None:
            self.model.trim_components(max_n_components)
        # Reset the target given the new model
        self.set_target(old_target)

    def __str__(self):
        str_out = 'Point Distribution Model \n' \
                  ' - centred:              {}\n' \
                  ' - # features:           {}\n' \
                  ' - # active components:  {}\n' \
                  ' - kept variance:        {:.2}  {:.1%}\n' \
                  ' - noise variance:       {:.2}  {:.1%}\n' \
                  ' - total # components:   {}\n' \
                  ' - components shape:     {}\n'.format(
            self.model.centred,  self.model.n_features, self.n_active_components,
            self.model.variance(), self.model.variance_ratio(),
            self.model.noise_variance(), self.model.noise_variance_ratio(),
            self.model.n_components, self.model.components.shape)
        return str_out


class GlobalPDM(PDM):
    r"""
    Class for building a Point Distribution Model that also stores a Global
    Alignment transform. The final transform couples the Global Alignment
    transform to a statistical linear model, so that its weights are fully
    specified by both the weights of statistical model and the weights of the
    similarity transform.

    Parameters
    ----------
    data : `list` of `menpo.shape.PointCloud` or `menpo.model.PCAModel` instance
        If a `list` of `menpo.shape.PointCloud`, then a `menpo.model.PCAModel`
        will be trained from those training shapes. Otherwise, a trained
        `menpo.model.PCAModel` instance can be provided.
    global_transform_cls : `class`
        The Global Similarity transform class
        (e.g. :map:`DifferentiableAlignmentSimilarity`).
    max_n_components : `int` or ``None``, optional
        The maximum number of components that the model will keep. If ``None``,
        then all the components will be kept.
    """
    def __init__(self, data, global_transform_cls, max_n_components=None):
        super(GlobalPDM, self).__init__(data, max_n_components=max_n_components)
        # Start the global_transform as an identity (first call to
        # from_vector_inplace() or set_target() will update this)
        mean = self.model.mean()
        self.global_transform = global_transform_cls(mean, mean)

    @property
    def n_global_parameters(self):
        r"""
        The number of parameters in the `global_transform`

        :type: `int`
        """
        return self.global_transform.n_parameters

    @property
    def global_parameters(self):
        r"""
        The parameters for the global transform.

        :type: ``(n_global_parameters,) `ndarray`
        """
        return self.global_transform.as_vector()

    def _new_target_from_state(self):
        r"""
        Return the appropriate target for the model weights provided,
        accounting for the effect of the global transform

        Returns
        -------
        new_target : `menpo.shape.PointCloud`
            A new target for the weights provided
        """
        return self.global_transform.apply(self.model.instance(self.weights))

    def _weights_for_target(self, target):
        r"""
        Return the appropriate model weights for target provided, accounting
        for the effect of the global transform. Note that this method
        updates the global transform to be in the correct state.

        Parameters
        ----------
        target : `menpo.shape.PointCloud`
            The target that the statistical model will try to reproduce

        Returns
        -------
        weights : ``(P,)`` `ndarray`
            Weights of the statistical model that generate the closest
            PointCloud to the requested target
        """
        self._update_global_transform(target)
        projected_target = self.global_transform.pseudoinverse().apply(target)
        # now we have the target in model space, project it to recover the
        # weights
        try:
            new_weights = self.model.project(projected_target, target.project_weight)
        except AttributeError:
            new_weights = self.model.project(projected_target)
        # TODO investigate the impact of this, could be problematic
        # the model can't perfectly reproduce the target we asked for -
        # reset the global_transform.target to what it CAN produce
        #refined_target = self._target_for_weights(new_weights)
        #self.global_transform.target = refined_target
        return new_weights

    def _update_global_transform(self, target):
        self.global_transform.set_target(target)

    def _as_vector(self):
        r"""
        Return the current parameters of this transform - this is the
        just the linear model's weights

        Returns
        -------
        params : ``(n_parameters,)`` `ndarray`
            The vector of parameters
        """
        return np.hstack([self.global_parameters, self.weights])

    def _from_vector_inplace(self, vector):
        # First, update the global transform
        global_parameters = vector[:self.n_global_parameters]
        self._update_global_weights(global_parameters)
        # Now extract the weights, and let super handle the update
        weights = vector[self.n_global_parameters:]
        PDM._from_vector_inplace(self, weights)

    def _update_global_weights(self, global_weights):
        r"""
        Hook that allows for overriding behavior when the global weights are
        set. Default implementation simply asks global_transform to
        update itself from vector.
        """
        self.global_transform._from_vector_inplace(global_weights)

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
        """
        # d_dp is always evaluated at the mean shape
        if points is None:
            points = self.model.mean().points

        # compute dX/dp

        # dX/dq is the Jacobian of the global transform evaluated at the
        # current target
        # (n_points, n_global_params, n_dims)
        dX_dq = self._global_transform_d_dp(points)

        # by application of the chain rule dX/db is the Jacobian of the
        # model transformed by the linear component of the global transform
        # (n_points, n_weights, n_dims)
        dS_db = PDM.d_dp(self, [])
        # (n_points, n_dims, n_dims)
        dX_dS = self.global_transform.d_dx(points)
        # (n_points, n_weights, n_dims)
        dX_db = np.einsum('ilj, idj -> idj', dX_dS, dS_db)

        # dX/dp is simply the concatenation of the previous two terms
        # (n_points, n_params, n_dims)
        return np.hstack((dX_dq, dX_db))

    def _global_transform_d_dp(self, points):
        return self.global_transform.d_dp(points)


class OrthoPDM(GlobalPDM):
    r"""
    Class for building a Point Distribution Model that also stores a Global
    Alignment transform. The final transform couples the Global Alignment
    transform to a statistical linear model, so that its weights are fully
    specified by both the weights of statistical model and the weights of the
    similarity transform.

    This transform (in contrast to the :map`GlobalPDM`) additionally
    orthonormalises both the global and the model basis against each other,
    ensuring that orthogonality and normalization is enforced across the unified
    bases.

    Parameters
    ----------
    data : `list` of `menpo.shape.PointCloud` or `menpo.model.PCAModel` instance
        If a `list` of `menpo.shape.PointCloud`, then a `menpo.model.PCAModel`
        will be trained from those training shapes. Otherwise, a trained
        `menpo.model.PCAModel` instance can be provided.
    max_n_components : `int` or ``None``, optional
        The maximum number of components that the model will keep. If ``None``,
        then all the components will be kept.
    """
    def __init__(self, data, max_n_components=None):
        from menpofit.transform import DifferentiableAlignmentSimilarity
        super(OrthoPDM, self).__init__(
            data,  DifferentiableAlignmentSimilarity,
            max_n_components=max_n_components)
        self._construct_similarity_model()
        # Set target from state (after orthonormalizing)
        self._sync_target_from_state()

    def _construct_similarity_model(self):
        # 1. Construct similarity model from the mean of the model
        model_mean = self.model.mean()
        self.similarity_model = similarity_2d_instance_model(model_mean)
        # 2. Orthonormalize model and similarity model
        self.model.orthonormalize_against_inplace(self.similarity_model)
        # The number of components may have changed. So re-create the weights
        # from the new number of active components
        self._weights = np.zeros(self.model.n_active_components)
        self.similarity_weights = self.similarity_model.project(model_mean)

    @property
    def global_parameters(self):
        r"""
        The parameters for the global transform.

        :type: ``(n_global_parameters,)`` `ndarray`
        """
        return self.similarity_weights

    def _update_global_transform(self, target):
        try:
            self.similarity_weights = self.similarity_model.project(target, target.project_weight)    # , target.project_weight
        except AttributeError:
            self.similarity_weights = self.similarity_model.project(target)
        self._update_global_weights(self.similarity_weights)

    def _update_global_weights(self, global_weights):
        self.similarity_weights = global_weights
        new_target = self.similarity_model.instance(global_weights)
        self.global_transform.set_target(new_target)

    def _global_transform_d_dp(self, points):
        return self.similarity_model.components.reshape(
            self.n_global_parameters, -1, self.n_dims).swapaxes(0, 1)

    def increment(self, shapes, n_shapes=None, forgetting_factor=1.0,
                  max_n_components=None, verbose=False):
        r"""
        Update the eigenvectors, eigenvalues and mean vector of this model
        by performing incremental PCA on the given samples.

        Parameters
        ----------
        shapes : `list` of `menpo.shape.PointCloud`
            List of new shapes to update the model from.
        n_shapes : `int` or ``None``, optional
            If `int`, then `shapes`  must be an iterator that yields `n_shapes`.
            If ``None``, then `shapes` has to be a list (so we know how large
            the data matrix needs to be).
        forgetting_factor : ``[0.0, 1.0]`` `float`, optional
            Forgetting factor that weights the relative contribution of new
            samples vs old samples. If 1.0, all samples are weighted equally
            and, hence, the results is the exact same as performing batch
            PCA on the concatenated list of old and new simples. If <1.0,
            more emphasis is put on the new samples. See [1] for details.
        max_n_components : `int` or ``None``, optional
            The maximum number of components that the model will keep.
            If ``None``, then all the components will be kept.
        verbose : `bool`, optional
            If ``True``, then information about the progress will be printed.

        References
        ----------
        .. [1] D. Ross, J. Lim, R.S. Lin, M.H. Yang. "Incremental Learning for
            Robust Visual Tracking". International Journal on Computer Vision,
            2007.
        """
        old_target = self.target
        aligned_shapes = align_shapes(shapes)
        self.model.increment(aligned_shapes, n_samples=n_shapes,
                             forgetting_factor=forgetting_factor,
                             verbose=verbose)
        if max_n_components is not None:
            self.model.trim_components(max_n_components)
        # Re-orthonormalize
        self._construct_similarity_model()
        # Reset the target given the new models
        self.set_target(old_target)

    def __str__(self):
        str_out = 'Point Distribution Model with Similarity Transform \n' \
                  ' - total # components:      {}\n' \
                  ' - # similarity components: {}\n' \
                  ' - # PCA components:        {}\n' \
                  ' - # active components:     {} + {} = {}\n' \
                  ' - centred:                 {}\n' \
                  ' - # features:              {}\n' \
                  ' - kept variance:           {:.2}  {:.1%}\n' \
                  ' - noise variance:          {:.2}  {:.1%}\n' \
                  ' - components shape:        {}\n'.format(
            self.similarity_model.n_components + self.model.n_components,
            self.similarity_model.n_components, self.model.n_components,
            self.similarity_model.n_components, self.n_active_components,
            self.n_parameters, self.model.centred, self.model.n_features,
            self.model.variance(), self.model.variance_ratio(),
            self.model.noise_variance(), self.model.noise_variance_ratio(),
            self.model.components.shape)
        return str_out
