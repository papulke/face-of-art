from __future__ import division
import warnings
import numpy as np
from scipy.stats import multivariate_normal

from menpo.base import name_of_callable
from menpo.feature import no_op
from menpo.visualize import print_dynamic, print_progress
from menpo.model import GMRFModel
from menpo.shape import (DirectedGraph, UndirectedGraph, Tree, PointTree,
                         PointDirectedGraph, PointUndirectedGraph)

from menpofit import checks
from menpofit.base import batch
from menpofit.modelinstance import OrthoPDM
from menpofit.builder import (compute_features, scale_images, align_shapes,
                              rescale_images_to_reference_shape,
                              extract_patches, MenpoFitBuilderWarning,
                              compute_reference_shape)


class GenerativeAPS(object):
    r"""
    Class for training a multi-scale Generative Active Pictorial Structures
    model.  Please see the references for a basic list of relevant papers.

    Parameters
    ----------
    images : `list` of `menpo.image.Image`
        The `list` of training images.
    group : `str` or ``None``, optional
        The landmark group that will be used to train the AAM. If ``None`` and
        the images only have a single landmark group, then that is the one
        that will be used. Note that all the training images need to have the
        specified landmark group.
    appearance_graph : `list` of graphs or a single graph or ``None``, optional
        The graph to be used for the appearance `menpo.model.GMRFModel` training.
        It must be a `menpo.shape.UndirectedGraph`. If ``None``, then a
        `menpo.model.PCAModel` is used instead.
    shape_graph : `list` of graphs or a single graph or ``None``, optional
        The graph to be used for the shape `menpo.model.GMRFModel` training. It
        must be a `menpo.shape.UndirectedGraph`. If ``None``, then the shape
        model is built using `menpo.model.PCAModel`.
    deformation_graph : `list` of graphs or a single graph or ``None``, optional
        The graph to be used for the deformation `menpo.model.GMRFModel`
        training. It must be either a `menpo.shape.DirectedGraph` or a
        `menpo.shape.Tree`. If ``None``, then the minimum spanning tree of the
        data is computed.
    holistic_features : `closure` or `list` of `closure`, optional
        The features that will be extracted from the training images. Note
        that the features are extracted before warping the images to the
        reference shape. If `list`, then it must define a feature function per
        scale. Please refer to `menpo.feature` for a list of potential features.
    reference_shape : `menpo.shape.PointCloud` or ``None``, optional
        The reference shape that will be used for building the APS. The purpose
        of the reference shape is to normalise the size of the training images.
        The normalization is performed by rescaling all the training images
        so that the scale of their ground truth shapes matches the scale of
        the reference shape. Note that the reference shape is rescaled with
        respect to the `diagonal` before performing the normalisation. If
        ``None``, then the mean shape will be used.
    diagonal : `int` or ``None``, optional
        This parameter is used to rescale the reference shape so that the
        diagonal of its bounding box matches the provided value. In other
        words, this parameter controls the size of the model at the highest
        scale. If ``None``, then the reference shape does not get rescaled.
    scales : `float` or `tuple` of `float`, optional
        The scale value of each scale. They must provided in ascending order,
        i.e. from lowest to highest scale. If `float`, then a single scale is
        assumed.
    patch_shape : (`int`, `int`) or `list` of (`int`, `int`), optional
        The shape of the patches to be extracted. If a `list` is provided,
        then it defines a patch shape per scale.
    patch_normalisation : `list` of `callable` or a single `callable`, optional
        The normalisation function to be applied on the extracted patches. If
        `list`, then it must have length equal to the number of scales. If a
        single patch normalization `callable`, then this is the one applied to
        all scales.
    use_procrustes : `bool`, optional
        If ``True``, then Generalized Procrustes Alignment is applied before
        building the deformation model.
    precision_dtype : `numpy.dtype`, optional
        The data type of the appearance GMRF's precision matrix. For example, it
        can be set to `numpy.float32` for single precision or to `numpy.float64`
        for double precision. Even though the precision matrix is stored as a
        `scipy.sparse` matrix, this parameter has a big impact on the amount of
        memory required by the model.
    max_shape_components : `int`, `float`, `list` of those or ``None``, optional
        The number of shape components to keep. If `int`, then it sets the exact
        number of components. If `float`, then it defines the variance
        percentage that will be kept. If `list`, then it should
        define a value per scale. If a single number, then this will be
        applied to all scales. If ``None``, then all the components are kept.
        Note that the unused components will be permanently trimmed.
    n_appearance_components : `list` of `int` or `int` or ``None``, optional
        The number of appearance components used for building the appearance
        `menpo.shape.GMRFModel`. If `list`, then it must have length equal to
        the number of scales. If a single `int`, then this is the one applied
        to all scales. If ``None``, the covariance matrix of each edge is
        inverted using `np.linalg.inv`. If `int`, it is inverted using
        truncated SVD using the specified number of components.
    can_be_incremented : `bool`, optional
        In case you intend to incrementally update the model in the future,
        then this flag must be set to ``True`` from the first place. Note
        that if ``True``, the appearance and deformation `menpo.shape.GMRFModel`
        models will occupy double memory.
    verbose : `bool`, optional
        If ``True``, then the progress of building the APS will be printed.
    batch_size : `int` or ``None``, optional
        If an `int` is provided, then the training is performed in an
        incremental fashion on image batches of size equal to the provided
        value. If ``None``, then the training is performed directly on the
        all the images.

    References
    ----------
    .. [1] E. Antonakos, J. Alabort-i-Medina, and S. Zafeiriou, "Active
        Pictorial Structures", Proceedings of the IEEE Conference on Computer
        Vision and Pattern Recognition (CVPR), Boston, MA, USA, pp. 1872-1882,
        June 2015.
    """
    def __init__(self, images, group=None, appearance_graph=None,
                 shape_graph=None, deformation_graph=None,
                 holistic_features=no_op, reference_shape=None, diagonal=None,
                 scales=(0.5, 1.0), patch_shape=(17, 17),
                 patch_normalisation=no_op, use_procrustes=True,
                 precision_dtype=np.float32, max_shape_components=None,
                 n_appearance_components=None, can_be_incremented=False,
                 verbose=False, batch_size=None):
        # Check parameters
        checks.check_diagonal(diagonal)
        scales = checks.check_scales(scales)
        n_scales = len(scales)
        holistic_features = checks.check_callable(holistic_features, n_scales)
        patch_shape = checks.check_patch_shape(patch_shape, n_scales)
        patch_normalisation = checks.check_callable(patch_normalisation,
                                                    n_scales)
        max_shape_components = checks.check_max_components(
            max_shape_components, n_scales, 'max_shape_components')
        n_appearance_components = checks.check_max_components(
            n_appearance_components, n_scales, 'n_appearance_components')

        # Assign attributes
        self.diagonal = diagonal
        self.scales = scales
        self.holistic_features = holistic_features
        self.patch_shape = patch_shape
        self.patch_normalisation = patch_normalisation
        self.reference_shape = reference_shape
        self.use_procrustes = use_procrustes
        self.is_incremental = can_be_incremented
        self.precision_dtype = precision_dtype
        self.max_shape_components = max_shape_components
        self.n_appearance_components = n_appearance_components

        # Check provided graphs
        self.appearance_graph = checks.check_graph(
            appearance_graph, UndirectedGraph, 'appearance_graph', n_scales)
        self.shape_graph = checks.check_graph(shape_graph, UndirectedGraph,
                                              'shape_graph', n_scales)
        self.deformation_graph = checks.check_graph(
            deformation_graph, [DirectedGraph, Tree], 'deformation_graph',
            n_scales)

        # Initialize models' lists
        self.shape_models = []
        self.appearance_models = []
        self.deformation_models = []

        # Train APS
        self._train(images, increment=False, group=group, batch_size=batch_size,
                    verbose=verbose)

    def _train(self, images, increment=False, group=None, batch_size=None,
               verbose=False):
        # If batch_size is not None, then we may have a generator, else we
        # assume we have a list.
        if batch_size is not None:
            # Create a generator of fixed sized batches. Will still work even
            # on an infinite list.
            image_batches = batch(images, batch_size)
        else:
            image_batches = [list(images)]

        for k, image_batch in enumerate(image_batches):
            if k == 0:
                if self.reference_shape is None:
                    # If no reference shape was given, use the mean of the first
                    # batch
                    if batch_size is not None:
                        warnings.warn('No reference shape was provided. The '
                                      'mean of the first batch will be the '
                                      'reference shape. If the batch mean is '
                                      'not representative of the true mean, '
                                      'this may cause issues.',
                                      MenpoFitBuilderWarning)
                    self.reference_shape = compute_reference_shape(
                        [i.landmarks[group].lms for i in image_batch],
                        self.diagonal, verbose=verbose)

            # After the first batch, we are incrementing the model
            if k > 0:
                increment = True

            if verbose:
                print('Computing batch {}'.format(k))

            # Train each batch
            self._train_batch(
                image_batch, increment=increment, group=group, verbose=verbose)

    def _train_batch(self, image_batch, increment=False, group=None,
                     verbose=False):
        # Rescale to existing reference shape
        image_batch = rescale_images_to_reference_shape(
            image_batch, group, self.reference_shape, verbose=verbose)

        # If the deformation graph was not provided (None given), then compute
        # the MST
        if None in self.deformation_graph:
            graph_shapes = [i.landmarks[group].lms for i in image_batch]
            deformation_mst = _compute_minimum_spanning_tree(
                graph_shapes, root_vertex=0, prefix='- ', verbose=verbose)
            self.deformation_graph = [deformation_mst if g is None else g
                                      for g in self.deformation_graph]

        # Build models at each scale
        if verbose:
            print_dynamic('- Building models\n')

        feature_images = []
        # for each scale (low --> high)
        for j in range(self.n_scales):
            if verbose:
                if len(self.scales) > 1:
                    scale_prefix = '  - Scale {}: '.format(j)
                else:
                    scale_prefix = '  - '
            else:
                scale_prefix = None

            # Handle holistic features
            if j == 0 and self.holistic_features[j] == no_op:
                # Saves a lot of memory
                feature_images = image_batch
            elif (j == 0 or self.holistic_features[j] is not
                  self.holistic_features[j - 1]):
                # Compute features only if this is the first pass through
                # the loop or the features at this scale are different from
                # the features at the previous scale
                feature_images = compute_features(image_batch,
                                                  self.holistic_features[j],
                                                  prefix=scale_prefix,
                                                  verbose=verbose)
            # handle scales
            if self.scales[j] != 1:
                # Scale feature images only if scale is different than 1
                scaled_images = scale_images(feature_images, self.scales[j],
                                             prefix=scale_prefix,
                                             verbose=verbose)
            else:
                scaled_images = feature_images

            # Extract potentially rescaled shapes
            scale_shapes = [i.landmarks[group].lms for i in scaled_images]

            # Apply procrustes to align the shapes
            aligned_shapes = align_shapes(scale_shapes)

            # Build the shape model using the aligned shapes
            if verbose:
                print_dynamic('{}Building shape model'.format(scale_prefix))

            if not increment:
                self.shape_models.append(self._build_shape_model(
                    aligned_shapes, self.shape_graph[j],
                    self.max_shape_components[j], verbose=verbose))
            else:
                self.shape_models[j].increment(aligned_shapes, verbose=verbose)

            # Build the deformation model
            if verbose:
                print_dynamic('{}Building deformation model'.format(
                    scale_prefix))

            if self.use_procrustes:
                deformation_shapes = aligned_shapes
            else:
                deformation_shapes = scale_shapes

            if not increment:
                self.deformation_models.append(self._build_deformation_model(
                    deformation_shapes, self.deformation_graph[j],
                    verbose=verbose))
            else:
                self.deformation_models[j].increment(deformation_shapes,
                                                     verbose=verbose)

            # Obtain warped images
            warped_images = self._warp_images(scaled_images, scale_shapes,
                                              j, scale_prefix, verbose)

            # Build the appearance model
            if verbose:
                print_dynamic('{}Building appearance model'.format(
                    scale_prefix))

            if not increment:
                self.appearance_models.append(self._build_appearance_model(
                    warped_images, self.appearance_graph[j],
                    self.n_appearance_components[j], verbose=verbose))
            else:
                self._increment_appearance_model(
                    warped_images, self.appearance_graph[j],
                    self.appearance_models[j], verbose=verbose)

            if verbose:
                print_dynamic('{}Done\n'.format(scale_prefix))

    def increment(self, images, group=None, batch_size=None, verbose=False):
        r"""
        Method that incrementally updates the APS model with a new batch of
        training images.

        Parameters
        ----------
        images : `list` of `menpo.image.Image`
            The `list` of training images.
        group : `str` or ``None``, optional
            The landmark group that will be used to train the APS. If ``None``
            and the images only have a single landmark group, then that is the
            one that will be used. Note that all the training images need to
            have the specified landmark group.
        batch_size : `int` or ``None``, optional
            If an `int` is provided, then the training is performed in an
            incremental fashion on image batches of size equal to the provided
            value. If ``None``, then the training is performed directly on the
            all the images.
        verbose : `bool`, optional
            If ``True``, then the progress of building the APS will be printed.
        """
        return self._train(images, increment=True, group=group,
                           verbose=verbose, batch_size=batch_size)

    def _build_shape_model(self, shapes, shape_graph, max_shape_components,
                           verbose=False):
        # if the provided graph is None, then apply PCA, else use the GMRF
        if shape_graph is not None:
            pca_model = GMRFModel(
                shapes, shape_graph, mode='concatenation', n_components=None,
                dtype=np.float64, sparse=False, incremental=self.is_incremental,
                verbose=verbose).principal_components_analysis()
            return OrthoPDM(pca_model, max_n_components=max_shape_components)
        else:
            return OrthoPDM(shapes, max_n_components=max_shape_components)

    def _build_deformation_model(self, shapes, deformation_graph,
                                 verbose=False):
        return GMRFModel(shapes, deformation_graph, mode='subtraction',
                         n_components=None, dtype=np.float64, sparse=False,
                         incremental=self.is_incremental, verbose=verbose)

    def _build_appearance_model(self, images, appearance_graph,
                                n_appearance_components, verbose=False):
        if appearance_graph is not None:
            return GMRFModel(images, appearance_graph, mode='concatenation',
                             n_components=n_appearance_components,
                             dtype=self.precision_dtype, sparse=True,
                             incremental=self.is_incremental, verbose=verbose)
        else:
            raise NotImplementedError('The full appearance model is not '
                                      'implemented yet.')

    def _increment_appearance_model(self, images, appearance_graph,
                                    appearance_model, verbose=False):
        if appearance_graph is not None:
            appearance_model.increment(images, verbose=verbose)
        else:
            raise NotImplementedError('The full appearance model is not '
                                      'implemented yet.')

    def _warp_images(self, images, shapes, scale_index, prefix, verbose):
        return extract_patches(
            images, shapes, self.patch_shape[scale_index],
            normalise_function=self.patch_normalisation[scale_index],
            prefix=prefix, verbose=verbose)

    @property
    def n_scales(self):
        """
        Returns the number of scales.

        :type: `int`
        """
        return len(self.scales)

    @property
    def _str_title(self):
        r"""
        Returns a string containing name of the model.

        :type: `str`
        """
        return 'Generative Active Pictorial Structures'

    def instance(self, shape_weights=None, scale_index=-1, as_graph=False):
        r"""
        Generates an instance of the shape model.

        Parameters
        ----------
        shape_weights : ``(n_weights,)`` `ndarray` or `list` or ``None``, optional
            The weights of the shape model that will be used to create a novel
            shape instance. If ``None``, the weights are assumed to be zero,
            thus the mean shape is used.
        scale_index : `int`, optional
            The scale to be used.
        as_graph : `bool`, optional
            If ``True``, then the instance will be returned as a
            `menpo.shape.PointTree` or a `menpo.shape.PointDirectedGraph`,
            depending on the type of the deformation graph.
        """
        if shape_weights is None:
            shape_weights = [0]
        sm = self.shape_models[scale_index].model
        shape_instance = sm.instance(shape_weights, normalized_weights=True)
        if as_graph:
            if isinstance(self.deformation_graph[scale_index], Tree):
                shape_instance = PointTree(
                    shape_instance.points,
                    self.deformation_graph[scale_index].adjacency_matrix,
                    self.deformation_graph[scale_index].root_vertex)
            else:
                shape_instance = PointDirectedGraph(
                    shape_instance.points,
                    self.deformation_graph[scale_index].adjacency_matrix)
        return shape_instance

    def random_instance(self, scale_index=-1, as_graph=False):
        r"""
        Generates a random instance of the APS.

        Parameters
        ----------
        scale_index : `int`, optional
            The scale to be used.
        as_graph : `bool`, optional
            If ``True``, then the instance will be returned as a
            `menpo.shape.PointTree` or a `menpo.shape.PointDirectedGraph`,
            depending on the type of the deformation graph.
        """
        shape_weights = np.random.randn(
            self.shape_models[scale_index].n_active_components)
        return self.instance(shape_weights, scale_index=scale_index,
                             as_graph=as_graph)

    def view_shape_models_widget(self, n_parameters=5,
                                 parameters_bounds=(-3.0, 3.0),
                                 mode='multiple', figure_size=(10, 8)):
        r"""
        Visualizes the shape models of the APS object using an interactive
        widget.

        Parameters
        ----------
        n_parameters : `int` or `list` of `int` or ``None``, optional
            The number of shape principal components to be used for the
            parameters sliders. If `int`, then the number of sliders per
            scale is the minimum between `n_parameters` and the number of
            active components per scale. If `list` of `int`, then a number of
            sliders is defined per scale. If ``None``, all the active
            components per scale will have a slider.
        parameters_bounds : ``(float, float)``, optional
            The minimum and maximum bounds, in std units, for the sliders.
        mode : {``single``, ``multiple``}, optional
            If ``'single'``, only a single slider is constructed along with a
            drop down menu. If ``'multiple'``, a slider is constructed for
            each parameter.
        figure_size : (`int`, `int`), optional
            The size of the rendered figure.
        """
        try:
            from menpowidgets import visualize_shape_model
            visualize_shape_model(
                [sm.model for sm in self.shape_models],
                n_parameters=n_parameters, parameters_bounds=parameters_bounds,
                figure_size=figure_size, mode=mode)
        except ImportError:
            from menpo.visualize.base import MenpowidgetsMissingError
            raise MenpowidgetsMissingError()

    def view_shape_graph_widget(self, scale_index=-1, figure_size=(10, 8)):
        r"""
        Visualize the shape graph using an interactive widget.

        Parameters
        ----------
        scale_index : `int`, optional
            The scale to be used.
        figure_size : (`int`, `int`), optional
            The size of the rendered figure.

        Raises
        ------
        ValueError
            Scale level {scale_index} uses a PCA shape model, so there is no
            graph
        """
        if self.shape_graph[scale_index] is not None:
            PointUndirectedGraph(
                self.shape_models[scale_index].model.mean().points,
                self.shape_graph[scale_index].adjacency_matrix).view_widget(
                figure_size=figure_size)
        else:
            raise ValueError("Scale level {} uses a PCA shape model, so there "
                             "is no graph".format(scale_index))

    def view_deformation_graph_widget(self, scale_index=-1,
                                      figure_size=(10, 8)):
        r"""
        Visualize the deformation graph using an interactive widget.

        Parameters
        ----------
        scale_index : `int`, optional
            The scale to be used.
        figure_size : (`int`, `int`), optional
            The size of the rendered figure.
        """
        if isinstance(self.deformation_graph[scale_index], Tree):
            dg = PointTree(self.shape_models[scale_index].model.mean().points,
                           self.deformation_graph[scale_index].adjacency_matrix,
                           self.deformation_graph[scale_index].root_vertex)
        else:
            dg = PointDirectedGraph(
                self.shape_models[scale_index].model.mean().points,
                self.deformation_graph[scale_index].adjacency_matrix)
        dg.view_widget(figure_size=figure_size)

    def view_appearance_graph_widget(self, scale_index=-1, figure_size=(10, 8)):
        r"""
        Visualize the appearance graph using an interactive widget.

        Parameters
        ----------
        scale_index : `int`, optional
            The scale to be used.
        figure_size : (`int`, `int`), optional
            The size of the rendered figure.

        Raises
        ------
        ValueError
            Scale level {scale_index} uses a PCA appearance model, so there
            is no graph
        """
        if self.appearance_graph[scale_index] is not None:
            PointUndirectedGraph(
                self.shape_models[scale_index].model.mean().points,
                self.appearance_graph[scale_index].adjacency_matrix).\
                view_widget(figure_size=figure_size)
        else:
            raise ValueError("Scale level {} uses a PCA appearance model, "
                             "so there is no graph".format(scale_index))

    def view_deformation_model(self, scale_index=-1, n_std=2,
                               render_colour_bar=False, colour_map='jet',
                               image_view=True, figure_id=None,
                               new_figure=False, render_graph_lines=True,
                               graph_line_colour='b', graph_line_style='-',
                               graph_line_width=1., ellipse_line_colour='r',
                               ellipse_line_style='-', ellipse_line_width=1.,
                               render_markers=True, marker_style='o',
                               marker_size=5, marker_face_colour='k',
                               marker_edge_colour='k', marker_edge_width=1.,
                               render_axes=False,
                               axes_font_name='sans-serif', axes_font_size=10,
                               axes_font_style='normal',
                               axes_font_weight='normal', crop_proportion=0.1,
                               figure_size=(10, 8)):
        r"""
        Visualize the deformation model by plotting a Gaussian ellipsis per
        graph edge.

        Parameters
        ----------
        scale_index : `int`, optional
            The scale to be used.
        n_std : `float`, optional
            This defines the size of the ellipses in terms of number of standard
            deviations.
        render_colour_bar : `bool`, optional
            If ``True``, then the ellipses will be coloured based on their
            normalized standard deviations and a colour bar will also appear on
            the side. If ``False``, then all the ellipses will have the same
            colour.
        colour_map : `str`, optional
            A valid Matplotlib colour map. For more info, please refer to
            `matplotlib.cm`.
        image_view : `bool`, optional
            If ``True`` the ellipses will be rendered in the image coordinates
            system.
        figure_id : `object`, optional
            The id of the figure to be used.
        new_figure : `bool`, optional
            If ``True``, a new figure is created.
        render_graph_lines : `bool`, optional
            Defines whether to plot the graph's edges.
        graph_line_colour : See Below, optional
            The colour of the lines of the graph's edges.
            Example options::

                {r, g, b, c, m, k, w}
                or
                (3, ) ndarray

        graph_line_style : ``{-, --, -., :}``, optional
            The style of the lines of the graph's edges.
        graph_line_width : `float`, optional
            The width of the lines of the graph's edges.
        ellipse_line_colour : See Below, optional
            The colour of the lines of the ellipses.
            Example options::

                {r, g, b, c, m, k, w}
                or
                (3, ) ndarray

        ellipse_line_style : ``{-, --, -., :}``, optional
            The style of the lines of the ellipses.
        ellipse_line_width : `float`, optional
            The width of the lines of the ellipses.
        render_markers : `bool`, optional
            If ``True``, the centers of the ellipses will be rendered.
        marker_style : See Below, optional
            The style of the centers of the ellipses. Example options ::

                {., ,, o, v, ^, <, >, +, x, D, d, s, p, *, h, H, 1, 2, 3, 4, 8}

        marker_size : `int`, optional
            The size of the centers of the ellipses in points.
        marker_face_colour : See Below, optional
            The face (filling) colour of the centers of the ellipses.
            Example options ::

                {r, g, b, c, m, k, w}
                or
                (3, ) ndarray

        marker_edge_colour : See Below, optional
            The edge colour of the centers of the ellipses.
            Example options ::

                {r, g, b, c, m, k, w}
                or
                (3, ) ndarray

        marker_edge_width : `float`, optional
            The edge width of the centers of the ellipses.
        render_axes : `bool`, optional
            If ``True``, the axes will be rendered.
        axes_font_name : See Below, optional
            The font of the axes. Example options ::

                {serif, sans-serif, cursive, fantasy, monospace}

        axes_font_size : `int`, optional
            The font size of the axes.
        axes_font_style : ``{normal, italic, oblique}``, optional
            The font style of the axes.
        axes_font_weight : See Below, optional
            The font weight of the axes.
            Example options ::

                {ultralight, light, normal, regular, book, medium, roman,
                 semibold,demibold, demi, bold, heavy, extra bold, black}

        crop_proportion : `float`, optional
            The proportion to be left around the centers' pointcloud.
        figure_size : (`float`, `float`) `tuple` or ``None`` optional
            The size of the figure in inches.
        """
        from menpo.visualize import plot_gaussian_ellipses

        mean_shape = self.shape_models[scale_index].model.mean().points
        deformation_graph = self.deformation_graph[scale_index]

        # get covariance matrices
        covariances = []
        means = []
        for e in range(deformation_graph.n_edges):
            # find vertices
            parent = deformation_graph.edges[e, 0]
            child = deformation_graph.edges[e, 1]

            # relative location mean
            means.append(mean_shape[child, :])

            # relative location cov
            s1 = -self.deformation_models[scale_index].precision[2 * child,
                                                                 2 * parent]
            s2 = -self.deformation_models[scale_index].precision[2 * child + 1,
                                                                 2 * parent + 1]
            s3 = -self.deformation_models[scale_index].precision[2 * child,
                                                                 2 * parent + 1]
            covariances.append(np.linalg.inv(np.array([[s1, s3], [s3, s2]])))

        # plot deformation graph
        if isinstance(deformation_graph, Tree):
            renderer = PointTree(
                mean_shape,
                deformation_graph.adjacency_matrix,
                deformation_graph.root_vertex).view(
                figure_id=figure_id, new_figure=new_figure,
                image_view=image_view, render_lines=render_graph_lines,
                line_colour=graph_line_colour, line_style=graph_line_style,
                line_width=graph_line_width, render_markers=render_markers,
                marker_style=marker_style, marker_size=marker_size,
                marker_face_colour=marker_face_colour,
                marker_edge_colour=marker_edge_colour,
                marker_edge_width=marker_edge_width, render_axes=render_axes,
                axes_font_name=axes_font_name, axes_font_size=axes_font_size,
                axes_font_style=axes_font_style,
                axes_font_weight=axes_font_weight, figure_size=figure_size)
        else:
            renderer = PointDirectedGraph(
                mean_shape,
                deformation_graph.adjacency_matrix).view(
                figure_id=figure_id, new_figure=new_figure,
                image_view=image_view, render_lines=render_graph_lines,
                line_colour=graph_line_colour, line_style=graph_line_style,
                line_width=graph_line_width, render_markers=render_markers,
                marker_style=marker_style, marker_size=marker_size,
                marker_face_colour=marker_face_colour,
                marker_edge_colour=marker_edge_colour,
                marker_edge_width=marker_edge_width, render_axes=render_axes,
                axes_font_name=axes_font_name, axes_font_size=axes_font_size,
                axes_font_style=axes_font_style,
                axes_font_weight=axes_font_weight, figure_size=figure_size)

        # plot ellipses
        renderer = plot_gaussian_ellipses(
            covariances, means, n_std=n_std,
            render_colour_bar=render_colour_bar,
            colour_bar_label='Normalized Standard Deviation',
            colour_map=colour_map, figure_id=renderer.figure_id,
            new_figure=False, image_view=image_view,
            line_colour=ellipse_line_colour, line_style=ellipse_line_style,
            line_width=ellipse_line_width, render_markers=render_markers,
            marker_edge_colour=marker_edge_colour,
            marker_face_colour=marker_face_colour,
            marker_edge_width=marker_edge_width, marker_size=marker_size,
            marker_style=marker_style, render_axes=render_axes,
            axes_font_name=axes_font_name, axes_font_size=axes_font_size,
            axes_font_style=axes_font_style, axes_font_weight=axes_font_weight,
            crop_proportion=crop_proportion, figure_size=figure_size)

        return renderer

    def __str__(self):
        return _aps_str(self)


def _compute_minimum_spanning_tree(shapes, root_vertex=0, prefix='',
                                   verbose=False):
    # initialize weights matrix
    n_vertices = shapes[0].n_points
    weights = np.zeros((n_vertices, n_vertices))

    # print progress if requested
    range1 = range(n_vertices-1)
    if verbose:
        range1 = print_progress(
            range1, end_with_newline=False,
            prefix='{}Deformation graph - Computing complete graph`s '
                   'weights'.format(prefix))

    # compute weights
    for i in range1:
        for j in range(i+1, n_vertices, 1):
            # create data matrix of edge
            diffs_x = [s.points[i, 0] - s.points[j, 0] for s in shapes]
            diffs_y = [s.points[i, 1] - s.points[j, 1] for s in shapes]
            coords = np.array([diffs_x, diffs_y])

            # compute mean and covariance
            m = np.mean(coords, axis=1)
            c = np.cov(coords)

            # get weight
            for im in range(len(shapes)):
                weights[i, j] += -np.log(multivariate_normal.pdf(coords[:, im],
                                                                 mean=m, cov=c))
            weights[j, i] = weights[i, j]

    # create undirected graph
    complete_graph = UndirectedGraph(weights)

    if verbose:
        print_dynamic('{}Deformation graph - Minimum spanning graph '
                      'computed.\n'.format(prefix))

    # compute minimum spanning graph
    return complete_graph.minimum_spanning_tree(root_vertex)


def _aps_str(aps):
    if aps.diagonal is not None:
        diagonal = aps.diagonal
    else:
        y, x = aps.reference_shape.range()
        diagonal = np.sqrt(x ** 2 + y ** 2)

    # Compute scale info strings
    scales_info = []
    lvl_str_tmplt = r"""   - Scale {}
     - Holistic feature: {}
     - Patch shape: {}
     - Appearance model class: {}
       - {}
       - {} features per point ({} in total)
       - {}
     - Shape model class: {}
       - {}
       - {} shape components
       - {} similarity transform parameters
     - Deformation model class: {}
       - {}"""
    for k, s in enumerate(aps.scales):
        comp_str = "No SVD used"
        if aps.appearance_models[k].n_components is not None:
            comp_str = "{} SVD components".format(aps.appearance_models[k].n_components)
        shape_model_str = "Trained using PCA"
        if aps.shape_graph[k] is not None:
            shape_model_str = "Trained using GMRF: {}".format(aps.shape_graph[k].__str__())
        scales_info.append(lvl_str_tmplt.format(
            s, name_of_callable(aps.holistic_features[k]),
            aps.patch_shape[k],
            name_of_callable(aps.appearance_models[k]),
            aps.appearance_models[k].graph.__str__(),
            aps.appearance_models[k].n_features_per_vertex,
            aps.appearance_models[k].n_features,
            comp_str,
            name_of_callable(aps.shape_models[k]),
            shape_model_str,
            aps.shape_models[k].model.n_components,
            aps.shape_models[k].n_global_parameters,
            name_of_callable(aps.deformation_models[k]),
            aps.deformation_models[k].graph.__str__()))
    scales_info = '\n'.join(scales_info)

    cls_str = r"""{class_title}
 - Images scaled to diagonal: {diagonal:.2f}
 - Scales: {scales}
{scales_info}
""".format(class_title=aps._str_title,
           diagonal=diagonal,
           scales=aps.scales,
           scales_info=scales_info)
    return cls_str
