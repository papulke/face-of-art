import numpy as np
from collections import Iterable

from menpo.image import Image

from menpofit.visualize import view_image_multiple_landmarks
from menpofit.error import euclidean_bb_normalised_error
from menpofit.error import euclidean_distance_indexed_normalised_error


def _rescale_shapes_to_reference(shapes, affine_transform, scale_transform):
    rescaled_shapes = []
    for shape in shapes:
        shape = scale_transform.apply(shape)
        rescaled_shapes.append(affine_transform.apply(shape))
    return rescaled_shapes


def _parse_iters(iters, n_shapes):
    if not (iters is None or isinstance(iters, int) or
                isinstance(iters, list)):
        raise ValueError('iters must be either int or list or None')
    if iters is None:
        iters = list(range(n_shapes))
    if isinstance(iters, int):
        iters = [iters]
    return iters


def _get_scale_of_iter(iter_i, reconstruction_indices):
    ids = np.array(reconstruction_indices)
    return np.nonzero(iter_i >= ids)[0][-1]


class Result(object):
    r"""
    Class for defining a basic fitting result. It holds the final shape of a
    fitting process and, optionally, the initial shape, ground truth shape
    and the image object.

    Parameters
    ----------
    final_shape : `menpo.shape.PointCloud`
        The final shape of the fitting process.
    image : `menpo.image.Image` or `subclass` or ``None``, optional
        The image on which the fitting process was applied. Note that a copy
        of the image will be assigned as an attribute. If ``None``, then no
        image is assigned.
    initial_shape : `menpo.shape.PointCloud` or ``None``, optional
        The initial shape that was provided to the fitting method to
        initialise the fitting process. If ``None``, then no initial shape is
        assigned.
    gt_shape : `menpo.shape.PointCloud` or ``None``, optional
        The ground truth shape associated with the image. If ``None``, then no
        ground truth shape is assigned.
    """
    def __init__(self, final_shape, image=None, initial_shape=None,
                 gt_shape=None):
        self._final_shape = final_shape
        self._initial_shape = initial_shape
        self._gt_shape = gt_shape
        # If image is provided, create a copy
        self._image = None
        if image is not None:
            self._image = Image(image.pixels)

    @property
    def is_iterative(self):
        r"""
        Flag whether the object is an iterative fitting result.

        :type: `bool`
        """
        return False

    @property
    def final_shape(self):
        r"""
        Returns the final shape of the fitting process.

        :type: `menpo.shape.PointCloud`
        """
        return self._final_shape

    @property
    def initial_shape(self):
        r"""
        Returns the initial shape that was provided to the fitting method to
        initialise the fitting process. In case the initial shape does not
        exist, then ``None`` is returned.

        :type: `menpo.shape.PointCloud` or ``None``
        """
        # return self.shapes[0]
        return self._initial_shape

    @property
    def gt_shape(self):
        r"""
        Returns the ground truth shape associated with the image. In case there
        is not an attached ground truth shape, then ``None`` is returned.

        :type: `menpo.shape.PointCloud` or ``None``
        """
        return self._gt_shape

    @property
    def image(self):
        r"""
        Returns the image that the fitting was applied on, if it was provided.
        Otherwise, it returns ``None``.

        :type: `menpo.shape.Image` or `subclass` or ``None``
        """
        return self._image

    def final_error(self, compute_error=None):
        r"""
        Returns the final error of the fitting process, if the ground truth
        shape exists. This is the error computed based on the `final_shape`.

        Parameters
        ----------
        compute_error: `callable`, optional
            Callable that computes the error between the fitted and
            ground truth shapes.

        Returns
        -------
        final_error : `float`
            The final error at the end of the fitting process.

        Raises
        ------
        ValueError
            Ground truth shape has not been set, so the final error cannot be
            computed
        """
        if compute_error is None:
            compute_error = euclidean_distance_indexed_normalised_error  # euclidean_bb_normalised_error
        if self.gt_shape is not None:
            return compute_error(self.final_shape, self.gt_shape, 36, 45)
        else:
            raise ValueError('Ground truth shape has not been set, so the '
                             'final error cannot be computed')

    def initial_error(self, compute_error=None):
        r"""
        Returns the initial error of the fitting process, if the ground truth
        shape and initial shape exist. This is the error computed based on the
        `initial_shape`.

        Parameters
        ----------
        compute_error: `callable`, optional
            Callable that computes the error between the initial and
            ground truth shapes.

        Returns
        -------
        initial_error : `float`
            The initial error at the beginning of the fitting process.

        Raises
        ------
        ValueError
            Initial shape has not been set, so the initial error cannot be
            computed
        ValueError
            Ground truth shape has not been set, so the initial error cannot be
            computed
        """
        if compute_error is None:
            compute_error = euclidean_distance_indexed_normalised_error  # euclidean_bb_normalised_error
        if self.initial_shape is None:
            raise ValueError('Initial shape has not been set, so the initial '
                             'error cannot be computed')
        elif self.gt_shape is None:
            raise ValueError('Ground truth shape has not been set, so the '
                             'initial error cannot be computed')
        else:
            return compute_error(self.initial_shape, self.gt_shape, 36, 45)

    def view(self, figure_id=None, new_figure=False, render_image=True,
             render_final_shape=True, render_initial_shape=False,
             render_gt_shape=False, subplots_enabled=True, channels=None,
             interpolation='bilinear', cmap_name=None, alpha=1., masked=True,
             final_marker_face_colour='r', final_marker_edge_colour='k',
             final_line_colour='r', initial_marker_face_colour='b',
             initial_marker_edge_colour='k', initial_line_colour='b',
             gt_marker_face_colour='y', gt_marker_edge_colour='k',
             gt_line_colour='y', render_lines=True, line_style='-',
             line_width=2, render_markers=True, marker_style='o', marker_size=4,
             marker_edge_width=1., render_numbering=False,
             numbers_horizontal_align='center',
             numbers_vertical_align='bottom', numbers_font_name='sans-serif',
             numbers_font_size=10, numbers_font_style='normal',
             numbers_font_weight='normal', numbers_font_colour='k',
             render_legend=True, legend_title='', legend_font_name='sans-serif',
             legend_font_style='normal', legend_font_size=10,
             legend_font_weight='normal', legend_marker_scale=None,
             legend_location=2, legend_bbox_to_anchor=(1.05, 1.),
             legend_border_axes_pad=None, legend_n_columns=1,
             legend_horizontal_spacing=None, legend_vertical_spacing=None,
             legend_border=True, legend_border_padding=None,
             legend_shadow=False, legend_rounded_corners=False,
             render_axes=False, axes_font_name='sans-serif', axes_font_size=10,
             axes_font_style='normal', axes_font_weight='normal',
             axes_x_limits=None, axes_y_limits=None, axes_x_ticks=None,
             axes_y_ticks=None, figure_size=(10, 8)):
        """
        Visualize the fitting result. The method renders the final fitted
        shape and optionally the initial shape, ground truth shape and the
        image, id they were provided.

        Parameters
        ----------
        figure_id : `object`, optional
            The id of the figure to be used.
        new_figure : `bool`, optional
            If ``True``, a new figure is created.
        render_image : `bool`, optional
            If ``True`` and the image exists, then it gets rendered.
        render_final_shape : `bool`, optional
            If ``True``, then the final fitting shape gets rendered.
        render_initial_shape : `bool`, optional
            If ``True`` and the initial fitting shape exists, then it gets
            rendered.
        render_gt_shape : `bool`, optional
            If ``True`` and the ground truth shape exists, then it gets
            rendered.
        subplots_enabled : `bool`, optional
            If ``True``, then the requested final, initial and ground truth
            shapes get rendered on separate subplots.
        channels : `int` or `list` of `int` or ``all`` or ``None``
            If `int` or `list` of `int`, the specified channel(s) will be
            rendered. If ``all``, all the channels will be rendered in subplots.
            If ``None`` and the image is RGB, it will be rendered in RGB mode.
            If ``None`` and the image is not RGB, it is equivalent to ``all``.
        interpolation : See Below, optional
            The interpolation used to render the image. For example, if
            ``bilinear``, the image will be smooth and if ``nearest``, the
            image will be pixelated. Example options ::

                {none, nearest, bilinear, bicubic, spline16, spline36, hanning,
                 hamming, hermite, kaiser, quadric, catrom, gaussian, bessel,
                 mitchell, sinc, lanczos}

        cmap_name: `str`, optional,
            If ``None``, single channel and three channel images default
            to greyscale and rgb colormaps respectively.
        alpha : `float`, optional
            The alpha blending value, between 0 (transparent) and 1 (opaque).
        masked : `bool`, optional
            If ``True``, then the image is rendered as masked.
        final_marker_face_colour : See Below, optional
            The face (filling) colour of the markers of the final fitting shape.
            Example options ::

                {r, g, b, c, m, k, w}
                or
                (3, ) ndarray

        final_marker_edge_colour : See Below, optional
            The edge colour of the markers of the final fitting shape.
            Example options ::

                {r, g, b, c, m, k, w}
                or
                (3, ) ndarray

        final_line_colour : See Below, optional
            The line colour of the final fitting shape.
            Example options ::

                {r, g, b, c, m, k, w}
                or
                (3, ) ndarray

        initial_marker_face_colour : See Below, optional
            The face (filling) colour of the markers of the initial shape.
            Example options ::

                {r, g, b, c, m, k, w}
                or
                (3, ) ndarray

        initial_marker_edge_colour : See Below, optional
            The edge colour of the markers of the initial shape.
            Example options ::

                {r, g, b, c, m, k, w}
                or
                (3, ) ndarray

        initial_line_colour : See Below, optional
            The line colour of the initial shape.
            Example options ::

                {r, g, b, c, m, k, w}
                or
                (3, ) ndarray

        gt_marker_face_colour : See Below, optional
            The face (filling) colour of the markers of the ground truth shape.
            Example options ::

                {r, g, b, c, m, k, w}
                or
                (3, ) ndarray

        gt_marker_edge_colour : See Below, optional
            The edge colour of the markers of the ground truth shape.
            Example options ::

                {r, g, b, c, m, k, w}
                or
                (3, ) ndarray

        gt_line_colour : See Below, optional
            The line colour of the ground truth shape.
            Example options ::

                {r, g, b, c, m, k, w}
                or
                (3, ) ndarray

        render_lines : `bool` or `list` of `bool`, optional
            If ``True``, the lines will be rendered. You can either provide a
            single value that will be used for all shapes or a list with a
            different value per shape in (`final`, `initial`, `groundtruth`)
            order.
        line_style : `str` or `list` of `str`, optional
            The style of the lines. You can either provide a single value that
            will be used for all shapes or a list with a different value per
            shape in (`final`, `initial`, `groundtruth`) order.
            Example options::

                {'-', '--', '-.', ':'}

        line_width : `float` or `list` of `float`, optional
            The width of the lines. You can either provide a single value that
            will be used for all shapes or a list with a different value per
            shape in (`final`, `initial`, `groundtruth`) order.
        render_markers : `bool` or `list` of `bool`, optional
            If ``True``, the markers will be rendered. You can either provide a
            single value that will be used for all shapes or a list with a
            different value per shape in (`final`, `initial`, `groundtruth`)
            order.
        marker_style : `str` or `list` of `str`, optional
            The style of the markers. You can either provide a single value that
            will be used for all shapes or a list with a different value per
            shape in (`final`, `initial`, `groundtruth`) order.
            Example options::

                {., ,, o, v, ^, <, >, +, x, D, d, s, p, *, h, H, 1, 2, 3, 4, 8}

        marker_size : `int` or `list` of `int`, optional
            The size of the markers in points. You can either provide a single
            value that will be used for all shapes or a list with a different
            value per shape in (`final`, `initial`, `groundtruth`) order.
        marker_edge_width : `float` or `list` of `float`, optional
            The width of the markers' edge. You can either provide a single
            value that will be used for all shapes or a list with a different
            value per shape in (`final`, `initial`, `groundtruth`) order.
        render_numbering : `bool`, optional
            If ``True``, the landmarks will be numbered.
        numbers_horizontal_align : ``{center, right, left}``, optional
            The horizontal alignment of the numbers' texts.
        numbers_vertical_align : ``{center, top, bottom, baseline}``, optional
            The vertical alignment of the numbers' texts.
        numbers_font_name : See Below, optional
            The font of the numbers. Example options ::

                {serif, sans-serif, cursive, fantasy, monospace}

        numbers_font_size : `int`, optional
            The font size of the numbers.
        numbers_font_style : ``{normal, italic, oblique}``, optional
            The font style of the numbers.
        numbers_font_weight : See Below, optional
            The font weight of the numbers.
            Example options ::

                {ultralight, light, normal, regular, book, medium, roman,
                 semibold, demibold, demi, bold, heavy, extra bold, black}

        numbers_font_colour : See Below, optional
            The font colour of the numbers.
            Example options ::

                {r, g, b, c, m, k, w}
                or
                (3, ) ndarray

        render_legend : `bool`, optional
            If ``True``, the legend will be rendered.
        legend_title : `str`, optional
            The title of the legend.
        legend_font_name : See below, optional
            The font of the legend. Example options ::

                {serif, sans-serif, cursive, fantasy, monospace}

        legend_font_style : ``{normal, italic, oblique}``, optional
            The font style of the legend.
        legend_font_size : `int`, optional
            The font size of the legend.
        legend_font_weight : See Below, optional
            The font weight of the legend.
            Example options ::

                {ultralight, light, normal, regular, book, medium, roman,
                 semibold, demibold, demi, bold, heavy, extra bold, black}

        legend_marker_scale : `float`, optional
            The relative size of the legend markers with respect to the original
        legend_location : `int`, optional
            The location of the legend. The predefined values are:

            =============== ==
            'best'          0
            'upper right'   1
            'upper left'    2
            'lower left'    3
            'lower right'   4
            'right'         5
            'center left'   6
            'center right'  7
            'lower center'  8
            'upper center'  9
            'center'        10
            =============== ==

        legend_bbox_to_anchor : (`float`, `float`) `tuple`, optional
            The bbox that the legend will be anchored.
        legend_border_axes_pad : `float`, optional
            The pad between the axes and legend border.
        legend_n_columns : `int`, optional
            The number of the legend's columns.
        legend_horizontal_spacing : `float`, optional
            The spacing between the columns.
        legend_vertical_spacing : `float`, optional
            The vertical space between the legend entries.
        legend_border : `bool`, optional
            If ``True``, a frame will be drawn around the legend.
        legend_border_padding : `float`, optional
            The fractional whitespace inside the legend border.
        legend_shadow : `bool`, optional
            If ``True``, a shadow will be drawn behind legend.
        legend_rounded_corners : `bool`, optional
            If ``True``, the frame's corners will be rounded (fancybox).
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
                 semibold, demibold, demi, bold, heavy, extra bold, black}

        axes_x_limits : `float` or (`float`, `float`) or ``None``, optional
            The limits of the x axis. If `float`, then it sets padding on the
            right and left of the Image as a percentage of the Image's width. If
            `tuple` or `list`, then it defines the axis limits. If ``None``, then
            the limits are set automatically.
        axes_y_limits : (`float`, `float`) `tuple` or ``None``, optional
            The limits of the y axis. If `float`, then it sets padding on the
            top and bottom of the Image as a percentage of the Image's height. If
            `tuple` or `list`, then it defines the axis limits. If ``None``, then
            the limits are set automatically.
        axes_x_ticks : `list` or `tuple` or ``None``, optional
            The ticks of the x axis.
        axes_y_ticks : `list` or `tuple` or ``None``, optional
            The ticks of the y axis.
        figure_size : (`float`, `float`) `tuple` or ``None`` optional
            The size of the figure in inches.

        Returns
        -------
        renderer : `class`
            The renderer object.
        """
        # Create image instance
        if self.image is None:
            image = Image(np.zeros((10, 10)))
            render_image = False
        else:
            image = Image(self.image.pixels)
        # Assign pointclouds to image
        groups = []
        face_colours = []
        edge_colours = []
        line_colours = []
        subplots_titles = {}
        if render_final_shape:
            image.landmarks['final'] = self.final_shape
            groups.append('final')
            face_colours.append(final_marker_face_colour)
            edge_colours.append(final_marker_edge_colour)
            line_colours.append(final_line_colour)
            subplots_titles['final'] = 'Final'
        if self.initial_shape is not None and render_initial_shape:
            image.landmarks['initial'] = self.initial_shape
            groups.append('initial')
            face_colours.append(initial_marker_face_colour)
            edge_colours.append(initial_marker_edge_colour)
            line_colours.append(initial_line_colour)
            subplots_titles['initial'] = 'Initial'
        if self.gt_shape is not None and render_gt_shape:
            image.landmarks['groundtruth'] = self.gt_shape
            groups.append('groundtruth')
            face_colours.append(gt_marker_face_colour)
            edge_colours.append(gt_marker_edge_colour)
            line_colours.append(gt_line_colour)
            subplots_titles['groundtruth'] = 'Groundtruth'
        # Render
        return view_image_multiple_landmarks(
                image, groups, with_labels=None, figure_id=figure_id,
                new_figure=new_figure, subplots_enabled=subplots_enabled,
                subplots_titles=subplots_titles, render_image=render_image,
                render_landmarks=True, masked=masked,
                channels=channels, interpolation=interpolation,
                cmap_name=cmap_name, alpha=alpha, image_view=True,
                render_lines=render_lines, line_style=line_style,
                line_width=line_width, line_colour=line_colours,
                render_markers=render_markers, marker_style=marker_style,
                marker_size=marker_size, marker_edge_width=marker_edge_width,
                marker_edge_colour=edge_colours,
                marker_face_colour=face_colours,
                render_numbering=render_numbering,
                numbers_horizontal_align=numbers_horizontal_align,
                numbers_vertical_align=numbers_vertical_align,
                numbers_font_name=numbers_font_name,
                numbers_font_size=numbers_font_size,
                numbers_font_style=numbers_font_style,
                numbers_font_weight=numbers_font_weight,
                numbers_font_colour=numbers_font_colour,
                render_legend=render_legend, legend_title=legend_title,
                legend_font_name=legend_font_name,
                legend_font_style=legend_font_style,
                legend_font_size=legend_font_size,
                legend_font_weight=legend_font_weight,
                legend_marker_scale=legend_marker_scale,
                legend_location=legend_location,
                legend_bbox_to_anchor=legend_bbox_to_anchor,
                legend_border_axes_pad=legend_border_axes_pad,
                legend_n_columns=legend_n_columns,
                legend_horizontal_spacing=legend_horizontal_spacing,
                legend_vertical_spacing=legend_vertical_spacing,
                legend_border=legend_border,
                legend_border_padding=legend_border_padding,
                legend_shadow=legend_shadow,
                legend_rounded_corners=legend_rounded_corners,
                render_axes=render_axes, axes_font_name=axes_font_name,
                axes_font_size=axes_font_size, axes_font_style=axes_font_style,
                axes_font_weight=axes_font_weight, axes_x_limits=axes_x_limits,
                axes_y_limits=axes_y_limits, axes_x_ticks=axes_x_ticks,
                axes_y_ticks=axes_y_ticks, figure_size=figure_size)

    def view_widget(self, browser_style='buttons', figure_size=(10, 8),
                    style='coloured'):
        r"""
        Visualizes the result object using an interactive widget.

        Parameters
        ----------
        browser_style : {``'buttons'``, ``'slider'``}, optional
            It defines whether the selector of the images will have the form of
            plus/minus buttons or a slider.
        figure_size : (`int`, `int`), optional
            The initial size of the rendered figure.
        style : {``'coloured'``, ``'minimal'``}, optional
            If ``'coloured'``, then the style of the widget will be coloured. If
            ``minimal``, then the style is simple using black and white colours.
        """
        try:
            from menpowidgets import visualize_fitting_result
            visualize_fitting_result(self, figure_size=figure_size, style=style,
                                     browser_style=browser_style)
        except ImportError:
            from menpo.visualize.base import MenpowidgetsMissingError
            raise MenpowidgetsMissingError()

    def __str__(self):
        out = "Fitting result of {} landmark points.".format(
                self.final_shape.n_points)
        if self.gt_shape is not None:
            if self.initial_shape is not None:
                out += "\nInitial error: {:.4f}".format(self.initial_error())
            out += "\nFinal error: {:.4f}".format(self.final_error())
        return out


class NonParametricIterativeResult(Result):
    r"""
    Class for defining a non-parametric iterative fitting result, i.e. the
    result of a method that does not optimize over a parametric shape model. It
    holds the shapes of all the iterations of the fitting procedure. It can
    optionally store the image on which the fitting was applied, as well as its
    ground truth shape.

    Parameters
    ----------
    shapes : `list` of `menpo.shape.PointCloud`
        The `list` of shapes per iteration. Note that the list does not
        include the initial shape. The last member of the list is the final
        shape.
    initial_shape : `menpo.shape.PointCloud` or ``None``, optional
        The initial shape from which the fitting process started. If ``None``,
        then no initial shape is assigned.
    image : `menpo.image.Image` or `subclass` or ``None``, optional
        The image on which the fitting process was applied. Note that a copy
        of the image will be assigned as an attribute. If ``None``, then no
        image is assigned.
    gt_shape : `menpo.shape.PointCloud` or ``None``, optional
        The ground truth shape associated with the image. If ``None``, then no
        ground truth shape is assigned.
    costs : `list` of `float` or ``None``, optional
        The `list` of cost per iteration. If ``None``, then it is assumed that
        the cost function cannot be computed for the specific algorithm. It must
        have the same length as `shapes`.
    """
    def __init__(self, shapes, initial_shape=None, image=None, gt_shape=None,
                 costs=None):
        super(NonParametricIterativeResult, self).__init__(
            final_shape=shapes[-1], image=image, initial_shape=initial_shape,
            gt_shape=gt_shape)
        self._n_iters = len(shapes)
        # If initial shape is provided, then add it in the beginning of shapes
        self._shapes = shapes
        if self.initial_shape is not None:
            self._shapes = [self.initial_shape] + self._shapes
        # Add costs as property
        self._costs = costs

    @property
    def is_iterative(self):
        r"""
        Flag whether the object is an iterative fitting result.

        :type: `bool`
        """
        return True

    @property
    def shapes(self):
        r"""
        Returns the `list` of shapes obtained at each iteration of the fitting
        process. The `list` includes the `initial_shape` (if it exists) and
        `final_shape`.

        :type: `list` of `menpo.shape.PointCloud`
        """
        return self._shapes

    @property
    def n_iters(self):
        r"""
        Returns the total number of iterations of the fitting process.

        :type: `int`
        """
        return self._n_iters

    def to_result(self, pass_image=True, pass_initial_shape=True,
                  pass_gt_shape=True):
        r"""
        Returns a :map:`Result` instance of the object, i.e. a fitting result
        object that does not store the iterations. This can be useful for
        reducing the size of saved fitting results.

        Parameters
        ----------
        pass_image : `bool`, optional
            If ``True``, then the image will get passed (if it exists).
        pass_initial_shape : `bool`, optional
            If ``True``, then the initial shape will get passed (if it exists).
        pass_gt_shape : `bool`, optional
            If ``True``, then the ground truth shape will get passed (if it
            exists).

        Returns
        -------
        result : :map:`Result`
            The final "lightweight" fitting result.
        """
        image = None
        if pass_image:
            image = self.image
        initial_shape = None
        if pass_initial_shape:
            initial_shape = self.initial_shape
        gt_shape = None
        if pass_gt_shape:
            gt_shape = self.gt_shape
        return Result(self.final_shape, image=image,
                      initial_shape=initial_shape, gt_shape=gt_shape)

    def errors(self, compute_error=None):
        r"""
        Returns a list containing the error at each fitting iteration, if the
        ground truth shape exists.

        Parameters
        ----------
        compute_error: `callable`, optional
            Callable that computes the error between the shape at each
            iteration and the ground truth shape.

        Returns
        -------
        errors : `list` of `float`
            The error at each iteration of the fitting process.

        Raises
        ------
        ValueError
            Ground truth shape has not been set, so the final error cannot be
            computed
        """
        if compute_error is None:
            compute_error = euclidean_bb_normalised_error
        if self.gt_shape is not None:
            return [compute_error(t, self.gt_shape)
                    for t in self.shapes]
        else:
            raise ValueError('Ground truth shape has not been set, so the '
                             'errors per iteration cannot be computed')

    def plot_errors(self, compute_error=None, figure_id=None,
                    new_figure=False, render_lines=True, line_colour='b',
                    line_style='-', line_width=2, render_markers=True,
                    marker_style='o', marker_size=4, marker_face_colour='b',
                    marker_edge_colour='k', marker_edge_width=1.,
                    render_axes=True, axes_font_name='sans-serif',
                    axes_font_size=10, axes_font_style='normal',
                    axes_font_weight='normal', axes_x_limits=0.,
                    axes_y_limits=None, axes_x_ticks=None,
                    axes_y_ticks=None, figure_size=(10, 6),
                    render_grid=True, grid_line_style='--',
                    grid_line_width=0.5):
        r"""
        Plot of the error evolution at each fitting iteration.

        Parameters
        ----------
        compute_error: `callable`, optional
            Callable that computes the error between the shape at each
            iteration and the ground truth shape.
        figure_id : `object`, optional
            The id of the figure to be used.
        new_figure : `bool`, optional
            If ``True``, a new figure is created.
        render_lines : `bool`, optional
            If ``True``, the line will be rendered.
        line_colour : `colour` or ``None`` (See below), optional
            The colour of the line. If ``None``, the colour is sampled from
            the jet colormap.
            Example `colour` options are ::

                {r, g, b, c, m, k, w}
                or
                (3, ) ndarray

        line_style : `str` (See below), optional
            The style of the lines. Example options::

                {-, --, -., :}

        line_width : `float`, optional
            The width of the lines.
        render_markers : `bool`, optional
            If ``True``, the markers will be rendered.
        marker_style : `str` (See below), optional
            The style of the markers.
            Example `marker` options ::

                {., ,, o, v, ^, <, >, +, x, D, d, s, p, *, h, H, 1, 2, 3, 4, 8}

        marker_size : `int`, optional
            The size of the markers in points.
        marker_face_colour : `colour` or ``None``, optional
            The face (filling) colour of the markers. If ``None``, the colour
            is sampled from the jet colormap.
            Example `colour` options are ::

                {r, g, b, c, m, k, w}
                or
                (3, ) ndarray

        marker_edge_colour : `colour` or ``None``, optional
            The edge colour of the markers. If ``None``, the colour
            is sampled from the jet colormap.
            Example `colour` options are ::

                {r, g, b, c, m, k, w}
                or
                (3, ) ndarray

        marker_edge_width : `float`, optional
            The width of the markers' edge.
        render_axes : `bool`, optional
            If ``True``, the axes will be rendered.
        axes_font_name : `str` (See below), optional
            The font of the axes.
            Example options ::

                {serif, sans-serif, cursive, fantasy, monospace}

        axes_font_size : `int`, optional
            The font size of the axes.
        axes_font_style : `str` (See below), optional
            The font style of the axes.
            Example options ::

                {normal, italic, oblique}

        axes_font_weight : `str` (See below), optional
            The font weight of the axes.
            Example options ::

                {ultralight, light, normal, regular, book, medium, roman,
                 semibold, demibold, demi, bold, heavy, extra bold, black}

        axes_x_limits : `float` or (`float`, `float`) or ``None``, optional
            The limits of the x axis. If `float`, then it sets padding on the
            right and left of the graph as a percentage of the curves' width. If
            `tuple` or `list`, then it defines the axis limits. If ``None``,
            then the limits are set automatically.
        axes_y_limits : `float` or (`float`, `float`) or ``None``, optional
            The limits of the y axis. If `float`, then it sets padding on the
            top and bottom of the graph as a percentage of the curves' height.
            If `tuple` or `list`, then it defines the axis limits. If ``None``,
            then the limits are set automatically.
        axes_x_ticks : `list` or `tuple` or ``None``, optional
            The ticks of the x axis.
        axes_y_ticks : `list` or `tuple` or ``None``, optional
            The ticks of the y axis.
        figure_size : (`float`, `float`) or ``None``, optional
            The size of the figure in inches.
        render_grid : `bool`, optional
            If ``True``, the grid will be rendered.
        grid_line_style : ``{'-', '--', '-.', ':'}``, optional
            The style of the grid lines.
        grid_line_width : `float`, optional
            The width of the grid lines.

        Returns
        -------
        renderer : `menpo.visualize.GraphPlotter`
            The renderer object.
        """
        from menpo.visualize import plot_curve
        errors = self.errors(compute_error=compute_error)
        return plot_curve(
                x_axis=list(range(len(errors))), y_axis=[errors], figure_id=figure_id,
                new_figure=new_figure, title='Fitting Errors per Iteration',
                x_label='Iteration', y_label='Fitting Error',
                axes_x_limits=axes_x_limits, axes_y_limits=axes_y_limits,
                axes_x_ticks=axes_x_ticks, axes_y_ticks=axes_y_ticks,
                render_lines=render_lines, line_colour=line_colour,
                line_style=line_style, line_width=line_width,
                render_markers=render_markers, marker_style=marker_style,
                marker_size=marker_size, marker_face_colour=marker_face_colour,
                marker_edge_colour=marker_edge_colour,
                marker_edge_width=marker_edge_width, render_legend=False,
                render_axes=render_axes, axes_font_name=axes_font_name,
                axes_font_size=axes_font_size, axes_font_style=axes_font_style,
                axes_font_weight=axes_font_weight, figure_size=figure_size,
                render_grid=render_grid,  grid_line_style=grid_line_style,
                grid_line_width=grid_line_width)

    def displacements(self):
        r"""
        A list containing the displacement between the shape of each iteration
        and the shape of the previous one.

        :type: `list` of `ndarray`
        """
        return [np.linalg.norm(s1.points - s2.points, axis=1)
                for s1, s2 in zip(self.shapes, self.shapes[1:])]

    def displacements_stats(self, stat_type='mean'):
        r"""
        A list containing a statistical metric on the displacements between
        the shape of each iteration and the shape of the previous one.

        Parameters
        ----------
        stat_type : ``{'mean', 'median', 'min', 'max'}``, optional
            Specifies a statistic metric to be extracted from the displacements.

        Returns
        -------
        displacements_stat : `list` of `float`
            The statistical metric on the points displacements for each
            iteration.

        Raises
        ------
        ValueError
            type must be 'mean', 'median', 'min' or 'max'
        """
        if stat_type == 'mean':
            return [np.mean(d) for d in self.displacements()]
        elif stat_type == 'median':
            return [np.median(d) for d in self.displacements()]
        elif stat_type == 'max':
            return [np.max(d) for d in self.displacements()]
        elif stat_type == 'min':
            return [np.min(d) for d in self.displacements()]
        else:
            raise ValueError("type must be 'mean', 'median', 'min' or 'max'")

    def plot_displacements(self, stat_type='mean', figure_id=None,
                           new_figure=False, render_lines=True, line_colour='b',
                           line_style='-', line_width=2, render_markers=True,
                           marker_style='o', marker_size=4,
                           marker_face_colour='b', marker_edge_colour='k',
                           marker_edge_width=1., render_axes=True,
                           axes_font_name='sans-serif', axes_font_size=10,
                           axes_font_style='normal', axes_font_weight='normal',
                           axes_x_limits=0., axes_y_limits=None,
                           axes_x_ticks=None, axes_y_ticks=None,
                           figure_size=(10, 6), render_grid=True,
                           grid_line_style='--', grid_line_width=0.5):
        r"""
        Plot of a statistical metric of the displacement between the shape of
        each iteration and the shape of the previous one.

        Parameters
        ----------
        stat_type : {``mean``, ``median``, ``min``, ``max``}, optional
            Specifies a statistic metric to be extracted from the displacements
            (see also `displacements_stats()` method).
        figure_id : `object`, optional
            The id of the figure to be used.
        new_figure : `bool`, optional
            If ``True``, a new figure is created.
        render_lines : `bool`, optional
            If ``True``, the line will be rendered.
        line_colour : `colour` or ``None`` (See below), optional
            The colour of the line. If ``None``, the colour is sampled from
            the jet colormap.
            Example `colour` options are ::

                {r, g, b, c, m, k, w}
                or
                (3, ) ndarray

        line_style : `str` (See below), optional
            The style of the lines. Example options::

                {-, --, -., :}

        line_width : `float`, optional
            The width of the lines.
        render_markers : `bool`, optional
            If ``True``, the markers will be rendered.
        marker_style : `str` (See below), optional
            The style of the markers.
            Example `marker` options ::

                {., ,, o, v, ^, <, >, +, x, D, d, s, p, *, h, H, 1, 2, 3, 4, 8}

        marker_size : `int`, optional
            The size of the markers in points.
        marker_face_colour : `colour` or ``None``, optional
            The face (filling) colour of the markers. If ``None``, the colour
            is sampled from the jet colormap.
            Example `colour` options are ::

                {r, g, b, c, m, k, w}
                or
                (3, ) ndarray

        marker_edge_colour : `colour` or ``None``, optional
            The edge colour of the markers. If ``None``, the colour
            is sampled from the jet colormap.
            Example `colour` options are ::

                {r, g, b, c, m, k, w}
                or
                (3, ) ndarray

        marker_edge_width : `float`, optional
            The width of the markers' edge.
        render_axes : `bool`, optional
            If ``True``, the axes will be rendered.
        axes_font_name : `str` (See below), optional
            The font of the axes.
            Example options ::

                {serif, sans-serif, cursive, fantasy, monospace}

        axes_font_size : `int`, optional
            The font size of the axes.
        axes_font_style : `str` (See below), optional
            The font style of the axes.
            Example options ::

                {normal, italic, oblique}

        axes_font_weight : `str` (See below), optional
            The font weight of the axes.
            Example options ::

                {ultralight, light, normal, regular, book, medium, roman,
                 semibold, demibold, demi, bold, heavy, extra bold, black}

        axes_x_limits : `float` or (`float`, `float`) or ``None``, optional
            The limits of the x axis. If `float`, then it sets padding on the
            right and left of the graph as a percentage of the curves' width. If
            `tuple` or `list`, then it defines the axis limits. If ``None``,
            then the limits are set automatically.
        axes_y_limits : `float` or (`float`, `float`) or ``None``, optional
            The limits of the y axis. If `float`, then it sets padding on the
            top and bottom of the graph as a percentage of the curves' height.
            If `tuple` or `list`, then it defines the axis limits. If ``None``,
            then the limits are set automatically.
        axes_x_ticks : `list` or `tuple` or ``None``, optional
            The ticks of the x axis.
        axes_y_ticks : `list` or `tuple` or ``None``, optional
            The ticks of the y axis.
        figure_size : (`float`, `float`) or ``None``, optional
            The size of the figure in inches.
        render_grid : `bool`, optional
            If ``True``, the grid will be rendered.
        grid_line_style : ``{'-', '--', '-.', ':'}``, optional
            The style of the grid lines.
        grid_line_width : `float`, optional
            The width of the grid lines.

        Returns
        -------
        renderer : `menpo.visualize.GraphPlotter`
            The renderer object.
        """
        from menpo.visualize import plot_curve
        # set labels
        if stat_type == 'max':
            name = 'Maximum'
        elif stat_type == 'min':
            name = 'Minimum'
        elif stat_type == 'mean':
            name = 'Mean'
        elif stat_type == 'median':
            name = 'Median'
        else:
            raise ValueError('stat_type must be one of {max, min, mean, '
                             'median}.')
        y_label = '{} Displacement'.format(name)
        title = '{} displacement per Iteration'.format(name)

        # plot
        displacements = self.displacements_stats(stat_type=stat_type)
        return plot_curve(
                x_axis=list(range(len(displacements))), y_axis=[displacements],
                figure_id=figure_id, new_figure=new_figure, title=title,
                x_label='Iteration', y_label=y_label,
                axes_x_limits=axes_x_limits, axes_y_limits=axes_y_limits,
                axes_x_ticks=axes_x_ticks, axes_y_ticks=axes_y_ticks,
                render_lines=render_lines, line_colour=line_colour,
                line_style=line_style, line_width=line_width,
                render_markers=render_markers, marker_style=marker_style,
                marker_size=marker_size, marker_face_colour=marker_face_colour,
                marker_edge_colour=marker_edge_colour,
                marker_edge_width=marker_edge_width, render_legend=False,
                render_axes=render_axes, axes_font_name=axes_font_name,
                axes_font_size=axes_font_size, axes_font_style=axes_font_style,
                axes_font_weight=axes_font_weight, figure_size=figure_size,
                render_grid=render_grid,  grid_line_style=grid_line_style,
                grid_line_width=grid_line_width)

    @property
    def costs(self):
        r"""
        Returns a `list` with the cost per iteration. It returns ``None`` if
        the costs are not computed.

        :type: `list` of `float` or ``None``
        """
        return self._costs

    def plot_costs(self, figure_id=None, new_figure=False, render_lines=True,
                   line_colour='b', line_style='-', line_width=2,
                   render_markers=True, marker_style='o', marker_size=4,
                   marker_face_colour='b', marker_edge_colour='k',
                   marker_edge_width=1., render_axes=True,
                   axes_font_name='sans-serif', axes_font_size=10,
                   axes_font_style='normal', axes_font_weight='normal',
                   axes_x_limits=0., axes_y_limits=None, axes_x_ticks=None,
                   axes_y_ticks=None, figure_size=(10, 6),
                   render_grid=True, grid_line_style='--',
                   grid_line_width=0.5):
        r"""
        Plot of the cost function evolution at each fitting iteration.

        Parameters
        ----------
        figure_id : `object`, optional
            The id of the figure to be used.
        new_figure : `bool`, optional
            If ``True``, a new figure is created.
        render_lines : `bool`, optional
            If ``True``, the line will be rendered.
        line_colour : `colour` or ``None``, optional
            The colour of the line. If ``None``, the colour is sampled from
            the jet colormap.
            Example `colour` options are ::

                    {'r', 'g', 'b', 'c', 'm', 'k', 'w'}
                    or
                    (3, ) ndarray

        line_style : ``{'-', '--', '-.', ':'}``, optional
            The style of the lines.
        line_width : `float`, optional
            The width of the lines.
        render_markers : `bool`, optional
            If ``True``, the markers will be rendered.
        marker_style : `marker`, optional
            The style of the markers.
            Example `marker` options ::

                    {'.', ',', 'o', 'v', '^', '<', '>', '+', 'x', 'D', 'd', 's',
                     'p', '*', 'h', 'H', '1', '2', '3', '4', '8'}

        marker_size : `int`, optional
            The size of the markers in points.
        marker_face_colour : `colour` or ``None``, optional
            The face (filling) colour of the markers. If ``None``, the colour
            is sampled from the jet colormap.
            Example `colour` options are ::

                    {'r', 'g', 'b', 'c', 'm', 'k', 'w'}
                    or
                    (3, ) ndarray

        marker_edge_colour : `colour` or ``None``, optional
            The edge colour of the markers.If ``None``, the colour
            is sampled from the jet colormap.
            Example `colour` options are ::

                    {'r', 'g', 'b', 'c', 'm', 'k', 'w'}
                    or
                    (3, ) ndarray

        marker_edge_width : `float`, optional
            The width of the markers' edge.
        render_axes : `bool`, optional
            If ``True``, the axes will be rendered.
        axes_font_name : See below, optional
            The font of the axes.
            Example options ::

                {'serif', 'sans-serif', 'cursive', 'fantasy', 'monospace'}

        axes_font_size : `int`, optional
            The font size of the axes.
        axes_font_style : ``{'normal', 'italic', 'oblique'}``, optional
            The font style of the axes.
        axes_font_weight : See below, optional
            The font weight of the axes.
            Example options ::

                {'ultralight', 'light', 'normal', 'regular', 'book', 'medium',
                 'roman', 'semibold', 'demibold', 'demi', 'bold', 'heavy',
                 'extra bold', 'black'}

        axes_x_limits : `float` or (`float`, `float`) or ``None``, optional
            The limits of the x axis. If `float`, then it sets padding on the
            right and left of the graph as a percentage of the curves' width. If
            `tuple` or `list`, then it defines the axis limits. If ``None``,
            then the limits are set automatically.
        axes_y_limits : `float` or (`float`, `float`) or ``None``, optional
            The limits of the y axis. If `float`, then it sets padding on the
            top and bottom of the graph as a percentage of the curves' height.
            If `tuple` or `list`, then it defines the axis limits. If ``None``,
            then the limits are set automatically.
        axes_x_ticks : `list` or `tuple` or ``None``, optional
            The ticks of the x axis.
        axes_y_ticks : `list` or `tuple` or ``None``, optional
            The ticks of the y axis.
        figure_size : (`float`, `float`) or ``None``, optional
            The size of the figure in inches.
        render_grid : `bool`, optional
            If ``True``, the grid will be rendered.
        grid_line_style : ``{'-', '--', '-.', ':'}``, optional
            The style of the grid lines.
        grid_line_width : `float`, optional
            The width of the grid lines.

        Returns
        -------
        renderer : `menpo.visualize.GraphPlotter`
            The renderer object.
        """
        from menpo.visualize import plot_curve
        costs = self.costs
        if costs is not None:
            return plot_curve(
                x_axis=list(range(len(costs))), y_axis=[costs],
                figure_id=figure_id, new_figure=new_figure,
                title='Cost per Iteration', x_label='Iteration',
                y_label='Cost Function', axes_x_limits=axes_x_limits,
                axes_y_limits=axes_y_limits, axes_x_ticks=axes_x_ticks,
                axes_y_ticks=axes_y_ticks, render_lines=render_lines,
                line_colour=line_colour, line_style=line_style,
                line_width=line_width, render_markers=render_markers,
                marker_style=marker_style, marker_size=marker_size,
                marker_face_colour=marker_face_colour,
                marker_edge_colour=marker_edge_colour,
                marker_edge_width=marker_edge_width, render_legend=False,
                render_axes=render_axes, axes_font_name=axes_font_name,
                axes_font_size=axes_font_size,
                axes_font_style=axes_font_style,
                axes_font_weight=axes_font_weight, figure_size=figure_size,
                render_grid=render_grid,  grid_line_style=grid_line_style,
                grid_line_width=grid_line_width)
        else:
            raise ValueError('costs are either not returned or not well '
                             'defined for the selected fitting algorithm')

    def view_iterations(self, figure_id=None, new_figure=False,
                        iters=None, render_image=True, subplots_enabled=False,
                        channels=None, interpolation='bilinear',
                        cmap_name=None, alpha=1., masked=True, render_lines=True,
                        line_style='-', line_width=2, line_colour=None,
                        render_markers=True, marker_edge_colour=None,
                        marker_face_colour=None, marker_style='o',
                        marker_size=4, marker_edge_width=1.,
                        render_numbering=False,
                        numbers_horizontal_align='center',
                        numbers_vertical_align='bottom',
                        numbers_font_name='sans-serif', numbers_font_size=10,
                        numbers_font_style='normal',
                        numbers_font_weight='normal',
                        numbers_font_colour='k', render_legend=True,
                        legend_title='', legend_font_name='sans-serif',
                        legend_font_style='normal', legend_font_size=10,
                        legend_font_weight='normal', legend_marker_scale=None,
                        legend_location=2, legend_bbox_to_anchor=(1.05, 1.),
                        legend_border_axes_pad=None, legend_n_columns=1,
                        legend_horizontal_spacing=None,
                        legend_vertical_spacing=None, legend_border=True,
                        legend_border_padding=None, legend_shadow=False,
                        legend_rounded_corners=False, render_axes=False,
                        axes_font_name='sans-serif', axes_font_size=10,
                        axes_font_style='normal', axes_font_weight='normal',
                        axes_x_limits=None, axes_y_limits=None,
                        axes_x_ticks=None, axes_y_ticks=None,
                        figure_size=(10, 8)):
        """
        Visualize the iterations of the fitting process.

        Parameters
        ----------
        figure_id : `object`, optional
            The id of the figure to be used.
        new_figure : `bool`, optional
            If ``True``, a new figure is created.
        iters : `int` or `list` of `int` or ``None``, optional
            The iterations to be visualized. If ``None``, then all the
            iterations are rendered.

            ======= ==================== =============
            No.     Visualised shape     Description
            ======= ==================== =============
            0       `self.initial_shape` Initial shape
            1       `self.shapes[1]`     Iteration 1
            i       `self.shapes[i]`     Iteration i
            n_iters `self.final_shape`   Final shape
            ======= ==================== =============

        render_image : `bool`, optional
            If ``True`` and the image exists, then it gets rendered.
        subplots_enabled : `bool`, optional
            If ``True``, then the requested final, initial and ground truth
            shapes get rendered on separate subplots.
        channels : `int` or `list` of `int` or ``all`` or ``None``
            If `int` or `list` of `int`, the specified channel(s) will be
            rendered. If ``all``, all the channels will be rendered in subplots.
            If ``None`` and the image is RGB, it will be rendered in RGB mode.
            If ``None`` and the image is not RGB, it is equivalent to ``all``.
        interpolation : `str` (See Below), optional
            The interpolation used to render the image. For example, if
            ``bilinear``, the image will be smooth and if ``nearest``, the
            image will be pixelated.
            Example options ::

                {none, nearest, bilinear, bicubic, spline16, spline36, hanning,
                 hamming, hermite, kaiser, quadric, catrom, gaussian, bessel,
                 mitchell, sinc, lanczos}

        cmap_name: `str`, optional,
            If ``None``, single channel and three channel images default
            to greyscale and rgb colormaps respectively.
        alpha : `float`, optional
            The alpha blending value, between 0 (transparent) and 1 (opaque).
        masked : `bool`, optional
            If ``True``, then the image is rendered as masked.
        render_lines : `bool` or `list` of `bool`, optional
            If ``True``, the lines will be rendered. You can either provide a
            single value that will be used for all shapes or a list with a
            different value per iteration shape.
        line_style : `str` or `list` of `str` (See below), optional
            The style of the lines. You can either provide a single value that
            will be used for all shapes or a list with a different value per
            iteration shape.
            Example options::

                {-, --, -., :}

        line_width : `float` or `list` of `float`, optional
            The width of the lines. You can either provide a single value that
            will be used for all shapes or a list with a different value per
            iteration shape.
        line_colour : `colour` or `list` of `colour` (See Below), optional
            The colour of the lines. You can either provide a single value that
            will be used for all shapes or a list with a different value per
            iteration shape.
            Example options ::

                {r, g, b, c, m, k, w}
                or
                (3, ) ndarray

        render_markers : `bool` or `list` of `bool`, optional
            If ``True``, the markers will be rendered. You can either provide a
            single value that will be used for all shapes or a list with a
            different value per iteration shape.
        marker_style : `str or `list` of `str` (See below), optional
            The style of the markers. You can either provide a single value that
            will be used for all shapes or a list with a different value per
            iteration shape.
            Example options ::

                {., ,, o, v, ^, <, >, +, x, D, d, s, p, *, h, H, 1, 2, 3, 4, 8}

        marker_size : `int` or `list` of `int`, optional
            The size of the markers in points. You can either provide a single
            value that will be used for all shapes or a list with a different
            value per iteration shape.
        marker_edge_colour : `colour` or `list` of `colour` (See Below), optional
            The edge colour of the markers. You can either provide a single
            value that will be used for all shapes or a list with a different
            value per iteration shape.
            Example options ::

                {r, g, b, c, m, k, w}
                or
                (3, ) ndarray

        marker_face_colour : `colour` or `list` of `colour` (See Below), optional
            The face (filling) colour of the markers. You can either provide a
            single value that will be used for all shapes or a list with a
            different value per iteration shape.
            Example options ::

                {r, g, b, c, m, k, w}
                or
                (3, ) ndarray

        marker_edge_width : `float` or `list` of `float`, optional
            The width of the markers' edge. You can either provide a single
            value that will be used for all shapes or a list with a different
            value per iteration shape.
        render_numbering : `bool`, optional
            If ``True``, the landmarks will be numbered.
        numbers_horizontal_align : `str` (See below), optional
            The horizontal alignment of the numbers' texts.
            Example options ::

                {center, right, left}

        numbers_vertical_align : `str` (See below), optional
            The vertical alignment of the numbers' texts.
            Example options ::

                {center, top, bottom, baseline}

        numbers_font_name : `str` (See below), optional
            The font of the numbers.
            Example options ::

                {serif, sans-serif, cursive, fantasy, monospace}

        numbers_font_size : `int`, optional
            The font size of the numbers.
        numbers_font_style : ``{normal, italic, oblique}``, optional
            The font style of the numbers.
        numbers_font_weight : `str` (See below), optional
            The font weight of the numbers.
            Example options ::

                {ultralight, light, normal, regular, book, medium, roman,
                 semibold, demibold, demi, bold, heavy, extra bold, black}

        numbers_font_colour : See Below, optional
            The font colour of the numbers.
            Example options ::

                {r, g, b, c, m, k, w}
                or
                (3, ) ndarray

        render_legend : `bool`, optional
            If ``True``, the legend will be rendered.
        legend_title : `str`, optional
            The title of the legend.
        legend_font_name : See below, optional
            The font of the legend.
            Example options ::

                {serif, sans-serif, cursive, fantasy, monospace}

        legend_font_style : `str` (See below), optional
            The font style of the legend.
            Example options ::

                {normal, italic, oblique}

        legend_font_size : `int`, optional
            The font size of the legend.
        legend_font_weight : `str` (See below), optional
            The font weight of the legend.
            Example options ::

                {ultralight, light, normal, regular, book, medium, roman,
                 semibold, demibold, demi, bold, heavy, extra bold, black}

        legend_marker_scale : `float`, optional
            The relative size of the legend markers with respect to the original
        legend_location : `int`, optional
            The location of the legend. The predefined values are:

            =============== ==
            'best'          0
            'upper right'   1
            'upper left'    2
            'lower left'    3
            'lower right'   4
            'right'         5
            'center left'   6
            'center right'  7
            'lower center'  8
            'upper center'  9
            'center'        10
            =============== ==

        legend_bbox_to_anchor : (`float`, `float`) `tuple`, optional
            The bbox that the legend will be anchored.
        legend_border_axes_pad : `float`, optional
            The pad between the axes and legend border.
        legend_n_columns : `int`, optional
            The number of the legend's columns.
        legend_horizontal_spacing : `float`, optional
            The spacing between the columns.
        legend_vertical_spacing : `float`, optional
            The vertical space between the legend entries.
        legend_border : `bool`, optional
            If ``True``, a frame will be drawn around the legend.
        legend_border_padding : `float`, optional
            The fractional whitespace inside the legend border.
        legend_shadow : `bool`, optional
            If ``True``, a shadow will be drawn behind legend.
        legend_rounded_corners : `bool`, optional
            If ``True``, the frame's corners will be rounded (fancybox).
        render_axes : `bool`, optional
            If ``True``, the axes will be rendered.
        axes_font_name : `str` (See below), optional
            The font of the axes.
            Example options ::

                {serif, sans-serif, cursive, fantasy, monospace}

        axes_font_size : `int`, optional
            The font size of the axes.
        axes_font_style : ``{normal, italic, oblique}``, optional
            The font style of the axes.
        axes_font_weight : `str` (See below), optional
            The font weight of the axes.
            Example options ::

                {ultralight, light, normal, regular, book, medium, roman,
                 semibold, demibold, demi, bold, heavy, extra bold, black}

        axes_x_limits : `float` or (`float`, `float`) or ``None``, optional
            The limits of the x axis. If `float`, then it sets padding on the
            right and left of the Image as a percentage of the Image's width. If
            `tuple` or `list`, then it defines the axis limits. If ``None``,
            then the limits are set automatically.
        axes_y_limits : (`float`, `float`) `tuple` or ``None``, optional
            The limits of the y axis. If `float`, then it sets padding on the
            top and bottom of the Image as a percentage of the Image's height.
            If `tuple` or `list`, then it defines the axis limits. If ``None``,
            then the limits are set automatically.
        axes_x_ticks : `list` or `tuple` or ``None``, optional
            The ticks of the x axis.
        axes_y_ticks : `list` or `tuple` or ``None``, optional
            The ticks of the y axis.
        figure_size : (`float`, `float`) `tuple` or ``None`` optional
            The size of the figure in inches.

        Returns
        -------
        renderer : `class`
            The renderer object.
        """
        # Parse iters
        iters = _parse_iters(iters, len(self.shapes))
        # Create image instance
        if self.image is None:
            image = Image(np.zeros((10, 10)))
            render_image = False
        else:
            image = Image(self.image.pixels)
        # Assign pointclouds to image
        n_digits = len(str(self.n_iters))
        groups = []
        subplots_titles = {}
        iters_offset = 1
        if self.initial_shape is not None:
            iters_offset = 0
        for j in iters:
            if j == 0 and self.initial_shape is not None:
                name = 'Initial'
                image.landmarks[name] = self.initial_shape
            elif j == len(self.shapes) - 1:
                name = 'Final'
                image.landmarks[name] = self.final_shape
            else:
                name = "iteration {:0{}d}".format(j + iters_offset, n_digits)
                image.landmarks[name] = self.shapes[j]
            groups.append(name)
            subplots_titles[name] = name
        # Render
        return view_image_multiple_landmarks(
                image, groups, with_labels=None, figure_id=figure_id,
                new_figure=new_figure, subplots_enabled=subplots_enabled,
                subplots_titles=subplots_titles, render_image=render_image,
                render_landmarks=True, masked=masked,
                channels=channels, interpolation=interpolation,
                cmap_name=cmap_name, alpha=alpha, image_view=True,
                render_lines=render_lines, line_style=line_style,
                line_width=line_width, line_colour=line_colour,
                render_markers=render_markers, marker_style=marker_style,
                marker_size=marker_size, marker_edge_width=marker_edge_width,
                marker_edge_colour=marker_edge_colour,
                marker_face_colour=marker_face_colour,
                render_numbering=render_numbering,
                numbers_horizontal_align=numbers_horizontal_align,
                numbers_vertical_align=numbers_vertical_align,
                numbers_font_name=numbers_font_name,
                numbers_font_size=numbers_font_size,
                numbers_font_style=numbers_font_style,
                numbers_font_weight=numbers_font_weight,
                numbers_font_colour=numbers_font_colour,
                render_legend=render_legend, legend_title=legend_title,
                legend_font_name=legend_font_name,
                legend_font_style=legend_font_style,
                legend_font_size=legend_font_size,
                legend_font_weight=legend_font_weight,
                legend_marker_scale=legend_marker_scale,
                legend_location=legend_location,
                legend_bbox_to_anchor=legend_bbox_to_anchor,
                legend_border_axes_pad=legend_border_axes_pad,
                legend_n_columns=legend_n_columns,
                legend_horizontal_spacing=legend_horizontal_spacing,
                legend_vertical_spacing=legend_vertical_spacing,
                legend_border=legend_border,
                legend_border_padding=legend_border_padding,
                legend_shadow=legend_shadow,
                legend_rounded_corners=legend_rounded_corners,
                render_axes=render_axes, axes_font_name=axes_font_name,
                axes_font_size=axes_font_size, axes_font_style=axes_font_style,
                axes_font_weight=axes_font_weight, axes_x_limits=axes_x_limits,
                axes_y_limits=axes_y_limits, axes_x_ticks=axes_x_ticks,
                axes_y_ticks=axes_y_ticks, figure_size=figure_size)


class ParametricIterativeResult(NonParametricIterativeResult):
    r"""
    Class for defining a parametric iterative fitting result, i.e. the
    result of a method that optimizes the parameters of a shape model. It
    holds the shapes and shape parameters of all the iterations of the
    fitting procedure. It can optionally store the image on which the
    fitting was applied, as well as its ground truth shape.

    .. note:: When using a method with a parametric shape model, the first step
              is to **reconstruct the initial shape** using the shape model. The
              generated reconstructed shape is then used as initialisation for
              the iterative optimisation. This step is not counted in the number
              of iterations.

    Parameters
    ----------
    shapes : `list` of `menpo.shape.PointCloud`
        The `list` of shapes per iteration. Note that the list does not
        include the initial shape. However, it includes the reconstruction of
        the initial shape. The last member of the list is the final shape.
    shape_parameters : `list` of `ndarray`
        The `list` of shape parameters per iteration. Note that the list
        includes the parameters of the projection of the initial shape. The last
        member of the list corresponds to the final shape's parameters. It must
        have the same length as `shapes`.
    initial_shape : `menpo.shape.PointCloud` or ``None``, optional
        The initial shape from which the fitting process started. If
        ``None``, then no initial shape is assigned.
    image : `menpo.image.Image` or `subclass` or ``None``, optional
        The image on which the fitting process was applied. Note that a copy
        of the image will be assigned as an attribute. If ``None``, then no
        image is assigned.
    gt_shape : `menpo.shape.PointCloud` or ``None``, optional
        The ground truth shape associated with the image. If ``None``, then
        no ground truth shape is assigned.
    costs : `list` of `float` or ``None``, optional
        The `list` of cost per iteration. If ``None``, then it is assumed that
        the cost function cannot be computed for the specific algorithm. It must
        have the same length as `shapes`.
    """
    def __init__(self, shapes, shape_parameters, initial_shape=None, image=None,
                 gt_shape=None, costs=None):
        # Assign shape parameters
        self._shape_parameters = shape_parameters
        # Get reconstructed initial shape
        self._reconstructed_initial_shape = shapes[0]
        # Call superclass
        super(ParametricIterativeResult, self).__init__(
                shapes=shapes, initial_shape=initial_shape, image=image,
                gt_shape=gt_shape, costs=costs)
        # Correct n_iters. The initial shape's reconstruction should not count
        # in the number of iterations.
        self._n_iters -= 1

    @property
    def shapes(self):
        r"""
        Returns the `list` of shapes obtained at each iteration of the fitting
        process. The `list` includes the `initial_shape` (if it exists),
        `reconstructed_initial_shape` and `final_shape`.

        :type: `list` of `menpo.shape.PointCloud`
        """
        return self._shapes

    @property
    def shape_parameters(self):
        r"""
        Returns the `list` of shape parameters obtained at each iteration of
        the fitting process. The `list` includes the parameters of the
        `reconstructed_initial_shape` and `final_shape`.

        :type: `list` of ``(n_params,)`` `ndarray`
        """
        return self._shape_parameters

    @property
    def reconstructed_initial_shape(self):
        r"""
        Returns the initial shape's reconstruction with the shape model that was
        used to initialise the iterative optimisation process.

        :type: `menpo.shape.PointCloud`
        """
        if self.initial_shape is not None:
            return self.shapes[1]
        else:
            return self.shapes[0]

    @property
    def _reconstruction_indices(self):
        r"""
        Returns a list with the indices of reconstructed shapes in the `shapes`
        list.

        :type: `list` of `int`
        """
        if self.initial_shape is not None:
            return [1]
        else:
            return [0]

    def reconstructed_initial_error(self, compute_error=None):
        r"""
        Returns the error of the reconstructed initial shape of the fitting
        process, if the ground truth shape exists. This is the error computed
        based on the `reconstructed_initial_shape`.

        Parameters
        ----------
        compute_error: `callable`, optional
            Callable that computes the error between the reconstructed initial
            and ground truth shapes.

        Returns
        -------
        reconstructed_initial_error : `float`
            The error that corresponds to the initial shape's reconstruction.

        Raises
        ------
        ValueError
            Ground truth shape has not been set, so the reconstructed initial
            error cannot be computed
        """
        if compute_error is None:
            compute_error = euclidean_bb_normalised_error
        if self.gt_shape is None:
            raise ValueError('Ground truth shape has not been set, so the '
                             'reconstructed initial error cannot be computed')
        else:
            return compute_error(self.reconstructed_initial_shape, self.gt_shape)

    def view_iterations(self, figure_id=None, new_figure=False,
                        iters=None, render_image=True, subplots_enabled=False,
                        channels=None, interpolation='bilinear',
                        cmap_name=None, alpha=1., masked=True, render_lines=True,
                        line_style='-', line_width=2, line_colour=None,
                        render_markers=True, marker_edge_colour=None,
                        marker_face_colour=None, marker_style='o',
                        marker_size=4, marker_edge_width=1.,
                        render_numbering=False,
                        numbers_horizontal_align='center',
                        numbers_vertical_align='bottom',
                        numbers_font_name='sans-serif', numbers_font_size=10,
                        numbers_font_style='normal',
                        numbers_font_weight='normal',
                        numbers_font_colour='k', render_legend=True,
                        legend_title='', legend_font_name='sans-serif',
                        legend_font_style='normal', legend_font_size=10,
                        legend_font_weight='normal', legend_marker_scale=None,
                        legend_location=2, legend_bbox_to_anchor=(1.05, 1.),
                        legend_border_axes_pad=None, legend_n_columns=1,
                        legend_horizontal_spacing=None,
                        legend_vertical_spacing=None, legend_border=True,
                        legend_border_padding=None, legend_shadow=False,
                        legend_rounded_corners=False, render_axes=False,
                        axes_font_name='sans-serif', axes_font_size=10,
                        axes_font_style='normal', axes_font_weight='normal',
                        axes_x_limits=None, axes_y_limits=None,
                        axes_x_ticks=None, axes_y_ticks=None,
                        figure_size=(10, 8)):
        """
        Visualize the iterations of the fitting process.

        Parameters
        ----------
        figure_id : `object`, optional
            The id of the figure to be used.
        new_figure : `bool`, optional
            If ``True``, a new figure is created.
        iters : `int` or `list` of `int` or ``None``, optional
            The iterations to be visualized. If ``None``, then all the
            iterations are rendered.

            ========= ==================================== ======================
            No.       Visualised shape                     Description
            ========= ==================================== ======================
            0           `self.initial_shape`               Initial shape
            1           `self.reconstructed_initial_shape` Reconstructed initial
            2           `self.shapes[2]`                   Iteration 1
            i           `self.shapes[i]`                   Iteration i-1
            n_iters+1 `self.final_shape`                   Final shape
            ========= ==================================== ======================

        render_image : `bool`, optional
            If ``True`` and the image exists, then it gets rendered.
        subplots_enabled : `bool`, optional
            If ``True``, then the requested final, initial and ground truth
            shapes get rendered on separate subplots.
        channels : `int` or `list` of `int` or ``all`` or ``None``
            If `int` or `list` of `int`, the specified channel(s) will be
            rendered. If ``all``, all the channels will be rendered in subplots.
            If ``None`` and the image is RGB, it will be rendered in RGB mode.
            If ``None`` and the image is not RGB, it is equivalent to ``all``.
        interpolation : `str` (See Below), optional
            The interpolation used to render the image. For example, if
            ``bilinear``, the image will be smooth and if ``nearest``, the
            image will be pixelated.
            Example options ::

                {none, nearest, bilinear, bicubic, spline16, spline36, hanning,
                 hamming, hermite, kaiser, quadric, catrom, gaussian, bessel,
                 mitchell, sinc, lanczos}

        cmap_name: `str`, optional,
            If ``None``, single channel and three channel images default
            to greyscale and rgb colormaps respectively.
        alpha : `float`, optional
            The alpha blending value, between 0 (transparent) and 1 (opaque).
        masked : `bool`, optional
            If ``True``, then the image is rendered as masked.
        render_lines : `bool` or `list` of `bool`, optional
            If ``True``, the lines will be rendered. You can either provide a
            single value that will be used for all shapes or a list with a
            different value per iteration shape.
        line_style : `str` or `list` of `str` (See below), optional
            The style of the lines. You can either provide a single value that
            will be used for all shapes or a list with a different value per
            iteration shape.
            Example options::

                {-, --, -., :}

        line_width : `float` or `list` of `float`, optional
            The width of the lines. You can either provide a single value that
            will be used for all shapes or a list with a different value per
            iteration shape.
        line_colour : `colour` or `list` of `colour` (See Below), optional
            The colour of the lines. You can either provide a single value that
            will be used for all shapes or a list with a different value per
            iteration shape.
            Example options ::

                {r, g, b, c, m, k, w}
                or
                (3, ) ndarray

        render_markers : `bool` or `list` of `bool`, optional
            If ``True``, the markers will be rendered. You can either provide a
            single value that will be used for all shapes or a list with a
            different value per iteration shape.
        marker_style : `str or `list` of `str` (See below), optional
            The style of the markers. You can either provide a single value that
            will be used for all shapes or a list with a different value per
            iteration shape.
            Example options ::

                {., ,, o, v, ^, <, >, +, x, D, d, s, p, *, h, H, 1, 2, 3, 4, 8}

        marker_size : `int` or `list` of `int`, optional
            The size of the markers in points. You can either provide a single
            value that will be used for all shapes or a list with a different
            value per iteration shape.
        marker_edge_colour : `colour` or `list` of `colour` (See Below), optional
            The edge colour of the markers. You can either provide a single
            value that will be used for all shapes or a list with a different
            value per iteration shape.
            Example options ::

                {r, g, b, c, m, k, w}
                or
                (3, ) ndarray

        marker_face_colour : `colour` or `list` of `colour` (See Below), optional
            The face (filling) colour of the markers. You can either provide a
            single value that will be used for all shapes or a list with a
            different value per iteration shape.
            Example options ::

                {r, g, b, c, m, k, w}
                or
                (3, ) ndarray

        marker_edge_width : `float` or `list` of `float`, optional
            The width of the markers' edge. You can either provide a single
            value that will be used for all shapes or a list with a different
            value per iteration shape.
        render_numbering : `bool`, optional
            If ``True``, the landmarks will be numbered.
        numbers_horizontal_align : `str` (See below), optional
            The horizontal alignment of the numbers' texts.
            Example options ::

                {center, right, left}

        numbers_vertical_align : `str` (See below), optional
            The vertical alignment of the numbers' texts.
            Example options ::

                {center, top, bottom, baseline}

        numbers_font_name : `str` (See below), optional
            The font of the numbers.
            Example options ::

                {serif, sans-serif, cursive, fantasy, monospace}

        numbers_font_size : `int`, optional
            The font size of the numbers.
        numbers_font_style : ``{normal, italic, oblique}``, optional
            The font style of the numbers.
        numbers_font_weight : `str` (See below), optional
            The font weight of the numbers.
            Example options ::

                {ultralight, light, normal, regular, book, medium, roman,
                 semibold, demibold, demi, bold, heavy, extra bold, black}

        numbers_font_colour : See Below, optional
            The font colour of the numbers.
            Example options ::

                {r, g, b, c, m, k, w}
                or
                (3, ) ndarray

        render_legend : `bool`, optional
            If ``True``, the legend will be rendered.
        legend_title : `str`, optional
            The title of the legend.
        legend_font_name : See below, optional
            The font of the legend.
            Example options ::

                {serif, sans-serif, cursive, fantasy, monospace}

        legend_font_style : `str` (See below), optional
            The font style of the legend.
            Example options ::

                {normal, italic, oblique}

        legend_font_size : `int`, optional
            The font size of the legend.
        legend_font_weight : `str` (See below), optional
            The font weight of the legend.
            Example options ::

                {ultralight, light, normal, regular, book, medium, roman,
                 semibold, demibold, demi, bold, heavy, extra bold, black}

        legend_marker_scale : `float`, optional
            The relative size of the legend markers with respect to the original
        legend_location : `int`, optional
            The location of the legend. The predefined values are:

            =============== ==
            'best'          0
            'upper right'   1
            'upper left'    2
            'lower left'    3
            'lower right'   4
            'right'         5
            'center left'   6
            'center right'  7
            'lower center'  8
            'upper center'  9
            'center'        10
            =============== ==

        legend_bbox_to_anchor : (`float`, `float`) `tuple`, optional
            The bbox that the legend will be anchored.
        legend_border_axes_pad : `float`, optional
            The pad between the axes and legend border.
        legend_n_columns : `int`, optional
            The number of the legend's columns.
        legend_horizontal_spacing : `float`, optional
            The spacing between the columns.
        legend_vertical_spacing : `float`, optional
            The vertical space between the legend entries.
        legend_border : `bool`, optional
            If ``True``, a frame will be drawn around the legend.
        legend_border_padding : `float`, optional
            The fractional whitespace inside the legend border.
        legend_shadow : `bool`, optional
            If ``True``, a shadow will be drawn behind legend.
        legend_rounded_corners : `bool`, optional
            If ``True``, the frame's corners will be rounded (fancybox).
        render_axes : `bool`, optional
            If ``True``, the axes will be rendered.
        axes_font_name : `str` (See below), optional
            The font of the axes.
            Example options ::

                {serif, sans-serif, cursive, fantasy, monospace}

        axes_font_size : `int`, optional
            The font size of the axes.
        axes_font_style : ``{normal, italic, oblique}``, optional
            The font style of the axes.
        axes_font_weight : `str` (See below), optional
            The font weight of the axes.
            Example options ::

                {ultralight, light, normal, regular, book, medium, roman,
                 semibold, demibold, demi, bold, heavy, extra bold, black}

        axes_x_limits : `float` or (`float`, `float`) or ``None``, optional
            The limits of the x axis. If `float`, then it sets padding on the
            right and left of the Image as a percentage of the Image's width. If
            `tuple` or `list`, then it defines the axis limits. If ``None``,
            then the limits are set automatically.
        axes_y_limits : (`float`, `float`) `tuple` or ``None``, optional
            The limits of the y axis. If `float`, then it sets padding on the
            top and bottom of the Image as a percentage of the Image's height.
            If `tuple` or `list`, then it defines the axis limits. If ``None``,
            then the limits are set automatically.
        axes_x_ticks : `list` or `tuple` or ``None``, optional
            The ticks of the x axis.
        axes_y_ticks : `list` or `tuple` or ``None``, optional
            The ticks of the y axis.
        figure_size : (`float`, `float`) `tuple` or ``None`` optional
            The size of the figure in inches.

        Returns
        -------
        renderer : `class`
            The renderer object.
        """
        # Parse iters
        iters = _parse_iters(iters, len(self.shapes))
        # Create image instance
        if self.image is None:
            image = Image(np.zeros((10, 10)))
            render_image = False
        else:
            image = Image(self.image.pixels)
        # Assign pointclouds to image
        n_digits = len(str(self.n_iters))
        groups = []
        subplots_titles = {}
        iters_offset = 0
        if self.initial_shape is not None:
            iters_offset = 1
        for j in iters:
            if j == 0 and self.initial_shape is not None:
                name = 'Initial'
                image.landmarks[name] = self.initial_shape
            elif j in self._reconstruction_indices:
                name = 'Reconstruction'
                image.landmarks[name] = self.shapes[j]
            elif j == len(self.shapes) - 1:
                name = 'Final'
                image.landmarks[name] = self.final_shape
            else:
                s = _get_scale_of_iter(j, self._reconstruction_indices)
                name = "iteration {:0{}d}".format(j - s + iters_offset, n_digits)
                image.landmarks[name] = self.shapes[j]
            groups.append(name)
            subplots_titles[name] = name
        # Render
        return view_image_multiple_landmarks(
                image, groups, with_labels=None, figure_id=figure_id,
                new_figure=new_figure, subplots_enabled=subplots_enabled,
                subplots_titles=subplots_titles, render_image=render_image,
                render_landmarks=True, masked=masked,
                channels=channels, interpolation=interpolation,
                cmap_name=cmap_name, alpha=alpha, image_view=True,
                render_lines=render_lines, line_style=line_style,
                line_width=line_width, line_colour=line_colour,
                render_markers=render_markers, marker_style=marker_style,
                marker_size=marker_size, marker_edge_width=marker_edge_width,
                marker_edge_colour=marker_edge_colour,
                marker_face_colour=marker_face_colour,
                render_numbering=render_numbering,
                numbers_horizontal_align=numbers_horizontal_align,
                numbers_vertical_align=numbers_vertical_align,
                numbers_font_name=numbers_font_name,
                numbers_font_size=numbers_font_size,
                numbers_font_style=numbers_font_style,
                numbers_font_weight=numbers_font_weight,
                numbers_font_colour=numbers_font_colour,
                render_legend=render_legend, legend_title=legend_title,
                legend_font_name=legend_font_name,
                legend_font_style=legend_font_style,
                legend_font_size=legend_font_size,
                legend_font_weight=legend_font_weight,
                legend_marker_scale=legend_marker_scale,
                legend_location=legend_location,
                legend_bbox_to_anchor=legend_bbox_to_anchor,
                legend_border_axes_pad=legend_border_axes_pad,
                legend_n_columns=legend_n_columns,
                legend_horizontal_spacing=legend_horizontal_spacing,
                legend_vertical_spacing=legend_vertical_spacing,
                legend_border=legend_border,
                legend_border_padding=legend_border_padding,
                legend_shadow=legend_shadow,
                legend_rounded_corners=legend_rounded_corners,
                render_axes=render_axes, axes_font_name=axes_font_name,
                axes_font_size=axes_font_size, axes_font_style=axes_font_style,
                axes_font_weight=axes_font_weight, axes_x_limits=axes_x_limits,
                axes_y_limits=axes_y_limits, axes_x_ticks=axes_x_ticks,
                axes_y_ticks=axes_y_ticks, figure_size=figure_size)


class MultiScaleNonParametricIterativeResult(NonParametricIterativeResult):
    r"""
    Class for defining a multi-scale non-parametric iterative fitting result,
    i.e. the result of a multi-scale method that does not optimise over a
    parametric shape model. It holds the shapes of all the iterations of
    the fitting procedure, as well as the scales. It can optionally store the
    image on which the fitting was applied, as well as its ground truth shape.

    Parameters
    ----------
    results : `list` of :map:`NonParametricIterativeResult`
        The `list` of non parametric iterative results per scale.
    scales : `list` of `float`
        The scale values (normally small to high).
    affine_transforms : `list` of `menpo.transform.Affine`
        The list of affine transforms per scale that transform the shapes into
        the original image space.
    scale_transforms : `list` of `menpo.shape.Scale`
        The list of scaling transforms per scale.
    image : `menpo.image.Image` or `subclass` or ``None``, optional
        The image on which the fitting process was applied. Note that a copy
        of the image will be assigned as an attribute. If ``None``, then no
        image is assigned.
    gt_shape : `menpo.shape.PointCloud` or ``None``, optional
        The ground truth shape associated with the image. If ``None``, then no
        ground truth shape is assigned.
    """
    def __init__(self, results, scales, affine_transforms, scale_transforms,
                 image=None, gt_shape=None):
        # Make sure results and scales are iterable
        if not isinstance(results, Iterable):
            results = [results]
        if not isinstance(scales, Iterable):
            scales = [scales]
        # Check that results and scales have the same length
        if len(results) != len(scales):
            raise ValueError('results and scales must have equal length ({} '
                             '!= {})'.format(len(results), len(scales)))
        # Get initial shape
        initial_shape = None
        if results[0].initial_shape is not None:
            initial_shape = _rescale_shapes_to_reference(
                shapes=[results[0].initial_shape],
                affine_transform=affine_transforms[0],
                scale_transform=scale_transforms[0])[0]
        # Create shapes list and n_iters_per_scale
        # If the result object has an initial shape, then it has to be
        # removed from the final shapes list
        n_iters_per_scale = []
        shapes = []
        for i in list(range(len(scales))):
            n_iters_per_scale.append(results[i].n_iters)
            if results[i].initial_shape is None:
                shapes += _rescale_shapes_to_reference(
                    shapes=results[i].shapes,
                    affine_transform=affine_transforms[i],
                    scale_transform=scale_transforms[i])
            else:
                shapes += _rescale_shapes_to_reference(
                    shapes=results[i].shapes[1:],
                    affine_transform=affine_transforms[i],
                    scale_transform=scale_transforms[i])
        # Call superclass
        super(MultiScaleNonParametricIterativeResult, self).__init__(
                shapes=shapes, initial_shape=initial_shape, image=image,
                gt_shape=gt_shape)
        # Get attributes
        self._n_iters_per_scale = n_iters_per_scale
        self._n_scales = len(scales)
        # Create costs list. We assume that if the costs of the first result
        # object is None, then the costs property of all objects is None.
        # Similarly, if the costs property of the the first object is not
        # None, then the same stands for the rest.
        self._costs = None
        if results[0].costs is not None:
            self._costs = []
            for r in results:
                self._costs += r.costs

    @property
    def n_iters_per_scale(self):
        r"""
        Returns the number of iterations per scale of the fitting process.

        :type: `list` of `int`
        """
        return self._n_iters_per_scale

    @property
    def n_scales(self):
        r"""
        Returns the number of scales used during the fitting process.

        :type: `int`
        """
        return self._n_scales


class MultiScaleParametricIterativeResult(MultiScaleNonParametricIterativeResult):
    r"""
    Class for defining a multi-scale parametric iterative fitting result, i.e.
    the result of a multi-scale method that optimizes over a parametric shape
    model. It holds the shapes of all the iterations of the fitting procedure,
    as well as the scales. It can optionally store the image on which the
    fitting was applied, as well as its ground truth shape.

    .. note:: When using a method with a parametric shape model, the first step
              is to **reconstruct the initial shape** using the shape model. The
              generated reconstructed shape is then used as initialisation for
              the iterative optimisation. This step is not counted in the number
              of iterations.

    Parameters
    ----------
    results : `list` of :map:`ParametricIterativeResult`
        The `list` of parametric iterative results per scale.
    scales : `list` of `float`
        The scale values (normally small to high).
    affine_transforms : `list` of `menpo.transform.Affine`
        The list of affine transforms per scale that transform the shapes into
        the original image space.
    scale_transforms : `list` of `menpo.shape.Scale`
        The list of scaling transforms per scale.
    image : `menpo.image.Image` or `subclass` or ``None``, optional
        The image on which the fitting process was applied. Note that a copy
        of the image will be assigned as an attribute. If ``None``, then no
        image is assigned.
    gt_shape : `menpo.shape.PointCloud` or ``None``, optional
        The ground truth shape associated with the image. If ``None``, then no
        ground truth shape is assigned.
    """
    def __init__(self, results, scales, affine_transforms, scale_transforms,
                 image=None, gt_shape=None):
        # Call superclass
        super(MultiScaleParametricIterativeResult, self).__init__(
            results=results, scales=scales, affine_transforms=affine_transforms,
            scale_transforms=scale_transforms, image=image, gt_shape=gt_shape)
        # Create shape parameters, and reconstructed initial shapes lists
        self._shape_parameters = []
        for r in results:
            self._shape_parameters += r.shape_parameters
        # Correct n_iters
        self._n_iters -= len(scales)

    @property
    def shape_parameters(self):
        r"""
        Returns the `list` of shape parameters obtained at each iteration of
        the fitting process. The `list` includes the parameters of the
        `initial_shape` (if it exists) and `final_shape`.

        :type: `list` of ``(n_params,)`` `ndarray`
        """
        return self._shape_parameters

    @property
    def reconstructed_initial_shapes(self):
        r"""
        Returns the result of the reconstruction step that takes place at each
        scale before applying the iterative optimisation.

        :type: `list` of `menpo.shape.PointCloud`
        """
        ids = self._reconstruction_indices
        return [self.shapes[i] for i in ids]

    @property
    def _reconstruction_indices(self):
        r"""
        Returns a list with the indices of reconstructed shapes in the `shapes`
        list.

        :type: `list` of `int`
        """
        initial_val = 0
        if self.initial_shape is not None:
            initial_val = 1
        ids = []
        for i in list(range(self.n_scales)):
            if i == 0:
                ids.append(initial_val)
            else:
                previous_val = ids[i - 1]
                ids.append(previous_val + self.n_iters_per_scale[i - 1] + 1)
        return ids

    def reconstructed_initial_error(self, compute_error=None):
        r"""
        Returns the error of the reconstructed initial shape of the fitting
        process, if the ground truth shape exists. This is the error computed
        based on the `reconstructed_initial_shapes[0]`.

        Parameters
        ----------
        compute_error: `callable`, optional
            Callable that computes the error between the reconstructed initial
            and ground truth shapes.

        Returns
        -------
        reconstructed_initial_error : `float`
            The error that corresponds to the initial shape's reconstruction.

        Raises
        ------
        ValueError
            Ground truth shape has not been set, so the reconstructed initial
            error cannot be computed
        """
        if compute_error is None:
            compute_error = euclidean_bb_normalised_error
        if self.gt_shape is None:
            raise ValueError('Ground truth shape has not been set, so the '
                             'reconstructed initial error cannot be computed')
        else:
            return compute_error(self.reconstructed_initial_shapes[0],
                                 self.gt_shape)

    def view_iterations(self, figure_id=None, new_figure=False,
                        iters=None, render_image=True, subplots_enabled=False,
                        channels=None, interpolation='bilinear',
                        cmap_name=None, alpha=1., masked=True, render_lines=True,
                        line_style='-', line_width=2, line_colour=None,
                        render_markers=True, marker_edge_colour=None,
                        marker_face_colour=None, marker_style='o',
                        marker_size=4, marker_edge_width=1.,
                        render_numbering=False,
                        numbers_horizontal_align='center',
                        numbers_vertical_align='bottom',
                        numbers_font_name='sans-serif', numbers_font_size=10,
                        numbers_font_style='normal',
                        numbers_font_weight='normal',
                        numbers_font_colour='k', render_legend=True,
                        legend_title='', legend_font_name='sans-serif',
                        legend_font_style='normal', legend_font_size=10,
                        legend_font_weight='normal', legend_marker_scale=None,
                        legend_location=2, legend_bbox_to_anchor=(1.05, 1.),
                        legend_border_axes_pad=None, legend_n_columns=1,
                        legend_horizontal_spacing=None,
                        legend_vertical_spacing=None, legend_border=True,
                        legend_border_padding=None, legend_shadow=False,
                        legend_rounded_corners=False, render_axes=False,
                        axes_font_name='sans-serif', axes_font_size=10,
                        axes_font_style='normal', axes_font_weight='normal',
                        axes_x_limits=None, axes_y_limits=None,
                        axes_x_ticks=None, axes_y_ticks=None,
                        figure_size=(10, 8)):
        """
        Visualize the iterations of the fitting process.

        Parameters
        ----------
        figure_id : `object`, optional
            The id of the figure to be used.
        new_figure : `bool`, optional
            If ``True``, a new figure is created.
        iters : `int` or `list` of `int` or ``None``, optional
            The iterations to be visualized. If ``None``, then all the
            iterations are rendered.

            ========= ==================================== ======================
            No.       Visualised shape                     Description
            ========= ==================================== ======================
            0           `self.initial_shape`               Initial shape
            1           `self.reconstructed_initial_shape` Reconstructed initial
            2           `self.shapes[2]`                   Iteration 1
            i           `self.shapes[i]`                   Iteration i-1
            n_iters+1 `self.final_shape`                   Final shape
            ========= ==================================== ======================

        render_image : `bool`, optional
            If ``True`` and the image exists, then it gets rendered.
        subplots_enabled : `bool`, optional
            If ``True``, then the requested final, initial and ground truth
            shapes get rendered on separate subplots.
        channels : `int` or `list` of `int` or ``all`` or ``None``
            If `int` or `list` of `int`, the specified channel(s) will be
            rendered. If ``all``, all the channels will be rendered in subplots.
            If ``None`` and the image is RGB, it will be rendered in RGB mode.
            If ``None`` and the image is not RGB, it is equivalent to ``all``.
        interpolation : `str` (See Below), optional
            The interpolation used to render the image. For example, if
            ``bilinear``, the image will be smooth and if ``nearest``, the
            image will be pixelated.
            Example options ::

                {none, nearest, bilinear, bicubic, spline16, spline36, hanning,
                 hamming, hermite, kaiser, quadric, catrom, gaussian, bessel,
                 mitchell, sinc, lanczos}

        cmap_name: `str`, optional,
            If ``None``, single channel and three channel images default
            to greyscale and rgb colormaps respectively.
        alpha : `float`, optional
            The alpha blending value, between 0 (transparent) and 1 (opaque).
        masked : `bool`, optional
            If ``True``, then the image is rendered as masked.
        render_lines : `bool` or `list` of `bool`, optional
            If ``True``, the lines will be rendered. You can either provide a
            single value that will be used for all shapes or a list with a
            different value per iteration shape.
        line_style : `str` or `list` of `str` (See below), optional
            The style of the lines. You can either provide a single value that
            will be used for all shapes or a list with a different value per
            iteration shape.
            Example options::

                {-, --, -., :}

        line_width : `float` or `list` of `float`, optional
            The width of the lines. You can either provide a single value that
            will be used for all shapes or a list with a different value per
            iteration shape.
        line_colour : `colour` or `list` of `colour` (See Below), optional
            The colour of the lines. You can either provide a single value that
            will be used for all shapes or a list with a different value per
            iteration shape.
            Example options ::

                {r, g, b, c, m, k, w}
                or
                (3, ) ndarray

        render_markers : `bool` or `list` of `bool`, optional
            If ``True``, the markers will be rendered. You can either provide a
            single value that will be used for all shapes or a list with a
            different value per iteration shape.
        marker_style : `str or `list` of `str` (See below), optional
            The style of the markers. You can either provide a single value that
            will be used for all shapes or a list with a different value per
            iteration shape.
            Example options ::

                {., ,, o, v, ^, <, >, +, x, D, d, s, p, *, h, H, 1, 2, 3, 4, 8}

        marker_size : `int` or `list` of `int`, optional
            The size of the markers in points. You can either provide a single
            value that will be used for all shapes or a list with a different
            value per iteration shape.
        marker_edge_colour : `colour` or `list` of `colour` (See Below), optional
            The edge colour of the markers. You can either provide a single
            value that will be used for all shapes or a list with a different
            value per iteration shape.
            Example options ::

                {r, g, b, c, m, k, w}
                or
                (3, ) ndarray

        marker_face_colour : `colour` or `list` of `colour` (See Below), optional
            The face (filling) colour of the markers. You can either provide a
            single value that will be used for all shapes or a list with a
            different value per iteration shape.
            Example options ::

                {r, g, b, c, m, k, w}
                or
                (3, ) ndarray

        marker_edge_width : `float` or `list` of `float`, optional
            The width of the markers' edge. You can either provide a single
            value that will be used for all shapes or a list with a different
            value per iteration shape.
        render_numbering : `bool`, optional
            If ``True``, the landmarks will be numbered.
        numbers_horizontal_align : `str` (See below), optional
            The horizontal alignment of the numbers' texts.
            Example options ::

                {center, right, left}

        numbers_vertical_align : `str` (See below), optional
            The vertical alignment of the numbers' texts.
            Example options ::

                {center, top, bottom, baseline}

        numbers_font_name : `str` (See below), optional
            The font of the numbers.
            Example options ::

                {serif, sans-serif, cursive, fantasy, monospace}

        numbers_font_size : `int`, optional
            The font size of the numbers.
        numbers_font_style : ``{normal, italic, oblique}``, optional
            The font style of the numbers.
        numbers_font_weight : `str` (See below), optional
            The font weight of the numbers.
            Example options ::

                {ultralight, light, normal, regular, book, medium, roman,
                 semibold, demibold, demi, bold, heavy, extra bold, black}

        numbers_font_colour : See Below, optional
            The font colour of the numbers.
            Example options ::

                {r, g, b, c, m, k, w}
                or
                (3, ) ndarray

        render_legend : `bool`, optional
            If ``True``, the legend will be rendered.
        legend_title : `str`, optional
            The title of the legend.
        legend_font_name : See below, optional
            The font of the legend.
            Example options ::

                {serif, sans-serif, cursive, fantasy, monospace}

        legend_font_style : `str` (See below), optional
            The font style of the legend.
            Example options ::

                {normal, italic, oblique}

        legend_font_size : `int`, optional
            The font size of the legend.
        legend_font_weight : `str` (See below), optional
            The font weight of the legend.
            Example options ::

                {ultralight, light, normal, regular, book, medium, roman,
                 semibold, demibold, demi, bold, heavy, extra bold, black}

        legend_marker_scale : `float`, optional
            The relative size of the legend markers with respect to the original
        legend_location : `int`, optional
            The location of the legend. The predefined values are:

            =============== ==
            'best'          0
            'upper right'   1
            'upper left'    2
            'lower left'    3
            'lower right'   4
            'right'         5
            'center left'   6
            'center right'  7
            'lower center'  8
            'upper center'  9
            'center'        10
            =============== ==

        legend_bbox_to_anchor : (`float`, `float`) `tuple`, optional
            The bbox that the legend will be anchored.
        legend_border_axes_pad : `float`, optional
            The pad between the axes and legend border.
        legend_n_columns : `int`, optional
            The number of the legend's columns.
        legend_horizontal_spacing : `float`, optional
            The spacing between the columns.
        legend_vertical_spacing : `float`, optional
            The vertical space between the legend entries.
        legend_border : `bool`, optional
            If ``True``, a frame will be drawn around the legend.
        legend_border_padding : `float`, optional
            The fractional whitespace inside the legend border.
        legend_shadow : `bool`, optional
            If ``True``, a shadow will be drawn behind legend.
        legend_rounded_corners : `bool`, optional
            If ``True``, the frame's corners will be rounded (fancybox).
        render_axes : `bool`, optional
            If ``True``, the axes will be rendered.
        axes_font_name : `str` (See below), optional
            The font of the axes.
            Example options ::

                {serif, sans-serif, cursive, fantasy, monospace}

        axes_font_size : `int`, optional
            The font size of the axes.
        axes_font_style : ``{normal, italic, oblique}``, optional
            The font style of the axes.
        axes_font_weight : `str` (See below), optional
            The font weight of the axes.
            Example options ::

                {ultralight, light, normal, regular, book, medium, roman,
                 semibold, demibold, demi, bold, heavy, extra bold, black}

        axes_x_limits : `float` or (`float`, `float`) or ``None``, optional
            The limits of the x axis. If `float`, then it sets padding on the
            right and left of the Image as a percentage of the Image's width. If
            `tuple` or `list`, then it defines the axis limits. If ``None``,
            then the limits are set automatically.
        axes_y_limits : (`float`, `float`) `tuple` or ``None``, optional
            The limits of the y axis. If `float`, then it sets padding on the
            top and bottom of the Image as a percentage of the Image's height.
            If `tuple` or `list`, then it defines the axis limits. If ``None``,
            then the limits are set automatically.
        axes_x_ticks : `list` or `tuple` or ``None``, optional
            The ticks of the x axis.
        axes_y_ticks : `list` or `tuple` or ``None``, optional
            The ticks of the y axis.
        figure_size : (`float`, `float`) `tuple` or ``None`` optional
            The size of the figure in inches.

        Returns
        -------
        renderer : `class`
            The renderer object.
        """
        # Parse iters
        iters = _parse_iters(iters, len(self.shapes))
        # Create image instance
        if self.image is None:
            image = Image(np.zeros((10, 10)))
            render_image = False
        else:
            image = Image(self.image.pixels)
        # Assign pointclouds to image
        n_digits = len(str(self.n_iters))
        groups = []
        subplots_titles = {}
        iters_offset = -2
        if self.initial_shape is not None:
            iters_offset = -1
        for j in iters:
            if j == 0 and self.initial_shape is not None:
                name = 'Initial'
                image.landmarks[name] = self.initial_shape
            elif j in self._reconstruction_indices:
                name = 'Reconstruction'
                image.landmarks[name] = self.shapes[j]
            elif j == len(self.shapes) - 1:
                name = 'Final'
                image.landmarks[name] = self.final_shape
            else:
                s = _get_scale_of_iter(j, self._reconstruction_indices)
                name = "iteration {:0{}d}".format(j - s + iters_offset, n_digits)
                image.landmarks[name] = self.shapes[j]
            groups.append(name)
            subplots_titles[name] = name
        # Render
        return view_image_multiple_landmarks(
            image, groups, with_labels=None, figure_id=figure_id,
            new_figure=new_figure, subplots_enabled=subplots_enabled,
            subplots_titles=subplots_titles, render_image=render_image,
            render_landmarks=True, masked=masked,
            channels=channels, interpolation=interpolation,
            cmap_name=cmap_name, alpha=alpha, image_view=True,
            render_lines=render_lines, line_style=line_style,
            line_width=line_width, line_colour=line_colour,
            render_markers=render_markers, marker_style=marker_style,
            marker_size=marker_size, marker_edge_width=marker_edge_width,
            marker_edge_colour=marker_edge_colour,
            marker_face_colour=marker_face_colour,
            render_numbering=render_numbering,
            numbers_horizontal_align=numbers_horizontal_align,
            numbers_vertical_align=numbers_vertical_align,
            numbers_font_name=numbers_font_name,
            numbers_font_size=numbers_font_size,
            numbers_font_style=numbers_font_style,
            numbers_font_weight=numbers_font_weight,
            numbers_font_colour=numbers_font_colour,
            render_legend=render_legend, legend_title=legend_title,
            legend_font_name=legend_font_name,
            legend_font_style=legend_font_style,
            legend_font_size=legend_font_size,
            legend_font_weight=legend_font_weight,
            legend_marker_scale=legend_marker_scale,
            legend_location=legend_location,
            legend_bbox_to_anchor=legend_bbox_to_anchor,
            legend_border_axes_pad=legend_border_axes_pad,
            legend_n_columns=legend_n_columns,
            legend_horizontal_spacing=legend_horizontal_spacing,
            legend_vertical_spacing=legend_vertical_spacing,
            legend_border=legend_border,
            legend_border_padding=legend_border_padding,
            legend_shadow=legend_shadow,
            legend_rounded_corners=legend_rounded_corners,
            render_axes=render_axes, axes_font_name=axes_font_name,
            axes_font_size=axes_font_size, axes_font_style=axes_font_style,
            axes_font_weight=axes_font_weight, axes_x_limits=axes_x_limits,
            axes_y_limits=axes_y_limits, axes_x_ticks=axes_x_ticks,
            axes_y_ticks=axes_y_ticks, figure_size=figure_size)
