import numpy as np

from menpofit.error import compute_cumulative_error


def _check_multi_argument(arg, n_objects, dtypes, error_str):
    # check argument's dtype with provided dtypes
    if not isinstance(dtypes, list):
        dtypes = [dtypes]
    is_dtype = False
    for t in dtypes:
        if (t is None and arg is None) or isinstance(arg, t):
            is_dtype = True
    # fix provided argument
    if is_dtype:
        arg = [arg] * n_objects
    elif isinstance(arg, list):
        if len(arg) == 1:
            arg *= n_objects
        elif len(arg) != n_objects:
            raise ValueError(error_str)
    else:
        raise ValueError(error_str)
    return arg


def view_image_multiple_landmarks(
        image, groups, with_labels=None, figure_id=None, new_figure=False,
        subplots_enabled=True, subplots_titles=None, render_image=True,
        render_landmarks=True, masked=True, channels=None,
        interpolation='bilinear', cmap_name=None, alpha=1.,
        image_view=True, render_lines=True, line_style='-', line_width=1,
        line_colour='r', render_markers=True, marker_style='o',
        marker_size=5, marker_edge_width=1, marker_edge_colour='k',
        marker_face_colour='r', render_numbering=False,
        numbers_horizontal_align='center', numbers_vertical_align='bottom',
        numbers_font_name='sans-serif', numbers_font_size=10,
        numbers_font_style='normal', numbers_font_weight='normal',
        numbers_font_colour='k', render_legend=True, legend_title='',
        legend_font_name='sans-serif', legend_font_style='normal',
        legend_font_size=10, legend_font_weight='normal',
        legend_marker_scale=None, legend_location=2,
        legend_bbox_to_anchor=(1.05, 1.), legend_border_axes_pad=None,
        legend_n_columns=1, legend_horizontal_spacing=None,
        legend_vertical_spacing=None, legend_border=True,
        legend_border_padding=None, legend_shadow=False,
        legend_rounded_corners=False, render_axes=False,
        axes_font_name='sans-serif', axes_font_size=10,
        axes_font_style='normal', axes_font_weight='normal',
        axes_x_limits=None, axes_y_limits=None, axes_x_ticks=None,
        axes_y_ticks=None, figure_size=(10, 8)):
    """
    Visualize an image with its attached landmark groups. The method can
    visualize multiple landmark groups on the same figure either on subplots
    or on the same plot.

    Parameters
    ----------
    image : `menpo.image.Image` or `menpo.image.MaskedImage` or subclass
        The image object.
    groups : `list`
        A list with the landmark groups to be visualized.
    with_labels : ``None`` or `list` of `str` or `list` of those, optional
        If not ``None``, only show the given label(s). You can either provide a
        single value that will be used for all landmark groups or a list with
        different value per landmark group.
    figure_id : `object`, optional
        The id of the figure to be used.
    new_figure : `bool`, optional
        If ``True``, a new figure is created.
    subplots_enabled : `bool`, optional
        If ``True``, then the requested landmark groups will be rendered on
        separate subplots.
    subplots_titles : ``None`` or `dict`, optional
        A dict with groups as keys and a subplot name value per key. If
        ``None``, then the group names are used.
    render_image : `bool`, optional
        If ``True``, then the image gets rendered.
    render_landmarks : `bool`, optional
        If ``True``, then the landmarks get rendered.
    masked : `bool`, optional
        If ``True``, then the image is rendered as masked.
    channels : `int` or `list` of `int` or ``all`` or ``None``
        If `int` or `list` of `int`, the specified channel(s) will be
        rendered. If ``all``, all the channels will be rendered in subplots.
        If ``None`` and the image is RGB, it will be rendered in RGB mode.
        If ``None`` and the image is not RGB, it is equivalent to ``all``.
    interpolation : See Below, optional
        The interpolation used to render the image. For example, if
        ``bilinear``, the image will be smooth and if ``nearest``, the
        image will be pixelated. Example options ::

            {'none', 'nearest', 'bilinear', 'bicubic', 'spline16', 'spline36',
             'hanning', 'hamming', 'hermite', 'kaiser', 'quadric', 'catrom',
             'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos'}

    cmap_name: `str`, optional,
        If ``None``, single channel and three channel images default
        to greyscale and rgb colormaps respectively.
    alpha : `float`, optional
        The alpha blending value, between 0 (transparent) and 1 (opaque).
    image_view : `bool`, optional
        If ``True``, then the landmarks are rendered in the image axes mode.
        It only gets applied if render_image is False.
    render_lines : `bool` or `list` of `bool`, optional
        If ``True``, the lines will be rendered. You can either provide a
        single value that will be used for all landmark groups or a list with
        a different value per landmark group.
    line_style : `str` or `list` of `str`, optional
        The style of the lines. You can either provide a single value that will
        be used for all landmark groups or a list with a different value per
        landmark group.
        Example options::

            {'-', '--', '-.', ':'}

    line_width : `float` or `list` of `float`, optional
        The width of the lines. You can either provide a single value that will
        be used for all landmark groups or a list with a different value per
        landmark group.
    line_colour : `colour` or `list` of `colour`, optional
        The colour of the lines. You can either provide a single value that
        will be used for all landmark groups or a list with a different value
        per landmark group.
        Example options::

            {'r', 'g', 'b', 'c', 'm', 'k', 'w'}
            or
            (3, ) ndarray

    render_markers : `bool`, optional
        If ``True``, the markers will be rendered. You can either provide a
        single value that will be used for all landmark groups or a list with
        a different value per landmark group.
    marker_style : `str` or `list` of `str`, optional
        The style of the markers. You can either provide a single value that
        will be used for all landmark groups or a list with a different value
        per landmark group.
        Example options::

            {'.', ',', 'o', 'v', '^', '<', '>', '+', 'x', 'D', 'd', 's', 'p',
             '*', 'h', 'H', '1', '2', '3', '4', '8'}

    marker_size : `float` or `list` of `float`, optional
        The size of the markers in points^2. You can either provide a single
        value that will be used for all landmark groups or a list with a
        different value per landmark group.
    marker_edge_width : `float` or `list` of `float`, optional
        The width of the markers' edge.  You can either provide a single
        value that will be used for all landmark groups or a list with a
        different value per landmark group.
    marker_edge_colour : `colour` or `list` of `colour`, optional
        The edge colour of the markers. You can either provide a single value
        that will be used for all landmark groups or a list with a different
        value per landmark group.
        Example options::

            {'r', 'g', 'b', 'c', 'm', 'k', 'w'}
            or
            (3, ) ndarray

    marker_face_colour : `colour` or `list` of `colour`, optional
        The face colour of the markers. You can either provide a single value
        that will be used for all landmark groups or a list with a different
        value per landmark group.
        Example options::

            {'r', 'g', 'b', 'c', 'm', 'k', 'w'}
            or
            (3, ) ndarray

    render_numbering : `bool`, optional
        If ``True``, the landmarks will be numbered.
    numbers_horizontal_align : ``{center, right, left}``, optional
        The horizontal alignment of the numbers' texts.
    numbers_vertical_align : ``{center, top, bottom, baseline}``, optional
        The vertical alignment of the numbers' texts.
    numbers_font_name : See Below, optional
        The font of the numbers. Example options ::

            {'serif', 'sans-serif', 'cursive', 'fantasy', 'monospace'}

    numbers_font_size : `int`, optional
        The font size of the numbers.
    numbers_font_style : ``{normal, italic, oblique}``, optional
        The font style of the numbers.
    numbers_font_weight : See Below, optional
        The font weight of the numbers.
        Example options ::

            {'ultralight', 'light', 'normal', 'regular', 'book', 'medium',
             'roman', 'semibold', 'demibold', 'demi', 'bold', 'heavy',
             'extra bold', 'black'}

    numbers_font_colour : See Below, optional
        The font colour of the numbers.
        Example options ::

            {'r', 'g', 'b', 'c', 'm', 'k', 'w'}
            or
            (3, ) ndarray

    render_legend : `bool`, optional
        If ``True``, the legend will be rendered.
    legend_title : `str`, optional
        The title of the legend.
    legend_font_name : See below, optional
        The font of the legend. Example options ::

            {'serif', 'sans-serif', 'cursive', 'fantasy', 'monospace'}

    legend_font_style : ``{normal, italic, oblique}``, optional
        The font style of the legend.
    legend_font_size : `int`, optional
        The font size of the legend.
    legend_font_weight : See Below, optional
        The font weight of the legend.
        Example options ::

            {'ultralight', 'light', 'normal', 'regular', 'book', 'medium',
             'roman', 'semibold', 'demibold', 'demi', 'bold', 'heavy',
             'extra bold', 'black'}

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

            {'serif', 'sans-serif', 'cursive', 'fantasy', 'monospace'}

    axes_font_size : `int`, optional
        The font size of the axes.
    axes_font_style : ``{normal, italic, oblique}``, optional
        The font style of the axes.
    axes_font_weight : See Below, optional
        The font weight of the axes.
        Example options ::

            {'ultralight', 'light', 'normal', 'regular', 'book', 'medium',
             'roman', 'semibold', 'demibold', 'demi', 'bold', 'heavy',
             'extra bold', 'black'}

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
    from menpo.visualize.viewmatplotlib import (MatplotlibSubplots,
                                                MatplotlibRenderer,
                                                _check_colours_list)
    from menpo.image import MaskedImage
    import matplotlib.pyplot as plt

    # If multiple channels were passed in, they must be restricted to the
    # first channel
    if isinstance(channels, list):
        channels = channels[0]

    # This makes the code shorter for dealing with masked images vs non-masked
    # images
    mask_arguments = ({'masked': masked} if isinstance(image, MaskedImage)
                      else {})

    # Parse arguments
    render_lines = _check_multi_argument(
            render_lines, len(groups), bool,
            'render_lines must be a bool or a list of bool with length equal '
            'to the number of landmark groups')
    line_style = _check_multi_argument(
            line_style, len(groups), str,
            'line_style must be a str or a list of str with length equal '
            'to the number of landmark groups')
    line_width = _check_multi_argument(
            line_width, len(groups), [int, float],
            'line_width must be an int/float or a list of int/float with '
            'length equal to the number of landmark groups')
    line_colour = _check_colours_list(
            True, line_colour, len(groups),
            'line_colour must be a colour value or a list of colours with '
            'length equal to the number of landmark groups')
    render_markers = _check_multi_argument(
            render_markers, len(groups), bool,
            'render_markers must be a bool or a list of bool with length equal '
            'to the number of landmark groups')
    marker_style = _check_multi_argument(
            marker_style, len(groups), str,
            'marker_style must be a str or a list of str with length equal '
            'to the number of landmark groups')
    marker_size = _check_multi_argument(
            marker_size, len(groups), [int, float],
            'marker_size must be an int/float or a list of int/float with '
            'length equal to the number of landmark groups')
    marker_edge_width = _check_multi_argument(
            marker_edge_width, len(groups), [int, float],
            'marker_edge_width must be an int/float or a list of int/float '
            'with length equal to the number of landmark groups')
    marker_edge_colour = _check_colours_list(
            True, marker_edge_colour, len(groups),
            'marker_edge_colour must be a colour value or a list of colours '
            'with length equal to the number of landmark groups')
    marker_face_colour = _check_colours_list(
            True, marker_face_colour, len(groups),
            'marker_face_colour must be a colour value or a list of colours '
            'with length equal to the number of landmark groups')
    if (with_labels is None or (isinstance(with_labels, list) and
                                isinstance(with_labels[0], str))):
        with_labels = [with_labels] * len(groups)
    elif isinstance(with_labels, list) and isinstance(with_labels[0], list):
        if len(with_labels) != len(groups):
            raise ValueError('with_labels must be a list of length equal to '
                             'the number of landmark groups')
    else:
        raise ValueError('with_labels must be a list of length equal to '
                         'the number of landmark groups')
    if subplots_titles is None:
        subplots_titles = {}
        for g in groups:
            subplots_titles[g] = g

    # Initialize renderer
    renderer = MatplotlibRenderer(figure_id=figure_id, new_figure=new_figure)

    # Render
    if render_image:
        # image will be displayed
        if render_landmarks and len(groups) > 0:
            # there are selected landmark groups and they will be displayed
            if subplots_enabled:
                # calculate subplots structure
                subplots = MatplotlibSubplots()._subplot_layout(len(groups))
            # show image with landmarks
            for k, group in enumerate(groups):
                if subplots_enabled:
                    # create subplot
                    plt.subplot(subplots[0], subplots[1], k + 1)
                    if render_legend:
                        # set subplot's title
                        plt.title(subplots_titles[group],
                                  fontname=legend_font_name,
                                  fontstyle=legend_font_style,
                                  fontweight=legend_font_weight,
                                  fontsize=legend_font_size)
                renderer = image.view_landmarks(
                        channels=channels, group=group,
                        with_labels=with_labels[k], without_labels=None,
                        figure_id=renderer.figure_id, new_figure=False,
                        render_lines=render_lines[k], line_style=line_style[k],
                        line_width=line_width[k], line_colour=line_colour[k],
                        render_markers=render_markers[k],
                        marker_style=marker_style[k],
                        marker_size=marker_size[k],
                        marker_edge_width=marker_edge_width[k],
                        marker_edge_colour=marker_edge_colour[k],
                        marker_face_colour=marker_face_colour[k],
                        render_numbering=render_numbering,
                        numbers_horizontal_align=numbers_horizontal_align,
                        numbers_vertical_align=numbers_vertical_align,
                        numbers_font_name=numbers_font_name,
                        numbers_font_size=numbers_font_size,
                        numbers_font_style=numbers_font_style,
                        numbers_font_weight=numbers_font_weight,
                        numbers_font_colour=numbers_font_colour,
                        render_legend=render_legend and not subplots_enabled,
                        legend_title=legend_title,
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
                        axes_font_size=axes_font_size,
                        axes_font_style=axes_font_style,
                        axes_font_weight=axes_font_weight,
                        axes_x_limits=axes_x_limits,
                        axes_y_limits=axes_y_limits,
                        axes_x_ticks=axes_x_ticks, axes_y_ticks=axes_y_ticks,
                        interpolation=interpolation, alpha=alpha,
                        cmap_name=cmap_name, figure_size=figure_size,
                        **mask_arguments)
            if not subplots_enabled:
                if render_legend:
                    # Options related to legend's font
                    prop = {'family': legend_font_name,
                            'size': legend_font_size,
                            'style': legend_font_style,
                            'weight': legend_font_weight}

                    # display legend on side
                    plt.gca().legend([subplots_titles[g] for g in groups],
                                     title=legend_title, prop=prop,
                                     loc=legend_location,
                                     bbox_to_anchor=legend_bbox_to_anchor,
                                     borderaxespad=legend_border_axes_pad,
                                     ncol=legend_n_columns,
                                     columnspacing=legend_horizontal_spacing,
                                     labelspacing=legend_vertical_spacing,
                                     frameon=legend_border,
                                     borderpad=legend_border_padding,
                                     shadow=legend_shadow,
                                     fancybox=legend_rounded_corners,
                                     markerscale=legend_marker_scale)
        else:
            # either there are not any landmark groups selected or they won't
            # be displayed
            renderer = image.view(
                    channels=channels, render_axes=render_axes,
                    axes_font_name=axes_font_name,
                    axes_font_size=axes_font_size,
                    axes_font_style=axes_font_style,
                    axes_font_weight=axes_font_weight,
                    axes_x_limits=axes_x_limits, axes_y_limits=axes_y_limits,
                    axes_x_ticks=axes_x_ticks, axes_y_ticks=axes_y_ticks,
                    figure_size=figure_size, interpolation=interpolation,
                    alpha=alpha, cmap_name=cmap_name, **mask_arguments)
    else:
        # image won't be displayed
        if render_landmarks and len(groups) > 0:
            # there are selected landmark groups and they will be displayed
            if subplots_enabled:
                # calculate subplots structure
                subplots = MatplotlibSubplots()._subplot_layout(len(groups))
            # not image, landmarks
            for k, group in enumerate(groups):
                if subplots_enabled:
                    # create subplot
                    plt.subplot(subplots[0], subplots[1], k + 1)
                    if render_legend:
                        # set subplot's title
                        plt.title(subplots_titles[group],
                                  fontname=legend_font_name,
                                  fontstyle=legend_font_style,
                                  fontweight=legend_font_weight,
                                  fontsize=legend_font_size)
                image.landmarks[group].lms.view(
                        image_view=image_view, render_lines=render_lines[k],
                        line_style=line_style[k], line_width=line_width[k],
                        line_colour=line_colour[k],
                        render_markers=render_markers[k],
                        marker_style=marker_style[k],
                        marker_size=marker_size[k],
                        marker_edge_width=marker_edge_width[k],
                        marker_edge_colour=marker_edge_colour[k],
                        marker_face_colour=marker_face_colour[k],
                        render_axes=render_axes, axes_font_name=axes_font_name,
                        axes_font_size=axes_font_size,
                        axes_font_style=axes_font_style,
                        axes_font_weight=axes_font_weight,
                        axes_x_limits=axes_x_limits,
                        axes_y_limits=axes_y_limits,
                        axes_x_ticks=axes_x_ticks, axes_y_ticks=axes_y_ticks,
                        figure_size=figure_size)
            if not subplots_enabled:
                if render_legend:
                    # Options related to legend's font
                    prop = {'family': legend_font_name,
                            'size': legend_font_size,
                            'style': legend_font_style,
                            'weight': legend_font_weight}

                    # display legend on side
                    plt.gca().legend([subplots_titles[g] for g in groups],
                                     title=legend_title, prop=prop,
                                     loc=legend_location,
                                     bbox_to_anchor=legend_bbox_to_anchor,
                                     borderaxespad=legend_border_axes_pad,
                                     ncol=legend_n_columns,
                                     columnspacing=legend_horizontal_spacing,
                                     labelspacing=legend_vertical_spacing,
                                     frameon=legend_border,
                                     borderpad=legend_border_padding,
                                     shadow=legend_shadow,
                                     fancybox=legend_rounded_corners,
                                     markerscale=legend_marker_scale)

    return renderer


def plot_cumulative_error_distribution(
        errors, error_range=None, figure_id=None, new_figure=False,
        title='Cumulative Error Distribution',
        x_label='Normalized Point-to-Point Error', y_label='Images Proportion',
        legend_entries=None, render_lines=True, line_colour=None,
        line_style='-', line_width=2, render_markers=True, marker_style='s',
        marker_size=7, marker_face_colour='w', marker_edge_colour=None,
        marker_edge_width=2, render_legend=True, legend_title=None,
        legend_font_name='sans-serif', legend_font_style='normal',
        legend_font_size=10, legend_font_weight='normal',
        legend_marker_scale=1., legend_location=2,
        legend_bbox_to_anchor=(1.05, 1.), legend_border_axes_pad=1.,
        legend_n_columns=1, legend_horizontal_spacing=1.,
        legend_vertical_spacing=1., legend_border=True,
        legend_border_padding=0.5, legend_shadow=False,
        legend_rounded_corners=False, render_axes=True,
        axes_font_name='sans-serif', axes_font_size=10,
        axes_font_style='normal', axes_font_weight='normal', axes_x_limits=None,
        axes_y_limits=None, axes_x_ticks=None, axes_y_ticks=None,
        figure_size=(10, 8), render_grid=True, grid_line_style='--',
        grid_line_width=0.5):
    r"""
    Plot the cumulative error distribution (CED) of the provided fitting errors.

    Parameters
    ----------
    errors : `list` of `lists`
        A `list` with `lists` of fitting errors. A separate CED curve will be
        rendered for each errors `list`.
    error_range : `list` of `float` with length 3, optional
        Specifies the horizontal axis range, i.e. ::

            error_range[0] = min_error
            error_range[1] = max_error
            error_range[2] = error_step

        If ``None``, then ``'error_range = [0., 0.101, 0.005]'``.
    figure_id : `object`, optional
        The id of the figure to be used.
    new_figure : `bool`, optional
        If ``True``, a new figure is created.
    title : `str`, optional
        The figure's title.
    x_label : `str`, optional
        The label of the horizontal axis.
    y_label : `str`, optional
        The label of the vertical axis.
    legend_entries : `list of `str` or ``None``, optional
        If `list` of `str`, it must have the same length as `errors` `list` and
        each `str` will be used to name each curve. If ``None``, the CED curves
        will be named as `'Curve %d'`.
    render_lines : `bool` or `list` of `bool`, optional
        If ``True``, the line will be rendered. If `bool`, this value will be
        used for all curves. If `list`, a value must be specified for each
        fitting errors curve, thus it must have the same length as `errors`.
    line_colour : `colour` or `list` of `colour` or ``None``, optional
        The colour of the lines. If not a `list`, this value will be
        used for all curves. If `list`, a value must be specified for each
        curve, thus it must have the same length as `y_axis`. If ``None``, the
        colours will be linearly sampled from jet colormap.
        Example `colour` options are ::

                {'r', 'g', 'b', 'c', 'm', 'k', 'w'}
                or
                (3, ) ndarray

    line_style : ``{'-', '--', '-.', ':'}`` or `list` of those, optional
        The style of the lines. If not a `list`, this value will be used for all
        curves. If `list`, a value must be specified for each curve, thus it
        must have the same length as `errors`.
    line_width : `float` or `list` of `float`, optional
        The width of the lines. If `float`, this value will be used for all
        curves. If `list`, a value must be specified for each curve, thus it
        must have the same length as `errors`.
    render_markers : `bool` or `list` of `bool`, optional
        If ``True``, the markers will be rendered. If `bool`, this value will be
        used for all curves. If `list`, a value must be specified for each
        curve, thus it must have the same length as `errors`.
    marker_style : `marker` or `list` of `markers`, optional
        The style of the markers. If not a `list`, this value will be used for
        all curves. If `list`, a value must be specified for each curve, thus it
        must have the same length as `errors`.
        Example `marker` options ::

                {'.', ',', 'o', 'v', '^', '<', '>', '+', 'x', 'D', 'd', 's',
                 'p', '*', 'h', 'H', '1', '2', '3', '4', '8'}

    marker_size : `int` or `list` of `int`, optional
        The size of the markers in points. If `int`, this value will be used
        for all curves. If `list`, a value must be specified for each curve, thus
        it must have the same length as `errors`.
    marker_face_colour : `colour` or `list` of `colour` or ``None``, optional
        The face (filling) colour of the markers. If not a `list`, this value
        will be used for all curves. If `list`, a value must be specified for
        each curve, thus it must have the same length as `errors`. If ``None``,
        the colours will be linearly sampled from jet colormap.
        Example `colour` options are ::

                {'r', 'g', 'b', 'c', 'm', 'k', 'w'}
                or
                (3, ) ndarray

    marker_edge_colour : `colour` or `list` of `colour` or ``None``, optional
        The edge colour of the markers. If not a `list`, this value will be used
        for all curves. If `list`, a value must be specified for each curve, thus
        it must have the same length as `errors`. If ``None``, the colours will
        be linearly sampled from jet colormap.
        Example `colour` options are ::

                {'r', 'g', 'b', 'c', 'm', 'k', 'w'}
                or
                (3, ) ndarray

    marker_edge_width : `float` or `list` of `float`, optional
        The width of the markers' edge. If `float`, this value will be used for
        all curves. If `list`, a value must be specified for each curve, thus it
        must have the same length as `errors`.
    render_legend : `bool`, optional
        If ``True``, the legend will be rendered.
    legend_title : `str`, optional
        The title of the legend.
    legend_font_name : See below, optional
        The font of the legend.
        Example options ::

            {'serif', 'sans-serif', 'cursive', 'fantasy', 'monospace'}

    legend_font_style : ``{'normal', 'italic', 'oblique'}``, optional
        The font style of the legend.
    legend_font_size : `int`, optional
        The font size of the legend.
    legend_font_weight : See below, optional
        The font weight of the legend.
        Example options ::

            {'ultralight', 'light', 'normal', 'regular', 'book', 'medium',
             'roman', 'semibold', 'demibold', 'demi', 'bold', 'heavy',
             'extra bold', 'black'}

    legend_marker_scale : `float`, optional
        The relative size of the legend markers with respect to the original
    legend_location : `int`, optional
        The location of the legend. The predefined values are:

        =============== ===
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
        =============== ===

    legend_bbox_to_anchor : (`float`, `float`), optional
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
        `tuple` or `list`, then it defines the axis limits. If ``None``, then
        the limits are set to ``(0., error_range[1])``.
    axes_y_limits : `float` or (`float`, `float`) or ``None``, optional
        The limits of the y axis. If `float`, then it sets padding on the
        top and bottom of the graph as a percentage of the curves' height. If
        `tuple` or `list`, then it defines the axis limits. If ``None``, then
        the limits are set to ``(0., 1.)``.
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

    Raises
    ------
    ValueError
        legend_entries list has different length than errors list

    Returns
    -------
    viewer : `menpo.visualize.GraphPlotter`
        The viewer object.
    """
    from menpo.visualize import plot_curve

    # make sure that errors is a list even with one list member
    if not isinstance(errors[0], list):
        errors = [errors]

    # create x and y axes lists
    if error_range is None:
        error_range = [0., 0.101, 0.005]
    x_axis = list(np.arange(error_range[0], error_range[1], error_range[2]))
    ceds = [compute_cumulative_error(e, x_axis) for e in errors]

    # parse legend_entries, axes_x_limits and axes_y_limits
    if legend_entries is None:
        legend_entries = ["Curve {}".format(k) for k in range(len(ceds))]
    if len(legend_entries) != len(ceds):
        raise ValueError('legend_entries list has different length than errors '
                         'list')
    if axes_x_limits is None:
        axes_x_limits = (0., x_axis[-1])
    if axes_y_limits is None:
        axes_y_limits = (0., 1.)

    # render
    return plot_curve(
            x_axis=x_axis, y_axis=ceds, figure_id=figure_id,
            new_figure=new_figure, legend_entries=legend_entries,
            title=title, x_label=x_label, y_label=y_label,
            axes_x_limits=axes_x_limits, axes_y_limits=axes_y_limits,
            axes_x_ticks=axes_x_ticks, axes_y_ticks=axes_y_ticks,
            render_lines=render_lines, line_colour=line_colour,
            line_style=line_style, line_width=line_width,
            render_markers=render_markers, marker_style=marker_style,
            marker_size=marker_size, marker_face_colour=marker_face_colour,
            marker_edge_colour=marker_edge_colour,
            marker_edge_width=marker_edge_width, render_legend=render_legend,
            legend_title=legend_title, legend_font_name=legend_font_name,
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
            axes_font_weight=axes_font_weight, figure_size=figure_size,
            render_grid=render_grid, grid_line_style=grid_line_style,
            grid_line_width=grid_line_width)
