from collections import OrderedDict
import numpy as np

from menpo.visualize import print_progress as menpo_print_progress


def print_progress(iterable, prefix='', n_items=None, offset=0,
                   show_bar=True, show_count=True, show_eta=True,
                   end_with_newline=True, verbose=True):
    r"""
    Print the remaining time needed to compute over an iterable.

    To use, wrap an existing iterable with this function before processing in
    a for loop (see example).

    The estimate of the remaining time is based on a moving average of the last
    100 items completed in the loop.

    This method is identical to `menpo.visualize.print_progress`, but adds a
    `verbose` flag which allows the printing to be skipped if necessary.

    Parameters
    ----------
    iterable : `iterable`
        An iterable that will be processed. The iterable is passed through by
        this function, with the time taken for each complete iteration logged.
    prefix : `str`, optional
        If provided a string that will be prepended to the progress report at
        each level.
    n_items : `int`, optional
        Allows for ``iterator`` to be a generator whose length will be assumed
        to be `n_items`. If not provided, then ``iterator`` needs to be
        `Sizable`.
    offset : `int`, optional
        Useful in combination with ``n_items`` - report back the progress as
        if `offset` items have already been handled. ``n_items``  will be left
        unchanged.
    show_bar : `bool`, optional
        If False, The progress bar (e.g. [=========      ]) will be hidden.
    show_count : `bool`, optional
        If False, The item count (e.g. (4/25)) will be hidden.
    show_eta : `bool`, optional
        If False, The estimated time to finish (e.g. - 00:00:03 remaining)
        will be hidden.
    end_with_newline : `bool`, optional
        If False, there will be no new line added at the end of the dynamic
        printing. This means the next print statement will overwrite the
        dynamic report presented here. Useful if you want to follow up a
        print_progress with a second print_progress, where the second
        overwrites the first on the same line.
    verbose : `bool`, optional
        Printing is performed only if set to ``True``.

    Raises
    ------
    ValueError
        ``offset`` provided without ``n_items``

    Examples
    --------
    This for loop: ::

        from time import sleep
        for i in print_progress(range(100)):
            sleep(1)

    prints a progress report of the form: ::

        [=============       ] 70% (7/10) - 00:00:03 remaining
    """
    if verbose:
        # Yield the images from the menpo print_progress (yield from would
        # be perfect here :( )
        for i in menpo_print_progress(iterable, prefix=prefix, n_items=n_items,
                                      offset=offset, show_bar=show_bar,
                                      show_count=show_count, show_eta=show_eta,
                                      end_with_newline=end_with_newline):
            yield i
    else:
        # Skip the verbosity!
        for i in iterable:
            yield i


def statistics_table(errors, method_names, auc_max_error, auc_error_step,
                     auc_min_error=0., stats_types=None, stats_names=None,
                     sort_by=None, precision=4):
    r"""
    Function that generates a table with statistical measures on the fitting
    results of various methods using pandas. It supports multiple types of
    statistical measures.

    **Note that the returned object is a pandas table which can be further
    converted to Latex tabular or simply a string.** See the examples for
    more details.

    Parameters
    ----------
    errors : `list` of `list` of `float`
        A `list` that contains `lists` of `float` with the errors per method.
    method_names : `list` of `str`
        The `list` with the names that will appear for each method. Note that
        it must have the same length as `errors`.
    auc_max_error : `float`
        The maximum error value for computing the area under the curve.
    auc_error_step : `float`
        The sampling step of the error bins for computing the area under the
        curve.
    auc_min_error : `float`, optional
        The minimum error value for computing the area under the curve.
    stats_types : `list` of `str` or ``None``, optional
        The types of statistical measures to compute. Possible options are:

        ======== ========================================================
        Value    Description
        ======== ========================================================
        `mean`   The mean value of the errors.
        `std`    The standard deviation of the errors.
        `median` The median value of the errors.
        `mad`    The median absolute deviation of the errors.
        `max`    The max value of the errors.
        `auc`    The area under the curve based on the CED of the errors.
        `fr`     The failure rate (percentage of images that failed).
        ======== ========================================================

        If ``None``, then all of them will be used with the above order.
    stats_names : `list` of `str`, optional
        The `list` with the names that will appear for each statistical measure
        type selected in `stats_types`. Note that it must have the same
        length as `stats_types`.
    sort_by : `str` or ``None``, optional
        The column to use for sorting the methods. If ``None``, then no
        sorting is performed and the methods will appear in the provided
        order of `method_names`. Possible options are:

        ======== ========================================================
        Value    Description
        ======== ========================================================
        `mean`   The mean value of the errors.
        `std`    The standard deviation of the errors.
        `median` The median value of the errors.
        `mad`    The median absolute deviation of the errors.
        `max`    The max value of the errors.
        `auc`    The area under the curve based on the CED of the errors.
        `fr`     The failure rate (percentage of images that failed).
        ======== ========================================================

    precision : `int`, optional
        The precision of the reported values, i.e. the number of decimals.

    Raises
    ------
    ValueError
        stat_type must be selected from [mean, std, median, mad, max, auc, fr]
    ValueError
        sort_by must be selected from [mean, std, median, mad, max, auc, fr]
    ValueError
        stats_types and stats_names must have the same length

    Returns
    -------
    table : `pandas.DataFrame`
        The pandas table. It can be further converted to various format,
        such as Latex tabular or `str`.

    Examples
    --------
    Let us create some errors for 3 methods sampled from Normal distributions
    with different mean and standard deviations: ::

        import numpy as np
        from menpofit.visualize import statistics_table

        method_names = ['Method_1', 'Method_2', 'Method_3']
        errors = [list(np.random.normal(0.07, 0.02, 400)),
                  list(np.random.normal(0.06, 0.03, 400)),
                  list(np.random.normal(0.08, 0.04, 400))]

    We can create a pandas `DataFrame` as: ::

        tab = statistics_table(errors, method_names, auc_max_error=0.1,
                               auc_error_step=0.001, sort_by='auc')
        tab

    Pandas offers excellent functionalities. For example, the table can be
    converted to an `str` as: ::

        print(tab.to_string())

    or to a Latex tabular as: ::

        print(tab.to_latex())

    """
    from menpofit.error import compute_statistical_measures
    import pandas as pn

    # Make sure errors is a list of lists
    if not isinstance(errors[0], list):
        errors = [errors]

    # Compute statistics
    means, stds, medians, mads, maxs, aucs, frs = compute_statistical_measures(
            errors, step_error=auc_error_step, max_error=auc_max_error,
            min_error=auc_min_error)

    # Check stats types
    supported_types = ['mean', 'std', 'median', 'mad', 'max', 'auc', 'fr']
    if stats_types is None:
        stats_types = supported_types

    # Check stats names
    if stats_names is None:
        stats_names = stats_types

    # Check stats_types and stats_names lists
    if len(stats_types) != len(stats_names):
        raise ValueError('stats_types and stats_names must have the same '
                         'length')

    # Create data dict
    data = OrderedDict()
    for stat_type, stat_name in zip(stats_types, stats_names):
        if stat_type not in supported_types:
            raise ValueError('stat_type must be selected from [mean, std, '
                             'median, mad, max, auc, fr]')
        if stat_type == 'mean':
            data[stat_name] = np.array(means)
        if stat_type == 'std':
            data[stat_name] = np.array(stds)
        if stat_type == 'median':
            data[stat_name] = np.array(medians)
        if stat_type == 'mad':
            data[stat_name] = np.array(mads)
        if stat_type == 'max':
            data[stat_name] = np.array(maxs)
        if stat_type == 'auc':
            data[stat_name] = np.array(aucs)
        if stat_type == 'fr':
            data[stat_name] = np.array(frs)

    # Create pandas table
    tab = pn.DataFrame(data, index=method_names)

    # Sort table
    ascending = True
    if sort_by is not None:
        if sort_by not in stats_types:
            raise ValueError('sort_by must be selected from [mean, std, '
                             'median, mad, max, auc, fr]')
        if sort_by == 'auc':
            ascending = False
        tab.sort_values(by=stats_names[stats_types.index(sort_by)],
                        inplace=True, ascending=ascending)

    # Set precision
    pn.set_option('precision', precision)

    return tab
