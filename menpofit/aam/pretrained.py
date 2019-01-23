from menpofit.io import load_fitter


def load_balanced_frontal_face_fitter():
    r"""
    Loads a frontal face patch-based AAM fitter that is a good compromise
    between model size, fitting time and fitting performance. The model returns
    68 facial landmark points (the standard IBUG68 markup).

    Note that the first time you invoke this function, menpofit will
    download the fitter from Menpo's server. The fitter will then be stored
    locally for future use.

    The model is a :map:`PatchAAM` trained using the following parameters:

        =================== =================================
        Parameter           Value
        =================== =================================
        `diagonal`          110
        `scales`            (0.5, 1.0)
        `patch_shape`       [(13, 13), (13, 13)]
        `holistic_features` `menpo.feature.fast_dsift()`
        `n_shape`           [5, 20]
        `n_appearance`      [30, 150]
        `lk_algorithm_cls`  :map:`WibergInverseCompositional`
        =================== =================================

    It is also using the following `sampling` grid:

    .. code-block:: python

      import numpy as np

      patch_shape = (13, 13)
      sampling_step = 4

      sampling_grid = np.zeros(patch_shape, dtype=np.bool)
      sampling_grid[::sampling_step, ::sampling_step] = True
      sampling = [sampling_grid, sampling_grid]

    Additionally, it is trained on LFPW trainset, HELEN trainset, IBUG and AFW
    datasets (3283 images in total), which are hosted in
    http://ibug.doc.ic.ac.uk/resources/facial-point-annotations/.

    Returns
    -------
    fitter : :map:`LucasKanadeAAMFitter`
        A pre-trained :map:`LucasKanadeAAMFitter` based on a :map:`PatchAAM`
        that performs facial landmark localization returning 68 points (iBUG68).
    """
    return load_fitter('balanced_frontal_face_aam')
