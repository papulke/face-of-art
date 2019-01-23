from menpofit import checks
from menpofit.fitter import MultiScaleParametricFitter

from .algorithm import InverseCompositional


class LucasKanadeATMFitter(MultiScaleParametricFitter):
    r"""
    Class for defining an ATM fitter using the Lucas-Kanade optimization.

    Parameters
    ----------
    atm : :map:`ATM` or `subclass`
        The trained ATM model.
    lk_algorithm_cls : `class`, optional
        The Lukas-Kanade optimisation algorithm that will get applied. The
        possible algorithms are:

        =========================== ============== =============
        Class                       Warp Direction Warp Update
        =========================== ============== =============
        :map:`ForwardCompositional` Forward        Compositional
        :map:`InverseCompositional` Inverse
        =========================== ============== =============

    n_shape : `int` or `float` or `list` of those or ``None``, optional
        The number of shape components that will be used. If `int`, then it
        defines the exact number of active components. If `float`, then it
        defines the percentage of variance to keep. If `int` or `float`, then
        the provided value will be applied for all scales. If `list`, then it
        defines a value per scale. If ``None``, then all the available
        components will be used. Note that this simply sets the active
        components without trimming the unused ones. Also, the available
        components may have already been trimmed to `max_shape_components`
        during training.
    sampling : `list` of `int` or `ndarray` or ``None``
        It defines a sampling mask per scale. If `int`, then it defines the
        sub-sampling step of the sampling mask. If `ndarray`, then it
        explicitly defines the sampling mask. If ``None``, then no
        sub-sampling is applied.
    """
    def __init__(self, atm, lk_algorithm_cls=InverseCompositional,
                 n_shape=None, sampling=None):
        # Store model
        self._model = atm

        # Check parameters
        checks.set_models_components(atm.shape_models, n_shape)
        self._sampling = checks.check_sampling(sampling, atm.n_scales)

        # Get list of algorithm objects per scale
        interfaces = atm.build_fitter_interfaces(self._sampling)
        algorithms = [lk_algorithm_cls(interface) for interface in interfaces]

        # Call superclass
        super(LucasKanadeATMFitter, self).__init__(
            scales=atm.scales, reference_shape=atm.reference_shape,
            holistic_features=atm.holistic_features, algorithms=algorithms)

    @property
    def atm(self):
        r"""
        The trained ATM model.

        :type: :map:`ATM` or `subclass`
        """
        return self._model

    def warped_images(self, image, shapes):
        r"""
        Given an input test image and a list of shapes, it warps the image
        into the shapes. This is useful for generating the warped images of a
        fitting procedure stored within a
        :map:`MultiScaleParametricIterativeResult`.

        Parameters
        ----------
        image : `menpo.image.Image` or `subclass`
            The input image to be warped.
        shapes : `list` of `menpo.shape.PointCloud`
            The list of shapes in which the image will be warped. The shapes
            are obtained during the iterations of a fitting procedure.

        Returns
        -------
        warped_images : `list` of `menpo.image.MaskedImage` or `ndarray`
            The warped images.
        """
        return self.algorithms[-1].interface.warped_images(image=image,
                                                           shapes=shapes)

    def __str__(self):
        # Compute scale info strings
        scales_info = []
        lvl_str_tmplt = r"""   - Scale {}
     - {} active shape components
     - {} similarity transform components"""
        for k, s in enumerate(self.scales):
            scales_info.append(lvl_str_tmplt.format(
                    s,
                    self.atm.shape_models[k].n_active_components,
                    self.atm.shape_models[k].n_global_parameters))
        scales_info = '\n'.join(scales_info)

        cls_str = r"""{class_title}
 - Scales: {scales}
{scales_info}
    """.format(class_title=self.algorithms[0].__str__(),
               scales=self.scales,
               scales_info=scales_info)
        return self.atm.__str__() + cls_str
