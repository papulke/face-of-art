import numpy as np

from menpo.feature import no_op
from menpo.base import name_of_callable

from menpofit.transform import DifferentiableAlignmentAffine
from menpofit.fitter import MultiScaleNonParametricFitter
from menpofit import checks

from .algorithm import InverseCompositional
from .residual import SSD
from .result import LucasKanadeResult


class LucasKanadeFitter(MultiScaleNonParametricFitter):
    r"""
    Class for defining a multi-scale Lucas-Kanade fitter that performs alignment
    with respect to a homogeneous transform. Please see the references for a
    basic list of relevant papers.

    Parameters
    ----------
    template : `menpo.image.Image`
        The template image.
    group : `str` or ``None``, optional
        The landmark group of the `template` that will be used as reference
        shape. If ``None`` and the `template` only has a single landmark
        group, then that is the one that will be used.
    holistic_features : `closure` or `list` of `closure`, optional
        The features that will be extracted from the training images. Note
        that the features are extracted before warping the images to the
        reference shape. If `list`, then it must define a feature function per
        scale. Please refer to `menpo.feature` for a list of potential features.
    diagonal : `int` or ``None``, optional
        This parameter is used to rescale the reference shape (specified by
        `group`) so that the diagonal of its bounding box matches the
        provided value. In other words, this parameter controls the size of
        the model at the highest scale. If ``None``, then the reference shape
        does not get rescaled.
    scales : `tuple` of `float`, optional
        The scale value of each scale. They must provided in ascending order,
        i.e. from lowest to highest scale.
    transform : `subclass` of :map:`DP` and :map:`DX`, optional
        A differential homogeneous transform object, e.g.
        :map:`DifferentiableAlignmentAffine`.
    algorithm_cls : `class`, optional
        The Lukas-Kanade optimisation algorithm that will get applied. The
        possible algorithms in `menpofit.lk.algorithm` are:

        ====================== ============== =============
        Class                  Warp Direction Warp Update
        ====================== ============== =============
        `ForwardAdditive`      Forward        Additive
        `ForwardCompositional` Forward        Compositional
        `InverseCompositional` Inverse
        ====================== ============== =============
    residual_cls : `class` subclass, optional
        The residual that will get applied. All possible residuals are:

        ========================== ============================================
        Class                      Description
        ========================== ============================================
        :map:`SSD`                 Sum of Squared Differences
        :map:`FourierSSD`          Sum of Squared Differences on Fourier domain
        :map:`ECC`                 Enhanced Correlation Coefficient
        :map:`GradientImages`      Image Gradient
        :map:`GradientCorrelation` Gradient Correlation
        ========================== ============================================

    References
    ----------
    .. [1] B.D. Lucas, and T. Kanade, "An iterative image registration
        technique with an application to stereo vision", International Joint
        Conference on Artificial Intelligence, pp. 674-679, 1981.
    .. [2] G.D. Evangelidis, and E.Z. Psarakis. "Parametric Image Alignment
        Using Enhanced Correlation Coefficient Maximization", IEEE Transactions
        on Pattern Analysis and Machine Intelligence, 30(10): 1858-1865, 2008.
    .. [3] A.B. Ashraf, S. Lucey, and T. Chen. "Fast Image Alignment in the
        Fourier Domain", IEEE Proceedings of International Conference on
        Computer Vision and Pattern Recognition, pp. 2480-2487, 2010.
    .. [4] G. Tzimiropoulos, S. Zafeiriou, and M. Pantic. "Robust and
        Efficient Parametric Face Alignment", IEEE Proceedings of International
        Conference on Computer Vision (ICCV), pp. 1847-1854, November 2011.
    """
    def __init__(self, template, group=None, holistic_features=no_op,
                 diagonal=None, transform=DifferentiableAlignmentAffine,
                 scales=(0.5, 1.0), algorithm_cls=InverseCompositional,
                 residual_cls=SSD):
        # Check arguments
        checks.check_diagonal(diagonal)
        scales = checks.check_scales(scales)
        holistic_features = checks.check_callable(holistic_features,
                                                  len(scales))
        # Assign attributes
        self.transform_cls = transform
        self.diagonal = diagonal

        # Make template masked for warping
        template = template.as_masked(copy=False)

        # Get reference shape
        if self.diagonal:
            template = template.rescale_landmarks_to_diagonal_range(
                self.diagonal, group=group)
        reference_shape = template.landmarks[group].lms

        # Call superclass
        super(LucasKanadeFitter, self).__init__(
            scales=list(scales), reference_shape=reference_shape,
            holistic_features=holistic_features, algorithms=[])

        # Create templates
        self.templates, self.sources = self._prepare_template(template,
                                                              group=group)

        # Get list of algorithm objects per scale
        self.algorithms = []
        for j, (t, s) in enumerate(zip(self.templates, self.sources)):
            transform = self.transform_cls(s, s)
            residual = residual_cls()
            self.algorithms.append(algorithm_cls(t, transform, residual))

    def _prepare_template(self, template, group=None):
        gt_shape = template.landmarks[group].lms
        templates, _, sources, _, _ = self._prepare_image(template, gt_shape,
                                                          gt_shape=gt_shape)
        return templates, sources

    def _fitter_result(self, image, algorithm_results, affine_transforms,
                       scale_transforms, gt_shape=None):
        r"""
        Function the creates the multi-scale fitting result object.

        Parameters
        ----------
        image : `menpo.image.Image` or subclass
            The image that was fitted.
        algorithm_results : `list` of :map:`LucasKanadeAlgorithmResult` or subclass
            The list of fitting result per scale.
        affine_transforms : `list` of `menpo.transform.Affine`
            The list of affine transforms per scale that are the inverses of the
            transformations introduced by the rescale wrt the reference shape as
            well as the feature extraction.
        scale_transforms : `list` of `menpo.shape.Scale`
            The list of inverse scaling transforms per scale.
        gt_shape : `menpo.shape.PointCloud`, optional
            The ground truth shape associated to the image.

        Returns
        -------
        fitting_result : :map:`LucasKanadeResult` or subclass
            The multi-scale fitting result containing the result of the fitting
            procedure.
        """
        return LucasKanadeResult(
            results=algorithm_results, scales=self.scales,
            affine_transforms=affine_transforms,
            scale_transforms=scale_transforms, image=image, gt_shape=gt_shape)

    def warped_images(self, image, shapes):
        r"""
        Given an input test image and a list of shapes, it warps the image
        into the shapes. This is useful for generating the warped images of a
        fitting procedure stored within a :map:`LucasKanadeResult`.

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
        return self.algorithms[-1].warped_images(image=image, shapes=shapes)

    def __str__(self):
        if self.diagonal is not None:
            diagonal = self.diagonal
        else:
            y, x = self.reference_shape.range()
            diagonal = np.sqrt(x ** 2 + y ** 2)

        # Compute scale info strings
        scales_info = []
        lvl_str_tmplt = r"""   - Scale {}
     - Holistic feature: {}
     - Template shape: {}"""
        for k, s in enumerate(self.scales):
            scales_info.append(lvl_str_tmplt.format(
                    s, name_of_callable(self.holistic_features[k]),
                    self.templates[k].shape))
        scales_info = '\n'.join(scales_info)

        cls_str = r"""Lucas-Kanade {class_title}
 - {residual}
 - Images warped with {transform} transform
 - Images scaled to diagonal: {diagonal:.2f}
 - Scales: {scales}
{scales_info}
""".format(class_title=self.algorithms[0],
           residual=self.algorithms[0].residual,
           transform=name_of_callable(self.transform_cls),
           diagonal=diagonal,
           scales=self.scales,
           scales_info=scales_info)
        return cls_str
