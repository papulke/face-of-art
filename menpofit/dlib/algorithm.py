from __future__ import division
import dlib

from menpo.visualize import print_dynamic

from menpofit.fitter import raise_costs_warning
from menpofit.result import NonParametricIterativeResult

from .conversion import (copy_dlib_options, pointcloud_to_dlib_rect,
                         bounding_box_pointcloud_to_dlib_fo_detection,
                         dlib_full_object_detection_to_pointcloud,
                         image_to_dlib_pixels)


class DlibAlgorithm(object):
    r"""
    DLib wrapper class for fitting a multi-scale Ensemble of Regression Trees
    model.

    Parameters
    ----------
    dlib_options : `list`
        List of the options expected by DLib, as specified in the
        `dlib.shape_predictor_training_options` method. Specifically:

        =========================== ============================================
        Value                       Description
        =========================== ============================================
        be_verbose                  Flag to enable info printing
        cascade_depth               Number of cascades
        feature_pool_region_padding Size of region within which to get features
        feature_pool_size           Number of pixels used to generate features
        lambda_param feature        Controls how tight the feature sampling is
        nu                          Regularisation parameter
        num_test_splits             Number of split features to sample per node
        num_trees_per_cascade_level The number of trees created for each cascade
        oversampling_amount         The number of random initialisations
        random_seed                 Seed of random number generator
        tree_depth                  The depth of the trees used in each cascade
        =========================== ============================================

    n_iterations : `int`, optional
        Number of iterations (cascades).
    """
    def __init__(self, dlib_options, n_iterations=10):
        self.dlib_model = None
        self._n_iterations = n_iterations
        self.dlib_options = copy_dlib_options(dlib_options)
        # T from Kazemi paper - Total number of cascades
        self.dlib_options.cascade_depth = int(self.n_iterations)

    @property
    def n_iterations(self):
        """
        Returns the number of iterations (cascades).

        :type: `int`
        """
        return self._n_iterations

    @n_iterations.setter
    def n_iterations(self, v):
        """
        Sets the number of iterations (cascades).

        Parameters
        ----------
        v : `int`
            The new number of iterations.
        """
        self._n_iterations = v
        # T from Kazemi paper - Total number of cascades
        self.dlib_options.cascade_depth = self._n_iterations

    def train(self, images, gt_shapes, bounding_boxes, prefix='',
              verbose=False):
        r"""
        Train an algorithm.

        Parameters
        ----------
        images : `list` of `menpo.image.Image`
            The `list` of training images.
        gt_shapes : `list` of `menpo.shape.PointCloud`
            The `list` of ground truth shapes that correspond to the images.
        bounding_boxes : `list` of `list` of `menpo.shape.PointDirectedGraph`
            The `list` of `list` of perturbed bounding boxes per image.
        prefix : `str`, optional
            Prefix str for verbose.
        verbose : `bool`, optional
            If ``True``, then information about the training is printed.
        """
        if verbose and self.dlib_options.oversampling_amount > 1:
            n_menpofit_peturbations = len(bounding_boxes[0])
            n_dlib_perturbations = self.dlib_options.oversampling_amount
            total_perturbations = (n_menpofit_peturbations *
                                   n_dlib_perturbations)
            print_dynamic('{}WARNING: Dlib oversampling is being used. '
                          '{} = {} * {} total perturbations will be generated '
                          'by Dlib!\n'.format(prefix, total_perturbations,
                                              n_menpofit_peturbations,
                                              n_dlib_perturbations))

        im_pixels = [image_to_dlib_pixels(im) for im in images]

        detections = []
        for bboxes, im, gt_s in zip(bounding_boxes, images, gt_shapes):
            fo_dets = [bounding_box_pointcloud_to_dlib_fo_detection(bb, gt_s)
                       for bb in bboxes]
            detections.append(fo_dets)

        if verbose:
            print_dynamic('{}Performing Dlib training - please see stdout '
                          'for verbose output provided by Dlib!'.format(prefix))

        # Perform DLIB training
        self.dlib_options.be_verbose = verbose
        self.dlib_model = dlib.train_shape_predictor(
            im_pixels, detections, self.dlib_options)

        for bboxes, pix, fo_dets in zip(bounding_boxes, im_pixels, detections):
            for bb, fo_det in zip(bboxes, fo_dets):
                # Perform prediction
                pred = dlib_full_object_detection_to_pointcloud(
                    self.dlib_model(pix, fo_det.rect))
                # Update bounding box in place
                bb._from_vector_inplace(pred.bounding_box().as_vector())

        if verbose:
            print_dynamic('{}Training Dlib done.\n'.format(prefix))

        return bounding_boxes

    def run(self, image, bounding_box, gt_shape=None, return_costs=False,
            **kwargs):
        r"""
        Run the predictor to an image given an initial bounding box.

        Parameters
        ----------
        image : `menpo.image.Image` or subclass
            The image to be fitted.
        bounding_box : `menpo.shape.PointDirectedGraph`
            The initial bounding box from which the fitting procedure
            will start.
        gt_shape : `menpo.shape.PointCloud` or ``None``, optional
            The ground truth shape associated to the image.
        return_costs : `bool`, optional
            If ``True``, then the cost function values will be computed
            during the fitting procedure. Then these cost values will be
            assigned to the returned `fitting_result`. *Note that this
            argument currently has no effect and will raise a warning if set
            to ``True``. This is because it is not possible to evaluate the
            cost function of this algorithm.*

        Returns
        -------
        fitting_result: `menpofit.result.NonParametricIterativeResult`
            The result of the fitting procedure.
        """
        # costs warning
        if return_costs:
            raise_costs_warning(self)

        # Perform prediction
        pix = image_to_dlib_pixels(image)
        rect = pointcloud_to_dlib_rect(bounding_box)
        pred = dlib_full_object_detection_to_pointcloud(
                self.dlib_model(pix, rect))
        return NonParametricIterativeResult(
            shapes=[pred], initial_shape=None, image=image, gt_shape=gt_shape)
