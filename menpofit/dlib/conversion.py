import dlib
import numpy as np

from menpo.shape import PointCloud, bounding_box


all_parts = lambda det: ((p.y, p.x) for p in det.parts())


def pointcloud_to_dlib_parts(pcloud):
    return [dlib.point(int(p[1]), int(p[0])) for p in pcloud.points]


def dlib_full_object_detection_to_pointcloud(full_object_detection):
    return PointCloud(np.vstack(all_parts(full_object_detection)), copy=False)


def dlib_rect_to_bounding_box(rect):
    return bounding_box((rect.top(), rect.left()),
                        (rect.bottom(), rect.right()))


def pointcloud_to_dlib_rect(pg):
    min_p, max_p = pg.bounds()
    return dlib.rectangle(left=int(min_p[1]), top=int(min_p[0]),
                          right=int(max_p[1]), bottom=int(max_p[0]))


def bounding_box_pointcloud_to_dlib_fo_detection(bbox, pcloud):
    return dlib.full_object_detection(
        pointcloud_to_dlib_rect(bbox.bounding_box()),
        pointcloud_to_dlib_parts(pcloud))


def copy_dlib_options(options):
    new_options = dlib.shape_predictor_training_options()
    for p in sorted(filter(lambda x: '__' not in x, dir(options))):
        setattr(new_options, p, getattr(options, p))
    return new_options


def image_to_dlib_pixels(im):
    pixels = np.array(im.as_PILImage())
    # Only supports RGB and Grayscale
    if im.n_channels > 3:
        pixels = pixels[..., 0]
    return pixels
