from menpo_functions import *
from data_loading_functions import *


def augment_menpo_img_ns(img, img_dir_ns, p_ns=0, ns_ind=None):
    """texture style image augmentation using stylized copies in *img_dir_ns*"""

    if p_ns > 0.5:
        ns_augs = glob(os.path.join(img_dir_ns, img.path.name.split('.')[0] + '*'))
        num_augs = len(ns_augs)
        if num_augs > 0:
            if ns_ind is None or ns_ind >= num_augs:
                ns_ind = np.random.randint(0, num_augs)
            ns_aug = mio.import_image(ns_augs[ns_ind])
            img.pixels = ns_aug.pixels
    return img


def augment_menpo_img_ns_dont_apply(img, img_dir_ns, p_ns=0, ns_ind=None):
    """texture style image augmentation using stylized copies in *img_dir_ns*"""

    if p_ns > 0.5:
        ns_augs = glob(os.path.join(img_dir_ns, img.path.name.split('.')[0] + '*'))
        num_augs = len(ns_augs)
        if num_augs > 0:
            if ns_ind is None or ns_ind >= num_augs:
                ns_ind = np.random.randint(0, num_augs)
            # ns_aug = mio.import_image(ns_augs[ns_ind])
            # ns_pixels = ns_aug.pixels
    return img


def augment_menpo_img_geom_dont_apply(img, p_geom=0):
    """geometric style image augmentation using random face deformations"""

    if p_geom > 0.5:
        lms_geom_warp = deform_face_geometric_style(img.landmarks['PTS'].points.copy(), p_scale=p_geom, p_shift=p_geom)
    return img


def augment_menpo_img_geom(img, p_geom=0):
    """geometric style image augmentation using random face deformations"""

    if p_geom > 0.5:
        lms_geom_warp = deform_face_geometric_style(img.landmarks['PTS'].points.copy(), p_scale=p_geom, p_shift=p_geom)
        img=warp_face_image_tps(img, PointCloud(lms_geom_warp))
    return img


def load_menpo_image_list(
        img_dir, train_crop_dir, img_dir_ns, mode, bb_dictionary=None, image_size=256, margin=0.25,
        bb_type='gt', test_data='full', augment_basic=True, augment_texture=False, p_texture=0,
        augment_geom=False, p_geom=0, verbose=False,ns_ind=None, dataset='training'):

    def crop_to_face_image_gt(img):
        return crop_to_face_image(img, bb_dictionary, gt=True, margin=margin, image_size=image_size)

    def crop_to_face_image_init(img):
        return crop_to_face_image(img, bb_dictionary, gt=False, margin=margin, image_size=image_size)

    def augment_menpo_img_ns_rand(img):
        return augment_menpo_img_ns(img, img_dir_ns, p_ns=1. * (np.random.rand() < p_texture),ns_ind=ns_ind)

    def augment_menpo_img_geom_rand(img):
        return augment_menpo_img_geom(img, p_geom=1. * (np.random.rand() < p_geom))

    if mode is 'TRAIN':
        if train_crop_dir is None:
            img_set_dir = os.path.join(img_dir, dataset+'_set')
            out_image_list = mio.import_images(img_set_dir, verbose=verbose, normalize=False)
            if bb_type is 'gt':
                out_image_list = out_image_list.map(crop_to_face_image_gt)
            elif bb_type is 'init':
                out_image_list = out_image_list.map(crop_to_face_image_init)
        else:
            img_set_dir = os.path.join(img_dir, train_crop_dir)
            out_image_list = mio.import_images(img_set_dir, verbose=verbose)

        if augment_texture and img_dir_ns is not None:
            out_image_list = out_image_list.map(augment_menpo_img_ns_rand)
        if augment_geom:
            out_image_list = out_image_list.map(augment_menpo_img_geom_rand)
        if augment_basic:
            out_image_list = out_image_list.map(augment_face_image)

    else:
        img_set_dir = os.path.join(img_dir, test_data + '_set')
        if test_data in ['full', 'challenging', 'common', 'training', 'test']:
            out_image_list = mio.import_images(img_set_dir, verbose=verbose, normalize=False)
            if bb_type is 'gt':
                out_image_list = out_image_list.map(crop_to_face_image_gt)
            elif bb_type is 'init':
                out_image_list = out_image_list.map(crop_to_face_image_init)
        else:
            out_image_list = mio.import_images(img_set_dir, verbose=verbose)

    return out_image_list


def load_menpo_image_list_no_geom(
        img_dir, train_crop_dir, img_dir_ns, mode, bb_dictionary=None, image_size=256, margin=0.25,
        bb_type='gt', test_data='full', augment_basic=True, augment_texture=False, p_texture=0,
        augment_geom=False, p_geom=0, verbose=False,ns_ind=None, dataset='training'):

    def crop_to_face_image_gt(img):
        return crop_to_face_image(img, bb_dictionary, gt=True, margin=margin, image_size=image_size)

    def crop_to_face_image_init(img):
        return crop_to_face_image(img, bb_dictionary, gt=False, margin=margin, image_size=image_size)

    def augment_menpo_img_ns_rand(img):
        return augment_menpo_img_ns(img, img_dir_ns, p_ns=1. * (np.random.rand() < p_texture),ns_ind=ns_ind)

    def augment_menpo_img_geom_rand(img):
        return augment_menpo_img_geom_dont_apply(img, p_geom=1. * (np.random.rand() < p_geom))

    if mode is 'TRAIN':
        if train_crop_dir is None:
            img_set_dir = os.path.join(img_dir, dataset+'_set')
            out_image_list = mio.import_images(img_set_dir, verbose=verbose, normalize=False)
            if bb_type is 'gt':
                out_image_list = out_image_list.map(crop_to_face_image_gt)
            elif bb_type is 'init':
                out_image_list = out_image_list.map(crop_to_face_image_init)
        else:
            img_set_dir = os.path.join(img_dir, train_crop_dir)
            out_image_list = mio.import_images(img_set_dir, verbose=verbose)

        if augment_texture and img_dir_ns is not None:
            out_image_list = out_image_list.map(augment_menpo_img_ns_rand)
        if augment_geom:
            out_image_list = out_image_list.map(augment_menpo_img_geom_rand)
        if augment_basic:
            out_image_list = out_image_list.map(augment_face_image)

    else:
        img_set_dir = os.path.join(img_dir, test_data + '_set')
        if test_data in ['full', 'challenging', 'common', 'training', 'test']:
            out_image_list = mio.import_images(img_set_dir, verbose=verbose, normalize=False)
            if bb_type is 'gt':
                out_image_list = out_image_list.map(crop_to_face_image_gt)
            elif bb_type is 'init':
                out_image_list = out_image_list.map(crop_to_face_image_init)
        else:
            out_image_list = mio.import_images(img_set_dir, verbose=verbose)

    return out_image_list


def load_menpo_image_list_no_texture(
        img_dir, train_crop_dir, img_dir_ns, mode, bb_dictionary=None, image_size=256, margin=0.25,
        bb_type='gt', test_data='full', augment_basic=True, augment_texture=False, p_texture=0,
        augment_geom=False, p_geom=0, verbose=False,ns_ind=None, dataset='training'):

    def crop_to_face_image_gt(img):
        return crop_to_face_image(img, bb_dictionary, gt=True, margin=margin, image_size=image_size)

    def crop_to_face_image_init(img):
        return crop_to_face_image(img, bb_dictionary, gt=False, margin=margin, image_size=image_size)

    def augment_menpo_img_ns_rand(img):
        return augment_menpo_img_ns_dont_apply(img, img_dir_ns, p_ns=1. * (np.random.rand() < p_texture),ns_ind=ns_ind)

    def augment_menpo_img_geom_rand(img):
        return augment_menpo_img_geom(img, p_geom=1. * (np.random.rand() < p_geom))

    if mode is 'TRAIN':
        if train_crop_dir is None:
            img_set_dir = os.path.join(img_dir, dataset+'_set')
            out_image_list = mio.import_images(img_set_dir, verbose=verbose, normalize=False)
            if bb_type is 'gt':
                out_image_list = out_image_list.map(crop_to_face_image_gt)
            elif bb_type is 'init':
                out_image_list = out_image_list.map(crop_to_face_image_init)
        else:
            img_set_dir = os.path.join(img_dir, train_crop_dir)
            out_image_list = mio.import_images(img_set_dir, verbose=verbose)

        if augment_texture and img_dir_ns is not None:
            out_image_list = out_image_list.map(augment_menpo_img_ns_rand)
        if augment_geom:
            out_image_list = out_image_list.map(augment_menpo_img_geom_rand)
        if augment_basic:
            out_image_list = out_image_list.map(augment_face_image)

    else:
        img_set_dir = os.path.join(img_dir, test_data + '_set')
        if test_data in ['full', 'challenging', 'common', 'training', 'test']:
            out_image_list = mio.import_images(img_set_dir, verbose=verbose, normalize=False)
            if bb_type is 'gt':
                out_image_list = out_image_list.map(crop_to_face_image_gt)
            elif bb_type is 'init':
                out_image_list = out_image_list.map(crop_to_face_image_init)
        else:
            out_image_list = mio.import_images(img_set_dir, verbose=verbose)

    return out_image_list


def load_menpo_image_list_no_artistic(
        img_dir, train_crop_dir, img_dir_ns, mode, bb_dictionary=None, image_size=256, margin=0.25,
        bb_type='gt', test_data='full', augment_basic=True, augment_texture=False, p_texture=0,
        augment_geom=False, p_geom=0, verbose=False,ns_ind=None, dataset='training'):

    def crop_to_face_image_gt(img):
        return crop_to_face_image(img, bb_dictionary, gt=True, margin=margin, image_size=image_size)

    def crop_to_face_image_init(img):
        return crop_to_face_image(img, bb_dictionary, gt=False, margin=margin, image_size=image_size)

    def augment_menpo_img_ns_rand(img):
        return augment_menpo_img_ns_dont_apply(img, img_dir_ns, p_ns=1. * (np.random.rand() < p_texture),ns_ind=ns_ind)

    def augment_menpo_img_geom_rand(img):
        return augment_menpo_img_geom_dont_apply(img, p_geom=1. * (np.random.rand() < p_geom))

    if mode is 'TRAIN':
        if train_crop_dir is None:
            img_set_dir = os.path.join(img_dir, dataset+'_set')
            out_image_list = mio.import_images(img_set_dir, verbose=verbose, normalize=False)
            if bb_type is 'gt':
                out_image_list = out_image_list.map(crop_to_face_image_gt)
            elif bb_type is 'init':
                out_image_list = out_image_list.map(crop_to_face_image_init)
        else:
            img_set_dir = os.path.join(img_dir, train_crop_dir)
            out_image_list = mio.import_images(img_set_dir, verbose=verbose)

        if augment_texture and img_dir_ns is not None:
            out_image_list = out_image_list.map(augment_menpo_img_ns_rand)
        if augment_geom:
            out_image_list = out_image_list.map(augment_menpo_img_geom_rand)
        if augment_basic:
            out_image_list = out_image_list.map(augment_face_image)

    else:
        img_set_dir = os.path.join(img_dir, test_data + '_set')
        if test_data in ['full', 'challenging', 'common', 'training', 'test']:
            out_image_list = mio.import_images(img_set_dir, verbose=verbose, normalize=False)
            if bb_type is 'gt':
                out_image_list = out_image_list.map(crop_to_face_image_gt)
            elif bb_type is 'init':
                out_image_list = out_image_list.map(crop_to_face_image_init)
        else:
            out_image_list = mio.import_images(img_set_dir, verbose=verbose)

    return out_image_list