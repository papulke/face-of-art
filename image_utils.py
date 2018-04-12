import numpy as np
import os
from scipy.io import loadmat
import cv2
from menpo.shape.pointcloud import PointCloud
import menpo.io as mio
import matplotlib.pyplot as plt
from scipy.ndimage import zoom


'''********* bounding box and image loading functions *********'''


def center_margin_bb(bb, img_bounds, margin=0.25):
    bb_size = ([bb[0, 2] - bb[0, 0], bb[0, 3] - bb[0, 1]])
    margins = (np.max(bb_size) * (1 + margin) - bb_size) / 2

    bb_new = np.zeros_like(bb)
    bb_new[0, 0] = np.maximum(bb[0, 0] - margins[0], 0)
    bb_new[0, 2] = np.minimum(bb[0, 2] + margins[0], img_bounds[1])
    bb_new[0, 1] = np.maximum(bb[0, 1] - margins[1], 0)
    bb_new[0, 3] = np.minimum(bb[0, 3] + margins[1], img_bounds[0])
    return bb_new


def load_bb_files(bb_file_dirs):
    bb_files_dict = {}
    for bb_file in bb_file_dirs:
        bb_mat = loadmat(bb_file)['bounding_boxes']
        num_imgs = np.max(bb_mat.shape)
        for i in range(num_imgs):
            name = bb_mat[0][i][0][0][0][0]
            bb_init = bb_mat[0][i][0][0][1] - 1  # matlab indicies
            bb_gt = bb_mat[0][i][0][0][2] - 1  # matlab indicies
            if str(name) in bb_files_dict.keys():
                print str(name), 'already loaded from: ', bb_file
            bb_files_dict[str(name)] = (bb_init, bb_gt)
    return bb_files_dict


def load_bb_dictionary(bb_dir, mode, test_data='full'):
    if mode is 'TRAIN':
        bb_dirs = \
            ['bounding_boxes_afw.mat', 'bounding_boxes_helen_trainset.mat', 'bounding_boxes_lfpw_trainset.mat']
    else:
        if test_data is 'common':
            bb_dirs = \
                ['bounding_boxes_helen_testset.mat', 'bounding_boxes_lfpw_testset.mat']
        elif test_data is 'challenging':
            bb_dirs = ['bounding_boxes_ibug.mat']
        elif test_data is 'full':
            bb_dirs = \
                ['bounding_boxes_ibug.mat', 'bounding_boxes_helen_testset.mat', 'bounding_boxes_lfpw_testset.mat']

    if mode is 'TEST' and test_data is 'test':
        bb_files_dict = None
    else:
        bb_dirs = [os.path.join(bb_dir, dataset) for dataset in bb_dirs]
        bb_files_dict = load_bb_files(bb_dirs)

    return bb_files_dict


def crop_to_face_image(img, bb_dictionary=None, gt=True, margin=0.25, image_size=256):
    name = img.path.name
    img_bounds = img.bounds()[1]

    if bb_dictionary is None:
        bb_menpo = img.landmarks['PTS'].bounding_box().points
        bb = np.array([[bb_menpo[0, 1], bb_menpo[0, 0], bb_menpo[2, 1], bb_menpo[2, 0]]])
    else:
        if gt:
            bb = bb_dictionary[name][1]  # ground truth
        else:
            bb = bb_dictionary[name][0]  # init from face detector

    bb = center_margin_bb(bb, img_bounds, margin=margin)

    bb_pointcloud = PointCloud(np.array([[bb[0, 1], bb[0, 0]],
                                         [bb[0, 3], bb[0, 0]],
                                         [bb[0, 3], bb[0, 2]],
                                         [bb[0, 1], bb[0, 2]]]))

    face_crop = img.crop_to_pointcloud(bb_pointcloud).resize([image_size, image_size])

    return face_crop


def augment_face_image(img, image_size=256, crop_size=248, angle_range=30, flip=True):
    lim = image_size - crop_size
    min_crop_inds = np.random.randint(0, lim, 2)
    max_crop_inds = min_crop_inds + crop_size
    flip_rand = np.random.random() > 0.5
    rot_angle = 2 * angle_range * np.random.random_sample() - angle_range

    if flip and flip_rand:
        rand_crop = img.crop(min_crop_inds, max_crop_inds).mirror(). \
            rotate_ccw_about_centre(rot_angle).resize([image_size, image_size])
    else:
        rand_crop = img.crop(min_crop_inds, max_crop_inds). \
            rotate_ccw_about_centre(rot_angle).resize([image_size, image_size])

    return rand_crop


def load_menpo_image_list(img_dir, mode, bb_dictionary=None, image_size=256, margin=0.25, bb_type='gt',
                          test_data='full'):
    def crop_to_face_image_gt(img, bb_dictionary=bb_dictionary, margin=margin, image_size=image_size):
        return crop_to_face_image(img, bb_dictionary, gt=True, margin=margin, image_size=image_size)

    def crop_to_face_image_init(img, bb_dictionary=bb_dictionary, margin=margin, image_size=image_size):
        return crop_to_face_image(img, bb_dictionary, gt=False, margin=margin, image_size=image_size)

    if mode is 'TRAIN':
        img_set_dir = os.path.join(img_dir, 'training_set')

    else:
        img_set_dir = os.path.join(img_dir, test_data + '_set')

    image_menpo_list = mio.import_images(img_set_dir, verbose=True)

    if bb_type is 'gt':
        face_crop_image_list = image_menpo_list.map(crop_to_face_image_gt)
    else:
        face_crop_image_list = image_menpo_list.map(crop_to_face_image_init)

    if mode is 'TRAIN':
        out_image_list = face_crop_image_list.map(augment_face_image)
    else:
        out_image_list = face_crop_image_list

    return out_image_list


'''********* heat-maps and image loading functions *********'''


# look for: ECT-FaceAlignment/caffe/src/caffe/layers/data_heatmap.cpp
def gaussian(x, y, x0, y0, sigma=6):
    return 1./(np.sqrt(2*np.pi)*sigma) * np.exp(-0.5 * ((x-x0)**2 + (y-y0)**2) / sigma**2)


def create_heat_maps(landmarks, num_landmarks=68, image_size=256, sigma=6):

    x, y = np.mgrid[0:image_size, 0:image_size]

    maps = np.zeros((image_size, image_size, num_landmarks))

    for i in range(num_landmarks):
        out = gaussian(x, y, landmarks[i,0], landmarks[i,1], sigma=sigma)
        maps[:, :, i] = (8./3)*sigma*out  # copied from ECT

    return maps


def load_data(img_list, batch_inds, image_size=256, c_dim=3, num_landmarks=68, sigma=6, scale='255', save_landmarks=False):

    num_inputs = len(batch_inds)
    batch_menpo_images = img_list[batch_inds]

    images = np.zeros([num_inputs, image_size, image_size, c_dim]).astype('float32')
    maps = np.zeros([num_inputs, image_size, image_size, num_landmarks]).astype('float32')
    maps_small = np.zeros([num_inputs, image_size / 4, image_size / 4, num_landmarks]).astype('float32')

    if save_landmarks:
        landmarks = np.zeros([num_inputs, num_landmarks, 2]).astype('float32')
    else:
        landmarks = None

    for ind, img in enumerate(batch_menpo_images):
        lms = img.landmarks['PTS'].points
        images[ind, :, :, :] = np.rollaxis(img.pixels, 0, 3)
        maps[ind, :, :, :] = create_heat_maps(lms, num_landmarks, image_size, sigma)
        maps_small[ind, :, :, :] = zoom(maps[ind, :, :, :], (0.25, 0.25, 1))
        if save_landmarks:
            landmarks[ind, :, :] = lms

    if scale is '255':
        images *= 255  # SAME AS ECT?
    elif scale is 'zero_center':
        images = 2 * images - 1

    return images, maps, maps_small, landmarks


def heat_maps_to_image(maps, landmarks, image_size=256):

    x, y = np.mgrid[0:image_size, 0:image_size]

    pixel_dist = np.sqrt(
        np.square(np.expand_dims(x, 2) - landmarks[:, 0]) + np.square(np.expand_dims(y, 2) - landmarks[:, 1]))

    nn_landmark = np.argmin(pixel_dist, 2)

    map_image = maps[x, y, nn_landmark]

    return map_image


def heat_maps_to_image(maps, landmarks=None, image_size=256, num_landmarks=68):

    if landmarks is None:
        landmarks = heat_maps_to_landmarks(maps, image_size=image_size, num_landmarks=num_landmarks)

    x, y = np.mgrid[0:image_size, 0:image_size]

    pixel_dist = np.sqrt(
        np.square(np.expand_dims(x, 2) - landmarks[:, 0]) + np.square(np.expand_dims(y, 2) - landmarks[:, 1]))

    nn_landmark = np.argmin(pixel_dist, 2)

    map_image = maps[x, y, nn_landmark]

    return map_image


def heat_maps_to_landmarks(maps, image_size=256, num_landmarks=68):

    landmarks = np.zeros((num_landmarks,2)).astype('float32')

    for m_ind in range(num_landmarks):
        landmarks[m_ind, :] = np.unravel_index(maps[:, :, m_ind].argmax(), (image_size, image_size))

    return landmarks


def print_training_params_to_file(init_locals):
    del init_locals['self']
    with open(os.path.join(init_locals['save_log_path'], 'Training_Parameters.txt'), 'w') as f:
        f.write('Training Parameters:\n\n')
        for key, value in init_locals.items():
            f.write('* %s: %s\n' % (key, value))


def create_img_with_landmarks(image, landmarks, image_size=256, num_landmarks=68, scale='255'):
    image = image.reshape(image_size, image_size, -1)

    if scale is 'zero_center':
        image = 127.5 * (image + 1)
    elif scale is '1':
        image *= 255

    landmarks = landmarks.reshape(num_landmarks, 2)
    landmarks = np.clip(landmarks, 0, image_size)

    for (y, x) in landmarks.astype('int'):
        cv2.circle(image, (x, y), 1, (255, 0, 0), -1)

    return image


def merge_images_landmarks_maps(images, maps, image_size=256, num_landmarks=68, num_samples=9, scale='255'):

    images = images[:num_samples]

    row = int(np.sqrt(num_samples))
    merged = np.zeros([row * image_size, row * image_size * 2, 3])

    cmap = plt.get_cmap('jet')

    for idx, img in enumerate(images):
        i = idx // row
        j = idx % row

        img_lamdmarks = heat_maps_to_landmarks(maps[idx, :, :, :], image_size=image_size, num_landmarks=num_landmarks)
        map_image = heat_maps_to_image(maps[idx, :, :, :], img_lamdmarks, image_size=image_size,
                                       num_landmarks=num_landmarks)

        rgba_map_image = cmap(map_image)
        map_image = np.delete(rgba_map_image, 3, 2) * 255

        img = create_img_with_landmarks(img, img_lamdmarks, image_size, num_landmarks, scale=scale)

        merged[i * image_size:(i + 1) * image_size, (j * 2) * image_size:(j * 2 + 1) * image_size, :] = img
        merged[i * image_size:(i + 1) * image_size, (j * 2 + 1) * image_size:(j * 2 + 2) * image_size, :] = map_image

    return merged


def merge_compare_maps(maps_small, maps, image_size=64, num_landmarks=68, num_samples=9):
    maps_small = maps_small[:num_samples]
    maps = maps[:num_samples]
    maps_rescale = zoom(maps, (1, 0.25, 0.25, 1))
    cmap = plt.get_cmap('jet')

    row = int(np.sqrt(num_samples))
    merged = np.zeros([row * image_size, row * image_size * 2, 3])

    for idx, map_small in enumerate(maps_small):
        i = idx // row
        j = idx % row

        map_image_small = heat_maps_to_image(map_small, image_size=image_size, num_landmarks=num_landmarks)
        map_image = heat_maps_to_image(maps_rescale[idx, :, :, :], image_size=image_size, num_landmarks=num_landmarks)

        rgba_map_image = cmap(map_image)
        map_image = np.delete(rgba_map_image, 3, 2) * 255

        rgba_map_image_small = cmap(map_image_small)
        map_image_small = np.delete(rgba_map_image_small, 3, 2) * 255

        merged[i * image_size:(i + 1) * image_size, (j * 2) * image_size:(j * 2 + 1) * image_size, :] = map_image_small
        merged[i * image_size:(i + 1) * image_size, (j * 2 + 1) * image_size:(j * 2 + 2) * image_size, :] = map_image

    return merged