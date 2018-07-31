import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from scipy.misc import imresize


def print_training_params_to_file(init_locals):
    del init_locals['self']
    with open(os.path.join(init_locals['save_log_path'], 'Training_Parameters.txt'), 'w') as f:
        f.write('Training Parameters:\n\n')
        for key, value in init_locals.items():
            f.write('* %s: %s\n' % (key, value))


# heat maps to landmarks without pre-allocation

def heat_maps_to_landmarks(maps, image_size=256, num_landmarks=68):

    landmarks = np.zeros((num_landmarks,2)).astype('float32')

    for m_ind in range(num_landmarks):
        landmarks[m_ind, :] = np.unravel_index(maps[:, :, m_ind].argmax(), (image_size, image_size))

    return landmarks


def batch_heat_maps_to_landmarks(batch_maps, batch_size, image_size=256, num_landmarks=68):
    batch_landmarks = np.zeros((batch_size,num_landmarks, 2)).astype('float32')
    for i in range(batch_size):

        heat_maps_to_landmarks_alloc_once(
            maps=batch_maps[i, :, :, :], landmarks=batch_landmarks[i, :, :], image_size=image_size,
            num_landmarks=num_landmarks)

    return batch_landmarks


# heat maps to landmarks with pre-allocation

def heat_maps_to_landmarks_alloc_once(maps, landmarks, image_size=256, num_landmarks=68):

    for m_ind in range(num_landmarks):
        landmarks[m_ind, :] = np.unravel_index(maps[:, :, m_ind].argmax(), (image_size, image_size))


def batch_heat_maps_to_landmarks_alloc_once(batch_maps, batch_landmarks, batch_size, image_size=256, num_landmarks=68):

    for i in range(batch_size):
        heat_maps_to_landmarks_alloc_once(
            maps=batch_maps[i, :, :, :], landmarks=batch_landmarks[i, :, :], image_size=image_size,
            num_landmarks=num_landmarks)


# sample images without pre-allocation

def normalize_map(map_in):
    map_min = map_in.min()
    return (map_in - map_min) / (map_in.max() - map_min)


def map_to_rgb(map_gray):
    cmap = plt.get_cmap('jet')
    rgba_map_image = cmap(map_gray)
    map_rgb = np.delete(rgba_map_image, 3, 2) * 255
    return map_rgb


def create_img_with_landmarks(image, landmarks, image_size=256, num_landmarks=68, scale=255, circle_size=2):
    image = image.reshape(image_size, image_size, -1)

    if scale is 0:
        image = 127.5 * (image + 1)
    elif scale is 1:
        image *= 255

    landmarks = landmarks.reshape(num_landmarks, 2)
    landmarks = np.clip(landmarks, 0, image_size-1)

    for (y, x) in landmarks.astype('int'):
        cv2.circle(image, (x, y), circle_size, (255, 0, 0), -1)

    return image


def heat_maps_to_image(maps, landmarks=None, image_size=256, num_landmarks=68):

    if landmarks is None:
        landmarks = heat_maps_to_landmarks(maps, image_size=image_size, num_landmarks=num_landmarks)

    x, y = np.mgrid[0:image_size, 0:image_size]

    pixel_dist = np.sqrt(
        np.square(np.expand_dims(x, 2) - landmarks[:, 0]) + np.square(np.expand_dims(y, 2) - landmarks[:, 1]))

    nn_landmark = np.argmin(pixel_dist, 2)

    map_image = maps[x, y, nn_landmark]
    map_image = (map_image-map_image.min())/(map_image.max()-map_image.min())  # normalize for visualization

    return map_image


def merge_images_landmarks_maps_gt(images, maps, maps_gt, landmarks=None, image_size=256, num_landmarks=68,
                                   num_samples=9, scale=255, circle_size=2, test_data='full', fast=False):

    images = images[:num_samples]
    if maps.shape[1] is not image_size:
        images = zoom(images, (1, 0.25, 0.25, 1))
        image_size /= 4
        image_size=int(image_size)
    if maps_gt.shape[1] is not image_size:
        maps_gt = zoom(maps_gt, (1, 0.25, 0.25, 1))

    cmap = plt.get_cmap('jet')

    row = int(np.sqrt(num_samples))
    merged = np.zeros([row * image_size, row * image_size * 3, 3])

    for idx, img in enumerate(images):
        i = idx // row
        j = idx % row

        if landmarks is None:
            img_landmarks = heat_maps_to_landmarks(maps[idx, :, :, :], image_size=image_size,
                                                   num_landmarks=num_landmarks)
        else:
            img_landmarks = landmarks[idx]

        if fast:
            map_image = np.amax(maps[idx, :, :, :], 2)
            map_image = (map_image - map_image.min()) / (map_image.max() - map_image.min())
        else:
            map_image = heat_maps_to_image(maps[idx, :, :, :], img_landmarks, image_size=image_size,
                                           num_landmarks=num_landmarks)
        rgba_map_image = cmap(map_image)
        map_image = np.delete(rgba_map_image, 3, 2) * 255

        if test_data not in ['full', 'challenging', 'common', 'training', 'test']:
            map_gt_image = map_image.copy()
        else:
            if fast:
                map_gt_image = np.amax(maps_gt[idx, :, :, :], 2)
                map_gt_image = (map_gt_image - map_gt_image.min()) / (map_gt_image.max() - map_gt_image.min())
            else:
                map_gt_image = heat_maps_to_image(maps_gt[idx, :, :, :], image_size=image_size,
                                                  num_landmarks=num_landmarks)
            rgba_map_gt_image = cmap(map_gt_image)
            map_gt_image = np.delete(rgba_map_gt_image, 3, 2) * 255

        img = create_img_with_landmarks(img, img_landmarks, image_size, num_landmarks, scale=scale,
                                        circle_size=circle_size)

        merged[i * image_size:(i + 1) * image_size, (j * 3) * image_size:(j * 3 + 1) * image_size, :] = img
        merged[i * image_size:(i + 1) * image_size, (j * 3 + 1) * image_size:(j * 3 + 2) * image_size,
        :] = map_image
        merged[i * image_size:(i + 1) * image_size, (j * 3 + 2) * image_size:(j * 3 + 3) * image_size,
        :] = map_gt_image

    return merged


def map_comapre_channels(images, maps1, maps2, image_size=64, num_landmarks=68, scale=255, test_data='full'):
        map1 = maps1[0]
        map2 = maps2[0]
        image = images[0]

        if image.shape[0] is not image_size:
            image = zoom(image, (0.25, 0.25, 1))
        if scale is 1:
            image *= 255
        elif scale is 0:
            image = 127.5 * (image + 1)

        row = np.ceil(np.sqrt(num_landmarks)).astype(np.int64)
        merged = np.zeros([row * image_size, row * image_size * 2, 3])

        for idx in range(num_landmarks):
            i = idx // row
            j = idx % row
            channel_map = map_to_rgb(normalize_map(map1[:, :, idx]))
            if test_data not in ['full', 'challenging', 'common', 'training', 'test']:
                channel_map2 = channel_map.copy()
            else:
                channel_map2 = map_to_rgb(normalize_map(map2[:, :, idx]))

            merged[i * image_size:(i + 1) * image_size, (j * 2) * image_size:(j * 2 + 1) * image_size, :] = channel_map
            merged[i * image_size:(i + 1) * image_size, (j * 2 + 1) * image_size:(j * 2 + 2) * image_size,
            :] = channel_map2

        i = (idx + 1) // row
        j = (idx + 1) % row
        merged[i * image_size:(i + 1) * image_size, (j * 2) * image_size:(j * 2 + 1) * image_size, :] = image

        return merged


# sample images with pre-allocation (currently not in use)

def heat_maps_to_image_rgb(maps, landmarks=None, image_size=256, num_landmarks=68, approx=False):
    cmap = plt.get_cmap('jet')
    if approx:
        map_image = np.amax(maps, 2)
    else:
        if landmarks is None:
            landmarks = heat_maps_to_landmarks(maps, image_size=image_size, num_landmarks=num_landmarks)
        x, y = np.mgrid[0:image_size, 0:image_size]

        pixel_dist = np.sqrt(
            np.square(np.expand_dims(x, 2) - landmarks[:, 0]) + np.square(np.expand_dims(y, 2) - landmarks[:, 1]))

        nn_landmark = np.argmin(pixel_dist, 2)

        map_image = maps[x, y, nn_landmark]
    map_image = (map_image-map_image.min())/(map_image.max()-map_image.min())  # normalize for visualization
    map_image = cmap(map_image)  # convert to RGB image
    map_image = np.delete(map_image, 3, 2) * 255

    return map_image


def merge_images_landmarks_maps_gt_alloc_once(
        images, maps, maps_gt, merged, landmarks, image_size=256, num_landmarks=68, num_samples=9, scale=255,
        circle_size=2, test_data='full', approx=False, add_landmarks=True):

    row = int(np.sqrt(num_samples))

    if maps.shape[1] is not image_size:
        image_size /= 4
        image_size = int(image_size)
        rescale_img = True
    else:
        rescale_img = False

    for idx in range(num_samples):
        i = idx // row
        j = idx % row

        if landmarks is None:
            img_landmarks = heat_maps_to_landmarks(maps[idx, :, :, :], image_size=image_size, num_landmarks=num_landmarks)
        else:
            img_landmarks = landmarks[idx]

        # image with landmarks
        if rescale_img:
            if add_landmarks:
                merged[i * image_size:(i + 1) * image_size, (j * 3) * image_size:(j * 3 + 1) * image_size, :] =\
                    create_img_with_landmarks(
                    imresize(images[idx],[image_size,image_size]), img_landmarks, image_size, num_landmarks, scale=scale,
                    circle_size=circle_size)
            else:
                merged[i * image_size:(i + 1) * image_size, (j * 3) * image_size:(j * 3 + 1) * image_size, :] =\
                    imresize(images[idx], [image_size, image_size])
        else:
            if add_landmarks:
                merged[i * image_size:(i + 1) * image_size, (j * 3) * image_size:(j * 3 + 1) * image_size, :] =\
                    create_img_with_landmarks(
                        images[idx].copy(), img_landmarks, image_size, num_landmarks, scale=scale, circle_size=circle_size)
            else:
                merged[i * image_size:(i + 1) * image_size, (j * 3) * image_size:(j * 3 + 1) * image_size, :] = images[idx]

        # pred map image
        merged[i * image_size:(i + 1) * image_size, (j * 3 + 1) * image_size:(j * 3 + 2) * image_size, :] = \
            heat_maps_to_image_rgb(
                maps[idx, :, :, :], img_landmarks, image_size=image_size, num_landmarks=num_landmarks, approx=approx)

        # pred map image
        if test_data not in ['full', 'challenging', 'common', 'training', 'test']:
            merged[i * image_size:(i + 1) * image_size, (j * 3 + 2) * image_size:(j * 3 + 3) * image_size, :] =\
                merged[i * image_size:(i + 1) * image_size, (j * 3 + 1) * image_size:(j * 3 + 2) * image_size, :]
        else:
            merged[i * image_size:(i + 1) * image_size, (j * 3 + 2) * image_size:(j * 3 + 3) * image_size, :] = \
                heat_maps_to_image_rgb(
                    maps_gt[idx, :, :, :], image_size=image_size, num_landmarks=num_landmarks, approx=approx)


def map_comapre_channels_alloc_once(image, map1, map2, merged, image_size=64, num_landmarks=68, scale=255, test_data='full'):

    row = int(np.ceil(np.sqrt(num_landmarks)))

    for idx in range(num_landmarks):
        i = idx // row
        j = idx % row

        map1_min = map1[:, :, idx].min()
        map1_max = map1[:, :, idx].max()

        merged[i * image_size:(i + 1) * image_size, (j * 2) * image_size:(j * 2 + 1) * image_size, :] =\
            map_to_rgb((map1[:, :, idx] - map1_min) / (map1_max - map1_min))

        if test_data not in ['full', 'challenging', 'common', 'training', 'test']:
            merged[i * image_size:(i + 1) * image_size, (j * 2 + 1) * image_size:(j * 2 + 2) * image_size, :] = \
                merged[i * image_size:(i + 1) * image_size, (j * 2) * image_size:(j * 2 + 1) * image_size, :]
        else:
            map2_min = map2[:, :, idx].min()
            map2_max = map2[:, :, idx].max()
            merged[i * image_size:(i + 1) * image_size, (j * 2 + 1) * image_size:(j * 2 + 2) * image_size, :] =\
                map_to_rgb((map2[:, :, idx] - map2_min) / (map2_max - map2_min))

    i = (idx + 1) // row
    j = (idx + 1) % row

    if scale is 1:
        image *= 255
    elif scale is 0:
        image = 127.5 * (image + 1)

    if image.shape[0] is not image_size:
        merged[i * image_size:(i + 1) * image_size, (j * 2) * image_size:(j * 2 + 1) * image_size, :] =\
            imresize(image, [image_size, image_size])
    else:
        merged[i * image_size:(i + 1) * image_size, (j * 2) * image_size:(j * 2 + 1) * image_size, :] = image
