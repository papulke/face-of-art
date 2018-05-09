import scipy.io
import scipy.misc
from glob import glob
import os
import numpy as np
from image_utils import *
from ops import *
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import contrib


class DeepHeatmapsModel(object):

    """facial landmark localization Network"""

    def __init__(self, mode='TRAIN', train_iter=500000, learning_rate=1e-8, image_size=256, c_dim=3, batch_size=10,
                 num_landmarks=68, augment_basic=True, augment_texture=False, augment_geom=False, img_path='data',
                 save_log_path='logs', save_sample_path='sample', save_model_path='model',
                 test_model_path='model/deep_heatmaps_primary-1000'):

        # values to print to save parameter:

        # optimizer parameters
        momentum = 0.95
        step = 80000  # for lr decay
        gamma = 0.1  # for lr decay

        # network init parameters
        weight_initializer = 'xavier'  # random_normal or xavier
        weight_initializer_std = 0.01
        bias_initializer = 0.0

        sigma = 1.5  # sigma for heatmap generation
        scale = '1'  # scale for image normalization '255' / '1' / '0'
        margin = 0.25  # for face crops
        bb_type = 'gt'  # gt/init

        valid_size = 100
        test_data = 'full'  # if mode is TEST, this choose the set to use full/common/challenging/test
        train_crop_dir = 'crop_gt_margin_0.25'
        img_dir_ns = os.path.join(img_path, train_crop_dir+'_ns')
        p_texture = 0.  # initial probability of artistic texture augmentation
        p_geom = 0.  # initial probability of artistic geometric augmentation
        artistic_step = 2  # increase probability of artistic augmentation every X epochs
        artistic_start = 3  # min epoch to start artistic augmentation
        basic_start = 1  # min epoch to start basic augmentation

        # sampling and logging parameters
        self.print_every = 10
        self.save_every = 5000
        self.sample_every = 1000
        self.sample_grid = 9
        self.log_histograms = False
        self.sample_to_log = True
        self.save_valid_images = True

        self.debug = False
        self.debug_data_size = 20
        self.compute_nme = True

        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True

        self.mode = mode
        self.train_iter = train_iter
        self.learning_rate = learning_rate

        self.image_size = image_size
        self.c_dim = c_dim
        self.batch_size = batch_size

        self.num_landmarks = num_landmarks

        self.save_log_path = save_log_path
        self.save_sample_path = save_sample_path
        self.save_model_path = save_model_path
        self.test_model_path = test_model_path
        self.img_path=img_path

        self.momentum = momentum
        self.step = step  # for lr decay
        self.gamma = gamma  # for lr decay

        self.weight_initializer = weight_initializer  # random_normal or xavier
        self.weight_initializer_std = weight_initializer_std
        self.bias_initializer = bias_initializer

        self.sigma = sigma  # sigma for heatmap generation
        self.scale = scale  # scale for image normalization '255' / '1' / '0'

        self.test_data = test_data  # if mode is TEST, this choose the set to use full/common/challenging/test
        self.train_crop_dir = train_crop_dir
        self.img_dir_ns = img_dir_ns
        self.augment_basic = augment_basic  # perform basic augmentation (rotation,flip,crop)
        self.augment_texture = augment_texture  # perform artistic texture augmentation (NS)
        self.p_texture = p_texture  # initial probability of artistic texture augmentation
        self.augment_geom = augment_geom  # perform artistic geometric augmentation
        self.p_geom = p_geom  # initial probability of artistic geometric augmentation
        self.artistic_step = artistic_step  # increase probability of artistic augmentation every X epochs
        self.artistic_start = artistic_start  # min epoch to start artistic augmentation
        self.basic_start = basic_start  # min epoch to start basic augmentation

        # load image, bb and landmark data using menpo
        self.bb_dir = os.path.join(img_path, 'Bounding_Boxes')
        if mode == 'TEST' and test_data == 'art':
            self.compute_nme = False
            self.img_menpo_list = mio.import_images(os.path.join(img_path + 'art_set'), verbose=True)
        else:
            self.bb_dictionary = load_bb_dictionary(self.bb_dir, mode, test_data=self.test_data)
            self.img_menpo_list = load_menpo_image_list_artistic_aug(
                img_path, train_crop_dir, img_dir_ns, mode,bb_dictionary=self.bb_dictionary,
                image_size=self.image_size, margin=margin, bb_type=bb_type, test_data=self.test_data,
                augment_basic=(augment_basic and basic_start == 0),
                augment_texture=augment_texture, p_texture=p_texture, augment_geom=augment_geom, p_geom=p_geom)

        if self.debug:
            self.img_menpo_list = self.img_menpo_list[:self.debug_data_size]

        if mode is 'TRAIN':
            train_params = locals()
            print_training_params_to_file(train_params)  # save init parameters

            # train - validation split
            self.valid_size = valid_size
            np.random.seed(0)
            img_inds = np.arange(len(self.img_menpo_list))
            np.random.shuffle(img_inds)

            val_inds = img_inds[:self.valid_size]
            self.train_inds = img_inds[self.valid_size:]
            self.valid_img_menpo_list = self.img_menpo_list[val_inds]

            self.valid_images_loaded, _, self.valid_gt_maps_loaded, self.valid_landmarks_loaded = \
                load_data(self.valid_img_menpo_list, np.arange(self.valid_size), image_size=self.image_size,
                          c_dim=self.c_dim, num_landmarks=self.num_landmarks, sigma=self.sigma, scale=self.scale,
                          save_landmarks=True, primary=True)
            if self.valid_size > self.sample_grid:
                self.valid_gt_maps_loaded =self.valid_gt_maps_loaded[:self.sample_grid]

            self.img_menpo_list = self.img_menpo_list[self.train_inds]

    def add_placeholders(self):

        if self.mode == 'TEST':
            self.images = tf.placeholder(
                tf.float32, [None, self.image_size, self.image_size, self.c_dim], 'images')

            self.heatmaps_small = tf.placeholder(
                tf.float32, [None, self.image_size/4, self.image_size/4, self.num_landmarks], 'heatmaps_small')
            if self.compute_nme:
                self.lms_small = tf.placeholder(tf.float32, [None, self.num_landmarks, 2], 'lms_small')
                self.pred_lms_small = tf.placeholder(tf.float32, [None, self.num_landmarks, 2], 'pred_lms_small')

        elif self.mode == 'TRAIN':
            self.images = tf.placeholder(
                tf.float32, [None, self.image_size, self.image_size, self.c_dim], 'train_images')

            self.heatmaps_small = tf.placeholder(
                tf.float32, [None, self.image_size/4, self.image_size/4, self.num_landmarks], 'train_heatmaps_small')

            if self.compute_nme:
                self.train_lms_small = tf.placeholder(tf.float32, [None, self.num_landmarks, 2], 'train_lms_small')
                self.train_pred_lms_small = tf.placeholder(tf.float32, [None, self.num_landmarks, 2], 'train_pred_lms_small')

            self.valid_lms_small = tf.placeholder(tf.float32, [None, self.num_landmarks, 2], 'valid_lms_small')
            self.valid_pred_lms_small = tf.placeholder(tf.float32, [None, self.num_landmarks, 2], 'valid_pred_lms_small')

            if self.sample_to_log:
                row = int(np.sqrt(self.sample_grid))
                self.log_image_map = tf.placeholder(
                    tf.uint8, [None,row * self.image_size / 4, 3 * row *self.image_size / 4, self.c_dim], 'sample_img_map')
                row = np.ceil(np.sqrt(self.num_landmarks)).astype(np.int64)
                self.log_map_channels = tf.placeholder(
                    tf.uint8, [None, row * self.image_size / 4, 2 * row * self.image_size / 4, self.c_dim],
                    'sample_map_channels')

    def heatmaps_network(self, input_images, reuse=None, name='pred_heatmaps'):

        with tf.name_scope(name):

            if self.weight_initializer == 'xavier':
                weight_initializer = contrib.layers.xavier_initializer()
            else:
                weight_initializer = tf.random_normal_initializer(stddev=self.weight_initializer_std)

            bias_init = tf.constant_initializer(self.bias_initializer)

            with tf.variable_scope('heatmaps_network'):
                with tf.name_scope('primary_net'):

                    l1 = conv_relu_pool(input_images, 5, 128, conv_ker_init=weight_initializer, conv_bias_init=bias_init,
                                        reuse=reuse, var_scope='conv_1')
                    l2 = conv_relu_pool(l1, 5, 128, conv_ker_init=weight_initializer, conv_bias_init=bias_init,
                                        reuse=reuse, var_scope='conv_2')
                    l3 = conv_relu(l2, 5, 128, conv_ker_init=weight_initializer, conv_bias_init=bias_init,
                                   reuse=reuse, var_scope='conv_3')

                    l4_1 = conv_relu(l3, 3, 128, conv_dilation=1, conv_ker_init=weight_initializer,
                                     conv_bias_init=bias_init, reuse=reuse, var_scope='conv_4_1')
                    l4_2 = conv_relu(l3, 3, 128, conv_dilation=2, conv_ker_init=weight_initializer,
                                     conv_bias_init=bias_init, reuse=reuse, var_scope='conv_4_2')
                    l4_3 = conv_relu(l3, 3, 128, conv_dilation=3, conv_ker_init=weight_initializer,
                                     conv_bias_init=bias_init, reuse=reuse, var_scope='conv_4_3')
                    l4_4 = conv_relu(l3, 3, 128, conv_dilation=4, conv_ker_init=weight_initializer,
                                     conv_bias_init=bias_init, reuse=reuse, var_scope='conv_4_4')

                    l4 = tf.concat([l4_1, l4_2, l4_3, l4_4], 3, name='conv_4')

                    l5_1 = conv_relu(l4, 3, 256, conv_dilation=1, conv_ker_init=weight_initializer,
                                     conv_bias_init=bias_init, reuse=reuse, var_scope='conv_5_1')
                    l5_2 = conv_relu(l4, 3, 256, conv_dilation=2, conv_ker_init=weight_initializer,
                                     conv_bias_init=bias_init, reuse=reuse, var_scope='conv_5_2')
                    l5_3 = conv_relu(l4, 3, 256, conv_dilation=3, conv_ker_init=weight_initializer,
                                     conv_bias_init=bias_init, reuse=reuse, var_scope='conv_5_3')
                    l5_4 = conv_relu(l4, 3, 256, conv_dilation=4, conv_ker_init=weight_initializer,
                                     conv_bias_init=bias_init, reuse=reuse, var_scope='conv_5_4')

                    l5 = tf.concat([l5_1, l5_2, l5_3, l5_4], 3, name='conv_5')

                    l6 = conv_relu(l5, 1, 512, conv_ker_init=weight_initializer,
                                   conv_bias_init=bias_init, reuse=reuse, var_scope='conv_6')
                    l7 = conv_relu(l6, 1, 256, conv_ker_init=weight_initializer,
                                   conv_bias_init=bias_init, reuse=reuse, var_scope='conv_7')
                    primary_out = conv(l7, 1, self.num_landmarks, conv_ker_init=weight_initializer,
                                            conv_bias_init=bias_init, reuse=reuse, var_scope='conv_8')

                self.all_layers = [l1, l2, l3, l4, l5, l6, l7, primary_out]

                return primary_out

    def build_model(self):
        self.pred_hm_p = self.heatmaps_network(self.images,name='heatmaps_prediction')

    def create_loss_ops(self):

        def l2_loss_norm_eyes(pred_landmarks, real_landmarks, normalize=True, name='NME_loss'):

            with tf.name_scope(name):
                with tf.name_scope('real_pred_landmarks_diff'):
                    landmarks_diff = pred_landmarks - real_landmarks

                if normalize:
                    with tf.name_scope('inter_pupil_dist'):
                        with tf.name_scope('left_eye'):
                            p1 = tf.reduce_mean(tf.slice(real_landmarks, [0, 42, 0], [-1, 6, 2]), axis=1)
                        with tf.name_scope('right_eye'):
                            p2 = tf.reduce_mean(tf.slice(real_landmarks, [0, 36, 0], [-1, 6, 2]), axis=1)
                        eps = 1e-6
                        eye_dist = tf.expand_dims(tf.expand_dims(
                            tf.sqrt(tf.reduce_sum(tf.square(p1 - p2), axis=1)) + eps, axis=1), axis=1)

                    norm_landmarks_diff = landmarks_diff / eye_dist
                    l2_landmarks_norm = tf.reduce_mean(tf.square(norm_landmarks_diff))

                    out = l2_landmarks_norm
                else:
                    l2_landmarks = tf.reduce_mean(tf.square(landmarks_diff))
                    out = l2_landmarks

                return out

        if self.mode is 'TRAIN':
            primary_maps_diff = self.pred_hm_p-self.heatmaps_small
            self.total_loss = 1000.*tf.reduce_mean(tf.square(primary_maps_diff))

            if self.compute_nme:
                self.nme_loss = l2_loss_norm_eyes(self.train_pred_lms_small,self.train_lms_small)
            else:
                self.nme_loss = tf.constant(0.)

            if self.valid_size > 0:
                self.valid_nme_loss = l2_loss_norm_eyes(self.valid_pred_lms_small,self.valid_lms_small)
            else:
                self.valid_nme_loss = tf.constant(0.)

        elif self.mode == 'TEST' and self.compute_nme:
            self.nme_loss = l2_loss_norm_eyes(self.pred_lms_small, self.lms_small)

    def predict_landmarks_in_batches(self,image_paths,session):

        num_batches = int(1.*len(image_paths)/self.batch_size)
        if num_batches == 0:
            batch_size = len(image_paths)
            num_batches = 1
        else:
            batch_size = self.batch_size

        img_inds = np.arange(len(image_paths))
        for j in range(num_batches):
            batch_inds = img_inds[j * self.batch_size:(j + 1) * self.batch_size]

            batch_images, _, _, batch_lms_small = \
                load_data(self.img_menpo_list, batch_inds, image_size=self.image_size, c_dim=self.c_dim,
                          num_landmarks=self.num_landmarks, sigma=self.sigma, scale=self.scale,
                          save_landmarks=self.compute_nme, primary=True)

            if j == 0:
                batch_maps_small_pred = session.run(self.pred_hm_p, {self.images: batch_images})
                all_pred_landmarks = batch_heat_maps_to_landmarks(
                    batch_maps_small_pred, self.batch_size, image_size=self.image_size / 4,
                    num_landmarks=self.num_landmarks)
                all_gt_landmarks=batch_lms_small
            else:
                batch_maps_small_pred = session.run(self.pred_hm_p, {self.images: batch_images})
                batch_pred_landmarks = batch_heat_maps_to_landmarks(
                    batch_maps_small_pred, self.batch_size, image_size=self.image_size / 4,
                    num_landmarks=self.num_landmarks)

                all_pred_landmarks = np.concatenate((all_pred_landmarks,batch_pred_landmarks),0)
                all_gt_landmarks = np.concatenate((all_gt_landmarks, batch_lms_small), 0)

        reminder = len(image_paths)-num_batches*batch_size

        if reminder >0:
            reminder_inds = img_inds[-reminder:]
            batch_images, _, _, batch_lms_small = \
                load_data(self.img_menpo_list, reminder_inds, image_size=self.image_size, c_dim=self.c_dim,
                          num_landmarks=self.num_landmarks, sigma=self.sigma, scale=self.scale,
                          save_landmarks=self.compute_nme, primary=True)

            batch_maps_small_pred = session.run(self.pred_hm_p, {self.images: batch_images})
            batch_pred_landmarks = batch_heat_maps_to_landmarks(
                batch_maps_small_pred, reminder, image_size=self.image_size / 4,
                num_landmarks=self.num_landmarks)

            all_pred_landmarks = np.concatenate((all_pred_landmarks, batch_pred_landmarks), 0)
            all_gt_landmarks = np.concatenate((all_gt_landmarks, batch_lms_small), 0)

        return all_pred_landmarks, all_gt_landmarks

    def predict_landmarks_in_batches_loaded(self,images,session):

        num_images=int(images.shape[0])
        num_batches = int(1.*num_images/self.batch_size)
        if num_batches == 0:
            batch_size = num_images
            num_batches = 1
        else:
            batch_size = self.batch_size

        for j in range(num_batches):
            batch_images = images[j * self.batch_size:(j + 1) * self.batch_size,:,:,:]

            if j == 0:
                batch_maps_small_pred = session.run(self.pred_hm_p, {self.images: batch_images})
                all_pred_landmarks = batch_heat_maps_to_landmarks(
                    batch_maps_small_pred, self.batch_size, image_size=self.image_size / 4,
                    num_landmarks=self.num_landmarks)
            else:
                batch_maps_small_pred = session.run(self.pred_hm_p, {self.images: batch_images})
                batch_pred_landmarks = batch_heat_maps_to_landmarks(
                    batch_maps_small_pred, self.batch_size, image_size=self.image_size / 4,
                    num_landmarks=self.num_landmarks)

                all_pred_landmarks = np.concatenate((all_pred_landmarks,batch_pred_landmarks),0)

        reminder = num_images-num_batches*batch_size
        if reminder >0:
            batch_images = images[-reminder:, :, :, :]
            batch_maps_small_pred = session.run(self.pred_hm_p, {self.images: batch_images})
            batch_pred_landmarks = batch_heat_maps_to_landmarks(
                batch_maps_small_pred, reminder, image_size=self.image_size / 4,
                num_landmarks=self.num_landmarks)

            all_pred_landmarks = np.concatenate((all_pred_landmarks, batch_pred_landmarks), 0)

        return all_pred_landmarks

    def create_summary_ops(self):

            var_summary = [tf.summary.histogram(var.name,var) for var in tf.trainable_variables()]
            grads = tf.gradients(self.total_loss, tf.trainable_variables())
            grads = list(zip(grads, tf.trainable_variables()))
            grad_summary = [tf.summary.histogram(var.name+'/grads',grad) for grad,var in grads]
            activ_summary = [tf.summary.histogram(layer.name, layer) for layer in self.all_layers]
            l_total = tf.summary.scalar('l_total', self.total_loss)
            l_nme = tf.summary.scalar('l_nme', self.nme_loss)
            l_v_nme = tf.summary.scalar('valid_l_nme', self.valid_nme_loss)

            self.p_geom_log = tf.placeholder(tf.float32, [])
            self.p_texture_log = tf.placeholder(tf.float32, [])
            p_geom = tf.summary.scalar('p_geom', self.p_geom_log)
            p_texture = tf.summary.scalar('p_texture', self.p_texture_log)

            if self.log_histograms:
                self.batch_summary_op = tf.summary.merge([p_texture, p_geom, l_total, l_nme, l_v_nme, var_summary,
                                                          grad_summary, activ_summary])
            else:
                self.batch_summary_op = tf.summary.merge([p_texture, p_geom, l_total, l_nme, l_v_nme])
            if self.sample_to_log:
                img_map_summary =tf.summary.image('compare_map_to_gt',self.log_image_map)
                map_channels_summary = tf.summary.image('compare_map_channels_to_gt', self.log_map_channels)
                self.img_summary = tf.summary.merge([img_map_summary, map_channels_summary])
                if self.valid_size > self.sample_grid:
                    img_map_summary_valid = tf.summary.image('compare_map_to_gt_valid', self.log_image_map)
                    map_channels_summary_valid = tf.summary.image('compare_map_channels_to_gt_valid', self.log_map_channels)
                    self.img_summary_valid = tf.summary.merge([img_map_summary_valid, map_channels_summary_valid])

    def eval(self):

        self.add_placeholders()
        # build model
        self.build_model()
        self.create_loss_ops()

        num_images = len(self.img_menpo_list)
        img_inds = np.arange(num_images)

        sample_iter = int(1. * num_images / self.sample_grid)

        with tf.Session(config=self.config) as sess:

            # load trained parameters
            print ('loading test model...')
            saver = tf.train.Saver()
            saver.restore(sess, self.test_model_path)

            _, model_name = os.path.split(self.test_model_path)

            for i in range(sample_iter):

                batch_inds = img_inds[i * self.sample_grid:(i + 1) * self.sample_grid]

                if self.test_data == 'art':
                    batch_images = load_art_data(self.img_menpo_list, batch_inds, image_size=self.image_size,
                                                 c_dim=self.c_dim, scale=self.scale)
                else:
                    batch_images, _, batch_maps_gt, _ = \
                        load_data(self.img_menpo_list, batch_inds, image_size=self.image_size, c_dim=self.c_dim,
                                  num_landmarks=self.num_landmarks, sigma=self.sigma, scale=self.scale,
                                  save_landmarks=False, primary=True)

                batch_maps_small_pred = sess.run(self.pred_hm_p, {self.images: batch_images})

                if self.test_data == 'art':
                    batch_maps_gt = batch_maps_small_pred.copy()

                sample_path_imgs = os.path.join(
                    self.save_sample_path, model_name +'-'+ self.test_data+'-sample-%d-to-%d-1.png' % (
                        i * self.sample_grid, (i + 1) * self.sample_grid))

                sample_path_channels = os.path.join(
                    self.save_sample_path, model_name +'-'+ self.test_data+ '-sample-%d-to-%d-3.png' % (
                        i * self.sample_grid, (i + 1) * self.sample_grid))

                merged_img = merge_images_landmarks_maps_gt(
                    batch_images.copy(), batch_maps_small_pred, batch_maps_gt, image_size=self.image_size,
                    num_landmarks=self.num_landmarks, num_samples=self.sample_grid, scale=self.scale, circle_size=0,
                    test_data=self.test_data)

                map_per_channel = map_comapre_channels(
                    batch_images.copy(), batch_maps_small_pred,batch_maps_gt, image_size=self.image_size / 4,
                    num_landmarks=self.num_landmarks, scale=self.scale, test_data=self.test_data)

                scipy.misc.imsave(sample_path_imgs, merged_img)
                scipy.misc.imsave(sample_path_channels, map_per_channel)

                print ('saved %s' % sample_path_imgs)
            if self.compute_nme:
                print ('\n Calculating NME on: ' + self.test_data + '...')
                pred_lms, lms_gt = self.predict_landmarks_in_batches(self.img_menpo_list, sess)
                nme = sess.run(self.nme_loss, {self.pred_lms_small: pred_lms, self.lms_small: lms_gt})
                print ('NME on ' + self.test_data + ': ' + str(nme))

    def train(self):
            tf.set_random_seed(1234)
            np.random.seed(1234)
            # build a graph
            # add placeholders
            self.add_placeholders()
            # build model
            self.build_model()
            # create loss ops
            self.create_loss_ops()
            # create summary ops
            self.create_summary_ops()

            # create optimizer and training op
            global_step = tf.Variable(0, trainable=False)
            lr = tf.train.exponential_decay(self.learning_rate,global_step, self.step, self.gamma, staircase=True)
            optimizer = tf.train.MomentumOptimizer(lr,self.momentum)

            train_op = optimizer.minimize(self.total_loss,global_step=global_step)

            with tf.Session(config=self.config) as sess:

                tf.global_variables_initializer().run()

                # create model saver and file writer
                summary_writer = tf.summary.FileWriter(logdir=self.save_log_path, graph=tf.get_default_graph())
                saver = tf.train.Saver()

                print
                print('*** Start Training ***')

                # set random seed
                epoch = 0

                num_train_images = len(self.img_menpo_list)

                img_inds = np.arange(num_train_images)
                np.random.shuffle(img_inds)

                p_texture = self.p_texture
                p_geom = self.p_geom
                artistic_reload = True
                basic_reload = True
                for step in range(self.train_iter + 1):

                    # get batch images
                    j = step % int(float(num_train_images) / float(self.batch_size))

                    if step > 0 and j == 0:
                        np.random.shuffle(img_inds)  # shuffle data if finished epoch
                        epoch += 1
                        artistic_reload = True

                    # add basic augmentation (if basic_start > 0 and augment_basic is True)
                    if basic_reload and (epoch >= self.basic_start) and self.basic_start > 0 and self.augment_basic:
                        basic_reload = False
                        self.img_menpo_list = reload_img_menpo_list_artistic_aug_train(
                            self.img_path, self.train_crop_dir, self.img_dir_ns, self.mode, self.train_inds,
                            debug=self.debug, debug_size=self.debug_data_size, image_size=self.image_size,
                            augment_basic=self.augment_basic, augment_texture=self.augment_texture, p_texture=p_texture,
                            augment_geom=self.augment_geom, p_geom=p_geom)
                        print "****** adding basic augmentation ******"

                    # increase artistic augmentation probability
                    if (epoch % self.artistic_step == 0 and epoch >= self.artistic_start) and\
                            (self.augment_geom or self.augment_texture) and artistic_reload:
                        artistic_reload = False
                        p_geom = 1.- 0.95 ** (epoch/self.artistic_step)
                        p_texture = 1. - 0.95 ** (epoch/self.artistic_step)

                        self.img_menpo_list = reload_img_menpo_list_artistic_aug_train(
                            self.img_path, self.train_crop_dir, self.img_dir_ns, self.mode, self.train_inds,
                            debug=self.debug, debug_size=self.debug_data_size, image_size=self.image_size,
                            augment_basic=(self.augment_basic and epoch >= self.basic_start),
                            augment_texture=self.augment_texture, p_texture=p_texture,
                            augment_geom=self.augment_geom, p_geom=p_geom)

                        print "****** increasing artistic augmentation probability ******"
                        print "****** p_texture:", p_texture, "p_geom:", p_geom, "******"

                    batch_inds = img_inds[j * self.batch_size:(j + 1) * self.batch_size]

                    batch_images, _, batch_maps_small, batch_lms_small =\
                        load_data(self.img_menpo_list, batch_inds, image_size=self.image_size, c_dim=self.c_dim,
                                  num_landmarks=self.num_landmarks, sigma=self.sigma, scale=self.scale,
                                  save_landmarks=self.compute_nme, primary=True)

                    feed_dict_train = {self.images: batch_images, self.heatmaps_small: batch_maps_small}

                    sess.run(train_op, feed_dict_train)

                    # save to log and print status
                    if step == 0 or (step + 1) % self.print_every == 0:

                        if self.compute_nme is False and self.valid_size == 0:
                            feed_dict_log = {self.images: batch_images, self.heatmaps_small: batch_maps_small,
                                             self.p_geom_log: p_geom, self.p_texture_log: p_texture}

                        elif self.compute_nme and self.valid_size == 0:
                            batch_maps_small_pred = sess.run(self.pred_hm_p, {self.images: batch_images})
                            pred_lms_small = batch_heat_maps_to_landmarks(
                                batch_maps_small_pred, self.batch_size, image_size=self.image_size/4,
                                num_landmarks=self.num_landmarks)

                            feed_dict_log = {
                                self.images: batch_images, self.heatmaps_small: batch_maps_small,
                                self.train_lms_small: batch_lms_small, self.train_pred_lms_small: pred_lms_small,
                                self.p_geom_log: p_geom, self.p_texture_log: p_texture}

                        elif self.compute_nme and self.valid_size > 0:
                            valid_pred_lms = self.predict_landmarks_in_batches_loaded(self.valid_images_loaded,sess)

                            batch_maps_small_pred = sess.run(self.pred_hm_p, {self.images: batch_images})
                            pred_lms_small = batch_heat_maps_to_landmarks(
                                batch_maps_small_pred, self.batch_size, image_size=self.image_size/4,
                                num_landmarks=self.num_landmarks)

                            feed_dict_log = {
                                self.images: batch_images, self.heatmaps_small: batch_maps_small,
                                self.train_lms_small: batch_lms_small, self.train_pred_lms_small: pred_lms_small,
                                self.valid_lms_small: self.valid_landmarks_loaded,
                                self.valid_pred_lms_small: valid_pred_lms,
                                self.p_geom_log: p_geom, self.p_texture_log: p_texture}

                        else:
                            valid_pred_lms = self.predict_landmarks_in_batches_loaded(self.valid_images_loaded,sess)
                            feed_dict_log = {
                                self.images: batch_images, self.heatmaps_small: batch_maps_small,
                                self.valid_lms_small: self.valid_landmarks_loaded,
                                self.valid_pred_lms_small: valid_pred_lms,
                                self.p_geom_log: p_geom, self.p_texture_log: p_texture}

                        summary, l_t, l_nme, l_v_nme = sess.run(
                            [self.batch_summary_op, self.total_loss, self.nme_loss, self.valid_nme_loss], feed_dict_log)

                        summary_writer.add_summary(summary, step)

                        print ('epoch: [%d] step: [%d/%d] primary loss: [%.6f] nme loss: [%.6f] valid nme loss: [%.6f]' % (
                            epoch, step + 1, self.train_iter, l_t, l_nme, l_v_nme))

                    # save model
                    if (step + 1) % self.save_every == 0:
                        saver.save(sess, os.path.join(self.save_model_path, 'deep_heatmaps'), global_step=step + 1)
                        print ('model/deep-heatmaps-%d saved' % (step + 1))

                    # save images
                    if step == 0 or (step + 1) % self.sample_every == 0:

                            if not self.compute_nme:
                                batch_maps_small_pred = sess.run(self.pred_hm_p,  {self.images: batch_images})

                            merged_img = merge_images_landmarks_maps_gt(
                                batch_images.copy(), batch_maps_small_pred, batch_maps_small,
                                image_size=self.image_size,
                                num_landmarks=self.num_landmarks, num_samples=self.sample_grid, scale=self.scale,
                                circle_size=0)

                            map_per_channel = map_comapre_channels(
                                batch_images.copy(), batch_maps_small_pred,batch_maps_small,
                                image_size=self.image_size/4, num_landmarks=self.num_landmarks, scale=self.scale)

                            if self.sample_to_log:
                                summary_img = sess.run(
                                    self.img_summary, {self.log_image_map: np.expand_dims(merged_img, 0),
                                                       self.log_map_channels: np.expand_dims(map_per_channel, 0)})

                                summary_writer.add_summary(summary_img, step)

                                if (self.valid_size > self.sample_grid) and self.save_valid_images:
                                    batch_maps_small_pred_val = sess.run(
                                        self.pred_hm_p, {self.images: self.valid_images_loaded[:self.sample_grid]})

                                    merged_img = merge_images_landmarks_maps_gt(
                                        self.valid_images_loaded[:self.sample_grid].copy(), batch_maps_small_pred_val,
                                        self.valid_gt_maps_loaded, image_size=self.image_size,
                                        num_landmarks=self.num_landmarks, num_samples=self.sample_grid,
                                        scale=self.scale, circle_size=0)

                                    map_per_channel = map_comapre_channels(
                                        self.valid_images_loaded[:self.sample_grid].copy(), batch_maps_small_pred_val,
                                        self.valid_gt_maps_loaded, image_size=self.image_size / 4,
                                        num_landmarks=self.num_landmarks, scale=self.scale)

                                    summary_img = sess.run(
                                        self.img_summary_valid, {self.log_image_map: np.expand_dims(merged_img, 0),
                                                           self.log_map_channels: np.expand_dims(map_per_channel, 0)})
                                    summary_writer.add_summary(summary_img, step)

                            else:
                                sample_path_imgs = os.path.join(self.save_sample_path,'epoch-%d-train-iter-%d-1.png'
                                                                % (epoch, step + 1))
                                sample_path_ch_maps = os.path.join(self.save_sample_path, 'epoch-%d-train-iter-%d-3.png'
                                                                   % (epoch, step + 1))
                                scipy.misc.imsave(sample_path_imgs, merged_img)
                                scipy.misc.imsave(sample_path_ch_maps, map_per_channel)

                print('*** Finished Training ***')
