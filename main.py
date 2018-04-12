import tensorflow as tf
from deep_heatmaps_model_ect import DeepHeatmapsModel

flags = tf.app.flags
flags.DEFINE_string('mode', 'TRAIN', "'TRAIN' or 'TEST'")
flags.DEFINE_string('save_model_path', 'model', "directory for saving the model")
flags.DEFINE_string('save_sample_path', 'sample', "directory for saving the sampled images")
flags.DEFINE_string('save_log_path', 'logs', "directory for saving the log file")
FLAGS = flags.FLAGS


def main(_):

    # create directories if not exist
    if not tf.gfile.Exists(FLAGS.save_model_path):
        tf.gfile.MakeDirs(FLAGS.save_model_path)
    if not tf.gfile.Exists(FLAGS.save_sample_path):
        tf.gfile.MakeDirs(FLAGS.save_sample_path)
    if not tf.gfile.Exists(FLAGS.save_log_path):
        tf.gfile.MakeDirs(FLAGS.save_log_path)

    model = DeepHeatmapsModel(mode=FLAGS.mode, train_iter=100, learning_rate=0.001, image_size=256, c_dim=3,
                              batch_size=4, num_landmarks=68,
                              img_path='/Users/arik/Desktop/DATA/face_data/conventional_landmark_detection_dataset',
                              save_log_path=FLAGS.save_log_path, save_sample_path=FLAGS.save_sample_path,
                              save_model_path=FLAGS.save_model_path, test_model_path = 'model/deep_heatmaps-1000')

    if FLAGS.mode == 'TRAIN':
        model.train()
    else:
        model.eval()

if __name__ == '__main__':
    tf.app.run()
