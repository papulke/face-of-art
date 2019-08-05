from menpo_functions import *
from deep_heatmaps_model_fusion_net import DeepHeatmapsModel
from scipy.misc import imsave

# *************** define parameters and paths ***************

data_dir = '~/AF_dataset2/'
test_data = 'Fernand_Leger'  # subdirectory containing portraits for landmark detection (under data_dir)

out_dir = 'out_pred_landmarks'  # directory for saving predicted landmarks
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

model_path = '~/model_foa/deep_heatmaps-60000'  # model for estimation stage
pdm_path = 'pdm_clm_models/pdm_models/'  # models for correction stage
clm_path = 'pdm_clm_models/clm_models/g_t_all'  # model for tuning stage

outline_tune = False  # if true use tuning stage on eyebrows+jaw, else use tuning stage on jaw only
# (see paper for details)

save_cropped_imgs = True  # save input images in their cropped version to out_dir.
# the predicted landmarks will match this version of the images


# *************** load images and model ***************

# load images
img_list = load_menpo_image_list(
    img_dir=data_dir, test_data=test_data, train_crop_dir=data_dir, img_dir_ns=data_dir, mode='TEST')

# load model
heatmap_model = DeepHeatmapsModel(
    mode='TEST', img_path=data_dir, test_model_path=model_path, test_data=test_data, menpo_verbose=False)


# *************** predict landmarks ***************

for i, img in enumerate(img_list):
    if i == 0:
        reuse = None
    else:
        reuse = True

    preds = heatmap_model.get_landmark_predictions(img_list=[img], pdm_models_dir=pdm_path,
                                                   clm_model_path=clm_path, reuse=reuse)

    if outline_tune:
        pred_lms = preds['ECpTp_out']
    else:
        pred_lms = preds['ECpTp_jaw']

    mio.export_landmark_file(PointCloud(pred_lms[0]), os.path.join(out_dir, img.path.stem + '.pts'),
                             overwrite=True)
    if save_cropped_imgs:
        imsave(os.path.join(out_dir, img.path.stem + '.png'), img.pixels_with_channels_at_back())
