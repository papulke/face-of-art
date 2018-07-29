# Using deep Face Heat Maps & Artistic Augmentation for Facial Landmark Detection in Art

## Getting Started

### Requirements

* python 2.7
* anaconda

### Download datasets

download datasets from [here](https://www.dropbox.com/sh/3r481u61mqd0pso/AAAyuhdUX0tomYdsYtn6QXZfa?dl=0)

for training you will need to download the following folders:
* training_set
* Bounding_Boxes
* crop_gt_margin_0.25 (to save time on cropping data to ground-truth face bounding-box with 25% margin)
* crop_gt_margin_0.25_ns (for using artistic texture augmentation)

for testing you will need:
* full_set
* common_set
* challenging_set
* test_set
* Bounding_Boxes


### Install

Create a virtual environment and install the following:
* opencv
* menpo
* menpofit
* tensorflow-gpu

```
conda create -n deep_face_heatmaps_env python=2.7 anaconda
source activate deep_face_heatmaps_env
conda install -c menpo opencv
conda install -c menpo menpo
conda install -c menpo menpofit
pip install tensorflow-gpu

```

Clone repository:

```
git clone https://github.com/papulke/deep_face_heatmaps
```

## Instructions

### Training

To train fusion network you need to run main_fusion.py (main_primary for primary network).
You can add the following flags:

#### mode and logging parameters
* mode - 'TRAIN' or 'TEST'
* print_every - print losses to screen + log every X steps
* save_every - save model every X steps
* sample_every - sample heatmaps + landmark predictions every X steps
* sample_grid - number of training images in sample
* sample_to_log - samples will be saved to tensorboard log (bool)
* valid_size - number of validation images to run
* log_valid_every - evaluate on valid set every X epochs
* debug_data_size - subset data size to test in debug mode
* debug - run in debug mode: use subset of the data

#### define paths
* output_dir - directory for saving models, logs and samples
* save_model_path - directory for saving the model
* save_sample_path - directory for saving the sampled images
* save_log_path - directory for saving the log file
* img_path - data directory
* test_model_path - saved model to test
* test_data - test set to use: full/common/challenging/test/art
* valid_data - validation set to use: full/common/challenging/test/art
* train_crop_dir - directory of train images cropped to bb (+margin)
* img_dir_ns - dir of train imgs cropped to bb + style transfer

#### pretrain parameters (for fine-tuning / resume training)
* pre_train_path - pretrained model path
* load_pretrain - load pretrained weight (bool)
* load_primary_only - fine-tuning using only primary network weights (bool) (only in fusion net)

#### input data parameters
* image_size - image size
* c_dim - color channels
* num_landmarks - number of face landmarks
* sigma - std for heatmap generation gaussian
* scale - scale for image normalization 255/1/0
* margin - margin for face crops - % of bb size
* bb_type - bb to use ('gt':for ground truth / 'init':for face detector output)
* approx_maps - use heatmap approximation - major speed up
* win_mult - gaussian filter size for approx maps: 2 * sigma * win_mult + 1

#### optimization parameters
* l_weight_primary - primary loss weight (only in fusion net)
* l_weight_fusion - fusion loss weight (only in fusion net)
* train_iter - maximum training iterations
* batch_size - batch size
* learning_rate - initial learning rate
* adam_optimizer - use adam optimizer (if False momentum optimizer is used)
* momentum - optimizer momentum (relevant only if adam_optimizer==False)
* step - step for lr decay
* gamma - exponential base for lr decay
* weight_initializer - weight initializer: 'random_normal' / 'xavier'
* weight_initializer_std - std for random_normal weight initializer
* bias_initializer - constant value for bias initializer

#### augmentation parameters
* augment_basic - use basic augmentation (bool)
* basic_start - min epoch to start basic augmentation
* augment_texture - use artistic texture augmentation (bool)
* p_texture - initial probability of artistic texture augmentation
* augment_geom - use artistic geometric augmentation (bool)
* p_geom - initial probability of artistic geometric augmentation
* artistic_step - step for increasing probability of artistic augmentation in epochs
* artistic_start - min epoch to start artistic augmentation


example for training a model with texture augmentation (100% of images) and geometric augmentation (70% of images):
```
source activate deep_face_heatmaps_env
python main_fusion.py --output_dir='test_artistic_aug' --augment_geom=True \
--augment_texture=True --p_texture=1. --p_geom=0.7
```

### Testing 

* model.eval
* eval script - primary / fusion

```
Give an example
```

## Acknowledgments

* menpo, menpofit
* ect
* mdm
* neural style transfer
* artists?
* art dataset kaggle
