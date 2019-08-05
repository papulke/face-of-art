# The Face of Art: Landmark Detection and Geometric Style in Portraits

## Getting Started

### Requirements

* python
* anaconda

### Download
download model from [here](https://www.dropbox.com/sh/hrxcyug1bmbj6cs/AAAxq_zI5eawcLjM8zvUwaXha?dl=0)
download datasets from [here](https://www.dropbox.com/sh/3r481u61mqd0pso/AAAyuhdUX0tomYdsYtn6QXZfa?dl=0) (TODO: remove before publish)

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

for python 2:
```
conda create -n deep_face_heatmaps_env python=2.7 anaconda
source activate deep_face_heatmaps_env
conda install -c menpo opencv
conda install -c menpo menpo
conda install -c menpo menpofit
pip install tensorflow-gpu

```

for python 3:
```
conda create -n deep_face_heatmaps_env python=3.5 anaconda
source activate deep_face_heatmaps_env
conda install -c menpo opencv
conda install -c menpo menpo
conda install -c menpo menpofit
pip3 install tensorflow-gpu

```

Clone repository:

```
git clone https://github.com/papulke/deep_face_heatmaps
```

## Instructions

### Training

To train fusion network you need to run main_fusion.py (or main_primary.py for primary network).

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
* debug_data_size - subset data size for debug mode
* debug - run in debug mode: use subset of the data

#### define paths
* output_dir - directory for saving models, logs and samples
* save_model_path - directory for saving the model
* save_sample_path - directory for saving the sampled images
* save_log_path - directory for saving the log file
* img_path - data directory (containing subdirectories of datasets and BBs)
* test_model_path - saved model to test
* test_data - test set to use: full/common/challenging/test
* valid_data - validation set to use: full/common/challenging/test
* train_crop_dir - directory of train images cropped to bb (+margin)
* img_dir_ns - dir of train imgs cropped to bb + style transfer
* epoch_data_dir - directory containing pre-augmented data for each epoch
* use_epoch_data - use pre-augmented data (bool)


#### pretrain parameters (for fine-tuning / resume training)
* pre_train_path - pretrained model path
* load_pretrain - load pretrained weight (bool)
* load_primary_only - fine-tuning using only primary network weights (bool)

#### input data parameters
* image_size - image size
* c_dim - color channels
* num_landmarks - number of face landmarks
* sigma - std for heatmap generation gaussian
* scale - scale for image normalization 255/1/0
* margin - margin for face crops - % of bb size
* bb_type - bb to use ('gt':for ground truth / 'init':for face detector output)
* approx_maps - use heatmap approximation - major speed up
* win_mult - gaussian filter size for heatmaps approximation will be calculated by: 2 * sigma * win_mult + 1

#### optimization parameters
* l_weight_primary - primary loss weight
* l_weight_fusion - fusion loss weight
* l_weight_upsample - upsample loss weight
* train_iter - maximum training iterations
* batch_size - batch size
* learning_rate - initial learning rate
* adam_optimizer - use adam optimizer (if False momentum optimizer is used)
* momentum - optimizer momentum (relevant only if adam_optimizer==False)
* step - step for lr decay
* gamma - exponential base for lr decay
* reg - scalar multiplier for weight decay (0 to disable)
* weight_initializer - weight initializer: 'random_normal' / 'xavier'
* weight_initializer_std - std for random_normal weight initializer
* bias_initializer - constant value for bias initializer

#### augmentation parameters
* augment_basic - use basic augmentation (bool)
* augment_texture - use artistic texture augmentation (bool)
* p_texture - probability of artistic texture augmentation
* augment_geom - use artistic geometric augmentation (bool)
* p_geom - probability of artistic geometric augmentation


example for training a model with texture augmentation (100% of images) and geometric augmentation (~70% of images):
```
python main_fusion.py --mode='TRAIN' --output_dir='test_artistic_aug' --augment_geom=True \
--augment_texture=True --p_texture=1. --p_geom=0.7
```

### Testing 

There are 3 options to test our models:
1. using main_fusion.py
2. using evaluate_model.py
3. using evaluate_and_compare_multiple_models.py

#### Evaluating using main files

Using this option you can sample heat-maps + predictions of the selected test data.
If ground-truth landmarks are provided, the normalized mean error will be calculated.

example:
```
python main_fusion.py --mode='TEST' --test_model_path='model/deep_heatmaps-100000' \
--test_data='challenging'
```

TODO: add details for: evaluate_model.py, evaluate_and_compare_multiple_models.py
<!-- #### Evaluating using evaluate_model
Using this option you can get normalized mean error statistics of the model on the selected test data.
This option will provide AUC measure, failure rate and CED plot.
You can add the following flags:
#### define paths
* img_dir - data directory (containing subdirectories of datasets and BBs)
* test_data - test set to use full/common/challenging/test
* model_path - pretrained model path
#### parameters used to train network
* network_type - network architecture 'Fusion'/'Primary'
* image_size - image size
* c_dim - color channels
* num_landmarks - number of face landmarks
* scale - scale for image normalization 255/1/0
* margin - margin for face crops - % of bb size
* bb_type - bb to use ('gt':for ground truth / 'init':for face detector output)
#### choose batch size and debug data size
* batch_size - batch size
* debug - run in debug mode - use subset of the data (bool)
* debug_data_size - subset data size to test in debug mode
#### statistics parameters
* max_error - error threshold to be considered as failure
* save_log - save statistics to log_dir (bool)
* log_path - directory for saving NME statistics
example:
```
python evaluate_model.py --model_path='model/deep_heatmaps-100000' --test_data='full' \
--network_type='Fusion' --max_error=0.07
```
#### Evaluating using evaluate_and_compare_multiple_models
Using this option you can create a unified CED plot of multiple input models.
in addition, AUC measures and failure rates will be printed to screen.
** NOTICE: 
* Each model should be placed in a different directory (using a meaningful name e.g: "fusion_lr_1e-6"/"primary_lr_1e-4"/"fusion_aug_texture" etc.). including the word primary/fusion in the directory names is a must!
* Each model directory should contain one saved model.
* All model directories should be placed in one directory (e.g: "models_to_compare")
* It is assumed that model meta files is provided
* It is assumed that all models were trained with the same: bb_type, scale, margin, num_landmarks, image_size and c_dim
example:
```
python evaluate_and_compare_multiple_models.py --models_dir='models_to_compare' \
--test_data='test'  --max_error=0.08 --log_path='logs/nme_statistics'
```--> 


## Acknowledgments

* [menpo](https://github.com/menpo/menpo)
* [menpofit](https://github.com/menpo/menpofit)
* [ect](https://github.com/HongwenZhang/ECT-FaceAlignment)
* [mdm](https://github.com/trigeorgis/mdm)
* [style transfer implementation](https://github.com/woodrush/neural-art-tf)
* [painter-by-numbers dataset](https://www.kaggle.com/c/painter-by-numbers/data)
