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
* crop_gt_margin_0.25_ns (for using artistic style texture augmentation)

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

Explain how to run the automated tests for this system

### Training

To train fusion network you need to run main_fusion.py (main_primary for primary network).
You can add the following flags:

#### define mode and paths
mode - TRAIN/TEST
save_model_path - directory for saving the model
save_sample_path - directory for saving the sampled images
save_log_path - directory for saving the log file
img_path - data directory
test_model_path - saved model to test
test_data - test set to use full/common/challenging/test/art

#### pretrain parameters
pre_train_path - pretrained model path
load_pretrain - load pretrained weight (True/False)
load_primary_only - load primary weight only (True/False)
image_size
c_dim - color channels
num_landmarks - number of face landmarks

#### optimization parameters
train_iter - maximum training iterations
batch_size
learning_rate - initial learning rate
momentum - optimizer momentum ? (remove from inputs)
step - step for lr decay
gamma - exponential base for lr decay

#### augmentation parameters
augment_basic - use basic augmentation (True/False)
basic_start - min epoch to start basic augmentation
augment_texture - use artistic texture augmentation (True/False)
p_texture - initial probability of artistic texture augmentation
augment_geom - use artistic geometric augmentation (True/False)
p_geom - initial probability of artistic geometric augmentation
artistic_step - increase probability of artistic augmentation every X epochs
artistic_start - min epoch to start artistic augmentation

example:
```
source activate deep_face_heatmaps_env
cd project_directory_path
python main_fusion.py --
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
