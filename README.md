# Using deep Face Heat Maps & Artistic Augmentation for Facial Landmark Detection in Art

## Getting Started

### Requirements

* python 2.7
* anaconda

### Download datasets
TODO: CHANGE LINK!!!!!!!!!!!!!!!!!

download datasets from [here](https://www.dropbox.com/sh/3r481u61mqd0pso/AAAyuhdUX0tomYdsYtn6QXZfa?dl=0)

for training you will need:
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

Create a virtual env named deep_face_heatmaps_env and install the following:
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


```
Give an example
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
