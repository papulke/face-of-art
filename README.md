# Deep Face Heat-Maps & Artistic image Augmentation for Robust Facial Landmark Detection in Art

## Getting Started

### Requirements

* python 2.7
* anaconda

### Download datasets

download datasets from [here] (https://www.dropbox.com/sh/3r481u61mqd0pso/AAAyuhdUX0tomYdsYtn6QXZfa?dl=0)

for training you will need:
* training set
* crop_gt_margin_0.25 (to save time on cropping data to ground-truth face bounding-box with 25% margin)
* crop_gt_margin_0.25_ns (for using artistic style texture augmentation)

for testing you will need:
* full set
* common set
* challenging set
* test set


### Install

Create a virtual env named: deep_face_heatmaps_env and install the following:
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

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo

## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
