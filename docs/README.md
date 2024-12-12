<div align="center">

# Motional<sup>TM</sup> nuPlan<sup>TM</sup>

**The world's first benchmark for autonomous vehicle planning.**

<p align="center"><img src="https://cdn.cookielaw.org/logos/8c60fe9e-585e-46b1-8f92-eba17239401e/d3e43cda-e0a4-42f2-9c04-0e1900c3f68f/894f42e4-cba8-48e4-8a15-e9c3ea937950/motional_logo_horiz_fullcolor_rgb.png" width="350px"></p>



______________________________________________________________________

<p align="center">
  <a href="https://www.nuplan.org/">Website</a> •
  <a href="https://www.nuscenes.org/nuplan#download">Download</a> •
  <a href="#citation">Citation</a><br>
  <a href="#changelog">Changelog</a> •
  <a href="#devkit-structure">Structure</a> •
  <a href="https://github.com/motional/nuplan-devkit/blob/master/docs/installation.md">Setup</a> <br>
  <a href="https://github.com/motional/nuplan-devkit/blob/master/tutorials/nuplan_framework.ipynb">Tutorial</a> •
  <a href="https://nuplan-devkit.readthedocs.io/en/latest/">Documentation</a> •
  <a href="https://eval.ai/web/challenges/challenge-page/1856/overview">Competition</a>
</p>

[![python](https://img.shields.io/badge/python-%20%203.9-blue.svg)]()
[![license](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/motional/nuplan-devkit/blob/master/LICENSE.txt)
[![Documentation Status](https://readthedocs.org/projects/nuplan-devkit/badge/?version=latest)](https://nuplan-devkit.readthedocs.io/en/latest/?badge=latest)

______________________________________________________________________

<br>

<p align="center"><img src="https://www.nuplan.org/static/media/nuPlan_final.3fde7586.png" width="500px"></p>

</div>

______________________________________________________________________
## Sensor Data Release
#### IMPORTANT: The file structure has changed! Please check [Dataset Setup](https://github.com/motional/nuplan-devkit/blob/master/docs/dataset_setup.md) page for the updated file structure.
- The nuPlan sensor data for the v1.1 dataset has been released. Please download the latest dataset from the nuPlan [page](https://www.nuscenes.org/nuplan#download).
- Due to the size of the sensor data, it will be released gradually. This first set of sensor data are the blobs corresponding to nuPlan mini.
- A short tutorial for the sensor data is provided `nuplan_sensor_data_tutorial.ipynb` to get you started.
______________________________________________________________________
## Planning challenges
#### IMPORTANT: The base docker image used in nuPlan submission has been updated. You should rebuild your submission container with the new `Dockerfile.submission`

- The Planning Challenge will be using devkit version 1.2 from now on. Submissions generated from version v1.1 should remain compatible. However, please double-check by submitting to the warm-up phase.
- The challenge will be presented as part of the [End-to-End Autonomous Driving](https://opendrivelab.com/event/cvpr23_ADworkshop) workshop at CVPR 2023
- The nuPlan Dataset v1.1 has been released. Please download the latest dataset from the nuPlan [page](https://www.nuscenes.org/nuplan#download).
______________________________________________________________________

## Changelog
- May 11th 2023
  * v1.2.2 Devkit: Upated the submission base images.
- May 9th 2023
  * v1.2.1 Devkit: Update to competition dates. Submission Deadline extended to May 26th, 2023.
- April 25th 2023
  * v1.2 Devkit: The nuPlan sensor data have been released! Improved feature caching and nuBoard dashboard functionality. Changed dataset file structure, data interfaces now allow retrieval of sensor data. Pinned several packages including hydra, numpy and sqlalchemy.
- January 20th 2023
  * v1.1 Devkit: The official nuPlan Challenge Release. Optimized training caching, simulation improvements, shapely 2.0 update. 
- Oct 13th 2022
  * v1.1 Dataset: Full nuPlan dataset - improved route plan, traffic light status, mission goal and more!
  * v1.0 Devkit: Update to nuplan-v1.1 dataset, metrics improvements, route plan fixes, documentation, IDMPlanner
- Sep 09 2022
  * v0.6 Devkit: Smart agents optimizations, nuBoard improvements, metrics improvements, submission pipeline deployment and documentation.
- Aug 26 2022
  * v0.5 Devkit: New map features, simulation improvements, open-loop detections with smart agents, iLQR tracker, metrics improvements and documentation.
- Aug 05 2022
  * v0.4 Devkit: Database optimization, PYGEOS warning suppression, metrics improvements, scenario filter for training.
- Jul 15 2022
  * v0.3 Devkit: Updates to the devit including but not limited to: nuBoard update, reduce RAM usage during caching and training, new map APIs, simulation and metrics runtime improvements.
- Jun 10 2022
  * v1.0 Dataset: Full nuPlan dataset release with over 1,300 hours of driving data (15,000+ logs) across 4 cities (Las Vegas, Pittsburgh, Boston, Singapore).
  * v0.2 Devkit: Multiple devkit updates with improvements across the whole framework (planning models, training, simulation, metrics, dashboard, tutorials and more).
- Dec 19 2021
  * v0.2 Dataset: Fixed bugs in the teaser nuPlan dataset.
- Dec 10 2021
  * v0.1 Dataset: Initial teaser nuPlan dataset release with over 200 hours of driving data (350+ logs) across Las Vegas.
  * v0.1 Devkit: Initial nuPlan devkit release.


______________________________________________________________________

## Devkit and dataset setup
Please refer to the [installation page](https://nuplan-devkit.readthedocs.io/en/latest/installation.html) for detailed instructions on how to setup the devkit.

Please refer to the [dataset page](https://nuplan-devkit.readthedocs.io/en/latest/dataset_setup.html) for detailed instructions on how to download and setup the dataset.

______________________________________________________________________

## Getting started
<p align="center"><img src="https://cdn.cookielaw.org/logos/8c60fe9e-585e-46b1-8f92-eba17239401e/d3e43cda-e0a4-42f2-9c04-0e1900c3f68f/808c47fb-8484-44eb-b369-d90d6bb4733e/motional_logo_stack_colorrev_rgb_black.png" width="350px"></p>
For those interested in participating in the Motional<sup>TM</sup> nuPlan<sup>TM</sup> Planning Competition, please refer to the competition landing [page](https://nuplan-devkit.readthedocs.io/en/latest/).

Please follow these steps to make yourself familiar with the nuPlan dataset:
- Familiarize yourself with the main [features of nuPlan](https://www.nuplan.org) and the [dataset description](https://www.nuplan.org/nuplan).
- Setup the devkit and dataset as described [above](#devkit-and-dataset-setup).
- Walk through the tutorials in [this folder](https://github.com/motional/nuplan-devkit/blob/master/tutorials/) or run them yourself using `jupyter notebook ~/nuplan-devkit/tutorials/<filename>.ipynb` and replacing `<filename>` with the tutorial's name. The following tutorials are available:
  - `nuplan_framework.ipynb`: Main tutorial for anyone who wants to dive right into ML planning.
  It describes how to 1) train an ML planner, 2) simulate it, 3) measure the performance and 4) visualize the results.
  - `nuplan_scenario_visualization.ipynb`: Tutorial for visualizing various scenario types contained within the nuPlan dataset (e.g. unprotected turns, lane changes, interactions with pedestrians and more).
  - `nuplan_planner_tutorial.ipynb`: Tutorial with instructions on how to develop and simulate a planner from scratch within the nuPlan framework.
  - `nuplan_advanced_model_training.ipynb`: This notebook will cover the details involved in training a planning model in the NuPlan framework. This notebook is a more detailed deep dive into the NuPlan architecture, and covers the extensibility points that can be used to build customized models in the NuPlan framework.

- Familiarize yourself with the nuPlan CLI, which gets installed by installing the devkit using `pip` (editable and not)
  by running:
  ```
  nuplan_cli --help
  nuplan_cli COMMAND --help
  ```
- Read the [nuPlan paper](https://www.nuplan.org/publications) to understand the details behind the dataset.

______________________________________________________________________

## Performance tuning guide
Training configurations are important to ensure your expected system performance, for example preprocessing cost, training speed, and numerical stability. If you encounter problems related to aforementioned aspects, please refer to [performance tuning guide](https://github.com/motional/nuplan-devkit/blob/master/docs/performance_tuning_guide.md) to find potential solutions.

______________________________________________________________________

## Devkit structure
Our code is organized in these directories:

```
nuplan_devkit
├── ci              - Continuous integration code - not relevant for average users.
├── docs            - Readmes and other documentation of the repo and dataset.
├── nuplan          - The main source folder.
│   ├── common      - Code shared by "database" and "planning".
│   ├── database    - The core devkit used to load and render nuPlan dataset and maps.
│   ├── planning    - The stand-alone planning framework for simulation, training and evaluation.
│   ├── submission  - The submission engine used for the planning challenge.
│   └── cli         - Command line interface tools for the nuPlan database.
└── tutorials       - Interactive tutorials, see "Getting started".
```
______________________________________________________________________


## Citation
Please use the following citation when referencing [nuPlan](https://arxiv.org/abs/2106.11810):
```
@INPROCEEDINGS{nuplan, 
  title={NuPlan: A closed-loop ML-based planning benchmark for autonomous vehicles},
  author={H. Caesar, J. Kabzan, K. Tan et al.,},
  booktitle={CVPR ADP3 workshop},
  year=2021
}
```
