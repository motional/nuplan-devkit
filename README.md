<div align="center">

# nuPlan
**The world's first benchmark for autonomous vehicle planning.**

______________________________________________________________________

<p align="center">
  <a href="https://www.nuplan.org/">Website</a> •
  <a href="https://www.nuscenes.org/nuplan#download">Download</a> •
  <a href="#citation">Citation</a><br>
  <a href="#changelog">Changelog</a> •
  <a href="#devkit-structure">Structure</a><br>
  <a href="https://github.com/motional/nuplan-devkit/blob/master/docs/installation.md">Setup</a> •
  <a href="https://github.com/motional/nuplan-devkit/blob/master/tutorials/nuplan_framework.ipynb">Tutorial</a>
</p>

[![python](https://img.shields.io/badge/python-3.8%20%7C%203.9-blue.svg)]()
[![license](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/motional/nuplan-devkit/blob/master/LICENSE.txt)

______________________________________________________________________

<br>

<p align="center"><img src="https://www.nuplan.org/static/media/nuPlan_final.3fde7586.png" width="500px"></p>

</div>

______________________________________________________________________
## Planning challenges
Stay tuned for the upcoming nuPlan ML planning challenges!
______________________________________________________________________

## Changelog
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
Please refer to the [installation page](https://github.com/motional/nuplan-devkit/blob/master/docs/installation.md) for detailed instructions on how to setup the devkit.

Please refer to the [dataset page](https://github.com/motional/nuplan-devkit/blob/master/docs/dataset_setup.md) for detailed instructions on how to download and setup the dataset.

______________________________________________________________________

## Getting started
Please follow these steps to make yourself familiar with the nuPlan dataset:
- Familiarize yourself with the main [features of nuPlan](https://www.nuplan.org) and the [dataset description](https://www.nuplan.org/nuplan).
- Setup the devkit and dataset as described [above](#devkit-and-dataset-setup).
- Walk through the tutorials in [this folder](https://github.com/motional/nuplan-devkit/blob/master/tutorials/) or run them yourself using `jupyter notebook ~/nuplan-devkit/tutorials/<filename>.ipynb` and replacing `<filename>` with the tutorial's name. The following tutorials are available:
  - `nuplan_framework.ipynb`: Main tutorial for anyone who wants to dive right into ML planning.
  It describes how to 1) train an ML planner, 2) simulate it, 3) measure the performance and 4) visualize the results.
  - `nuplan_scenario_visualization.ipynb`: Tutorial for visualizing various scenario types contained within the nuPlan dataset (e.g. unprotected turns, lane changes, interactions with pedestrians and more).
  - `nuplan_planner_tutorial.ipynb`: Tutorial with instructions on how to develop and simulate a planner from scratch within the nuPlan framework.

- Familiarize yourself with the nuPlan CLI, which gets installed by installing the devkit using `pip` (editable and not)
  by running:
  ```
  nuplan_cli --help
  nuplan_cli COMMAND --help
  ```
- Read the [nuPlan paper](https://www.nuplan.org/publications) to understand the details behind the dataset.

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
│   └── planning    - The stand-alone planning framework for simulation, training and evaluation.
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
