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

## Changelog
- Dec. 10, 2021: Devkit v0.1.0: Release of the initial teaser dataset (v0.1) and corresponding devkit and maps (v0.1). See [Teaser release](#teaser-release) for more information.

______________________________________________________________________

## Teaser release
On Dec. 10 2021 we released the nuPlan teaser dataset and devkit. This is meant to be a **public beta** version. 
We are aware of several limitations of the current dataset and devkit. 
Nevertheless we have chosen to make this teaser available to the public for early consultation and to receive feedback on how to improve it.
We appreciate your feedback as a [Github issue](https://github.com/motional/nuplan-devkit/issues).

**Note:** All interfaces are subject to change for the full release! No backward compatibility can be guaranteed.

Below is a list of upcoming features for the full release:
- The teaser dataset includes 200h of data from Las Vegas, we will be releasing the full 1500h dataset that also includes data from Singapore, Boston or Pittsburgh in early 2022.
- The full release will also include the sensor data for 150h (10% of the total dataset).
- Localization, perception scenario tags and traffic lights will be improved in future releases.
- The full release will have an improved dashboard, closed-loop training, advanced planning baselines, end-to-end planners, ML smart agents, RL environment, as well as more metrics and scenarios.

______________________________________________________________________

## Devkit structure
Our code is organized in these directories:

```
nuplan_devkit
├── ci              - Continuous integration code. Not relevant for average users.
├── docs            - Readmes and other documentation of the repo and dataset.
├── nuplan          - The main source folder.
│   ├── common      - Code shared by "database" and "planning".
│   ├── database    - The core devkit used to load and render nuPlan dataset and maps.
│   └── planning    - The stand-alone planning framework for simulation, training and evaluation.
└── tutorials       - Interactive tutorials, see "Getting started".
```

______________________________________________________________________

## Devkit setup
Please refer to the [installation page](https://github.com/motional/nuplan-devkit/blob/master/docs/installation.md) for detailed instructions on how to setup the devkit.

______________________________________________________________________

## Dataset setup
To download nuPlan you need to go to the [Download page](https://nuplan.org/nuplan#download), 
create an account and agree to the [Terms of Use](https://www.nuplan.org/terms-of-use).
After logging in you will see multiple archives. 
For the devkit to work you will need to download *all* archives.
Please unpack the archives to the `~/nuplan/dataset` folder.
Eventually you should have the following folder structure:
```
~/nuplan/dataset    -   The dataset folder. Can be read-only.
    nuplan_v*.db	  -	  SQLite database that includes all metadata
    maps	          -	  Folder for all map files
    <missing>       -   Sensor data will be added in the future
~/nuplan/exp        -   The experiment and cache folder. Must have read and write access.
```
If you want to use another folder, you can set the corresponding [environment variable](https://github.com/motional/nuplan-devkit/blob/master/docs/installation.md) or specify the `data_root` parameter of the NuPlanDB class (see tutorial).

______________________________________________________________________

## Getting started
Please follow these steps to make yourself familiar with the nuPlan dataset:
- Familiarize yourself with the main [features of nuPlan](https://www.nuplan.org) and the [dataset description](https://www.nuplan.org/nuplan).
- Setup the [devkit](#dataset-setup) and [dataset](#dataset-setup) as described above.
- Walk through the tutorials in [this folder](https://github.com/motional/nuplan-devkit/blob/master/tutorials/) or run them yourself using:
```
jupyter notebook ~/nuplan-devkit/tutorials/<filename>.ipynb
```
Replace &lt;filename&gt; with one of the following:
- `nuplan_framework.ipynb`: This is the main tutorial for anyone who wants to dive right into ML planning.
  It describes how to 1) train an ML planner, 2) simulate it, 3) measure the performance and 4) visualize the results.


- Read the [nuPlan paper](https://www.nuplan.org/publications) to understand the details behind the dataset.

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
