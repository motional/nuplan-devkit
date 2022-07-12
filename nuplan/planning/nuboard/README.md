<div align="center">

# nuBoard 
Welcome to the nuBoard repository!

nuBoard is a Python repo based on Bokeh. It is a visualization tool for nuPlan metrics and simulation. 
</div>

______________________________________________________________________

## nuBoard structure
Our code is organized in these directories:

```
nuboard
├── base              - Data and base classes shared by other code. 
├── resource          - Resources such as images, javascript and css files.
├── tabs              - Python code for each page.
├── templates         - HTML code for each page.
├── utils             - Common fucntions used by other code.
├── nuboard.py        - Code to start a Bokeh application server.
└── style.py          - Default color and size for templates.
```

______________________________________________________________________

## Usage

To start using nuBoard, please run the following command in cli:
- `python nuplan_devkit/nuplan/planning/script/run_nuboard.py`
- nuBoard takes nuboard files from simulation as input. 
To import nuboard files, we have two options:
  - GUI: Use the `upload file` button in the side navigation. However, this method is not supported if: 
    - Run nuBoard on a remote server. 
    - Move simulation data to different folders. 
  - Command line: Set `simulation_path` in the command:
    - `python nuplan_devkit/nuplan/planning/script/run_nuboard.py simulation_path="[path_1, path_2]"`

______________________________________________________________________
## Pages
There are three pages in nuBoard, which are `Overview`, `Histograms`, and `Scenarios`.
- Overview:
  - Show a summary of aggregated metric scores for each scenario type and planner. 
- Histograms:
  - Display distributions of available metric scores for different planners across all scenarios.
- Scenarios:
  - Visualize simulation and metric scores for a selected scenario.
______________________________________________________________________
