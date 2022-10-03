# Installation
We provide step-by-step instructions to install the nuPlan devkit. 
For a high-level overview, please read the [general readme](https://github.com/motional/nuplan-devkit#readme) first.
- [Download](#download-the-devkit)
- [Install Python](#install-python)
- [Install virtual environment](#install-virtual-environment)
- [Install the devkit](#install-the-devkit)
- [Install required packages](#install-required-packages)
- [Setup environment variables](#setup-environment-variables)

## Download the devkit
Download the devkit and move inside the folder:
```
cd && git clone https://github.com/motional/nuplan-devkit.git && cd nuplan-devkit
```
The above will download the files to your home directory. While you can change this to an arbitrary directory, the rest of our tutorials assumes that you are using the home directory. 

-----
## Install Python
The devkit is tested for Python 3.9 on Ubuntu.

- For **Ubuntu**: If the right Python version is not already installed on your system, install it by running:
   ```
   sudo apt install python-pip
   sudo add-apt-repository ppa:deadsnakes/ppa
   sudo apt-get update
   sudo apt-get install python3.9
   sudo apt-get install python3.9-dev
   ```
- For **Mac OS** download and install from `https://www.python.org/downloads/mac-osx/`.

-----
## Install virtual environment
Next we setup a virtual environment. We recommend Conda for this purpose.

### Install miniconda
See the [official Miniconda page](https://conda.io/en/latest/miniconda.html).

### Create a Conda environment
We create a new Conda environment using environment.yml.
```
conda env create -f environment.yml
```

### Activate the environment
If you are inside the Conda environment, your shell prompt should look like: `(nuplan) user@computer:~$`
Going forward, we shall always assume the Conda environment is active.
If that is not the case, you can enable the virtual environment using:
```
conda activate nuplan 
```
To deactivate the virtual environment, use:
```
conda deactivate
```

-----
## Install the devkit
### Option A: Install PIP package from remote
**Note:** This option is not yet supported. It will be added soon.

**For beginners**, the easiest option is to install the PIP package:
```
pip install nuplan-devkit
```
This installs the devkit and all of its dependencies.

### Option B: Install PIP package from local
**We recommend** that you install the local devkit as a PIP package:
```
pip install -e .
```
This installs the devkit and all of its dependencies.
Note that the editable mode (`-e`) is optional and means that the code is used in-place, rather than being copied elsewhere, and can be modified for easy development.

### Option C: Run source code directly
**Alternatively**, if you don't want to use the pip package, you can manually add the `nuplan-devkit` directory to your `PYTHONPATH` environmental variable, by adding the following to your `~/.bashrc`:
```
export PYTHONPATH="${PYTHONPATH}:$HOME/nuplan-devkit"
```
To activate these changes you then need to run:
```
source ~/.bashrc
```
To install the dependencies, run the following command:
```
pip install -r requirements_torch.txt
pip install -r requirements.txt
``` 

-----
## Change default directories
As described in the [general readme](https://github.com/motional/nuplan-devkit/blob/master/README.md), the default nuPlan directories are:
```
~/nuplan/dataset    -   The dataset folder. Can be read-only.
~/nuplan/exp        -   The experiment and cache folder. Must have read and write access.
```
If you want to change these on your system, you need to set the corresponding environment variables in your `~/.bashrc`, e.g.:
```
export NUPLAN_DATA_ROOT="$HOME/nuplan/dataset"
export NUPLAN_MAPS_ROOT="$HOME/nuplan/dataset/maps"
export NUPLAN_EXP_ROOT="$HOME/nuplan/exp"
```
This step is also required if you want to run any of the unit tests in the devkit.

-----
That's it you should be good to go!
