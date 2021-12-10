import os
from typing import List

import setuptools

# Change directory to allow installation from anywhere
script_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_folder)

# Get requirements
requirements_path = os.path.join(script_folder, 'requirements.txt')
requirements = []
with open(requirements_path, 'r') as f:
    requirements += f.read().splitlines()


def get_dirlist(_root_dir: str) -> List[str]:
    dirlist = []

    with os.scandir(_root_dir) as rit:
        for entry in rit:
            if not entry.name.startswith('.') and entry.is_dir():
                relative_dir = entry.path.replace(sdk_dir + '/', '')
                dirlist.append(relative_dir)
                dirlist += get_dirlist(entry.path)

    return dirlist


# Get subfolders recursively
sdk_dir = script_folder
packages = [d.replace('/', '.').replace('{}.'.format(sdk_dir), '') for d in get_dirlist(sdk_dir)]

# Filter out Python cache folders
packages = [p for p in packages if not p.endswith('__pycache__') and not p.endswith('.egg-info')]

# Installs
setuptools.setup(
    name='nuplan-devkit',
    version='0.1.0',
    author='The nuPlan team @ Motional',
    author_email='nuscenes@motional.com',
    description='The official devkit of the nuPlan dataset (www.nuPlan.org).',
    url='https://github.com/motional/nuplan-devkit',
    python_requires='>=3.9',
    install_requires=requirements,
    packages=packages,
    package_data={'': ['*.json']},
    include_package_data=True,
    classifiers=[
        'Programming Language :: Python :: 3.9',
        'Operating System :: OS Independent',
        'License :: Free for non-commercial use'
    ],
    license='apache-2.0'
)
