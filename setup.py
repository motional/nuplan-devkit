import os

import setuptools

# Change directory to allow installation from anywhere
script_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_folder)

# Get requirements
requirements_path = os.path.join(script_folder, 'requirements.txt')
requirements = []
with open(requirements_path, 'r') as f:
    requirements += f.read().splitlines()

# Installs
setuptools.setup(
    name='nuplan-devkit',
    version='0.2.0',
    author='The nuPlan team @ Motional',
    author_email='nuscenes@motional.com',
    description='The official devkit of the nuPlan dataset (www.nuPlan.org).',
    url='https://github.com/motional/nuplan-devkit',
    python_requires='>=3.9',
    install_requires=requirements,
    packages=setuptools.find_packages(script_folder, exclude=['*test', 'bazel-nuplan_devkit*', 'bazel-nuplan-devkit*']),
    package_data={'': ['*.json']},
    include_package_data=True,
    classifiers=[
        'Programming Language :: Python :: 3.9',
        'Operating System :: OS Independent',
        'License :: Free for non-commercial use',
    ],
    entry_points={"console_scripts": ["nuplan_cli = nuplan.cli.nuplan_cli:main"]},
    license='apache-2.0',
)
