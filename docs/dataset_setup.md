# Dataset setup

## Download
To download nuPlan you need to go to the [Download page](https://nuplan.org/nuplan#download), 
create an account and agree to the [Terms of Use](https://www.nuplan.org/terms-of-use).
After logging in you will see multiple archives. 
For the devkit to work you will need to download *all* archives.
Please unpack the archives to the `~/nuplan/dataset` folder.

## Filesystem hierarchy
The final hierarchy should look as follows (depending on the splits downloaded above):
```angular2html
~/nuplan
├── exp
│   └── ${USER}
│       ├── cache
│       │   └── <cached_tokens>
│       └── exp
│           └── my_nuplan_experiment
└── dataset
    ├── maps
    │   ├── nuplan-maps-v1.0.json
    │   ├── sg-one-north
    │   │   └── 9.17.1964
    │   │       └── map.gpkg
    │   ├── us-ma-boston
    │   │   └── 9.12.1817
    │   │       └── map.gpkg
    │   ├── us-nv-las-vegas-strip
    │   │   └── 9.15.1915
    │   │       └── map.gpkg
    │   └── us-pa-pittsburgh-hazelwood
    │       └── 9.17.1937
    │           └── map.gpkg
    └── nuplan-v1.0
        ├── mini
        │   ├── 2021.05.12.22.00.38_veh-35_01008_01518.db
        │   ├── 2021.06.09.17.23.18_veh-38_00773_01140.db
        │   ├── ...
        │   └── 2021.10.11.08.31.07_veh-50_01750_01948.db
        └── trainval
            ├── 2021.05.12.22.00.38_veh-35_01008_01518.db
            ├── 2021.06.09.17.23.18_veh-38_00773_01140.db
            ├── ...
            └── 2021.10.11.08.31.07_veh-50_01750_01948.db
```

If you want to use another folder, you can set the corresponding [environment variable](https://github.com/motional/nuplan-devkit/blob/master/docs/installation.md).
