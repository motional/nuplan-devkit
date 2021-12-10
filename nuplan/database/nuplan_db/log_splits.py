"""
This file contains mappings from split name (e.g. "val", "test") to log names. Each log ultimately corresponds to
a list of tokens that will be used by the splitters in database/nuplan_db/log_splits.py to form all the
usual splits (e.g. train/val/test) as well as convenient stratifications of these main splits (e.g.
train.United_States/val.Singapore/train.lv/train.lv.mini)
"""
from typing import Dict, List


def nuplan_36logs_log_splits() -> Dict[str, List[str]]:
    """
    Returns the train, val and test splits for the 1000 hr dataset.
    :return: The train, val, and test splits.
    """
    train = [
        '2021.05.12.23.36.44_g1p-veh-2035',
        '2021.05.13.17.57.34_g1p-veh-2030',
        '2021.05.13.19.18.32_g1p-veh-2030',
        '2021.05.13.19.37.43_g1p-veh-2030',
        '2021.05.13.21.34.01_g1p-veh-2030',
        '2021.05.14.00.01.18_g1p-veh-2030',
        '2021.05.14.16.27.17_g1p-veh-2035',
        '2021.05.14.16.44.42_g1p-veh-2035',
        '2021.05.14.17.13.58_g1p-veh-2030',
        '2021.05.14.18.15.19_g1p-veh-2035',
        '2021.05.14.22.06.56_g1p-veh-2030',
        '2021.05.17.16.40.09_g1p-veh-2035',
        '2021.05.17.17.09.35_g1p-veh-2024',
        '2021.05.17.18.36.26_g1p-veh-2024',
        '2021.05.17.21.22.41_g1p-veh-2035',
        '2021.05.17.23.17.13_g1p-veh-2035',
        '2021.05.18.13.20.19_g1p-veh-2025',
        '2021.05.18.14.29.38_g1p-veh-2024',
        '2021.05.18.17.30.42_g1p-veh-2035',
        '2021.05.18.17.38.02_g1p-veh-2024',
        '2021.05.18.20.57.37_g1p-veh-2035',
        '2021.05.18.21.31.22_g1p-veh-2030',
        '2021.05.18.22.36.36_g1p-veh-2035',
        '2021.05.18.23.20.21_g1p-veh-2030',
        '2021.05.19.14.07.59_g1p-veh-2025',
        '2021.05.20.17.51.23_g1p-veh-2024',
        '2021.05.21.12.42.04_g1p-veh-2035',
        '2021.05.21.13.15.49_g1p-veh-2025',
        '2021.05.21.17.47.35_g1p-veh-2035',
        '2021.05.21.18.35.08_g1p-veh-2047',
    ]
    val = [
        '2021.05.12.22.28.35_g1p-veh-2035',
        '2021.05.13.22.40.44_g1p-veh-2030',
        '2021.05.14.19.40.15_g1p-veh-2030',
        '2021.05.17.22.28.24_g1p-veh-2030',
        '2021.05.18.19.20.18_g1p-veh-2030',
        '2021.05.20.14.22.28_g1p-veh-2030',
    ]
    test: List[str] = []

    return {
        'train': train,
        'val': val,
        'test': test,
    }
