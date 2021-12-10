"""
Shared templates for the database schemas.
"""
import copy
from collections import OrderedDict
from typing import Dict, Tuple

base_tables: Dict[str, Tuple[str, Dict[str, Tuple[str, str]]]] = OrderedDict([])

tables = copy.deepcopy(base_tables)


# =========
# Semantics
# =========
tables.update(
    {'category':
        ("A category within our taxonomy. Includes both things (e.g. cars) or stuff (e.g. lanes, sidewalks). \n"
         "Subcategories are delineated by a period.",
            OrderedDict([
                ("token", ("<str>", "Unique record identifier.")),
                ("name", ("<str>", "Category name. Subcategories indicated by period. E.g. 'vehicle.car'.")),
                ("description", ("<str>", "Category description.")),
            ]))})

# =====================
# Sensor, logs and maps
# =====================
tables.update(
    {'log':
        ("Information about the log from which the data was extracted.",
            OrderedDict([
                ("token", ("<str>", "Unique record identifier.")),
                ("vehicle_name", ("<str>", "Vehicle name.")),
                ("vehicle_type", ("<str>", "Vehicle type.")),
                ("date", ("<str>", "Date (YYYY-MM-DD).")),
                ("timestamp", ("<int>", "Unix timestamp for when log started.")),
                ("logfile", ("<str>", "Original log file name.")),
                ("location", ("<str>", "Area where log was captured, e.g. 'One North'.")),
                ("map_version", ("<str>", "Name of map version used in this log. E.g. 'onenorth-2.2.4'.")),
            ]))})

tables.update(
    {'camera':
        ("Defines a calibrated camera used to record a particular log.",
            OrderedDict([
                ("token", ("<str>", "Unique record identifier.")),
                ("log_token", ("<str>", "Foreign key.")),
                ("channel", ("<str>", "Log channel name for this camera.")),
                ("model", ("<str>", "Camera model, e.g. 'Basler', 'Sony IMX'.")),
                ("translation", ("<float> [3]", "Coordinate system origin: x, y, z.")),
                ("rotation", ("<float> [4]", "Coordinate system orientation in quaternions.")),
                ("intrinsic", ("<float> [3, 3]", "Intrinsic camera calibration matrix.")),
                ("distortion", ("<float> [*]", "Distortion per convention of the CalTech camera calibration toolbox. "
                                               "Can be 5-10 coefficients.")),
                ("width", ("<int>", "Image width in pixels.")),
                ("height", ("<int>", "Image height in pixels.")),
            ]))})

tables.update(
    {'lidar':
        ("Defines a calibrated lidar used to record a particular log.",
            OrderedDict([
                ("token", ("<str>", "Unique record identifier.")),
                ("log_token", ("<str>", "Foreign key.")),
                ("channel", ("<str>", "Log channel name.")),
                ("model", ("<str>", "The lidar model.")),
                ("translation", ("<float> [3]", "Coordinate system origin: x, y, z.")),
                ("rotation", ("<float> [4]", "Coordinate system orientation in quaternions.")),
                ("max_nbr_points", ("<int>", "Maximum number of points in the captured lidar point clouds.")),
            ]))})


# ===========
# Extractions
# ===========

tables.update(
    {'ego_pose':
        ("Ego vehicle pose at a particular timestamp. Given with respect to global coordinate system.",
            OrderedDict([
                ("token", ("<str>", "Unique record identifier.")),
                ("timestamp", ("<int>", "Unix timestamp.")),
                ("x", ("<float>", "Ego vehicle location center_x.")),
                ("y", ("<float>", "Ego vehicle location center_y.")),
                ("z", ("<float>", "Ego vehicle location center_z.")),
                ("qw", ("<float>", "Ego vehicle orientation in quaternions.")),
                ("qx", ("<float>", "Ego vehicle orientation in quaternions.")),
                ("qy", ("<float>", "Ego vehicle orientation in quaternions.")),
                ("qz", ("<float>", "Ego vehicle orientation in quaternions.")),
                ("vx", ("<float>", "Ego vehicle velocity x.")),
                ("vy", ("<float>", "Ego vehicle velocity y.")),
                ("vz", ("<float>", "Ego vehicle velocity z.")),
                ("acceleration_x", ("<float>", "Ego vehicle acceleration x.")),
                ("acceleration_y", ("<float>", "Ego vehicle acceleration y.")),
                ("acceleration_z", ("<float>", "Ego vehicle acceleration z.")),
                ("angular_rate_x", ("<float>", "Ego vehicle angular rate x.")),
                ("angular_rate_y", ("<float>", "Ego vehicle angular rate y.")),
                ("angular_rate_x", ("<float>", "Ego vehicle angular rate z.")),
                ("epsg", ("<int>", "Ego vehicle epsg.")),
                ("health", ("<bool>", "Ego vehicle health.")),
                ("log_token", ("<str>", "Foreign key.")),
            ]))})
