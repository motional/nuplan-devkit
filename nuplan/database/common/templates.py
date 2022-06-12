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
    {
        'category': (
            "Taxonomy of object categories (e.g. 'vehicle', 'bicycle', 'pedestrian', 'traffic_cone', 'barrier', "
            "'czone_sign', 'generic_object').",
            OrderedDict(
                [
                    ("token", ("<str>", "Unique record identifier.")),
                    ("name", ("<str>", "Category name.")),
                    ("description", ("<str>", "Category description.")),
                ]
            ),
        )
    }
)

# =====================
# Sensor, logs and maps
# =====================
tables.update(
    {
        'log': (
            "Information about the log from which the data was extracted.",
            OrderedDict(
                [
                    ("token", ("<str>", "Unique record identifier.")),
                    ("vehicle_name", ("<str>", "Vehicle name.")),
                    ("date", ("<str>", "Date (YYYY-MM-DD).")),
                    ("timestamp", ("<int>", "Unix timestamp for when the log started.")),
                    ("logfile", ("<str>", "Original log file name.")),
                    ("location", ("<str>", "Area where log was captured, e.g. singapore.")),
                    ("map_version", ("<str>", "Name of map version used in this log.")),
                ]
            ),
        )
    }
)

tables.update(
    {
        'camera': (
            "The camera table contains information about the calibration and other settings of a particular camera "
            "in a particular log.",
            OrderedDict(
                [
                    ("token", ("<str>", "Unique record identifier.")),
                    (
                        "log_token",
                        ("<str>", "Foreign key. Identifies the log that uses the configuration specified here."),
                    ),
                    (
                        "channel",
                        (
                            "<str>",
                            "The camera name, which describes it's position on the car (e.g. 'CAM_F0', 'CAM_R0', 'CAM_R1', 'CAM_R2', 'CAM_B0', 'CAM_L0', 'CAM_L1', 'CAM_L2').",
                        ),
                    ),
                    ("model", ("<str>", "The camera model used.")),
                    (
                        "translation",
                        (
                            "<float> [3]",
                            "The extrinsic translation of the camera relative to the ego vehicle coordinate frame. Coordinate system origin in meters: x, y, z.",
                        ),
                    ),
                    (
                        "rotation",
                        (
                            "<float> [4]",
                            "The extrinsic rotation of the camera relative to the ego vehicle coordinate frame. Coordinate system orientation as quaternion: w, x, y, z.",
                        ),
                    ),
                    ("intrinsic", ("<float> [3, 3]", "Intrinsic camera calibration matrix.")),
                    (
                        "distortion",
                        (
                            "<float> [*]",
                            "The camera distortion parameters according to the Caltech model (k1, k2, p1, p2, k3)",
                        ),
                    ),
                    ("width", ("<int>", "The width of the camera image in pixels.")),
                    ("height", ("<int>", "The height of the camera image in pixels.")),
                ]
            ),
        )
    }
)

tables.update(
    {
        'lidar': (
            "The lidar table contains information about the calibration and other settings of a particular lidar in a "
            "particular  log.",
            OrderedDict(
                [
                    ("token", ("<str>", "Unique record identifier.")),
                    (
                        "log_token",
                        ("<str>", "Foreign key. Identifies the log that uses the configuration specified here."),
                    ),
                    ("channel", ("<str>", "Log channel name.")),
                    ("model", ("<str>", "The lidar model.")),
                    (
                        "translation",
                        (
                            "<float> [3]",
                            "The extrinsic translation of the lidar relative to the ego vehicle coordinate frame. Coordinate system origin in meters: x, y, z.",
                        ),
                    ),
                    (
                        "rotation",
                        (
                            "<float> [4]",
                            "The extrinsic rotation of the lidar relative to the ego vehicle coordinate frame. Coordinate system orientation as quaternion: w, x, y, z.",
                        ),
                    ),
                ]
            ),
        )
    }
)


# ===========
# Extractions
# ===========

tables.update(
    {
        'ego_pose': (
            "Ego vehicle pose at a particular timestamp.",
            OrderedDict(
                [
                    ("token", ("<str>", "Unique record identifier.")),
                    ("log_token", ("<str>", "Foreign key. Identifies the log which this ego pose is a part of.")),
                    ("timestamp", ("<int>", "Unix timestamp.")),
                    ("x", ("<float>", "Ego vehicle location center_x in global coordinates(in meters).")),
                    ("y", ("<float>", "Ego vehicle location center_y in global coordinates(in meters).")),
                    ("z", ("<float>", "Ego vehicle location center_z in global coordinates(in meters).")),
                    ("qw", ("<float>", "Ego vehicle orientation in quaternions in global coordinates.")),
                    ("qx", ("<float>", "Ego vehicle orientation in quaternions in global coordinates.")),
                    ("qy", ("<float>", "Ego vehicle orientation in quaternions in global coordinates.")),
                    ("qz", ("<float>", "Ego vehicle orientation in quaternions in global coordinates.")),
                    ("vx", ("<float>", "Ego vehicle velocity x in local coordinates(in m/s).")),
                    ("vy", ("<float>", "Ego vehicle velocity y in local coordinates(in m/s).")),
                    ("vz", ("<float>", "Ego vehicle velocity z in local coordinates(in m/s).")),
                    ("acceleration_x", ("<float>", "Ego vehicle acceleration x in local coordinates(in m/s2).")),
                    ("acceleration_y", ("<float>", "Ego vehicle acceleration y in local coordinates(in m/s2).")),
                    ("acceleration_z", ("<float>", "Ego vehicle acceleration z in local coordinates(in m/s2).")),
                    ("angular_rate_x", ("<float>", "Ego vehicle angular rate x in local coordinates.")),
                    ("angular_rate_y", ("<float>", "Ego vehicle angular rate y in local coordinates.")),
                    ("angular_rate_x", ("<float>", "Ego vehicle angular rate z in local coordinates.")),
                    (
                        "epsg",
                        (
                            "<int>",
                            "Ego vehicle epsg. Epsg is a latitude/longitude coordinate system based on "
                            "the Earth's center of mass.",
                        ),
                    ),
                ]
            ),
        )
    }
)
