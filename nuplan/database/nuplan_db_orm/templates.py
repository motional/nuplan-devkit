"""
Templates for the multi-modal database schema.
"""

import copy
from collections import OrderedDict

from nuplan.database.common.templates import tables as base_tables

tables = copy.deepcopy(base_tables)

# ===========
# Extractions
# ===========

tables.update(
    {
        'image': (
            "The image table stores metadata to retrieve a single image taken from a camera.  "
            "It does not store the image itself.",
            OrderedDict(
                [
                    ("token", ("<str>", "Unique record identifier.")),
                    (
                        "next_token",
                        ("<str>", "Foreign key. Record that follows this in time. Empty if last image of log."),
                    ),
                    (
                        "prev_token",
                        ("<str>", "Foreign key. Record that precedes this in time. Empty if first image of log."),
                    ),
                    (
                        "ego_pose_token",
                        ("<str>", "Foreign key. Indicates the ego pose at the time that the " "image was captured."),
                    ),
                    (
                        "camera_token",
                        ("<str>", "Foreign key. Indicates the camera settings used to capture the image."),
                    ),
                    ("filename_jpg", ("<str>", "Relative path to image file.")),
                    ("timestamp", ("<int>", "Unix timestamp.")),
                ]
            ),
        )
    }
)

tables.update(
    {
        'lidar_pc': (
            "The lidar_pc table stores metadata to retrieve a single pointcloud taken from a lidar. It does not store "
            "the pointcloud itself. The pointcloud combines sweeps from multiple different lidars that are aggregated "
            "on the car.",
            OrderedDict(
                [
                    ("token", ("<str>", "Unique record identifier.")),
                    (
                        "next_token",
                        ("<str>", "Foreign key. Record that follows this in time. Empty if end of log."),
                    ),
                    (
                        "prev_token",
                        ("<str>", "Foreign key. Record that precedes this in time. Empty if start of log."),
                    ),
                    ("scene_token", ("<str>", "Foreign key. References the scene that contains this lidar_pc.")),
                    (
                        "ego_pose_token",
                        (
                            "<str>",
                            "Foreign key. Indicates the ego pose at the time that the " "pointcloud was captured.",
                        ),
                    ),
                    (
                        "lidar_token",
                        ("<str>", "Foreign key. Indicates the lidar settings used to capture the " "pointcloud."),
                    ),
                    ("filename", ("<str>", "Relative path to pointcloud blob.")),
                    ("timestamp", ("<int>", "Unix timestamp.")),
                ]
            ),
        )
    }
)

# ===========
# Annotations
# ===========

tables.update(
    {
        'track': (
            "An object track, e.g. particular vehicle. This table is an enumeration of all unique object "
            "instances we observed. \n",
            OrderedDict(
                [
                    ("token", ("<str>", "Unique record identifier.")),
                    ("category_token", ("<str>", "Foreign key. Object instance category.")),
                    ("width", ("<float>", "Bounding box size width(in meters).")),
                    ("length", ("<float>", "Bounding box size length(in meters).")),
                    ("height", ("<float>", "Bounding box size height(in meters).")),
                ]
            ),
        )
    }
)

tables.update(
    {
        'lidar_box': (
            "The lidar_box table stores individual object annotations in the form of 3d bounding boxes. These boxes are "
            "extracted from an Offline Perception system and tracked across time using the `track` table. Since the "
            "perception system uses lidar as an input modality, all lidar_boxes are linked to the lidar_pc table.",
            OrderedDict(
                [
                    ("token", ("<str>", "Unique record identifier.")),
                    (
                        "lidar_pc_token",
                        ("<str>", "Foreign key. References the lidar_pc where this object box was detected."),
                    ),
                    (
                        "track_token",
                        ("<str>", "Foreign key. References the object track that this box was associated with."),
                    ),
                    (
                        "next_token",
                        ("<str>", "Foreign key. Record that follows this in time. Empty if end of track."),
                    ),
                    (
                        "prev_token",
                        ("<str>", "Foreign key. Record that precedes this in time. Empty if start of track."),
                    ),
                    ("x", ("<float>", "Bounding box location center_x in global coordinates(in meters).")),
                    ("y", ("<float>", "Bounding box location center_y in global coordinates(in meters).")),
                    ("z", ("<float>", "Bounding box location center_z in global coordinates(in meters).")),
                    ("width", ("<float>", "Bounding box size width(in meters).")),
                    ("length", ("<float>", "Bounding box size length(in meters).")),
                    ("height", ("<float>", "Bounding box size height(in meters).")),
                    (
                        "vx",
                        (
                            "<float>",
                            "Bounding box velocity v_x(in m/s). This quantity is estimated by a Neural "
                            "Network and may be inconsistent with future positions..",
                        ),
                    ),
                    (
                        "vy",
                        (
                            "<float>",
                            "Bounding box velocity v_y(in m/s). This quantity is estimated by a Neural "
                            "Network and may be inconsistent with future positions..",
                        ),
                    ),
                    (
                        "vz",
                        (
                            "<float>",
                            "Bounding box velocity v_z(in m/s). This quantity is estimated by a Neural "
                            "Network and may be inconsistent with future positions..",
                        ),
                    ),
                    ("yaw", ("<float>", "Bounding box orientation yaw.")),
                    ("confidence", ("<float>", "Bounding box confidence.")),
                ]
            ),
        )
    }
)

# ===========
# Scenarios
# ===========
tables.update(
    {
        'scene': (
            "A `scene` is a snippet of upto 20s duration from a `log`. Every scene stores a goal for the ego vehicle "
            "that is a future ego pose from beyond that scene. In addition we provide a sequence of road blocks to"
            " navigate towards the goal.",
            OrderedDict(
                [
                    ("token", ("<str>", "Unique record identifier.")),
                    ("log_token", ("<str>", "Foreign key. The log that this scene is a part of.")),
                    ("name", ("<str>", "Unique Scene Name.")),
                    (
                        "goal_ego_pose_token",
                        ("<str>", "Foreign key. A future ego pose that serves as the goal for this scene."),
                    ),
                    (
                        "roadblock_ids",
                        (
                            "<str>",
                            "A sequence of roadblock ids separated by commas. The ids can be looked up in the Map API.",
                        ),
                    ),
                ]
            ),
        )
    }
)

tables.update(
    {
        'scenario_tag': (
            "An instance of a scenario extracted from the database. Scenarios are linked to lidar_pcs and represent "
            "the point in time when a scenario miner was triggered,  e.g. when simultaneously being in two lanes for "
            "CHANGING_LANE. Some scenario types optionally can refer to an agent that the ego is interacting with "
            "(e.g. in STOPPING_WITH_LEAD).",
            OrderedDict(
                [
                    ("token", ("<str>", "Unique record identifier.")),
                    ("lidar_pc_token", ("<str>", "Foreign key. The lidar_pc at which this scenario was triggered.")),
                    (
                        "type",
                        (
                            "<str>",
                            "Type of Scenario. Ex on_intersection, starting_unprotected_cross_turn etc. "
                            "There are around 70 of these scenario types.",
                        ),
                    ),
                    (
                        "agent_track_token",
                        (
                            "<str>",
                            "Foreign key. Token of the agent interacting with the ego vehicle. "
                            "Can be none if there is no interacting agent.",
                        ),
                    ),
                ]
            ),
        )
    }
)

# ===========
# Traffic Light Status
# ===========

tables.update(
    {
        'traffic_light_status': (
            "We use the observed motion of agents in the environment to estimate the status of a traffic light. "
            "For simplicity the status is stored in a particular lane connector "
            "(an idealized path across an intersection), rather than in the traffic light itself.",
            OrderedDict(
                [
                    ("token", ("<str>", "Unique record identifier.")),
                    (
                        "lidar_pc_token",
                        (
                            "<str>",
                            "Foreign key. The traffic light status is based on the motion of the agents detected in this lidar_pc.",
                        ),
                    ),
                    ("lane_connector_id", ("<int>", "ID of lane connector in the map.")),
                    ("status", ("<str>", "Status of traffic light. Can be green, red or unknown.")),
                ]
            ),
        )
    }
)
