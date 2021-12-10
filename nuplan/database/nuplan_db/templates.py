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
    {'image':
        ("An image.",
            OrderedDict([
                ("token", ("<str>", "Unique record identifier.")),
                ("next_token", ("<str>", "Foreign key. Record that follows this in time. Empty if end of extraction.")),
                ("prev_token", ("<str>", "Foreign key. Record that precedes this in time. Empty if start of "
                                         "extraction.")),
                ("ego_pose_token", ("<str>", "Foreign key.")),
                ("camera_token", ("<str>", "Foreign key.")),
                ("filename_jpg", ("<str>", "Relative path to image file.")),
                ("timestamp", ("<int>", "Unix time stamp.")),
            ]))})

tables.update(
    {'lidar_pc':
        ("A lidar point cloud.",
            OrderedDict([
                ("token", ("<str>", "Unique record identifier.")),
                ("next_token", ("<str>", "Foreign key. Record that follows this in time. Empty if end of extraction.")),
                ("prev_token", ("<str>", "Foreign key. Record that precedes this in time. Empty if start of "
                                         "extraction.")),
                ("scene_token", ("<str>", "Foreign key.")),
                ("ego_pose_token", ("<str>", "Foreign key.")),
                ("lidar_token", ("<str>", "Foreign key.")),
                ("filename", ("<str>", "Relative path to data blob.")),
                ("timestamp", ("<int>", "Unix time stamp.")),
            ]))})

# ===========
# Annotations
# ===========

tables.update(
    {'track':
        ("An object track, e.g. particular vehicle. This table is an enumeration of all object "
         "instances we observed. \n",
            OrderedDict([
                ("token", ("<str>", "Unique record identifier.")),
                ("category_token", ("<str>", "Foreign key. Object instance category.")),
                ("width", ("<float>", "Bounding box size width.")),
                ("length", ("<float>", "Bounding box size length.")),
                ("height", ("<float>", "Bounding box size height.")),
                ("confidence", ("<float>", "Bounding box confidence.")),
            ]))})

tables.update(
    {'lidar_box':
        ("A 3D geometry defining the position of an object seen in a sample. Location is given in the world coordinate "
         "system.",
            OrderedDict([
                ("token", ("<str>", "Unique record identifier.")),
                ("lidar_pc_token", ("<str>", "Foreign key.")),
                ("track_token", ("<str>", "Foreign key.")),
                ("x", ("<float>", "Bounding box location center_x.")),
                ("y", ("<float>", "Bounding box location center_y.")),
                ("z", ("<float>", "Bounding box location center_z.")),
                ("width", ("<float>", "Bounding box size width.")),
                ("length", ("<float>", "Bounding box size length.")),
                ("height", ("<float>", "Bounding box size height.")),
                ("vx", ("<float>", "Bounding box velocity v_x.")),
                ("vy", ("<float>", "Bounding box velocity v_y.")),
                ("vz", ("<float>", "Bounding box velocity v_z.")),
                ("roll", ("<float>", "Bounding box orientation roll.")),
                ("pitch", ("<float>", "Bounding box orientation pitch.")),
                ("yaw", ("<float>", "Bounding box orientation yaw.")),
                ("confidence", ("<float>", "Bounding box confidence.")),
            ]))})

# ===========
# Scenarios
# ===========
tables.update(
    {'scene':
     (
         "Identified Scene occurances.",
         OrderedDict([
             ("token", ("<str>", "Unique record identifier.")),
             ("log_token", ("<str>", "Foreign key.")),
             ("name", ("<str>", "Unique Scene Name.")),
             ("goal_ego_pose_token", ("<str>", "Foreign key.")),
         ]))})

tables.update(
    {'scenario_tag':
     (
         "Tags associated with each Scenario.",
         OrderedDict([
             ("token", ("<str>", "Unique record identifier.")),
             ("lidar_pc_token", ("<str>", "Foreign key.")),
             ("type", ("<str>", "Type of Scenario.")),
             ("agent_track_token", ("<str>", "Foreign key.")),
             ("map_object_id", ("<int>", "Map Object ID.")),
         ]))})

# ===========
# Traffic Light Status
# ===========

tables.update(
    {'traffic_light_status':
     (
         "Status of Traffic Lights.",
         OrderedDict([
             ("token", ("<str>", "Unique record identifier.")),
             ("lidar_pc_token", ("<str>", "Foreign key.")),
             ("stop_line_id", ("<int>", "ID of stopline.")),
             ("lane_connector_id", ("<int>", "ID of Lane Connector.")),
             ("status", ("<str>", "Status of Traffic Light.")),
         ]))})
