from __future__ import annotations

import logging
from functools import cached_property
from typing import Any, List, Optional, Set, Tuple, Type, cast

from nuplan.database import nuplan_db_orm
from nuplan.database.common.db import DB, Table
from nuplan.database.maps_db.gpkg_mapsdb import GPKGMapsDB
from nuplan.database.nuplan_db_orm.camera import Camera
from nuplan.database.nuplan_db_orm.category import Category
from nuplan.database.nuplan_db_orm.ego_pose import EgoPose
from nuplan.database.nuplan_db_orm.image import Image
from nuplan.database.nuplan_db_orm.lidar import Lidar
from nuplan.database.nuplan_db_orm.lidar_box import LidarBox
from nuplan.database.nuplan_db_orm.lidar_pc import LidarPc
from nuplan.database.nuplan_db_orm.log import Log
from nuplan.database.nuplan_db_orm.scenario_tag import ScenarioTag
from nuplan.database.nuplan_db_orm.scene import Scene
from nuplan.database.nuplan_db_orm.templates import tables as nuplandb_table_templates
from nuplan.database.nuplan_db_orm.track import Track
from nuplan.database.nuplan_db_orm.traffic_light_status import TrafficLightStatus

logger = logging.getLogger(__name__)

MICROSECONDS_IN_A_SECOND = 1000000


class NuPlanDB(DB):
    """
    Database for loading and accessing nuPlan .db files.

    It provides lookups and get methods to access the SQL database tables and metadata.
    In addition, it provides functionality for automatically downloading a database from a remote (e.g. S3)
    if not present in the local filesystem and storing it.

    A database file is in the form of "<log_date>_<vehicle_number>_<snippet_start>_<snippet_end>.db"
    for example "2021.05.24.12.28.29_veh-12_04802_04907.db" - each database represents a log snippet of
    variable duration (e.g. 60sec or 30min) that was manually driven by an expert driver.

    The nuPlan dataset comprises of thousands of .db files.
    These can be collectively loaded and accessed from the `NuPlanDBWrapper` class and be used in training/simulation.
    """

    def __init__(
        self,
        data_root: str,
        load_path: str,
        maps_db: Optional[GPKGMapsDB] = None,
        verbose: bool = False,
    ):
        """
        Load database and create reverse indexes and shortcuts.
        :param data_root: Local data root for loading (or storing if downloaded) the database.
        :param load_path: Local or remote (S3) filename of the database to be loaded
        :param maps_db: Map database associated with this database.
        :param verbose: Whether to print status messages during load.
        """
        self._data_root = data_root
        self._load_path = load_path
        self._maps_db = maps_db
        self._verbose = verbose

        # Initialize parent class
        table_names = list(nuplandb_table_templates.keys())
        nuplandb_models_dict = {}
        nuplandb_models_dict["default"] = "models"
        nuplandb_models_dict["Camera"] = "camera"
        nuplandb_models_dict["Category"] = "category"
        nuplandb_models_dict["Image"] = "image"
        nuplandb_models_dict["Lidar"] = "lidar"
        nuplandb_models_dict["Log"] = "log"
        nuplandb_models_dict["Track"] = "track"
        nuplandb_models_dict["TrafficLightStatus"] = "traffic_light_status"
        nuplandb_models_dict["LidarBox"] = "lidar_box"
        nuplandb_models_dict["Scene"] = "scene"
        nuplandb_models_dict["ScenarioTag"] = "scenario_tag"
        nuplandb_models_dict["LidarPc"] = "lidar_pc"
        nuplandb_models_dict["EgoPose"] = "ego_pose"
        super().__init__(table_names, nuplan_db_orm, data_root, load_path, verbose, nuplandb_models_dict)

    def __reduce__(self) -> Tuple[Type[NuPlanDB], Tuple[Any, ...]]:
        """
        Hints on how to reconstruct the object when pickling.
        :return: Object type and constructor arguments to be used.
        """
        return self.__class__, (self._data_root, self._load_path, self._maps_db, self._verbose)

    @property
    def load_path(self) -> str:
        """Get the path from which the db file was loaded."""
        return self._load_path

    @property
    def maps_db(self) -> Optional[GPKGMapsDB]:
        """Get the MapsDB objectd attached to the database."""
        return self._maps_db

    @property
    def log_name(self) -> str:
        """Get the name of the log contained within the database."""
        return cast(str, self.log.logfile)

    @property
    def map_name(self) -> str:
        """Get the name of the map associated with the log of the database."""
        return cast(str, self.log.map_version)

    @property
    def category(self) -> Table[Category]:
        """
        Get Category table.
        :return: The category table.
        """
        return self.tables["category"]

    @property
    def log(self) -> Log:
        """
        Get first and only entry in the log table.
        :return: The log entry in the log table.
        """
        return self.tables["log"][0]

    @property
    def camera(self) -> Table[Camera]:
        """
        Get Camera table.
        :return: The camera table.
        """
        return self.tables["camera"]

    @property
    def lidar(self) -> Table[Lidar]:
        """
        Get Lidar table.
        :return: The lidar table.
        """
        return self.tables["lidar"]

    @property
    def ego_pose(self) -> Table[EgoPose]:
        """
        Get Ego Pose table.
        :return: The ego pose table.
        """
        return self.tables["ego_pose"]

    @property
    def image(self) -> Table[Image]:
        """
        Get Image table.
        :return: The image table.
        """
        return self.tables["image"]

    @property
    def lidar_pc(self) -> Table[LidarPc]:
        """
        Get Lidar Pc table.
        :return: The lidar pc table.
        """
        return self.tables["lidar_pc"]

    @property
    def lidar_box(self) -> Table[LidarBox]:
        """
        Get Lidar Box table.
        :return: The lidar box table.
        """
        return self.tables["lidar_box"]

    @property
    def track(self) -> Table[Track]:
        """
        Get Track table.
        :return: The track table.
        """
        return self.tables["track"]

    @property
    def scene(self) -> Table[Scene]:
        """
        Get Scene table.
        :return: The scene table.
        """
        return self.tables["scene"]

    @property
    def scenario_tag(self) -> Table[ScenarioTag]:
        """
        Get Scenario Tag table.
        :return: The scenario tag table.
        """
        return self.tables["scenario_tag"]

    @property
    def traffic_light_status(self) -> Table[TrafficLightStatus]:
        """
        Get Traffic Light Status table.
        :return: The traffic light status table.
        """
        return self.tables["traffic_light_status"]

    @cached_property
    def cam_channels(self) -> Set[str]:
        """
        Get list of camera channels.
        :return: The list of camera channels.
        """
        return {cam.channel for cam in self.camera}

    @cached_property
    def lidar_channels(self) -> Set[str]:
        """
        Get list of lidar channels.
        :return: The list of lidar channels.
        """
        return {lidar.channel for lidar in self.lidar}

    def get_unique_scenario_tags(self) -> List[str]:
        """Retrieve all unique scenario tags in the database."""
        return sorted({tag[0] for tag in self.session.query(ScenarioTag.type).distinct().all()})
