from __future__ import annotations

import logging
import os
from collections import defaultdict
from functools import reduce
from typing import Any, Dict, List, Optional, Set, Tuple, Type

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import PIL.Image
from cachetools import LRUCache, cached
from nuplan.database.common.blob_store.creator import BlobStoreCreator
from nuplan.database.common.db import DB, Table
from nuplan.database.maps_db.gpkg_mapsdb import GPKGMapsDB
from nuplan.database.nuplan_db import models as nuplandb_models
from nuplan.database.nuplan_db.models import Camera, Category, EgoPose, Image, Lidar, LidarBox, LidarPc, Log, \
    ScenarioTag, Scene, Track, TrafficLightStatus
from nuplan.database.nuplan_db.templates import tables as nuplandb_table_templates
from nuplan.database.utils.geometry import view_points

logger = logging.getLogger(__name__)

MICROSECONDS_IN_A_SECOND = 1000000


class NuPlanDB(DB):
    """
    Database class for the nuPlan database. It provides some simple lookups and get methods.
    """

    def __init__(self,
                 version: str,
                 data_root: str,
                 map_version: str = 'nuplan-maps-v0.1',
                 map_root: Optional[str] = None,
                 verbose: bool = False):
        """
        Loads database and creates reverse indexes and shortcuts.
        :param version: Version to load (e.g. "nuplan_v0.1_mini").
        :param data_root: Path to the NuPlanDB tables and blobs.
        :param map_version: Version to load (e.g. "nuplan-maps-v0.1").
        :param map_root: Root folder of the maps.
        :param verbose: Whether to print status messages during load.
        """
        self._map_version = map_version
        # Set default map folder
        if map_root is None:
            self._map_root = os.path.join(data_root, 'maps')
        else:
            self._map_root = map_root
        self._verbose = verbose

        # Initialize parent class
        table_names = list(nuplandb_table_templates.keys())
        maps_db = GPKGMapsDB(self._map_version, self._map_root)
        blob_store = BlobStoreCreator.create_nuplandb(data_root=data_root)
        super().__init__(table_names, nuplandb_models, data_root, version, verbose,
                         blob_store, maps_db)

        # Initialize NuPlanDBExplorer class
        self._explorer = NuPlanDBExplorer(self)

    def __reduce__(self) -> Tuple[Type[NuPlanDB], Tuple[Any, ...]]:
        """
        Hints on how to reconstruct the object when pickling.
        :return: Object type and constructor arguments to be used.
        """

        return self.__class__, (self._version, self._data_root, self._map_version, self._map_root, self._verbose)

    def __getstate__(self) -> Dict[str, Any]:
        """
        Called by pickle.dump/dumps to save class state.
        Don't save mapsdb or blobstore because they're not pickleable, re-create object when restoring.
        :return: The object state.
        """
        state = dict()
        state['_version'] = self._version
        state['_data_root'] = self._data_root
        state['_map_version'] = self._map_version
        state['_map_root'] = self._map_root
        state['_verbose'] = self._verbose

        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """
        Called by pickle.load/loads to restore class state.
        :param state: The object state.
        """
        db = NuPlanDB(
            version=state['_version'],
            data_root=state['_data_root'],
            map_version=state['_map_version'],
            map_root=state['_map_root'],
            verbose=state['_verbose'])
        self.__dict__.update(db.__dict__)

    # Explicitly assign tables to help the IDE determine valid class members.
    @property
    def category(self) -> Table[Category]:
        """
        Get Category table.
        :return: The category table.
        """
        return self.tables['category']

    @property
    def log(self) -> Table[Log]:
        """
        Get Log table.
        :return: The log table.
        """
        return self.tables['log']

    @property
    def camera(self) -> Table[Camera]:
        """
        Get Camera table.
        :return: The camera table.
        """
        return self.tables['camera']

    @property
    def lidar(self) -> Table[Lidar]:
        """
        Get Lidar table.
        :return: The lidar table.
        """
        return self.tables['lidar']

    @property
    def ego_pose(self) -> Table[EgoPose]:
        """
        Get Ego Pose table.
        :return: The ego pose table.
        """
        return self.tables['ego_pose']

    @property
    def image(self) -> Table[Image]:
        """
        Get Image table.
        :return: The image table.
        """
        return self.tables['image']

    @property
    def lidar_pc(self) -> Table[LidarPc]:
        """
        Get Lidar Pc table.
        :return: The lidar pc table.
        """
        return self.tables['lidar_pc']

    @property
    def lidar_box(self) -> Table[LidarBox]:
        """
        Get Lidar Box table.
        :return: The lidar box table.
        """
        if 'lidar_box' not in self.tables:
            self.tables['lidar_box'] = Table[LidarBox](LidarBox, self)
        return self.tables['lidar_box']

    @property
    def track(self) -> Table[Track]:
        """
        Get Track table.
        :return: The track table.
        """
        if 'track' not in self.tables:
            self.tables['track'] = Table[Track](Track, self)
        return self.tables['track']

    @property
    def scene(self) -> Table[Scene]:
        """
        Get Scene table.
        :return: The scene table.
        """
        if 'scene' not in self.tables:
            self.tables['scene'] = Table[Scene](Scene, self)
        return self.tables['scene']

    @property
    def scenario_tag(self) -> Table[ScenarioTag]:
        """
        Get Scenario Tag table.
        :return: The scenario tag table.
        """
        if 'scenario_tag' not in self.tables:
            self.tables['scenario_tag'] = Table[ScenarioTag](ScenarioTag, self)
        return self.tables['scenario_tag']

    @property
    def traffic_light_status(self) -> Table[TrafficLightStatus]:
        """
        Get Traffic Light Status table.
        :return: The traffic light status table.
        """
        if 'traffic_light_status' not in self.tables:
            self.tables['traffic_light_status'] = Table[TrafficLightStatus](TrafficLightStatus, self)
        return self.tables['traffic_light_status']

    @property  # type: ignore
    @cached(cache=LRUCache(maxsize=1))
    def cam_channels(self) -> Set[str]:
        """
        Get list of camera channels.
        :return: The list of camera channels.
        """
        return {cam.channel for cam in self.camera}

    @property  # type: ignore
    @cached(cache=LRUCache(maxsize=1))
    def lidar_channels(self) -> Set[str]:
        """
        Get list of lidar channels.
        :return: The list of lidar channels.
        """
        return {lidar.channel for lidar in self.lidar}

    def list_categories(self) -> None:
        """ Print list of categories. """
        self._explorer.list_categories()

    def render_pointcloud_in_image(self, lidar_pc: LidarPc, **kwargs: Any) -> None:
        """
        Render point cloud in image.

        :param lidar_pc: Lidar PC record.
        :kwargs: Optional configurations.
        """
        self._explorer.render_pointcloud_in_image(lidar_pc, **kwargs)


class NuPlanDBExplorer:
    """
    Helper class to list and visualize NuPlanDB data. These are meant to serve as tutorials and templates for
    working with the data.
    """

    def __init__(self, nuplandb: NuPlanDB):
        """
        :param nuplandb: NuPlanDB instance.
        """
        self.nuplandb = nuplandb

    def unique_scenario_tags(self) -> List[str]:
        """
        Get list of all the unique ScenarioTag types in the DB.
        :return: The list of all the unique scenario tag types.
        """
        return [tag[0] for tag in self.nuplandb.session.query(ScenarioTag.type).distinct().all()]

    def list_categories(self) -> None:
        """ Print categories, counts and stats. """

        logger.info('\nCompiling category summary ... ')

        # Retrieve category name and object sizes from DB.
        length_name = self.nuplandb.session.query(LidarBox.length, Category.name). \
            join(Track, LidarBox.track_token == Track.token).join(Category, Track.category_token == Category.token)
        width_name = self.nuplandb.session.query(LidarBox.width, Category.name). \
            join(Track, LidarBox.track_token == Track.token).join(Category, Track.category_token == Category.token)
        height_name = self.nuplandb.session.query(LidarBox.height, Category.name). \
            join(Track, LidarBox.track_token == Track.token).join(Category, Track.category_token == Category.token)

        # Group by category name
        length_categories = defaultdict(list)
        for size, name in length_name:
            length_categories[name].append(size)

        width_categories = defaultdict(list)
        for size, name in width_name:
            width_categories[name].append(size)

        height_categories = defaultdict(list)
        for size, name in height_name:
            height_categories[name].append(size)

        logger.info(f"{'name':>50} {'count':>10} {'width':>10} {'len':>10} {'height':>10} \n {'-'*101:>10}")
        for name, stats in sorted(length_categories.items()):
            length_stats = np.array(stats)
            width_stats = np.array(width_categories[name])
            height_stats = np.array(height_categories[name])
            logger.info(f"{name[:50]:>50} {length_stats.shape[0]:>10.2f} "
                        f"{np.mean(length_stats):>5.2f} {np.std(length_stats):>5.2f} "
                        f"{np.mean(width_stats):>5.2f} {np.std(width_stats):>5.2f} {np.mean(height_stats):>5.2f} "
                        f"{np.std(height_stats):>5.2f}")

    def map_pointcloud_to_image(self, lidar_pc: LidarPc, img: Image, color_channel: int = 2,
                                max_radius: float = np.inf) -> \
            Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], PIL.Image.Image]:
        """
        Given a lidar and camera sample_data, load point-cloud and map it to the image plane.
        :param lidar_pc: Lidar sample_data record.
        :param img: Camera sample_data record.
        :param color_channel: Set to 2 for coloring dots by depth, 3 for intensity.
        :param max_radius: Max xy radius of lidar points to include in visualization.
            Set to np.inf to include all points.
        :return (pointcloud <np.float: 2, n)>, coloring <np.float: n>, image <Image>).
        """

        assert isinstance(lidar_pc, LidarPc), 'first input must be a lidar_pc modality'
        assert isinstance(img, Image), 'second input must be a camera modality'

        # Load files.
        pc = lidar_pc.load()
        im = img.load_as(img_type='pil')

        # Filter lidar points to be inside desired range.
        radius = np.sqrt(pc.points[0] ** 2 + pc.points[1] ** 2)
        keep = radius <= max_radius
        pc.points = pc.points[:, keep]

        # Transform pc to img.
        transform = reduce(np.dot, [img.camera.trans_matrix_inv, img.ego_pose.trans_matrix_inv,
                                    lidar_pc.ego_pose.trans_matrix, lidar_pc.lidar.trans_matrix])
        pc.transform(transform)

        # Grab the coloring (depth or intensity).
        coloring = pc.points[color_channel, :]
        depths = pc.points[2, :]

        # Take the actual picture (matrix multiplication with camera - matrix + renormalization).
        points = view_points(pc.points[:3, :], img.camera.intrinsic_np, normalize=True)

        # Finally filter away points outside the image.
        mask = np.ones(depths.shape[0], dtype=bool)
        mask = np.logical_and(mask, depths > 0)
        mask = np.logical_and(mask, points[0, :] > 1)
        mask = np.logical_and(mask, points[0, :] < im.size[0] - 1)
        mask = np.logical_and(mask, points[1, :] > 1)
        mask = np.logical_and(mask, points[1, :] < im.size[1] - 1)

        points = points[:, mask]
        coloring = coloring[mask]

        return points, coloring, im

    def render_pointcloud_in_image(self, lidar_pc: LidarPc, dot_size: int = 5, color_channel: int = 2,
                                   max_radius: float = np.inf, image_channel: str = 'CAM_F0') -> None:
        """
        Scatter-plots pointcloud on top of image.
        :param sample: LidarPc Sample.
        :param dot_size: Scatter plot dot size.
        :param color_channel: Set to 2 for coloring dots by height, 3 for intensity.
        :param max_radius: Max xy radius of lidar points to include in visualization.
            Set to np.inf to include all points.
        :param image_channel: Which image to render.
        """
        image = lidar_pc.closest_image([image_channel])[0]

        points, coloring, im = self.map_pointcloud_to_image(lidar_pc, image, color_channel=color_channel,
                                                            max_radius=max_radius)
        plt.figure(figsize=(9, 16))
        plt.imshow(im)
        plt.scatter(points[0, :], points[1, :], c=coloring, s=dot_size)
        plt.axis('off')
