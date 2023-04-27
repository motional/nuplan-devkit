import os
from functools import lru_cache
from typing import Union, cast

from nuplan.database.maps_db.gpkg_mapsdb import GPKGMapsDB
from nuplan.database.maps_db.imapsdb import IMapsDB
from nuplan.database.nuplan_db_orm.camera import Camera
from nuplan.database.nuplan_db_orm.ego_pose import EgoPose
from nuplan.database.nuplan_db_orm.image import Image
from nuplan.database.nuplan_db_orm.lidar import Lidar
from nuplan.database.nuplan_db_orm.lidar_box import LidarBox
from nuplan.database.nuplan_db_orm.lidar_pc import LidarPc
from nuplan.database.nuplan_db_orm.nuplandb import NuPlanDB
from nuplan.database.nuplan_db_orm.nuplandb_wrapper import NuPlanDBWrapper, discover_log_dbs
from nuplan.database.nuplan_db_orm.track import Track

DEFAULT_TEST_DB_INDEX = 1  # default database index to load when only a single database is required
DEFAULT_TEST_CAMERA_INDEX = 0  # default camera for testing
DEFAULT_TEST_EGO_POSE_INDEX = 0  # default ego pose for testing
DEFAULT_TEST_LIDAR_INDEX = 0  # default lidar pc for testing
DEFAULT_TEST_LIDAR_PC_INDEX = 1000  # default lidar pc for testing
DEFAULT_TEST_LIDAR_BOX_INDEX = 5000  # default lidar box for testing
DEFAULT_TEST_LIDAR_BOX_INDEX_VEHICLE = 5004  # default lidar box for testing
DEFAULT_TEST_TRACK_INDEX = 100  # default track for testing
DEFAULT_TEST_LIDAR_PC_WITH_BLOB_TOKEN = 100  # default lidar pc with blob for testing - sensor blobs not yet supported
DEFAULT_TEST_IMAGE_WITH_BLOB_TOKEN = 0  # default image with blob for testing - sensor blobs not yet supported

# Environment variables required to load the nuPlan dataset for testing
NUPLAN_DATA_ROOT = os.environ["NUPLAN_DATA_ROOT"]
NUPLAN_MAPS_ROOT = os.environ["NUPLAN_MAPS_ROOT"]
NUPLAN_MAP_VERSION = os.environ["NUPLAN_MAP_VERSION"]

# Use a subset of the mini split for testing purposes
NUPLAN_DB_FILES = [
    # Mini train
    f"{NUPLAN_DATA_ROOT}/nuplan-v1.1/splits/mini/2021.07.16.20.45.29_veh-35_01095_01486.db",
    f"{NUPLAN_DATA_ROOT}/nuplan-v1.1/splits/mini/2021.10.06.17.43.07_veh-28_00508_00877.db",
    f"{NUPLAN_DATA_ROOT}/nuplan-v1.1/splits/mini/2021.08.17.18.54.02_veh-45_00665_01065.db",
    f"{NUPLAN_DATA_ROOT}/nuplan-v1.1/splits/mini/2021.10.11.02.57.41_veh-50_01522_02088.db",
    # Mini val
    f"{NUPLAN_DATA_ROOT}/nuplan-v1.1/splits/mini/2021.06.08.16.31.33_veh-38_01589_02072.db",
    f"{NUPLAN_DATA_ROOT}/nuplan-v1.1/splits/mini/2021.08.09.17.55.59_veh-28_00021_00307.db",
    f"{NUPLAN_DATA_ROOT}/nuplan-v1.1/splits/mini/2021.06.07.18.53.26_veh-26_00005_00427.db",
    f"{NUPLAN_DATA_ROOT}/nuplan-v1.1/splits/mini/2021.10.05.07.10.04_veh-52_01442_01802.db",
    # Mini test
    f"{NUPLAN_DATA_ROOT}/nuplan-v1.1/splits/mini/2021.06.28.16.57.59_veh-26_00016_00484.db",
    f"{NUPLAN_DATA_ROOT}/nuplan-v1.1/splits/mini/2021.08.30.14.54.34_veh-40_00439_00835.db",
    f"{NUPLAN_DATA_ROOT}/nuplan-v1.1/splits/mini/2021.09.16.15.12.03_veh-42_01037_01434.db",
    f"{NUPLAN_DATA_ROOT}/nuplan-v1.1/splits/mini/2021.10.06.07.26.10_veh-52_00006_00398.db",
]


@lru_cache(maxsize=1)
def get_test_nuplan_db_wrapper() -> NuPlanDBWrapper:
    """Get a nuPlan DB wrapper object with default settings to be used in testing."""
    return get_test_nuplan_db_wrapper_nocache()


def get_test_nuplan_db_wrapper_nocache() -> NuPlanDBWrapper:
    """
    Gets a nuPlan DB wrapper object with default settings to be used in testing.
    This object will not be cached.
    """
    return NuPlanDBWrapper(
        data_root=NUPLAN_DATA_ROOT,
        map_root=NUPLAN_MAPS_ROOT,
        db_files=NUPLAN_DB_FILES,
        map_version=NUPLAN_MAP_VERSION,
    )


@lru_cache(maxsize=1)
def get_test_nuplan_db_path() -> str:
    """Get a single nuPlan DB path to be used in testing."""
    paths = discover_log_dbs(NUPLAN_DB_FILES)
    return cast(str, paths[DEFAULT_TEST_DB_INDEX])


@lru_cache(maxsize=1)
def get_test_nuplan_db() -> NuPlanDB:
    """Get a nuPlan DB object with default settings to be used in testing."""
    return get_test_nuplan_db_nocache()


def get_test_nuplan_db_nocache() -> NuPlanDB:
    """
    Get a nuPlan DB object with default settings to be used in testing.
    Forces the data to be read from disk.
    """
    load_path = get_test_nuplan_db_path()
    maps_db = get_test_maps_db()

    return NuPlanDB(
        data_root=NUPLAN_DATA_ROOT,
        load_path=load_path,
        maps_db=maps_db,
    )


@lru_cache(maxsize=1)
def get_test_nuplan_camera() -> Camera:
    """Get a nuPlan camera object with default settings to be used in testing."""
    db = get_test_nuplan_db()
    return db.camera[DEFAULT_TEST_CAMERA_INDEX]


@lru_cache(maxsize=1)
def get_test_nuplan_egopose() -> EgoPose:
    """Get a nuPlan egopose object with default settings to be used in testing."""
    db = get_test_nuplan_db()
    return db.ego_pose[DEFAULT_TEST_EGO_POSE_INDEX]


@lru_cache(maxsize=1)
def get_test_nuplan_lidarpc(index: Union[int, str] = DEFAULT_TEST_LIDAR_PC_INDEX) -> LidarPc:
    """Get a nuPlan lidarpc object with default settings to be used in testing."""
    db = get_test_nuplan_db()
    return db.lidar_pc[index]


@lru_cache(maxsize=1)
def get_test_nuplan_lidarpc_with_blob() -> LidarPc:
    """Get a nuPlan lidarpc object with blob with default settings to be used in testing."""
    db = get_test_nuplan_db()
    return db.lidar_pc[DEFAULT_TEST_LIDAR_PC_WITH_BLOB_TOKEN]


@lru_cache(maxsize=1)
def get_test_nuplan_image() -> Image:
    """Get a nuPlan image object with default settings to be used in testing."""
    db = get_test_nuplan_db()
    return db.image[DEFAULT_TEST_IMAGE_WITH_BLOB_TOKEN]


@lru_cache(maxsize=1)
def get_test_nuplan_lidar() -> Lidar:
    """Get a nuPlan lidar object with default settings to be used in testing."""
    db = get_test_nuplan_db()
    return db.lidar[DEFAULT_TEST_LIDAR_INDEX]


@lru_cache(maxsize=1)
def get_test_nuplan_lidar_box() -> LidarBox:
    """Get a nuPlan lidar box object with default settings to be used in testing."""
    db = get_test_nuplan_db()
    return db.lidar_box[DEFAULT_TEST_LIDAR_BOX_INDEX]


@lru_cache(maxsize=1)
def get_test_nuplan_lidar_box_vehicle() -> LidarBox:
    """Get a nuPlan lidar box object with default settings to be used in testing."""
    db = get_test_nuplan_db()
    return db.lidar_box[DEFAULT_TEST_LIDAR_BOX_INDEX_VEHICLE]


@lru_cache(maxsize=1)
def get_test_nuplan_track() -> Track:
    """Get a nuPlan track object with default settings to be used in testing."""
    db = get_test_nuplan_db()
    return db.track[DEFAULT_TEST_TRACK_INDEX]


@lru_cache(maxsize=1)
def get_test_maps_db() -> IMapsDB:
    """Get a nuPlan maps DB object with default settings to be used in testing."""
    return GPKGMapsDB(
        map_version=NUPLAN_MAP_VERSION,
        map_root=NUPLAN_MAPS_ROOT,
    )
