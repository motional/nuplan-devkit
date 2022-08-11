import logging
from collections import defaultdict
from typing import List

import numpy as np
import numpy.typing as npt

from nuplan.database.nuplan_db_orm.category import Category
from nuplan.database.nuplan_db_orm.lidar_box import LidarBox
from nuplan.database.nuplan_db_orm.nuplandb import NuPlanDB
from nuplan.database.nuplan_db_orm.scenario_tag import ScenarioTag
from nuplan.database.nuplan_db_orm.track import Track

logger = logging.getLogger(__name__)


def unique_scenario_tags(db: NuPlanDB) -> List[str]:
    """
    Get list of all the unique ScenarioTag types in the DB.
    :param db: Database to use for printing metadata.
    :return: The list of all the unique scenario tag types.
    """
    return [tag[0] for tag in db.session.query(ScenarioTag.type).distinct().all()]


def list_categories(db: NuPlanDB) -> None:
    """
    Print categories, counts and stats.
    :param db: Database to use for printing metadata.
    """
    logger.info('\nCompiling category summary ... ')

    # Retrieve category name and object sizes from DB.
    length_name = (
        db.session.query(LidarBox.length, Category.name)
        .join(Track, LidarBox.track_token == Track.token)
        .join(Category, Track.category_token == Category.token)
    )
    width_name = (
        db.session.query(LidarBox.width, Category.name)
        .join(Track, LidarBox.track_token == Track.token)
        .join(Category, Track.category_token == Category.token)
    )
    height_name = (
        db.session.query(LidarBox.height, Category.name)
        .join(Track, LidarBox.track_token == Track.token)
        .join(Category, Track.category_token == Category.token)
    )

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
        length_stats: npt.NDArray[np.float32] = np.array(stats)
        width_stats: npt.NDArray[np.float32] = np.array(width_categories[name])
        height_stats: npt.NDArray[np.float32] = np.array(height_categories[name])
        logger.info(
            f"{name[:50]:>50} {length_stats.shape[0]:>10.2f} "
            f"{np.mean(length_stats):>5.2f} {np.std(length_stats):>5.2f} "
            f"{np.mean(width_stats):>5.2f} {np.std(width_stats):>5.2f} {np.mean(height_stats):>5.2f} "
            f"{np.std(height_stats):>5.2f}"
        )
