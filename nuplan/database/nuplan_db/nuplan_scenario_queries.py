import pickle
import sqlite3
from typing import Generator, List, Optional, Set, Tuple, Type, Union

import numpy as np
from pyquaternion import Quaternion

from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.scene_object import SceneObjectMetadata
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.common.actor_state.static_object import StaticObject
from nuplan.common.actor_state.tracked_objects import TrackedObject
from nuplan.common.actor_state.tracked_objects_types import AGENT_TYPES, TrackedObjectType
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.common.actor_state.waypoint import Waypoint
from nuplan.common.maps.maps_datatypes import TrafficLightStatusData, TrafficLightStatusType, Transform
from nuplan.common.utils.helpers import get_unique_incremental_track_id
from nuplan.database.nuplan_db.camera import Camera
from nuplan.database.nuplan_db.image import Image
from nuplan.database.nuplan_db.lidar_pc import LidarPc
from nuplan.database.nuplan_db.nuplan_db_utils import SensorDataSource
from nuplan.database.nuplan_db.query_session import execute_many, execute_one
from nuplan.database.nuplan_db.sensor_data_table_row import SensorDataTableRow
from nuplan.database.utils.label.utils import local2agent_type, raw_mapping


def _parse_tracked_object_row(row: sqlite3.Row) -> TrackedObject:
    """
    A convenience method to parse a TrackedObject from a sqlite3 row.
    :param row: The row from the DB query.
    :return: The parsed TrackedObject.
    """
    category_name = row["category_name"]
    pose = StateSE2(row["x"], row["y"], row["yaw"])
    oriented_box = OrientedBox(pose, width=row["width"], length=row["length"], height=row["height"])

    # These next two are globals
    label_local = raw_mapping["global2local"][category_name]
    tracked_object_type = TrackedObjectType[local2agent_type[label_local]]

    if tracked_object_type in AGENT_TYPES:
        return Agent(
            tracked_object_type=tracked_object_type,
            oriented_box=oriented_box,
            velocity=StateVector2D(row["vx"], row["vy"]),
            predictions=[],  # to be filled in later
            angular_velocity=np.nan,
            metadata=SceneObjectMetadata(
                token=row["token"].hex(),
                track_token=row["track_token"].hex(),
                track_id=get_unique_incremental_track_id(str(row["track_token"].hex())),
                timestamp_us=row["timestamp"],
                category_name=category_name,
            ),
        )
    else:
        return StaticObject(
            tracked_object_type=tracked_object_type,
            oriented_box=oriented_box,
            metadata=SceneObjectMetadata(
                token=row["token"].hex(),
                track_token=row["track_token"].hex(),
                track_id=get_unique_incremental_track_id(str(row["track_token"].hex())),
                timestamp_us=row["timestamp"],
                category_name=category_name,
            ),
        )


def get_sensor_token_by_index_from_db(log_file: str, sensor_source: SensorDataSource, index: int) -> Optional[str]:
    """
    Get the N-th sensor token ordered chronologically by timestamp from a particular channel.
    This is primarily used for unit testing.
    If the index does not exist (e.g. index = 10,000 in a log file with 1000 entries),
        then the result will be None.
    Only non-negative integer indexes are supported.
    :param log_file: The db file to query.
    :param sensor_source: Parameters for querying the correct table.
    :param index: The 0-indexed integer index of the lidarpc token to retrieve.
    :return: The token, if it exists.
    """
    if index < 0:
        raise ValueError(f"Index of {index} was supplied to get_lidarpc_token_by_index_from_db(), which is negative.")
    sensor_token = get_sensor_token(log_file, sensor_source.sensor_table, sensor_source.channel)

    query = f"""
    WITH ordered AS
    (
        SELECT  token,
                lidar_token,
                ROW_NUMBER() OVER (ORDER BY timestamp ASC) AS row_num
        FROM {sensor_source.table}
    )
    SELECT token
    FROM ordered
    WHERE (row_num - 1) = ?
        AND {sensor_source.sensor_token_column} = ?;
    """

    result = execute_one(query, [index, bytearray.fromhex(sensor_token)], log_file)
    return None if result is None else str(result["token"].hex())


def get_end_sensor_time_from_db(log_file: str, sensor_source: SensorDataSource) -> int:
    """
    Get the timestamp of the last sensor data recorded in the log file.
    :param log_file: The db file to query.
    :param sensor_source: Parameters for querying the correct table.
    :return: The timestamp of the last sensor data.
    """
    query = f"""
    SELECT MAX(timestamp) AS max_time
    FROM {sensor_source.table};
    """

    result = execute_one(query, [], log_file)
    return int(result["max_time"])


def get_sensor_data_token_timestamp_from_db(
    log_file: str, sensor_source: SensorDataSource, token: str
) -> Optional[int]:
    """
    Get the timestamp associated with an individual lidar_pc token.
    :param log_file: The db file to query.
    :param sensor_source: Parameters for querying the correct table.
    :param token: The token for which to grab the timestamp.
    :return: The timestamp associated with the token, if found.
    """
    query = f"""
    SELECT timestamp
    FROM {sensor_source.table}
    WHERE token = ?;
    """
    result = execute_one(query, (bytearray.fromhex(token),), log_file)
    return None if result is None else int(result["timestamp"])


def get_sensor_token_map_name_from_db(log_file: str, sensor_source: SensorDataSource, token: str) -> Optional[str]:
    """
    Get the map name for a provided sensor token.
    :param log_file: The db file to query.
    :param sensor_source: Parameters for querying the correct table.
    :param token: The token for which to get the map name.
    :return: The map name for the token, if found.
    """
    query = f"""
    SELECT map_version
    FROM log AS l
    INNER JOIN {sensor_source.sensor_table} AS sensor
        ON sensor.log_token = l.token
    INNER JOIN {sensor_source.table} AS sensor_data
        ON sensor_data.{sensor_source.sensor_token_column} = sensor.token
    WHERE sensor_data.token = ?;
    """

    result = execute_one(query, (bytearray.fromhex(token),), log_file)
    return None if result is None else result["map_version"]


def get_sampled_sensor_tokens_in_time_window_from_db(
    log_file: str, sensor_source: SensorDataSource, start_timestamp: int, end_timestamp: int, subsample_interval: int
) -> Generator[str, None, None]:
    """
    For every token in a window defined by [start_timestamp, end_timestamp], retrieve every `subsample_interval`-th sensor token, ordered in increasing order by timestamp.

    E.g. for this table
    ```
    token | timestamp
    -----------------
    1     | 0
    2     | 1
    3     | 2
    4     | 3
    5     | 4
    6     | 5
    7     | 6
    ```

    query with start_timestamp=1, end_timestamp=5, subsample_interval=2, table=lidar_pc, will return tokens
    [1, 3, 5].

    :param log_file: The db file to query.
    :param sensor_source: Parameters for querying the correct table.
    :param start_timestamp: The start of the window to sample, inclusive.
    :param end_timestamp: The end of the window to sample, inclusive.
    :param subsample_interval: The interval at which to sample.
    :return: A generator of lidar_pc tokens that fit the provided parameters.
    """
    sensor_token = get_sensor_token(log_file, sensor_source.sensor_table, sensor_source.channel)

    query = f"""
    WITH numbered AS
    (
        SELECT token, timestamp, ROW_NUMBER() OVER (ORDER BY timestamp ASC) AS row_num
        FROM {sensor_source.table}
        WHERE timestamp >= ?
        AND timestamp <= ?
        AND {sensor_source.sensor_token_column} == ?
    )
    SELECT token
    FROM numbered
    WHERE ((row_num - 1) % ?) = 0
    ORDER BY timestamp ASC;
    """
    for row in execute_many(
        query, (start_timestamp, end_timestamp, bytearray.fromhex(sensor_token), subsample_interval), log_file
    ):
        yield row["token"].hex()


def get_sensor_data_from_sensor_data_tokens_from_db(
    log_file: str,
    sensor_source: SensorDataSource,
    sensor_class: Type[SensorDataTableRow],
    tokens: Union[Generator[str, None, None], List[str]],
) -> Generator[SensorDataTableRow, None, None]:
    """
    Given a collection of sensor tokens, builds the corresponding sensor_class objects.
    This function makes no restrictions on the ordering of returned values.
    :param sensor_source: Parameters for querying the correct table.
    :param sensor_class: Class holding a row of the SensorData table.
    :param log_file: The db file to query.
    :param tokens: The tokens for which to build the sensor_class objects.
    :return: A generator yielding sensor_class objects.
    """
    if not isinstance(tokens, list):
        tokens = list(tokens)

    query = f"""
        SELECT *
        FROM {sensor_source.table}
        WHERE token IN ({('?,'*len(tokens))[:-1]});
    """

    for row in execute_many(query, [bytearray.fromhex(t) for t in tokens], log_file):
        yield sensor_class.from_db_row(row)


def get_sensor_transform_matrix_for_sensor_data_token_from_db(
    log_file: str, sensor_source: SensorDataSource, sensor_data_token: str
) -> Optional[Transform]:
    """
    Get the associated lidar transform matrix from the DB for the given lidarpc_token.
    :param log_file: The log file to query.
    :param sensor_source: Parameters for querying the correct table.
    :param sensor_data_token: The sensor data token to query.
    :return: The transform matrix. Reuturns None if the matrix does not exist in the DB (e.g. for a token that does not exist).
    """
    query = f"""
        SELECT  sensor.translation,
                sensor.rotation
        FROM {sensor_source.sensor_table} AS sensor
        INNER JOIN {sensor_source.table} AS sensor_data
            ON sensor_data.{sensor_source.sensor_token_column} = sensor.token
        WHERE sensor_data.token = ?;
    """

    row = execute_one(query, (bytearray.fromhex(sensor_data_token),), log_file)
    if row is None:
        return None

    translation = pickle.loads(row["translation"])
    rotation = pickle.loads(row["rotation"])

    output = Quaternion(rotation).transformation_matrix
    output[:3, 3] = np.array(translation)

    return output


def get_mission_goal_for_sensor_data_token_from_db(
    log_file: str, sensor_source: SensorDataSource, token: str
) -> Optional[StateSE2]:
    """
    Get the goal pose for a given lidar_pc token.
    :param log_file: The db file to query.
    :param sensor_source: Parameters for querying the correct table.
    :param token: The token for which to query the goal state.
    :return: The goal state.
    """
    query = f"""
        SELECT  ep.x,
                ep.y,
                ep.qw,
                ep.qx,
                ep.qy,
                ep.qz
        FROM ego_pose AS ep
        INNER JOIN scene AS s
            ON s.goal_ego_pose_token = ep.token
        INNER JOIN {sensor_source.table} AS sensor_data
            ON sensor_data.scene_token = s.token
        WHERE sensor_data.token = ?
    """

    row = execute_one(query, (bytearray.fromhex(token),), log_file)
    if row is None:
        return None

    q = Quaternion(row["qw"], row["qx"], row["qy"], row["qz"])
    return StateSE2(row["x"], row["y"], q.yaw_pitch_roll[0])


def get_roadblock_ids_for_lidarpc_token_from_db(log_file: str, lidarpc_token: str) -> Optional[List[str]]:
    """
    Get the scene roadblock ids from the db for a given lidar_pc token.
    :param log_file: The db file to query.
    :param lidarpc_token: The token for which to query the current state.
    :return: List of roadblock ids as str.
    """
    query = """
        SELECT  s.roadblock_ids
        FROM scene AS s
        INNER JOIN lidar_pc AS lp
            ON lp.scene_token = s.token
        WHERE lp.token = ?
    """
    # Each row is a space-separated list of route roadblock IDS, e.g. "123 234 345"
    row = execute_one(query, (bytearray.fromhex(lidarpc_token),), log_file)
    if row is None:
        return None
    return str(row["roadblock_ids"]).split(" ")


def get_statese2_for_lidarpc_token_from_db(log_file: str, token: str) -> Optional[StateSE2]:
    """
    Get the ego pose as a StateSE2 from the db for a given lidar_pc token.
    :param log_file: The db file to query.
    :param token: The token for which to query the current state.
    :return: The current ego state, as a StateSE2 object.
    """
    query = """
        SELECT  ep.x,
                ep.y,
                ep.qw,
                ep.qx,
                ep.qy,
                ep.qz
        FROM ego_pose AS ep
        INNER JOIN lidar_pc AS lp
            ON lp.ego_pose_token = ep.token
        WHERE lp.token = ?
    """

    row = execute_one(query, (bytearray.fromhex(token),), log_file)
    if row is None:
        return None

    q = Quaternion(row["qw"], row["qx"], row["qy"], row["qz"])
    return StateSE2(row["x"], row["y"], q.yaw_pitch_roll[0])


def get_sampled_lidarpcs_from_db(
    log_file: str,
    initial_token: str,
    sensor_source: SensorDataSource,
    sample_indexes: Union[Generator[int, None, None], List[int]],
    future: bool,
) -> Generator[LidarPc, None, None]:
    """
    Given an anchor token, return the tokens of either the previous or future tokens, sampled by the provided indexes.

    The result is always sorted by timestamp ascending.

    For example, given the following table:
    token | timestamp
    -----------------
    0     | 0
    1     | 1
    2     | 2
    3     | 3
    4     | 4
    5     | 5
    6     | 6
    7     | 7
    8     | 8
    9     | 9
    10    | 10

    Some sample results:
    initial token | sample_indexes | future | returned tokens
    ---------------------------------------------------------
    5             | [0, 1, 2]      | True   | [5, 6, 7]
    5             | [0, 1, 2]      | False  | [3, 4, 5]
    7             | [0, 3, 12]     | False  | [4, 7]
    0             | [11]           | True   | []

    :param log_file: The db file to query.
    :param initial_token: The token on which to base the query.
    :param sensor_source: Parameters for querying the correct table.
    :param sample_indexes: The indexes for which to sample.
    :param future: If true, the indexes represent future times. If false, they represent previous times.
    :return: A generator of LidarPC objects representing the requested indexes
    """
    if not isinstance(sample_indexes, list):
        sample_indexes = list(sample_indexes)

    sensor_token = get_sensor_token(log_file, sensor_source.sensor_table, sensor_source.channel)

    order_direction = "ASC" if future else "DESC"
    order_cmp = ">=" if future else "<="

    query = f"""
        WITH initial_lidarpc AS
        (
            SELECT token, timestamp
            FROM lidar_pc
            WHERE token = ?
        ),
        ordered AS
        (
            SELECT  lp.token,
                    lp.next_token,
                    lp.prev_token,
                    lp.ego_pose_token,
                    lp.lidar_token,
                    lp.scene_token,
                    lp.filename,
                    lp.timestamp,
                    ROW_NUMBER() OVER (ORDER BY lp.timestamp {order_direction}) AS row_num
            FROM lidar_pc AS lp
            CROSS JOIN initial_lidarpc AS il
            WHERE   lp.timestamp {order_cmp} il.timestamp
            AND lp.lidar_token = ?
        )
        SELECT  token,
                next_token,
                prev_token,
                ego_pose_token,
                lidar_token,
                scene_token,
                filename,
                timestamp
        FROM ordered

        -- ROW_NUMBER() starts at 1, where consumers will expect sample_indexes to be 0-indexed
        WHERE (row_num - 1) IN ({('?,'*len(sample_indexes))[:-1]})

        ORDER BY timestamp ASC;
    """

    args = [bytearray.fromhex(initial_token), bytearray.fromhex(sensor_token)] + sample_indexes  # type: ignore
    for row in execute_many(query, args, log_file):
        yield LidarPc.from_db_row(row)


def get_sampled_ego_states_from_db(
    log_file: str,
    initial_token: str,
    sensor_source: SensorDataSource,
    sample_indexes: Union[Generator[int, None, None], List[int]],
    future: bool,
) -> Generator[EgoState, None, None]:
    """
    Given an anchor token, retrieve the ego states associated with tokens order by time, sampled by the provided indexes.

    The result is always sorted by timestamp ascending.

    For example, given the following table:
    token | timestamp | ego_state
    -----------------------------
    0     | 0         | A
    1     | 1         | B
    2     | 2         | C
    3     | 3         | D
    4     | 4         | E
    5     | 5         | F
    6     | 6         | G
    7     | 7         | H
    8     | 8         | I
    9     | 9         | J
    10    | 10        | K

    Some sample results:
    initial token | sample_indexes | future | returned states
    ---------------------------------------------------------
    5             | [0, 1, 2]      | True   | [F, G, H]
    5             | [0, 1, 2]      | False  | [D, E, F]
    7             | [0, 3, 12]     | False  | [E, H]
    0             | [11]           | True   | []

    :param log_file: The db file to query.
    :param initial_token: The token on which to base the query.
    :param sample_indexes: The indexes for which to sample.
    :param future: If true, the indexes represent future times. If false, they represent previous times.
    :return: A generator of EgoState objects associated with the given LidarPCs.
    """
    if not isinstance(sample_indexes, list):
        sample_indexes = list(sample_indexes)

    sensor_token = get_sensor_token(log_file, sensor_source.sensor_table, sensor_source.channel)

    order_direction = "ASC" if future else "DESC"
    order_cmp = ">=" if future else "<="

    # TODO: We can remove dependency from lidar_pc if instead of accessing lp.scene_token we do a join on ego_pose
    query = f"""
        WITH initial_lidarpc AS
        (
            SELECT token, timestamp
            FROM lidar_pc
            WHERE token = ?
        ),
        ordered AS
        (
            SELECT  lp.token,
                    lp.next_token,
                    lp.prev_token,
                    lp.ego_pose_token,
                    lp.lidar_token,
                    lp.scene_token,
                    lp.filename,
                    lp.timestamp,
                    ROW_NUMBER() OVER (ORDER BY lp.timestamp {order_direction}) AS row_num
            FROM lidar_pc AS lp
            CROSS JOIN initial_lidarpc AS il
            WHERE   lp.timestamp {order_cmp} il.timestamp
            AND lidar_token = ?
        )
        SELECT  ep.x,
                ep.y,
                ep.qw,
                ep.qx,
                ep.qy,
                ep.qz,
                -- ego_pose and lidar_pc timestamps are not the same, even when linked by token!
                -- use the lidar_pc timestamp for compatibility with older code.
                o.timestamp,
                ep.vx,
                ep.vy,
                ep.acceleration_x,
                ep.acceleration_y
        FROM ego_pose AS ep
        INNER JOIN ordered AS o
            ON o.ego_pose_token = ep.token

        -- ROW_NUMBER() starts at 1, where consumers will expect sample_indexes to be 0-indexed
        WHERE (o.row_num - 1) IN ({('?,'*len(sample_indexes))[:-1]})

        ORDER BY o.timestamp ASC;
    """

    args = [bytearray.fromhex(initial_token), bytearray.fromhex(sensor_token)] + sample_indexes  # type: ignore
    for row in execute_many(query, args, log_file):
        q = Quaternion(row["qw"], row["qx"], row["qy"], row["qz"])
        yield EgoState.build_from_rear_axle(
            StateSE2(row["x"], row["y"], q.yaw_pitch_roll[0]),
            tire_steering_angle=0.0,
            vehicle_parameters=get_pacifica_parameters(),
            time_point=TimePoint(row["timestamp"]),
            rear_axle_velocity_2d=StateVector2D(row["vx"], y=row["vy"]),
            rear_axle_acceleration_2d=StateVector2D(x=row["acceleration_x"], y=row["acceleration_y"]),
        )


def get_ego_state_for_lidarpc_token_from_db(log_file: str, token: str) -> EgoState:
    """
    Get the ego state associated with an individual lidar_pc token from the db.

    :param log_file: The log file to query.
    :param token: The lidar_pc token to query.
    :return: The EgoState associated with the LidarPC.
    """
    query = """
        SELECT  ep.x,
                ep.y,
                ep.qw,
                ep.qx,
                ep.qy,
                ep.qz,
                -- ego_pose and lidar_pc timestamps are not the same, even when linked by token!
                -- use lidar_pc timestamp for backwards compatibility.
                lp.timestamp,
                ep.vx,
                ep.vy,
                ep.acceleration_x,
                ep.acceleration_y
        FROM ego_pose AS ep
        INNER JOIN lidar_pc AS lp
            ON lp.ego_pose_token = ep.token
        WHERE lp.token = ?
    """

    row = execute_one(query, (bytearray.fromhex(token),), log_file)
    if row is None:
        return None

    q = Quaternion(row["qw"], row["qx"], row["qy"], row["qz"])
    return EgoState.build_from_rear_axle(
        StateSE2(row["x"], row["y"], q.yaw_pitch_roll[0]),
        tire_steering_angle=0.0,
        vehicle_parameters=get_pacifica_parameters(),
        time_point=TimePoint(row["timestamp"]),
        rear_axle_velocity_2d=StateVector2D(row["vx"], y=row["vy"]),
        rear_axle_acceleration_2d=StateVector2D(x=row["acceleration_x"], y=row["acceleration_y"]),
    )


def get_traffic_light_status_for_lidarpc_token_from_db(
    log_file: str, token: str
) -> Generator[TrafficLightStatusData, None, None]:
    """
    Get the traffic light information associated with a given lidar_pc.
    :param log_file: The log file to query.
    :param token: The lidar_pc token for which to obtain the traffic light information.
    :return: The traffic light status data associated with the given lidar_pc.
    """
    query = """
        SELECT  CASE WHEN tl.status == "green" THEN 0
                     WHEN tl.status == "yellow" THEN 1
                     WHEN tl.status == "red" THEN 2
                     ELSE 3
                END AS status,
                tl.lane_connector_id,
                lp.timestamp AS timestamp
        FROM lidar_pc AS lp
        INNER JOIN traffic_light_status AS tl
            ON lp.token = tl.lidar_pc_token
        WHERE lp.token = ?
    """

    for row in execute_many(query, (bytearray.fromhex(token),), log_file):
        yield TrafficLightStatusData(
            status=TrafficLightStatusType(row["status"]),
            lane_connector_id=row["lane_connector_id"],
            timestamp=row["timestamp"],
        )


def get_tracked_objects_within_time_interval_from_db(
    log_file: str, start_timestamp: int, end_timestamp: int, filter_track_tokens: Optional[Set[str]] = None
) -> Generator[TrackedObject, None, None]:
    """
    Gets all of the tracked objects between the provided timestamps, inclusive.
    Optionally filters on a user-provided set of track tokens.

    This query will not obtain the future waypoints.
    For that, call `get_future_waypoints_for_agents_from_db()`
    with the tokens of the agents of interest.

    :param log_file: The log file to query.
    :param start_timestamp: The starting timestamp for which to query, in uS.
    :param end_timestamp: The ending timestamp for which to query, in uS.
    :param filter_track_tokens: If provided, only agents with `track_tokens` present in the provided set will be returned.
      If not provided, then all agents present at every time stamp will be returned.
    :return: A generator of TrackedObjects, sorted by TimeStamp, then TrackedObject.
    """
    args: List[Union[int, bytearray]] = [start_timestamp, end_timestamp]

    filter_clause = ""
    if filter_track_tokens is not None:
        filter_clause = """
            AND lb.track_token IN ({('?,'*len(filter_track_tokens))[:-1]})
        """
        for token in filter_track_tokens:
            args.append(bytearray.fromhex(token))

    query = f"""
        SELECT  c.name AS category_name,
                lb.x,
                lb.y,
                lb.z,
                lb.yaw,
                lb.width,
                lb.length,
                lb.height,
                lb.vx,
                lb.vy,
                lb.token,
                lb.track_token,
                lp.timestamp
        FROM lidar_box AS lb
        INNER JOIN track AS t
            ON t.token = lb.track_token
        INNER JOIN category AS c
            ON c.token = t.category_token
        INNER JOIN lidar_pc AS lp
            ON lp.token = lb.lidar_pc_token
        WHERE lp.timestamp >= ?
            AND lp.timestamp <= ?
            {filter_clause}
        ORDER BY lp.timestamp ASC, lb.track_token ASC;
    """
    for row in execute_many(query, args, log_file):
        yield _parse_tracked_object_row(row)


def get_tracked_objects_for_lidarpc_token_from_db(log_file: str, token: str) -> Generator[TrackedObject, None, None]:
    """
    Get all tracked objects for a given lidar_pc.
    This includes both agents and static objects.
    The values are returned in random order.

    For agents, this query will not obtain the future waypoints.
    For that, call `get_future_waypoints_for_agents_from_db()`
        with the tokens of the agents of interest.

    :param log_file: The log file to query.
    :param token: The lidar_pc token for which to obtain the objects.
    :return: The tracked objects associated with the token.
    """
    query = """
        SELECT  c.name AS category_name,
                lb.x,
                lb.y,
                lb.z,
                lb.yaw,
                lb.width,
                lb.length,
                lb.height,
                lb.vx,
                lb.vy,
                lb.token,
                lb.track_token,
                lp.timestamp
        FROM lidar_box AS lb
        INNER JOIN track AS t
            ON t.token = lb.track_token
        INNER JOIN category AS c
            ON c.token = t.category_token
        INNER JOIN lidar_pc AS lp
            ON lp.token = lb.lidar_pc_token
        WHERE lp.token = ?
    """

    for row in execute_many(query, (bytearray.fromhex(token),), log_file):
        yield _parse_tracked_object_row(row)


def get_future_waypoints_for_agents_from_db(
    log_file: str, track_tokens: Union[Generator[str, None, None], List[str]], start_timestamp: int, end_timestamp: int
) -> Generator[Tuple[str, Waypoint], None, None]:
    """
    Obtain the future waypoints for the selected agents from the DB in the provided time window.
    Results are sorted by track token, then by timestamp in ascending order.

    :param log_file: The log file to query.
    :param track_tokens: The track_tokens for which to query.
    :param start_timestamp: The starting timestamp for which to query.
    :param end_timestamp: The maximal time for which to query.
    :return: A generator of tuples of (track_token, Waypoint), sorted by track_token, then by timestamp in ascending order.
    """
    if not isinstance(track_tokens, list):
        track_tokens = list(track_tokens)

    query = f"""
        SELECT  lb.x,
                lb.y,
                lb.z,
                lb.yaw,
                lb.width,
                lb.length,
                lb.height,
                lb.vx,
                lb.vy,
                lb.track_token,
                lp.timestamp
        FROM lidar_box AS lb
        INNER JOIN lidar_pc AS lp
            ON lp.token = lb.lidar_pc_token
        WHERE   lp.timestamp >= ?
            AND lp.timestamp <= ?
            AND lb.track_token IN
            ({('?,'*len(track_tokens))[:-1]})
        ORDER BY lb.track_token ASC, lp.timestamp ASC;
    """

    args = [start_timestamp, end_timestamp] + [bytearray.fromhex(t) for t in track_tokens]  # type: ignore

    for row in execute_many(query, args, log_file):
        pose = StateSE2(row["x"], row["y"], row["yaw"])
        oriented_box = OrientedBox(pose, width=row["width"], height=row["height"], length=row["length"])
        velocity = StateVector2D(row["vx"], row["vy"])

        yield (row["track_token"].hex(), Waypoint(TimePoint(row["timestamp"]), oriented_box, velocity))


def get_scenarios_from_db(
    log_file: str,
    filter_tokens: Optional[List[str]],
    filter_types: Optional[List[str]],
    filter_map_names: Optional[List[str]],
    include_invalid_mission_goals: bool = True,
    include_cameras: bool = False,
) -> Generator[sqlite3.Row, None, None]:
    """
    Get the scenarios present in the db file that match the specified filter criteria.
    If a filter is None, then it will be elided from the query.
    Results are sorted by timestamp ascending
    :param log_file: The log file to query.
    :param filter_tokens: If provided, the set of allowable tokens to return.
    :param filter_types: If provided, the set of allowable scenario types to return.
    :param filter_map_names: If provided, the set of allowable map names to return.
    :param include_cameras: If true, filter for lidar_pcs that has corresponding images.
    :param include_invalid_mission_goals: If true, then scenarios without a valid mission goal will be included
        (i.e. get_mission_goal_for_sensor_data_token_from_db(token) returns None)
        If False, then these scenarios will be filtered.
    :sensor_data_source: Table specification for data sourcing.
    :return: A sqlite3.Row object with the following fields:
        * token: The initial lidar_pc token of the scenario.
        * timestamp: The timestamp of the initial lidar_pc of the scenario.
        * map_name: The map name from which the scenario came.
        * scenario_type: One of the mapped scenario types for the scenario.
            This can be None if there are no matching rows in scenario_types table.
            If there are multiple matches, then one is selected from the set of allowable filter clauses at random.
    """
    filter_clauses = []
    args: List[Union[str, bytearray]] = []
    if filter_types is not None:
        filter_clauses.append(
            f"""
        st.type IN ({('?,'*len(filter_types))[:-1]})
        """
        )
        args += filter_types

    if filter_tokens is not None:
        filter_clauses.append(
            f"""
        lp.token IN ({('?,'*len(filter_tokens))[:-1]})
        """
        )
        args += [bytearray.fromhex(t) for t in filter_tokens]

    if filter_map_names is not None:
        filter_clauses.append(
            f"""
        l.map_version IN ({('?,'*len(filter_map_names))[:-1]})
        """
        )
        args += filter_map_names

    if len(filter_clauses) > 0:
        filter_clause = "WHERE " + " AND ".join(filter_clauses)
    else:
        filter_clause = ""

    if include_invalid_mission_goals:
        invalid_goals_joins = ""
    else:
        invalid_goals_joins = """
        ---Join on ego_pose to filter scenarios that do not have a valid mission goal
        INNER JOIN scene AS invalid_goal_scene
            ON invalid_goal_scene.token = lp.scene_token
        INNER JOIN ego_pose AS invalid_goal_ego_pose
            ON invalid_goal_scene.goal_ego_pose_token = invalid_goal_ego_pose.token
        """

    if include_cameras:
        matching_camera_clause = """
        INNER JOIN image AS img
            ON img.ego_pose_token = lp.ego_pose_token
        """
    else:
        matching_camera_clause = ""

    query = f"""
        WITH ordered_scenes AS
        (
            SELECT  token,
                    ROW_NUMBER() OVER (ORDER BY name ASC) AS row_num
            FROM scene
        ),
        num_scenes AS
        (
            SELECT  COUNT(*) AS cnt
            FROM scene
        ),
        valid_scenes AS
        (
            SELECT  o.token
            FROM ordered_scenes AS o
            CROSS JOIN num_scenes AS n

            -- Define "valid" scenes as those that have at least 2 before and 2 after
            -- Note that the token denotes the beginning of a scene
            WHERE o.row_num >= 3 AND o.row_num < n.cnt - 1
        )
        SELECT  lp.token,
                lp.timestamp,
                l.map_version AS map_name,

                -- scenarios can have multiple tags
                -- Pick one arbitrarily from the list of acceptable tags
                MAX(st.type) AS scenario_type
        FROM lidar_pc AS lp
        LEFT OUTER JOIN scenario_tag AS st
            ON lp.token = st.lidar_pc_token
        INNER JOIN lidar AS ld
            ON ld.token = lp.lidar_token
        INNER JOIN log AS l
            ON ld.log_token = l.token
        INNER JOIN valid_scenes AS vs
            ON lp.scene_token = vs.token
        {matching_camera_clause}
        {invalid_goals_joins}
        {filter_clause}
        GROUP BY    lp.token,
                    lp.timestamp,
                    l.map_version
        ORDER BY lp.timestamp ASC;
    """

    for row in execute_many(query, args, log_file):
        yield row


def get_lidarpc_tokens_with_scenario_tag_from_db(log_file: str) -> Generator[Tuple[str, str], None, None]:
    """
    Get the LidarPc tokens that are tagged with a scenario from the DB, sorted by scenario_type in ascending order.
    :param log_file: The log file to query.
    :return: A generator of (scenario_tag, token) tuples where `token` is tagged with `scenario_tag`
    """
    query = """
    SELECT  st.type,
            lp.token
    FROM lidar_pc AS lp
    LEFT OUTER JOIN scenario_tag AS st
        ON lp.token=st.lidar_pc_token
    WHERE st.type IS NOT NULL
    ORDER BY st.type ASC NULLS LAST;
    """

    for row in execute_many(query, (), log_file):
        yield (str(row["type"]), row["token"].hex())


def get_sensor_token(log_file: str, table: str, channel: str) -> str:
    """
    Get the sensor token of a particular channel for the given table.
    :param log_file: The DB file.
    :param table: The sensor table to query.
    :param channel: The channel to select.
    :return: The token of the sensor with the given channel.
    """
    q1 = f"""
        SELECT token
        FROM {table}
        WHERE channel == '{channel}';
    """
    row = execute_one(q1, (), log_file)

    if row is None:
        raise RuntimeError(f"Channel {channel} not found in table {table}!")

    return str(row['token'].hex())


def get_images_from_lidar_tokens(
    log_file: str,
    tokens: List[str],
    channels: List[str],
    lookahead_window_us: int = 50000,
    lookback_window_us: int = 50000,
) -> Generator[Image, None, None]:
    """
    Get the images from the given channels for the given lidar_pc_tokens.
    Note: Both lookahead_window_us and lookback_window_us is defaulted to 50000us (0.05s). This means the search window
          is 0.1s centered around the queried lidar_pc timestamp. This is because lidar_pc are stored at 20hz and images
          are at 10hz for NuPlanDB. Hence, we can search the entire duration between lidar_pcs.
          Consider the example below where we want to query for images from the lidar_pc '4'. '|' represents a sample.

          iteration: 0    1    2    3   [4]   5    6
          timestamp: 0   0.05 0.1  0.15 0.2  0.25 0.3
          lidar_pc:  |    |    |    |    |    |    |
          Images:    |         |         |         |
          search window:            [---------]

          We set the search window to lookahead_window_us + lookback_window_us = 0.1s centered around lidar_pc '4'.
          This should guarantee that we retrieve the correct images associated with the queried lidar_pc.

    :param log_file: The log file to query.
    :param tokens: corresponding lidar_pc.
    :param channels: The channel to select.
    :param lookahead_window_us: [us] The time duration to look ahead relative to the lidar_pc for matching images.
    :param lookback_window_us: [us] The time duration to look back relative to the lidar_pc for matching images.
    :return: Images as a SensorDataTableRow.
    """
    query = f"""
            SELECT
                img.token,
                img.next_token,
                img.prev_token,
                img.ego_pose_token,
                img.camera_token,
                img.filename_jpg,
                img.timestamp,
                cam.channel
            FROM image AS img
              INNER JOIN lidar_pc AS lpc
                ON  img.timestamp <= lpc.timestamp + ?
                AND img.timestamp >= lpc.timestamp - ?
              INNER JOIN camera AS cam
                ON cam.token = img.camera_token
            WHERE cam.channel IN ({('?,'*len(channels))[:-1]}) AND lpc.token IN ({('?,'*len(tokens))[:-1]})
            ORDER BY lpc.timestamp ASC;
    """
    args = [lookahead_window_us, lookback_window_us]
    args += channels  # type: ignore
    args += [bytearray.fromhex(t) for t in tokens]  # type: ignore

    for row in execute_many(query, args, log_file):
        yield Image.from_db_row(row)


def get_cameras(
    log_file: str,
    channels: List[str],
) -> Generator[Camera, None, None]:
    """
    Get the cameras for the given channels.
    :param log_file: The log file to query.
    :param channels: The channel to select.
    :return: Cameras as a SensorDataTableRow.
    """
    query = f"""
            SELECT *
            FROM camera AS cam
            WHERE cam.channel IN ({('?,'*len(channels))[:-1]})
    """
    for row in execute_many(query, channels, log_file):
        yield Camera.from_db_row(row)
