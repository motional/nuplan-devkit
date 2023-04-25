import pickle
import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from nuplan.database.common.data_types import CameraIntrinsic, Rotation, Translation


@dataclass(frozen=True)
class DBGenerationParameters:
    """
    Encapsulates the parameters used to generate a synthetic NuPlan DB.
    """

    # The number of Lidar rows to generate
    num_lidars: int

    # The number of Camera rows to generate
    num_cameras: int

    # The number of LidarPC rows to generate for each lidar and Image rows for each Camera
    num_sensor_data_per_sensor: int

    # The ratio of lidar pc to an image. Example, 2 results in generating 2 lidar_pc for each image
    num_lidarpc_per_image_ratio: int

    # The number of scenes to include in the generated.
    # This must be <= number of lidar_pcs, and num_lidar_pcs % num_scenes must equal 0
    num_scenes: int

    # The number of traffic lights to generate per lidar_pc
    num_traffic_lights_per_lidar_pc: int

    # The number of moving agents to generate per lidar_pc
    num_agents_per_lidar_pc: int

    # The number of non-moving agents to generate per lidar_pc
    num_static_objects_per_lidar_pc: int

    # Tags scenes indexed by key with the selected tags.
    # For example, this dict
    #   {4: ["foo", "bar"], 5: ["bar"]}
    # Will create 3 tags - (scene_4, "foo"), (scene_4, "bar"), (scene_5, "bar")
    scene_scenario_tag_mapping: Dict[int, List[str]]

    # The file path to which to write the db.
    # TODO: can we eventually support ":memory:"?
    file_path: str

    def __post_init__(self) -> None:
        """
        Sanity checks to ensure that the class contains a valid configuration.
        """
        if self.num_scenes > self.num_sensor_data_per_sensor or self.num_sensor_data_per_sensor % self.num_scenes != 0:
            raise ValueError("Number of scenes must be less than number of point clouds, and must be an equal divisor.")

    def total_object_count(self) -> int:
        """
        Gets the total number of objects per lidar_pc in the configuration.
        :return: The number of objects per lidar_pc in the configuration.
        """
        return self.num_agents_per_lidar_pc + self.num_static_objects_per_lidar_pc


def int_to_str_token(val: Optional[int]) -> Optional[str]:
    """
    Convert an int to a string token used for DB access functions.
    :param val: The val to convert.
    :return: None if the input is None. Else, a string version of the input value to be used with db functions as a token.
    """
    return None if val is None else "{:08d}".format(val)


def str_token_to_int(val: Optional[str]) -> Optional[int]:
    """
    Convert a string token previously genreated with int_to_str_token() back to an int.
    :param val: The token to convert.
    :return: None if the input is None. Else, the int version of the string.
        The output is undefined if the token was not generated with int_to_str_token().
    """
    return None if val is None else int(val)


def _int_to_token(val: Optional[int]) -> Optional[bytearray]:
    """
    Convert an int directly to a token bytearray.
    Intended for use only in this file.
    :param val: The int to convert.
    :return: The token bytearray.
    """
    return None if val is None else bytearray.fromhex("{:08d}".format(val))


def _execute_non_query(query_text: str, file_path: str) -> None:
    """
    Connect to a SQLite DB and runs a query that returns no results.
    E.g. a CREATE TABLE statement.
    :param query_text: The query text to run.
    :param file_path: The file on which to run the query.
    """
    connection = sqlite3.connect(file_path)
    cursor = connection.cursor()
    try:
        cursor.execute(query_text)
    finally:
        cursor.close()
        connection.close()


def _execute_bulk_insert(query_text: str, values: List[Any], file_path: str) -> None:
    """
    Connect to a SQLite DB and runs a query that inserts many rows into the DB.
    This function will commit the changes after a successful execution.
    :param query_text: The query text to run.
    :param values: The values to insert.
    :param file_path: The file on which to run the query.
    """
    connection = sqlite3.connect(file_path)
    cursor = connection.cursor()
    try:
        cursor.executemany(query_text, values)
        cursor.execute("commit;")
    finally:
        cursor.close()
        connection.close()


def _generate_camera_data(
    db_generation_parameters: DBGenerationParameters,
    token_dicts: Dict[str, List[Dict[str, Any]]],
    offset_image_token: int,
    offset_camera_token: int,
    offset_ego_pose_token: int,
) -> None:
    """
    Generate all the mappings for the camera table based on the provided parameters.
    :param db_generation_parameters: The generation parameters to use.
    :param token_dicts: Dicts containing the FK information for each table
    :param offset_image_token: Offset to mark the range of the image tokens
    :param offset_camera_token: Offset to mark the range of the camera tokens
    :param offset_ego_pose_token: Offset to mark the range of the ego pose tokens
    """
    for sensor_idx in range(db_generation_parameters.num_cameras):
        camera_token = offset_camera_token + sensor_idx
        camera_channel_name = f"camera_{sensor_idx}"
        for sensor_data_idx in range(db_generation_parameters.num_sensor_data_per_sensor):
            # Only add one image for every num_lidarpc_per_image_ratio lidar_pcs
            if sensor_data_idx % db_generation_parameters.num_lidarpc_per_image_ratio == 0:
                ego_pose_token = sensor_data_idx + offset_ego_pose_token  # Ego pose token use for joining with LidarPC
                image_timestamp = sensor_data_idx * 1e6
                image_token = (
                    sensor_data_idx
                    + sensor_idx * db_generation_parameters.num_sensor_data_per_sensor
                    + offset_image_token
                )
                image_entry = {
                    "image_token": image_token,
                    "prev_image_token": None if sensor_data_idx == 0 else token_dicts["image"][-1]["image_token"],
                    "next_image_token": image_token + db_generation_parameters.num_lidarpc_per_image_ratio,
                    "ego_pose_token": ego_pose_token,
                    "camera_token": camera_token,
                    "image_timestamp": image_timestamp + 10 + sensor_idx,
                }
                token_dicts["image"].append(image_entry)

        # The last image entry has to point to None
        token_dicts["image"][-1]["next_image_token"] = None
        camera_entry = {
            "token": camera_token,
            "channel": camera_channel_name,
        }
        token_dicts["camera"].append(camera_entry)


def _generate_mapping_keys(db_generation_parameters: DBGenerationParameters) -> Dict[str, List[Dict[str, Any]]]:
    """
    Generate all the FK mappings between the generated tables based on the provided parameters.
    :param db_generation_parameters: The generation parameters to use.
    :return: Dicts containing the FK information for each table.
    """
    token_dicts: Dict[str, List[Dict[str, Any]]] = {
        "lidar_pc": [],
        "lidar": [],
        "image": [],
        "camera": [],
        "ego_pose": [],
        "track": [],
        "lidar_box": [],
        "scene": [],
        "traffic_light_status": [],
        "category": [],
        "scenario_tag": [],
    }

    num_pc_tokens_per_scene = int(
        db_generation_parameters.num_sensor_data_per_sensor / db_generation_parameters.num_scenes
    )

    # Most of these offsets are chosen arbitrarily, listed here for readability
    # The idea is to make it less likely that an errant join "accidentally" works
    # And it becomes clear which columns are supposed to join with each other

    base_token_multiplier = 100000
    # Offset to mark the range of the different tokens
    offset_scene_token = 1 * base_token_multiplier
    offset_ego_pose_token = 3 * base_token_multiplier
    offset_traffic_light_status_token = 4 * base_token_multiplier
    offset_lidar_box_token = 5 * base_token_multiplier
    offset_track_token_token = 6 * base_token_multiplier
    offset_lidar_token = 7 * base_token_multiplier
    offset_scenario_tag_token = 8 * base_token_multiplier
    offset_agents_token = 9 * base_token_multiplier
    offset_static_objects_token = 9 * base_token_multiplier + 5
    offset_camera_token = 10 * base_token_multiplier
    offset_image_token = 11 * base_token_multiplier

    # Step sizes of the tokens
    step_traffic_light_status_pc = 10000
    step_lidar_box_pc = 10000

    _generate_camera_data(
        db_generation_parameters, token_dicts, offset_image_token, offset_camera_token, offset_ego_pose_token
    )

    for sensor_idx in range(db_generation_parameters.num_lidars):
        lidar_token = offset_lidar_token + sensor_idx
        # To keep existing tests valid, first channel is called "channel" then others "channel_1" etc
        lidar_channel_name = "channel" if sensor_idx < 1 else f"channel_{sensor_idx}"

        for sensor_data_idx in range(db_generation_parameters.num_sensor_data_per_sensor):
            lidar_pc_token = sensor_data_idx + sensor_idx * db_generation_parameters.num_sensor_data_per_sensor

            scene_token = int(sensor_data_idx / num_pc_tokens_per_scene) + offset_scene_token

            ego_pose_token = sensor_data_idx + offset_ego_pose_token  # So can be joined across Lidar, as in NuplanDB
            timestamp_ego_pose = sensor_data_idx * 1e6  # This is the same for all lidars
            timestamp_lidar_pc = (
                timestamp_ego_pose + sensor_idx
            )  # This is different as the lidar scan time varies slightly

            next_lidar_pc = {
                "lidar_pc_token": lidar_pc_token,
                "prev_lidar_pc_token": None if sensor_data_idx == 0 else sensor_data_idx - 1,
                "next_lidar_pc_token": None
                if (sensor_data_idx == db_generation_parameters.num_sensor_data_per_sensor - 1)
                else sensor_data_idx + 1,
                "scene_token": scene_token,
                "ego_pose_token": ego_pose_token,
                "lidar_token": lidar_token,
                "lidar_pc_timestamp": timestamp_lidar_pc,
            }
            token_dicts["lidar_pc"].append(next_lidar_pc)

            # Ego pose timestamp can be slightly off from lidar_pc timestamp.
            # This should never be used by most queries.
            # So adjust the value by a slight amount to make it obvious when the wrong column is selected.
            ego_pose_entry = {"token": ego_pose_token, "timestamp": timestamp_ego_pose + 333}
            if ego_pose_entry not in token_dicts['ego_pose']:
                token_dicts["ego_pose"].append(ego_pose_entry)

            lidar_token_entry = {"token": lidar_token, "channel": lidar_channel_name}
            if lidar_token_entry not in token_dicts['lidar']:
                token_dicts["lidar"].append(lidar_token_entry)

            for traffic_light_idx in range(db_generation_parameters.num_traffic_lights_per_lidar_pc):
                statuses = ["green", "red", "yellow", "unknown"]

                # TODO: which lidar_pc does this refer to?
                traffic_light_status_entry = {
                    "token": offset_traffic_light_status_token
                    + (sensor_data_idx * step_traffic_light_status_pc)
                    + traffic_light_idx,
                    "lidar_pc_token": sensor_data_idx,
                    "lane_connector_id": traffic_light_idx,
                    "status": statuses[(offset_traffic_light_status_token + traffic_light_idx) % len(statuses)],
                }

                if traffic_light_status_entry not in token_dicts["traffic_light_status"]:
                    token_dicts["traffic_light_status"].append(traffic_light_status_entry)

            for object_idx in range(db_generation_parameters.total_object_count()):
                # We assume multiple lidars will be able to match the same track
                lidar_box_entry = {
                    "token": offset_lidar_box_token + (sensor_data_idx * step_lidar_box_pc) + object_idx,
                    "lidar_pc_token": sensor_data_idx,
                    "track_token": offset_track_token_token + object_idx,
                    "prev_token": None
                    if sensor_data_idx == 0
                    else offset_lidar_box_token + ((sensor_data_idx - 1) * step_lidar_box_pc) + object_idx,
                    "next_token": None
                    if sensor_data_idx == (db_generation_parameters.num_sensor_data_per_sensor - 1)
                    else offset_lidar_box_token + ((sensor_data_idx + 1) * step_lidar_box_pc) + object_idx,
                }
                if lidar_box_entry not in token_dicts["lidar_box"]:
                    token_dicts["lidar_box"].append(lidar_box_entry)

            if sensor_data_idx % num_pc_tokens_per_scene == 0:
                scene_token_entry = {
                    "token": scene_token,
                    "ego_pose_token": ego_pose_token + num_pc_tokens_per_scene - 1,
                    "name": "scene-{:03d}".format(int(sensor_data_idx / num_pc_tokens_per_scene)),
                }
                if scene_token_entry not in token_dicts['scene']:
                    token_dicts['scene'].append(scene_token_entry)

                # Tag the first lidar_pc in each scene with the proper tags.
                scene_idx = sensor_data_idx // num_pc_tokens_per_scene
                if scene_idx in db_generation_parameters.scene_scenario_tag_mapping and lidar_channel_name == 'channel':
                    tags = db_generation_parameters.scene_scenario_tag_mapping[scene_idx]
                    for tag in tags:
                        row = {
                            "token": offset_scenario_tag_token + len(token_dicts["scenario_tag"]),
                            "lidar_pc_token": lidar_pc_token,
                            "type": tag,
                        }
                        token_dicts["scenario_tag"].append(row)

    for lidar_pc_idx in range(db_generation_parameters.total_object_count()):
        token_dicts["track"].append(
            {
                "token": offset_track_token_token + lidar_pc_idx,
                "category_token": offset_agents_token
                if lidar_pc_idx < db_generation_parameters.num_agents_per_lidar_pc
                else offset_static_objects_token,
            }
        )

    return token_dicts


def _generate_lidar_pc_table(mapping_key_dicts: List[Dict[str, Any]], file_path: str) -> None:
    """
    Generate the lidar_pc table based on the provided FK information.
    :param mapping_key_dicts: A list of row FK information.
    :param file_path: The file path into which to create the tables.
    """
    assert len(mapping_key_dicts) > 0, "No data provided to _generate_lidar_pc_table"
    rows = []
    for idx, row in enumerate(mapping_key_dicts):
        rows.append(
            (
                _int_to_token(row["lidar_pc_token"]),
                _int_to_token(row["next_lidar_pc_token"]),
                _int_to_token(row["prev_lidar_pc_token"]),
                _int_to_token(row["ego_pose_token"]),
                _int_to_token(row["lidar_token"]),
                _int_to_token(row["scene_token"]),
                f"pc_{idx}.dat",
                row["lidar_pc_timestamp"],
            )
        )

    query = """
    CREATE TABLE lidar_pc (
        token BLOB NOT NULL,
        next_token BLOB,
        prev_token BLOB,
        ego_pose_token BLOB NOT NULL,
        lidar_token BLOB NOT NULL,
        scene_token BLOB,
        filename VARCHAR(128),
        timestamp INTEGER,
        PRIMARY KEY (token)
    );
    """

    _execute_non_query(query, file_path)

    query = f"""
    INSERT INTO lidar_pc (
        token,
        next_token,
        prev_token,
        ego_pose_token,
        lidar_token,
        scene_token,
        filename,
        timestamp
    )
    VALUES({('?,'*len(rows[0]))[:-1]});
    """

    _execute_bulk_insert(query, rows, file_path)


def _generate_lidar_table(mapping_key_dicts: List[Dict[str, Any]], file_path: str) -> None:
    """
    Generate the lidar table based on the provided FK information.
    :param mapping_key_dicts: A list of row FK information.
    :param file_path: The file path into which to create the tables.
    """
    assert len(mapping_key_dicts) > 0, "No data provided to _generate_lidar_table"
    rows = []
    for idx, row in enumerate(mapping_key_dicts):
        rows.append(
            (
                _int_to_token(row["token"]),
                _int_to_token(0),
                row["channel"],
                "model",
                pickle.dumps(Translation([idx, idx + 1, idx + 2])),
                pickle.dumps(Rotation([0, 0, 0, 0])),
            )
        )

    query = """
    CREATE TABLE lidar (
        token BLOB NOT NULL,
        log_token BLOB NOT NULL,
        channel VARCHAR(64),
        model VARCHAR(64),
        translation BLOB,
        rotation BLOB,
        PRIMARY KEY (token)
    );
    """

    _execute_non_query(query, file_path)

    query = f"""
    INSERT INTO lidar (
        token,
        log_token,
        channel,
        model,
        translation,
        rotation
    )
    VALUES({('?,'*len(rows[0]))[:-1]});
    """

    _execute_bulk_insert(query, rows, file_path)


def _generate_ego_pose_table(mapping_key_dicts: List[Dict[str, Any]], file_path: str) -> None:
    """
    Generate the ego_pose table based on the provided FK information.
    :param mapping_key_dicts: A list of row FK information.
    :param file_path: The file path into which to create the tables.
    """
    assert len(mapping_key_dicts) > 0, "No data passed to _generate_ego_pose_table"

    rows = []
    for idx, mapping_dict in enumerate(mapping_key_dicts):
        rows.append(
            (
                _int_to_token(mapping_dict["token"]),
                mapping_dict["timestamp"],
                idx,
                idx + 1,
                idx + 2,
                idx + 3,
                idx + 4,
                idx + 5,
                idx + 6,
                idx + 7,
                idx + 8,
                idx + 9,
                idx + 10,
                idx + 11,
                idx + 12,
                idx + 13,
                idx + 14,
                idx + 15,
                idx + 16,
                _int_to_token(0),  # TODO: if log token is ever needed, fill this in
            )
        )

    query = """
    CREATE TABLE ego_pose (
        token BLOB NOT NULL,
        timestamp INTEGER,
        x FLOAT,
        y FLOAT,
        z FLOAT,
        qw FLOAT,
        qx FLOAT,
        qy FLOAT,
        qz FLOAT,
        vx FLOAT,
        vy FLOAT,
        vz FLOAT,
        acceleration_x FLOAT,
        acceleration_y FLOAT,
        acceleration_z FLOAT,
        angular_rate_x FLOAT,
        angular_rate_y FLOAT,
        angular_rate_z FLOAT,
        epsg INTEGER,
        log_token BLOB NOT NULL,
        PRIMARY KEY (token)
    );
    """

    _execute_non_query(query, file_path)

    query = f"""
    INSERT INTO ego_pose (
        token,
        timestamp,
        x,
        y,
        z,
        qw,
        qx,
        qy,
        qz,
        vx,
        vy,
        vz,
        acceleration_x,
        acceleration_y,
        acceleration_z,
        angular_rate_x,
        angular_rate_y,
        angular_rate_z,
        epsg,
        log_token
    )
    VALUES({('?,'*len(rows[0]))[:-1]});
    """

    _execute_bulk_insert(query, rows, file_path)


def _generate_scene_table(mapping_key_dicts: List[Dict[str, Any]], file_path: str) -> None:
    """
    Generate the scene table based on the provided FK information.
    :param mapping_key_dicts: A list of row FK information.
    :param file_path: The file path into which to create the tables.
    """
    assert len(mapping_key_dicts) > 0, "No data passed to _generate_scene_table"
    rows = []
    for idx, mapping_dict in enumerate(mapping_key_dicts):
        rows.append(
            (
                _int_to_token(mapping_dict["token"]),
                _int_to_token(0),  # TODO: fill in if we ever need log_token
                mapping_dict["name"],
                _int_to_token(mapping_dict["ego_pose_token"]),
                f"{idx} {idx+1} {idx+2}",
            )
        )

    query = """
    CREATE TABLE scene (
        token BLOB NOT NULL,
        log_token BLOB NOT NULL,
        name TEXT,
        goal_ego_pose_token BLOB,
        roadblock_ids TEXT,
        PRIMARY KEY (token)
    );
    """

    _execute_non_query(query, file_path)

    query = f"""
    INSERT INTO scene (
        token,
        log_token,
        name,
        goal_ego_pose_token,
        roadblock_ids
    )
    VALUES({('?,'*len(rows[0]))[:-1]});
    """

    _execute_bulk_insert(query, rows, file_path)


def _generate_traffic_light_status_table(mapping_key_dicts: List[Dict[str, Any]], file_path: str) -> None:
    """
    Generates the traffic_light_status table based on the provided FK information.
    :param mapping_key_dicts: A list of row FK information.
    :param file_path: The file path into which to create the tables.
    """
    assert len(mapping_key_dicts) > 0, "No data passed to _generate_traffic_light_status_table"
    rows = []
    for idx, mapping_dict in enumerate(mapping_key_dicts):
        rows.append(
            (
                _int_to_token(mapping_dict["token"]),
                _int_to_token(mapping_dict["lidar_pc_token"]),
                mapping_dict["lane_connector_id"],
                mapping_dict["status"],
            )
        )

    query = """
    CREATE TABLE traffic_light_status (
        token BLOB NOT NULL,
        lidar_pc_token BLOB NOT NULL,
        lane_connector_id INTEGER,
        status VARCHAR(8),
        PRIMARY KEY (token)
    );
    """

    _execute_non_query(query, file_path)

    query = f"""
    INSERT INTO traffic_light_status (
        token,
        lidar_pc_token,
        lane_connector_id,
        status
    )
    VALUES({('?,'*len(rows[0]))[:-1]});
    """

    _execute_bulk_insert(query, rows, file_path)


def _generate_lidar_box_table(mapping_key_dicts: List[Dict[str, Any]], file_path: str) -> None:
    """
    Generate the lidar_box table based on the provided FK information.
    :param mapping_key_dicts: A list of row FK information.
    :param file_path: The file path into which to create the tables.
    """
    assert len(mapping_key_dicts) > 0, "No data passed to _generate_lidar_box_table"
    rows = []
    for idx, row in enumerate(mapping_key_dicts):
        rows.append(
            (
                _int_to_token(row["token"]),
                _int_to_token(row["lidar_pc_token"]),
                _int_to_token(row["track_token"]),
                _int_to_token(row["next_token"]),
                _int_to_token(row["prev_token"]),
                idx,
                idx + 1,
                idx + 2,
                idx + 3,
                idx + 4,
                idx + 5,
                idx + 6,
                idx + 7,
                idx + 8,
                idx + 9,
                idx + 10,
            )
        )

    query = """
    CREATE TABLE lidar_box (
        token BLOB NOT NULL,
        lidar_pc_token BLOB NOT NULL,
        track_token BLOB NOT NULL,
        next_token BLOB,
        prev_token BLOB,
        x FLOAT,
        y FLOAT,
        z FLOAT,
        width FLOAT,
        length FLOAT,
        height FLOAT,
        vx FLOAT,
        vy FLOAT,
        vz FLOAT,
        yaw FLOAT,
        confidence FLOAT,
        PRIMARY KEY (token)
    );
    """

    _execute_non_query(query, file_path)

    query = f"""
    INSERT INTO lidar_box (
        token,
        lidar_pc_token,
        track_token,
        next_token,
        prev_token,
        x,
        y,
        z,
        width,
        length,
        height,
        vx,
        vy,
        vz,
        yaw,
        confidence
    )
    VALUES({('?,'*len(rows[0]))[:-1]});
    """

    _execute_bulk_insert(query, rows, file_path)


def _generate_category_table(file_path: str) -> None:
    """
    Generate the category table based on the provided FK information.
    :param mapping_key_dicts: A list of row FK information.
    :param file_path: The file path into which to create the tables.
    """
    categories = ["vehicle", "bicycle", "pedestrian", "traffic_cone", "barrier", "czone_sign", "generic_object"]

    rows = [(_int_to_token(idx + 900000), cat, cat + ".") for idx, cat in enumerate(categories)]

    query = """
    CREATE TABLE category (
        token BLOB NOT NULL,
        name VARCHAR(64),
        description TEXT,
        PRIMARY KEY (token)
    );
    """

    _execute_non_query(query, file_path)

    query = f"""
    INSERT INTO category (
        token,
        name,
        description
    )
    VALUES({('?,'*len(rows[0]))[:-1]});
    """

    _execute_bulk_insert(query, rows, file_path)


def _generate_scenario_tag_table(mapping_key_dicts: List[Dict[str, Any]], file_path: str) -> None:
    """
    Generate the scenario_tag table based on the provided FK information.
    :param mapping_key_dicts: A list of row FK information.
    :param file_path: The file path into which to create the tables.
    """
    assert len(mapping_key_dicts) > 0, "No data passed to _generate_scenario_tag_table"
    rows = []
    for idx, row in enumerate(mapping_key_dicts):
        rows.append((_int_to_token(row["token"]), _int_to_token(row["lidar_pc_token"]), row["type"], _int_to_token(0)))

    query = """
        CREATE TABLE scenario_tag (
            token BLOB NOT NULL,
            lidar_pc_token BLOB NOT NULL,
            type TEXT,
            agent_track_token BLOB,
            PRIMARY KEY (token)
        );
    """

    _execute_non_query(query, file_path)

    query = f"""
    INSERT INTO scenario_tag (
        token,
        lidar_pc_token,
        type,
        agent_track_token
    )
    VALUES ({('?,'*len(rows[0]))[:-1]})
    """

    _execute_bulk_insert(query, rows, file_path)


def _generate_track_table(mapping_key_dicts: List[Dict[str, Any]], file_path: str) -> None:
    """
    Generate the track table based on the provided FK information.
    :param mapping_key_dicts: A list of row FK information.
    :param file_path: The file path into which to create the tables.
    """
    assert len(mapping_key_dicts) > 0, "No data passed to _generate_track_table"
    rows = []
    for idx, row in enumerate(mapping_key_dicts):
        rows.append((_int_to_token(row["token"]), _int_to_token(row["category_token"]), idx, idx + 1, idx + 2))

    query = """
    CREATE TABLE track (
        token BLOB NOT NULL,
        category_token BLOB NOT NULL,
        width FLOAT,
        length FLOAT,
        height FLOAT,
        PRIMARY KEY (token)
    );
    """

    _execute_non_query(query, file_path)

    query = f"""
    INSERT INTO track (
        token,
        category_token,
        width,
        length,
        height
    )
    VALUES ({('?,'*len(rows[0]))[:-1]})
    """

    _execute_bulk_insert(query, rows, file_path)


def _generate_log_table(file_path: str) -> None:
    """
    Generates the log table based on the provided FK information.
    :param mapping_key_dicts: A list of row FK information.
    :param file_path: The file path into which to create the tables.
    """
    rows = [(_int_to_token(0), "vehicle_name", "today", 0, "logfile", "location", "map_version")]

    query = """
    CREATE TABLE log (
        token BLOB NOT NULL,
        vehicle_name VARCHAR(64),
        date VARCHAR(64),
        timestamp INTEGER,
        logfile VARCHAR(64),
        location VARCHAR(64),
        map_version VARCHAR(64),
        PRIMARY KEY (token)
    );
    """
    _execute_non_query(query, file_path)

    query = f"""
    INSERT INTO log (
        token,
        vehicle_name,
        date,
        timestamp,
        logfile,
        location,
        map_version
    )
    VALUES ({('?,'*len(rows[0]))[:-1]})
    """

    _execute_bulk_insert(query, rows, file_path)


def _generate_image_table(mapping_key_dicts: List[Dict[str, Any]], file_path: str) -> None:
    """
    Generate the image table based on the provided FK information.
    :param mapping_key_dicts: A list of row FK information.
    :param file_path: The file path into which to create the tables.
    """
    assert len(mapping_key_dicts) > 0, "No data provided to _generate_image_table"
    rows = []
    for idx, row in enumerate(mapping_key_dicts):
        rows.append(
            (
                _int_to_token(row["image_token"]),
                _int_to_token(row["next_image_token"]),
                _int_to_token(row["prev_image_token"]),
                _int_to_token(row["ego_pose_token"]),
                _int_to_token(row["camera_token"]),
                f"image_{idx}_channel_{row['camera_token']}.dat",
                row["image_timestamp"],
            )
        )

    query = """
    CREATE TABLE image (
        token BLOB NOT NULL,
        next_token BLOB,
        prev_token BLOB,
        ego_pose_token BLOB NOT NULL,
        camera_token BLOB NOT NULL,
        filename_jpg VARCHAR(128),
        timestamp INTEGER,
        PRIMARY KEY (token)
    );
    """

    _execute_non_query(query, file_path)

    query = f"""
    INSERT INTO image (
        token,
        next_token,
        prev_token,
        ego_pose_token,
        camera_token,
        filename_jpg,
        timestamp
    )
    VALUES({('?,'*len(rows[0]))[:-1]});
    """

    _execute_bulk_insert(query, rows, file_path)


def _generate_camera_table(mapping_key_dicts: List[Dict[str, Any]], file_path: str) -> None:
    """
    Generate the camera table based on the provided FK information.
    :param mapping_key_dicts: A list of row FK information.
    :param file_path: The file path into which to create the tables.
    """
    assert len(mapping_key_dicts) > 0, "No data provided to _generate_lidar_table"
    rows = []
    for idx, row in enumerate(mapping_key_dicts):
        rows.append(
            (
                _int_to_token(row["token"]),
                _int_to_token(0),
                row["channel"],
                "model",
                pickle.dumps(Translation([idx, idx + 1, idx + 2])),
                pickle.dumps(Rotation([0, 0, 0, 0])),
                pickle.dumps(CameraIntrinsic([[0, 0, 0], [0, 0, 0], [0, 0, 0]])),
                pickle.dumps([0, 1, 2, 3, 4]),
                1.23,
                3.21,
            )
        )

    query = """
    CREATE TABLE camera (
        token BLOB NOT NULL,
        log_token BLOB NOT NULL,
        channel VARCHAR(64),
        model VARCHAR(64),
        translation BLOB,
        rotation BLOB,
        intrinsic BLOB,
        distortion BLOB,
        width FLOAT,
        height FLOAT,
        PRIMARY KEY (token)
    );
    """

    _execute_non_query(query, file_path)

    query = f"""
    INSERT INTO camera (
        token,
        log_token,
        channel,
        model,
        translation,
        rotation,
        intrinsic,
        distortion,
        width,
        height
    )
    VALUES({('?,'*len(rows[0]))[:-1]});
    """

    _execute_bulk_insert(query, rows, file_path)


def generate_minimal_nuplan_db(parameters: DBGenerationParameters) -> None:
    """
    Generate a synthetic nuplan_db based on the supplied generation parameters.
    :param parameters: The parameters to use for generation.
    """
    mapping_keys = _generate_mapping_keys(parameters)

    _generate_lidar_pc_table(mapping_keys["lidar_pc"], parameters.file_path)
    _generate_lidar_table(mapping_keys["lidar"], parameters.file_path)
    _generate_image_table(mapping_keys["image"], parameters.file_path)
    _generate_camera_table(mapping_keys["camera"], parameters.file_path)
    _generate_ego_pose_table(mapping_keys["ego_pose"], parameters.file_path)
    _generate_scene_table(mapping_keys["scene"], parameters.file_path)
    _generate_traffic_light_status_table(mapping_keys["traffic_light_status"], parameters.file_path)
    _generate_lidar_box_table(mapping_keys["lidar_box"], parameters.file_path)
    _generate_track_table(mapping_keys["track"], parameters.file_path)
    _generate_scenario_tag_table(mapping_keys["scenario_tag"], parameters.file_path)
    _generate_category_table(parameters.file_path)
    _generate_log_table(parameters.file_path)
