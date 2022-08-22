import pickle
import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from nuplan.database.common.data_types import Rotation, Translation


@dataclass(frozen=True)
class DBGenerationParameters:
    """
    Encapsulates the parameters used to generate a synthetic NuPlan DB.
    """

    # The number of LidarPC rows to generate
    num_lidar_pcs: int

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
        if self.num_scenes > self.num_lidar_pcs or self.num_lidar_pcs % self.num_scenes != 0:
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


def _generate_mapping_keys(db_generation_parameters: DBGenerationParameters) -> Dict[str, List[Dict[str, Any]]]:
    """
    Generate all the FK mappings between the generated tables based on the provided parameters.
    :param db_generation_parameters: The generation parameters to use.
    :return: Dicts containing the FK information for each table.
    """
    token_dicts: Dict[str, List[Dict[str, Any]]] = {
        "lidar_pc": [],
        "lidar": [],
        "ego_pose": [],
        "track": [],
        "lidar_box": [],
        "scene": [],
        "traffic_light_status": [],
        "category": [],
        "scenario_tag": [],
    }

    num_pc_tokens_per_scene = int(db_generation_parameters.num_lidar_pcs / db_generation_parameters.num_scenes)

    for lidar_pc_idx in range(db_generation_parameters.num_lidar_pcs):
        # Most of these offsets are chosen arbitrarily
        # The idea is to make it less likely that an errant join "accidentally" works
        # And it becomes clear which columns are supposed to join with each other
        lidar_pc_token = lidar_pc_idx
        scene_token = int(lidar_pc_idx / num_pc_tokens_per_scene) + 100000
        ego_pose_token = lidar_pc_idx + 300000
        traffic_light_status_base_token_id = 400000
        traffic_light_status_pc_step = 10000
        lidar_box_base_token_id = 500000
        lidar_box_pc_step = 10000
        track_token_base_token_id = 600000
        lidar_token = lidar_pc_idx + 700000
        scenario_tag_base_token_id = 800000
        timestamp = lidar_pc_idx * 1e6

        next_lidar_pc = {
            "lidar_pc_token": lidar_pc_token,
            "prev_lidar_pc_token": None if lidar_pc_idx == 0 else lidar_pc_idx - 1,
            "next_lidar_pc_token": None
            if (lidar_pc_idx == db_generation_parameters.num_lidar_pcs - 1)
            else lidar_pc_idx + 1,
            "scene_token": scene_token,
            "ego_pose_token": ego_pose_token,
            "lidar_token": lidar_token,
            "lidar_pc_timestamp": timestamp,
        }
        token_dicts["lidar_pc"].append(next_lidar_pc)

        # Ego pose timestamp can be slightly off from lidar_pc timestamp.
        # This should never be used by most queries.
        # So adjust the value by a slight amount to make it obvious when the wrong column is selected.
        token_dicts["ego_pose"].append({"token": ego_pose_token, "timestamp": timestamp + 333})

        token_dicts["lidar"].append(
            {
                "token": lidar_token,
            }
        )

        for traffic_light_idx in range(db_generation_parameters.num_traffic_lights_per_lidar_pc):
            statuses = ["green", "red", "yellow", "unknown"]

            token_dicts["traffic_light_status"].append(
                {
                    "token": traffic_light_status_base_token_id
                    + (lidar_pc_idx * traffic_light_status_pc_step)
                    + traffic_light_idx,
                    "lidar_pc_token": lidar_pc_token,
                    "lane_connector_id": traffic_light_idx,
                    "status": statuses[(traffic_light_status_base_token_id + traffic_light_idx) % len(statuses)],
                }
            )

        for traffic_light_idx in range(db_generation_parameters.total_object_count()):
            token_dicts["lidar_box"].append(
                {
                    "token": lidar_box_base_token_id + (lidar_pc_idx * lidar_box_pc_step) + traffic_light_idx,
                    "lidar_pc_token": lidar_pc_token,
                    "track_token": track_token_base_token_id + traffic_light_idx,
                    "prev_token": None
                    if lidar_pc_idx == 0
                    else lidar_box_base_token_id + ((lidar_pc_idx - 1) * lidar_box_pc_step) + traffic_light_idx,
                    "next_token": None
                    if lidar_pc_idx == (db_generation_parameters.num_lidar_pcs - 1)
                    else lidar_box_base_token_id + ((lidar_pc_idx + 1) * lidar_box_pc_step) + traffic_light_idx,
                }
            )

        if lidar_pc_idx % num_pc_tokens_per_scene == 0:
            token_dicts["scene"].append(
                {
                    "token": scene_token,
                    "ego_pose_token": ego_pose_token + num_pc_tokens_per_scene - 1,
                    "name": "scene-{:03d}".format(int(lidar_pc_idx / num_pc_tokens_per_scene)),
                }
            )

            # Tag the first lidar_pc in each scene with the proper tags.
            scene_idx = int(lidar_pc_idx / num_pc_tokens_per_scene)
            if scene_idx in db_generation_parameters.scene_scenario_tag_mapping:
                tags = db_generation_parameters.scene_scenario_tag_mapping[scene_idx]
                for tag in tags:
                    row = {
                        "token": scenario_tag_base_token_id + len(token_dicts["scenario_tag"]),
                        "lidar_pc_token": lidar_pc_token,
                        "type": tag,
                    }
                    token_dicts["scenario_tag"].append(row)

    for lidar_pc_idx in range(db_generation_parameters.total_object_count()):
        token_dicts["track"].append(
            {
                "token": track_token_base_token_id + lidar_pc_idx,
                "category_token": 900000 if lidar_pc_idx < db_generation_parameters.num_agents_per_lidar_pc else 900005,
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
                "channel",
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


def generate_minimal_nuplan_db(parameters: DBGenerationParameters) -> None:
    """
    Generate a synthetic nuplan_db based on the supplied generation parameters.
    :param parameters: The parameters to use for generation.
    """
    mapping_keys = _generate_mapping_keys(parameters)

    _generate_lidar_pc_table(mapping_keys["lidar_pc"], parameters.file_path)
    _generate_lidar_table(mapping_keys["lidar"], parameters.file_path)
    _generate_ego_pose_table(mapping_keys["ego_pose"], parameters.file_path)
    _generate_scene_table(mapping_keys["scene"], parameters.file_path)
    _generate_traffic_light_status_table(mapping_keys["traffic_light_status"], parameters.file_path)
    _generate_lidar_box_table(mapping_keys["lidar_box"], parameters.file_path)
    _generate_track_table(mapping_keys["track"], parameters.file_path)
    _generate_scenario_tag_table(mapping_keys["scenario_tag"], parameters.file_path)
    _generate_category_table(parameters.file_path)
    _generate_log_table(parameters.file_path)
