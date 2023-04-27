import os
import tempfile
import unittest
from pathlib import Path
from typing import IO
from unittest.mock import MagicMock, Mock, call

import pandas as pd

from nuplan.common.utils.distributed_scenario_filter import DistributedMode, DistributedScenarioFilter
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.scenario_builder.abstract_scenario_builder import AbstractScenarioBuilder
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.utils.multithreading.worker_pool import WorkerPool


class TestDistributedScenarioFilter(unittest.TestCase):
    """
    Test the distributed scenario filter that is intended to be used to split work across multiple nodes
    """

    def setUp(self) -> None:
        """
        Build some useful mocks to use in a variety of functions
        """
        # Mock Scenario Builder stuff
        self.scenario_builder_mock = MagicMock(AbstractScenarioBuilder)
        self.mock_scenarios = [MagicMock(AbstractScenario), MagicMock(AbstractScenario)]
        self.scenario_builder_mock.get_scenarios = MagicMock()
        self.scenario_builder_mock.get_scenarios.return_value = self.mock_scenarios
        self.build_scenario_builder_mock = MagicMock()
        self.build_scenario_builder_mock.return_value = self.scenario_builder_mock
        self.scenario_filter_mock = MagicMock(ScenarioFilter)
        self.build_scenario_filter_mock = MagicMock()
        self.build_scenario_filter_mock.return_value = self.scenario_filter_mock

        # Mock Configuration
        self.mock_dbs = ["file_1", "file_2"]
        self.cfg_mock = MagicMock()
        self.cfg_mock.scenario_builder = MagicMock()
        self.cfg_mock.scenario_builder.db_files = self.mock_dbs
        self.cfg_mock.scenario_filter = MagicMock()

        # Mock Scenarios
        self.mock_scenarios[0].token = "a"
        self.mock_scenarios[0].log_name = "1.log"
        self.mock_scenarios[1].token = "b"
        self.mock_scenarios[1].log_name = "2.log"

        # Worker Mock
        self.worker_mock = MagicMock(WorkerPool)

        # Mock for testing the dist filter's get_scenario method
        self.dist_filter_get_scenarios = DistributedScenarioFilter(self.cfg_mock, self.worker_mock, 0, 2, "path")
        self.dist_filter_get_scenarios._get_log_db_files_for_single_node = MagicMock()
        self.dist_filter_get_scenarios._get_scenarios_from_list_of_log_files = MagicMock()
        self.dist_filter_get_scenarios._get_repartition_tokens = MagicMock()
        self.dist_filter_get_scenarios._get_repartition_tokens.return_value = ["a"], ["1.log"]

    def test_get_scenarios_scenario_based(self) -> None:
        """
        Test that get_scenarios does full repartitioning in this case
        """
        with unittest.mock.patch(
            "nuplan.common.utils.distributed_scenario_filter.build_scenario_builder", self.build_scenario_builder_mock
        ), unittest.mock.patch(
            "nuplan.common.utils.distributed_scenario_filter.build_scenario_filter", self.build_scenario_filter_mock
        ), unittest.mock.patch(
            "nuplan.common.utils.distributed_scenario_filter.OmegaConf.set_struct"
        ):

            self.dist_filter_get_scenarios._distributed_mode = DistributedMode.SCENARIO_BASED
            scenarios = self.dist_filter_get_scenarios.get_scenarios()
            self.assertEqual(self.mock_scenarios, scenarios)

            # on single node we shouldn't call any of the splitting logic
            self.dist_filter_get_scenarios._get_log_db_files_for_single_node.assert_called()
            self.dist_filter_get_scenarios._get_scenarios_from_list_of_log_files.assert_called()
            self.dist_filter_get_scenarios._get_repartition_tokens.assert_called()

            # Confirm that in this case we have updated the cfg
            self.assertListEqual(self.cfg_mock.scenario_filter.scenario_tokens, ["a"])
            self.assertListEqual(self.cfg_mock.scenario_builder.db_files, ["1.log"])

            # Confirm that we call get scenarios with the config we started with
            self.build_scenario_builder_mock.assert_called_with(cfg=self.cfg_mock)
            self.build_scenario_filter_mock.assert_called_with(cfg=self.cfg_mock.scenario_filter)

    def test_get_scenarios_multiple_nodes_log_file_mode(self) -> None:
        """
        Test that get_scenarios we only call the methods that get a chunk of log files + gets the scenarios from that chunk
        """
        with unittest.mock.patch(
            "nuplan.common.utils.distributed_scenario_filter.build_scenario_builder", self.build_scenario_builder_mock
        ), unittest.mock.patch(
            "nuplan.common.utils.distributed_scenario_filter.build_scenario_filter", self.build_scenario_filter_mock
        ):
            self.dist_filter_get_scenarios._distributed_mode = DistributedMode.LOG_FILE_BASED

            # In this mode we should just return the scenarios we get from the log file
            mock_scenarios = [MagicMock()]
            self.dist_filter_get_scenarios._get_scenarios_from_list_of_log_files.return_value = mock_scenarios
            scenarios = self.dist_filter_get_scenarios.get_scenarios()
            self.assertEqual(mock_scenarios, scenarios)

            # on single node we shouldn't call any of the splitting logic
            self.dist_filter_get_scenarios._get_log_db_files_for_single_node.assert_called()
            self.dist_filter_get_scenarios._get_scenarios_from_list_of_log_files.assert_called()
            self.dist_filter_get_scenarios._get_repartition_tokens.assert_not_called()

            # Confirm that we don't call scenario builder in this mode (handled inside of the other methods in this case)
            self.build_scenario_builder_mock.assert_not_called()
            self.build_scenario_filter_mock.assert_not_called()

    def test_get_scenarios_single_node(self) -> None:
        """
        Test that get_scenarios just returns the scenarios built by the scenario builder in this case.
        """
        with unittest.mock.patch(
            "nuplan.common.utils.distributed_scenario_filter.build_scenario_builder", self.build_scenario_builder_mock
        ), unittest.mock.patch(
            "nuplan.common.utils.distributed_scenario_filter.build_scenario_filter", self.build_scenario_filter_mock
        ):

            self.dist_filter_get_scenarios._distributed_mode = DistributedMode.SINGLE_NODE
            scenarios = self.dist_filter_get_scenarios.get_scenarios()
            self.assertEqual(self.mock_scenarios, scenarios)

            # on single node we shouldn't call any of the splitting logic
            self.dist_filter_get_scenarios._get_log_db_files_for_single_node.assert_not_called()
            self.dist_filter_get_scenarios._get_scenarios_from_list_of_log_files.assert_not_called()
            self.dist_filter_get_scenarios._get_repartition_tokens.assert_not_called()

            # Confirm that we call get scenarios with the config we started with
            self.build_scenario_builder_mock.assert_called_with(cfg=self.cfg_mock)
            self.build_scenario_filter_mock.assert_called_with(cfg=self.cfg_mock.scenario_filter)

    def test_get_repartition_tokens(self) -> None:
        """
        Test that we make all of the expected calls, in the expected order, to repartition the tokens.
        """
        with unittest.mock.patch(
            "nuplan.common.utils.distributed_scenario_filter.get_unique_job_id"
        ) as id, unittest.mock.patch("nuplan.common.utils.distributed_scenario_filter.distributed_sync") as dist:

            # Setup mocks of all the functions we call (that are themselves tested separately
            dist_filter = DistributedScenarioFilter(self.cfg_mock, self.worker_mock, 0, 1, "path", timeout_seconds=5)
            dist_filter._write_token_csv_file = MagicMock()
            dist_filter._get_all_generated_csv = MagicMock()
            dist_filter._get_token_and_log_chunk_on_single_node = MagicMock()

            # Setup return values for mocks
            id.return_value = "1"
            dist_filter._get_all_generated_csv.return_value = [("a", "1"), ("b", "2")]
            dist_filter._get_token_and_log_chunk_on_single_node.return_value = ["a", "b"], ["path/1.db", "path/2.db"]

            # We create a manager mock to verify the order of the calls
            # order is (1) write csv, (2) sync, (3) get_all_csvs, (4) chunk the token/log file set
            manager = Mock()
            manager.attach_mock(dist_filter._write_token_csv_file, "write_csv")
            manager.attach_mock(dist, "sync")
            manager.attach_mock(dist_filter._get_all_generated_csv, "get_csvs")
            manager.attach_mock(dist_filter._get_token_and_log_chunk_on_single_node, "chunk")

            # call the function
            output = dist_filter._get_repartition_tokens(scenarios=self.mock_scenarios)
            self.assertEqual(output, (["a", "b"], ["path/1.db", "path/2.db"]))

            # check that we called dist_sync with the correct param
            expected_calls = [
                call.write_csv(self.mock_scenarios, Path("path/tokens/1")),  # write our csv
                call.sync(Path("path/barrier/1"), timeout_seconds=5),  # then sync
                call.get_csvs(Path("path/tokens/1")),  # then get all of the csv for all machines
                call.chunk([("a", "1"), ("b", "2")], Path(".")),  # then get our share of those tokens
            ]
            self.assertListEqual(manager.mock_calls, expected_calls)

    def test_get_all_generated_csv_s3(self) -> None:
        """
        Test that we get all of the tokens from the csv files we have created when running in mocked s3.
        """
        with unittest.mock.patch(
            "nuplan.common.utils.distributed_scenario_filter.expand_s3_dir"
        ) as expand, unittest.mock.patch(
            "nuplan.common.utils.distributed_scenario_filter.split_s3_path"
        ) as split, unittest.mock.patch(
            "nuplan.common.utils.distributed_scenario_filter.S3Store"
        ) as store:

            with tempfile.TemporaryDirectory() as tmp_dir_str:
                dist_filter = DistributedScenarioFilter(self.cfg_mock, self.worker_mock, 0, 1, "s3://dummy/path")
                dist_filter._write_token_csv_file(self.mock_scenarios, tmp_dir_str)
                split.return_value = "bucket", "file"
                expand.return_value = [os.path.join(tmp_dir_str, "0.csv")]

                def mock_get(path: str) -> IO[str]:
                    """
                    Mock get for the s3 store we mock, just opens the file as a local file.
                    """
                    return open(path)

                store.return_value = MagicMock()
                store.return_value.get = mock_get

                filter_output = dist_filter._get_all_generated_csv("s3://dummy/path")
                self.assertEqual(filter_output, [["a", "1.log"], ["b", "2.log"]])

    def test_get_all_generated_csv_local(self) -> None:
        """
        Test that we get all of the tokens from the csv files we have created when running locally.
        """
        with tempfile.TemporaryDirectory() as tmp_dir_str:
            # Simulate an example with two workers running
            dist_filter = DistributedScenarioFilter(self.cfg_mock, self.worker_mock, 0, 2, tmp_dir_str)
            dist_filter_2 = DistributedScenarioFilter(self.cfg_mock, self.worker_mock, 1, 2, tmp_dir_str)
            dist_filter._write_token_csv_file(self.mock_scenarios[:1], tmp_dir_str)
            dist_filter_2._write_token_csv_file(self.mock_scenarios[1:], tmp_dir_str)

            # At this stage, the filters should both see all of the tokens, so these should be equal
            filter_1_output = dist_filter._get_all_generated_csv(tmp_dir_str)
            filter_2_output = dist_filter_2._get_all_generated_csv(tmp_dir_str)
            self.assertListEqual(filter_1_output, filter_2_output)

            # Check that we get the expected tokens
            expected_token_set = {("a", "1.log"), ("b", "2.log")}
            self.assertEqual(
                len(filter_1_output), len(expected_token_set)
            )  # Make sure that the tokens in the filter output are unique
            self.assertSetEqual({tuple(i) for i in filter_1_output}, expected_token_set)

    def test_get_token_and_log_chunk_on_single_node(self) -> None:
        """
        Test that we correctly chunk the tokens and associated log names on each node.
        """
        with unittest.mock.patch("nuplan.common.utils.distributed_scenario_filter.check_s3_path_exists"):
            db_files_path = Path("s3://dummy/path")
            token_distribution = [("a", "1"), ("b", "1"), ("c", "2"), ("d", "2")]

            # If we only have one node, we expect that we get all of the tokens and logs
            dist_filter = DistributedScenarioFilter(self.cfg_mock, self.worker_mock, 0, 1, "")
            tokens, log_files = dist_filter._get_token_and_log_chunk_on_single_node(token_distribution, db_files_path)
            self.assertSetEqual(set(tokens), {"a", "b", "c", "d"})
            self.assertSetEqual(set(log_files), {"s3://dummy/path/1.db", "s3://dummy/path/2.db"})

            # If we have two nodes, we expect that we get the first half of the tokens
            dist_filter = DistributedScenarioFilter(self.cfg_mock, self.worker_mock, 0, 2, "")
            tokens, log_files = dist_filter._get_token_and_log_chunk_on_single_node(token_distribution, db_files_path)
            self.assertSetEqual(set(tokens), {"a", "b"})
            self.assertSetEqual(set(log_files), {"s3://dummy/path/1.db"})

    def test_write_token_csv_file(self) -> None:
        """
        Test that we correctly write out a csv file for the current node for the list of scenarios provided
        """
        dist_filter = DistributedScenarioFilter(self.cfg_mock, self.worker_mock, 0, 1, "")
        with tempfile.TemporaryDirectory() as tmp_dir_str:
            dist_filter._write_token_csv_file(self.mock_scenarios, tmp_dir_str)

            # expected_path is directory/node_rank.csv
            expected_path = os.path.join(tmp_dir_str, "0.csv")
            self.assertTrue(os.path.exists(expected_path))
            csv_out = pd.read_csv(expected_path).to_dict()

            # output should have two columns, one for token and one for log name.
            # The values should match what we mocked above
            self.assertEqual(csv_out, {'0': {0: 'a', 1: 'b'}, '1': {0: '1.log', 1: '2.log'}})

    def test_get_scenarios_from_list_of_log_files(self) -> None:
        """
        Test that we build a scenario builder with the proper db files updated, and successfully get scenarios from it
        """
        dist_filter = DistributedScenarioFilter(self.cfg_mock, self.worker_mock, 0, 1, "")
        with unittest.mock.patch(
            "nuplan.common.utils.distributed_scenario_filter.build_scenario_builder", self.build_scenario_builder_mock
        ), unittest.mock.patch(
            "nuplan.common.utils.distributed_scenario_filter.build_scenario_filter", self.build_scenario_filter_mock
        ):
            scenarios = dist_filter._get_scenarios_from_list_of_log_files(["file_3"])

            # config should be updated with the new list of dbs
            self.assertListEqual(self.cfg_mock.scenario_builder.db_files, ["file_3"])

            # We should build the scenario builder with the updated config and get scenarios from it
            self.build_scenario_filter_mock.assert_called_with(self.cfg_mock.scenario_filter)
            self.build_scenario_builder_mock.assert_called_with(self.cfg_mock)
            self.scenario_builder_mock.get_scenarios.assert_called_with(self.scenario_filter_mock, self.worker_mock)

            # the scenarios we should get should be the ones we mocked to the "get_scenarios" function
            self.assertEqual(scenarios, self.mock_scenarios)

    def test_get_log_db_files_for_single_node_non_distributed(self) -> None:
        """
        Test that in a non-distributed context we simply return all the db files in the config
        """
        dist_filter = DistributedScenarioFilter(self.cfg_mock, self.worker_mock, 0, 1, "")
        logs = dist_filter._get_log_db_files_for_single_node()
        self.assertListEqual(logs, self.mock_dbs)

    def test_get_log_db_files_for_single_node_distributed(self) -> None:
        """
        Test that in a distributed context we call the proper functions and chunk the data as expected
        """
        with unittest.mock.patch(
            "nuplan.common.utils.distributed_scenario_filter.get_db_filenames_from_load_path"
        ) as get, unittest.mock.patch("nuplan.common.utils.distributed_scenario_filter.check_s3_path_exists") as check:
            get.side_effect = lambda x: x
            check.return_value = True
            dist_filter = DistributedScenarioFilter(self.cfg_mock, self.worker_mock, 0, 2, "")
            logs = dist_filter._get_log_db_files_for_single_node()

            # We have two files and two nodes so we expect that our node of rank 0 should get the first file
            self.assertListEqual(logs, self.mock_dbs[:1])


if __name__ == "__main__":
    unittest.main()
