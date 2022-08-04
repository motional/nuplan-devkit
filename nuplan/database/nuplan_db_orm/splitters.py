import abc
import random
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Optional, Set, Union

from nuplan.database.common.db import DBSplitterInterface
from nuplan.database.nuplan_db_orm.log import Log
from nuplan.database.nuplan_db_orm.nuplandb import NuPlanDB

Sample = Any  # TODO: replace with lidar_pc


def _get_logs(db: NuPlanDB, split2log: Dict[str, List[str]], split_name: str) -> List[Log]:
    """
    For all the given split `split_name`, convert its corresponding log names into Log objects.
    :param db: NuPlanDB.
    :param split2log: Mapping from a split name to its corresponding data. The data is given as a list of log names
        (example of log name: '2021.07.16.20.45.29_veh-35_01095_01486').
    :param split_name: The split in which we want to get the Log objects. (example of split_name: "val").
    :return: List of logs.
    """
    logs = []
    for log_name in split2log[split_name]:
        log = db.log.select_one(logfile=log_name)
        if log is not None:
            logs.append(log)
    return logs


def _get_samples_from_logs(logs: List[Log], broken_extractions: Set[str]) -> List[Sample]:
    """
    Returns all the non-broken samples of a list of logs.
    For definitions of 'sample' and 'extraction', please take a look at README.md.

    :param logs: List of logs from which to extract samples.
    :param broken_extractions: List of extractions whose samples should be excluded from the output of this function.
    :return: List of non-broken samples associated with the given logs.
    """
    samples = []
    for log in logs:
        # For each non-broken extraction in a log, add all the samples of that extraction.
        for extraction in log.extractions:
            if extraction.token in broken_extractions:
                continue
            for sample in extraction.samples:
                samples.append(sample)
    return samples


def _set_splits_samples(
    split2samples: DefaultDict[str, List[Sample]],
    db: NuPlanDB,
    split2log: Dict[str, List[str]],
    broken_extractions: Optional[Set[str]] = None,
    sort_train: bool = True,
) -> None:
    """
    Populates split2samples with all the main splits and the ones defined in split2log, converting log names into
    the list of non-broken samples they contain in the database.

    :param split2samples: Main dictionary containing a mapping from split name to its corresponding data.
        The data is given as a list of samples.
    :param db: NuPlanDB.
    :param split2log: Mapping from a split name to its corresponding data. The data is given as a list of log names
        (example of log name: '2019.07.03.03.29.58_mvp-veh-8')
    :param broken_extractions: List of extractions whose samples should be excluded from the samples populated in
        `split2samples`.
    :param sort_train: Whether or not to sort the train split with respect to sample tokens. (This is useful
        to guarantee that randomly subsampled splits from train will not differ if they have the same random seed.)
    """
    broken_extractions = broken_extractions or set()

    for split_name in split2log.keys():
        logs = _get_logs(db, split2log, split_name)
        split2samples[split_name] = _get_samples_from_logs(logs, broken_extractions=broken_extractions)

    if "val" in split2samples or "test" in split2samples:
        split2samples['valtest'] = split2samples['val'] + split2samples['test']

    if "train" not in split2samples:
        # If a custom 'train' split is provided, keep it. If not then use all the remaining data to create it.
        split2samples['train'] = [rec for rec in split2samples['all'] if rec not in split2samples['valtest']]

    if sort_train:
        split2samples['train'].sort(key=lambda sample: str(sample.token))


def _get_all_samples(
    db: NuPlanDB,
    vehicle_type: str,
    broken_extractions: Optional[Set[str]] = None,
    excluded_drive_log_tags: Optional[Set[str]] = None,
) -> List[Sample]:
    """
    Returns all non-broken samples associated with one vehicle.

    :param db: NuPlanDB.
    :param vehicle_type: name of the vehicle which we should get data from.
    :param broken_extractions: List of extractions whose samples should be excluded from the output of this function.
    :param excluded_drive_log_tags: Logs that have ANY of those drive log tags will be excluded.
    :return: List of non-broken samples associated with given vehicle_type.
    """
    broken_extractions = broken_extractions or set()
    excluded_drive_log_tags = excluded_drive_log_tags or set()
    logs = db.log.select_many(vehicle_type=vehicle_type)
    # Logs that have ANY of those drive log tags will be excluded.
    logs = [log for log in logs if not set(log.drive_log_tags).intersection(excluded_drive_log_tags)]
    return _get_samples_from_logs(logs, broken_extractions)


def _set_location_splits(
    split2samples: DefaultDict[str, List[Sample]], db: NuPlanDB, core_splits_names: List[str]
) -> None:
    """
    Populates split2samples with splits by region and country done on top of core splits.

    For example:
        "train" -> "train", "train.United_States", "train.bs", , "train.lv", "train.ptc", "train.Singapore", "train.sg".

    :param split2samples: Main dictionary containing a mapping from split name to its corresponding data. The data is
     given as a list of samples. This function assumes core splits already exists (e.g. "train", "val, "test").
    :param db: NuPlanDB.
    :param core_splits_names: Split names on which the location splits should be done. (e.g. "train", "val, "test").
    """
    for split_name in core_splits_names:
        region_accumulator = []
        for region in db.regions:
            region_split_name = split_name + '.' + region
            country_split_name = split_name + '.' + db.country(region)

            split2samples[region_split_name] = [
                rec for rec in split2samples[split_name] if rec.extraction.log.location in db.locations(region)
            ]
            if country_split_name not in split2samples:
                split2samples[country_split_name] = []
            split2samples[country_split_name] += split2samples[region_split_name]

            region_accumulator += split2samples[region_split_name]

        # Assert that regions covered all samples.
        assert set(region_accumulator) == set(split2samples[split_name])


def _set_subsampled_splits(
    split2samples: DefaultDict[str, List[Sample]],
    db: NuPlanDB,
    core_splits_names: List[str],
    random_seed: Union[str, int],
    n_samples_per_region: int,
    split_suffix: str,
) -> None:
    """
    Populates split2samples with core splits.
    :param split2samples: Main dictionary containing a mapping from split name to its corresponding data. The data is
     given as a list of samples.
    :param db: NuPlanDB.
    :param core_splits_names: Name of the core splits, such as, ['test', 'val', 'valtest', 'train'].
    :param random_seed: Random seed to use for picking tokens.
    :param n_samples_per_region: number of samples for each region.
    :param split_suffix: suffix of the split name, such as 'mini', 'dev'.
    """
    st0 = random.getstate()  # Store previous random state.
    random.seed(random_seed)
    for split_name in core_splits_names:
        for region in db.regions:
            temp = split2samples[split_name + '.' + region].copy()
            random.shuffle(temp)
            split2samples[split_name + '.' + split_suffix] += temp[:n_samples_per_region]
    random.setstate(st0)  # Restore previous random state.


def _set_mini_splits(split2samples: DefaultDict[str, List[Sample]], db: NuPlanDB, core_splits_names: List[str]) -> None:
    """
    Populates split2samples with mini splits done on top of core splits.

    For example:
        "train" -> "train", "train.mini"

    :param split2samples: Main dictionary containing a mapping from split name to its corresponding data. The data is
     given as a list of samples. This function assumes the existence the following splits:
      - core splits (e.g. "train", "val, "test").
      - location splits (e.g. "train.bs", "val.United_States").
    :param db: NuPlanDB.
    :param core_splits_names: Name of the core splits to be considered.
    """
    return _set_subsampled_splits(
        split2samples, db, core_splits_names, random_seed="42", n_samples_per_region=100, split_suffix="mini"
    )


def _set_dev_splits(split2samples: DefaultDict[str, List[Sample]], db: NuPlanDB, core_splits_names: List[str]) -> None:
    """
    Populates split2samples with smaller evaluation splits done on top of core splits, to use in dev. experiments.
    For example:
        "train" -> "train", "train.dev"

    :param split2samples: Main dictionary containing a mapping from split name to its corresponding data. The data is
     given as a list of samples. This function assumes the existence the following splits:
      - core splits (e.g. "train", "val, "test")
      - location splits (e.g. "train.bs", "val.United_States").
    :param db: NuPlanDB.
    :param core_splits_names: Name of the core splits to be considered.
    """
    return _set_subsampled_splits(
        split2samples, db, core_splits_names, random_seed="42", n_samples_per_region=250, split_suffix="dev"
    )


def _convert_to_tokens(split2samples: DefaultDict[str, List[Sample]]) -> DefaultDict[str, List[str]]:
    """
    Convert all Samples provided into their corresponding tokens.

    :param split2samples: Main dictionary containing a mapping from split name to its corresponding data. The data is
     given as a list of samples.
    :return: Same mapping as split2samples except the data is in tokens format.
    """
    split2tokens = defaultdict(list)
    for split in split2samples:
        split2tokens[split] = [record.token for record in split2samples[split]]
    return split2tokens


class BaseNuPlanDBSplitter(DBSplitterInterface):
    """Base class for all NuPlanDB splitters."""

    def __init__(self, db: NuPlanDB):
        """
        :param db: NuPlanDB instance.
        """
        self._db = db

        # Update the DB manual reference count to prevent tables from being detached
        #   while the splitter is using it.
        self._db.add_ref()

    def __del__(self) -> None:
        """
        Called when the splitter is being destroyed.
        """
        # Splitter is no longer using the DB. Allow the tables to be detached and GC'd
        #   no other classes are using it.
        self._db.remove_ref()

    def __repr__(self) -> str:
        """
        Get the string representation.
        :return: The string representation.
        """
        return "{}(NuPlanDB('{}'))".format(self.__class__.__name__, self._db.name)

    def list(self) -> List[str]:
        """
        Get the list of the splits.
        :return: The list of splits.
        """
        return list(self._splits.keys())

    def split(self, split_name: str) -> List[str]:
        """
        Get list of tokens for the split.
        :return: The list of tokens for the split.
        """
        return sorted(self._splits[split_name])

    def logs(self, split_name: str) -> List[str]:
        """
        Get list of logs for the split.
        :return: The list of logs for the split.
        """
        sample_tokens = self.split(split_name)
        return list({self._db.sample[token].extraction.log.logfile for token in sample_tokens})

    @property
    @abc.abstractmethod
    def _splits(self) -> DefaultDict[str, List[str]]:
        """
        Returns a dictionary that maps from split name to list of NuPlanDB tokens.
        :return: A dictionary that maps from split name to list of NuPlanDB tokens.
        """
        pass
