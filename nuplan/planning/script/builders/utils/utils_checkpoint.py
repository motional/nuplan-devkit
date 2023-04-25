import pathlib
from datetime import datetime
from typing import Optional


def find_last_checkpoint_in_dir(group_dir: pathlib.Path, experiment_uid: pathlib.Path) -> Optional[pathlib.Path]:
    """
    Extract last checkpoint from a experiment
    :param group_dir: defined by ${group}/${experiment_name}/${job_name} from hydra
    :param experiment_uid: date time which will be used as ${group}/${experiment_name}/${job_name}/${experiment_uid}
    return checkpoint dir if existent, otherwise None
    """
    last_checkpoint_dir = group_dir / experiment_uid / 'checkpoints'

    if not last_checkpoint_dir.exists():
        return None

    checkpoints = list(last_checkpoint_dir.iterdir())
    last_epoch = max(int(path.stem[6:]) for path in checkpoints if path.stem.startswith('epoch'))
    return last_checkpoint_dir / f'epoch={last_epoch}.ckpt'


def extract_last_checkpoint_from_experiment(output_dir: pathlib.Path, date_format: str) -> Optional[pathlib.Path]:
    """
    Extract last checkpoint from latest experiment
    :param output_dir: of the current experiment, we assume that parent folder has previous experiments of the same type
    :param date_format: format time used for folders
    :return path to latest checkpoint, return None in case no checkpoint was found
    """
    date_times = [datetime.strptime(dir.name, date_format) for dir in output_dir.parent.iterdir() if dir != output_dir]
    date_times.sort(reverse=True)

    for date_time in date_times:
        checkpoint = find_last_checkpoint_in_dir(output_dir.parent, pathlib.Path(date_time.strftime(date_format)))
        if checkpoint:
            return checkpoint
    return None
