import logging
import os

logger = logging.getLogger(__name__)

DEFAULT_DATA_ROOT = os.path.expanduser("~/nuplan/dataset")
DEFAULT_EXP_ROOT = os.path.expanduser("~/nuplan/exp")


def set_default_path() -> None:
    """
    This function sets the default paths as environment variables if none are set.
    These can then be used by Hydra, unless the user overwrites them from the command line.
    """
    if "NUPLAN_DATA_ROOT" not in os.environ:
        logger.info(f'Setting default NUPLAN_DATA_ROOT: {DEFAULT_DATA_ROOT}')
        os.environ["NUPLAN_DATA_ROOT"] = DEFAULT_DATA_ROOT

    if "NUPLAN_EXP_ROOT" not in os.environ:
        logger.info(f'Setting default NUPLAN_EXP_ROOT: {DEFAULT_EXP_ROOT}')
        os.environ["NUPLAN_EXP_ROOT"] = DEFAULT_EXP_ROOT
