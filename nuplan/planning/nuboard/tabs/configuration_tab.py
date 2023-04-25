import base64
import io
import logging
import pathlib
import pickle
from pathlib import Path
from typing import Any, List

from bokeh.document.document import Document
from bokeh.models import CheckboxGroup, FileInput

from nuplan.planning.nuboard.base.base_tab import BaseTab
from nuplan.planning.nuboard.base.data_class import NuBoardFile
from nuplan.planning.nuboard.base.experiment_file_data import ExperimentFileData
from nuplan.planning.nuboard.style import configuration_tab_style

logger = logging.getLogger(__name__)


class ConfigurationTab:
    """Configuration tab for nuboard."""

    def __init__(self, doc: Document, experiment_file_data: ExperimentFileData, tabs: List[BaseTab]):
        """
        Configuration tab about configurating nuboard.
        :param experiment_file_data: Experiment file data.
        :param tabs: A list of tabs to be updated when configuration is changed.
        """
        self._doc = doc
        self._tabs = tabs
        self.experiment_file_data = experiment_file_data

        self._file_path_input = FileInput(
            accept=NuBoardFile.extension(),
            css_classes=["file-path-input"],
            margin=configuration_tab_style["file_path_input_margin"],
            name="file_path_input",
        )
        self._file_path_input.on_change("value", self._add_experiment_file)
        self._experiment_file_path_checkbox_group = CheckboxGroup(
            labels=self.experiment_file_path_stems,
            active=[index for index in range(len(self.experiment_file_data.file_paths))],
            name="experiment_file_path_checkbox_group",
            css_classes=["experiment-file-path-checkbox-group"],
        )
        self._experiment_file_path_checkbox_group.on_click(self._click_experiment_file_path_checkbox)
        if self.experiment_file_data.file_paths:
            self._file_paths_on_change()

    @property
    def experiment_file_path_stems(self) -> List[str]:
        """Return a list of file path stems."""
        experiment_paths = []
        for file_path in self.experiment_file_data.file_paths:
            metric_path = file_path.current_path / file_path.metric_folder
            if metric_path.exists():
                experiment_file_path_stem = file_path.current_path
            else:
                experiment_file_path_stem = file_path.metric_main_path

            if isinstance(experiment_file_path_stem, str):
                experiment_file_path_stem = pathlib.Path(experiment_file_path_stem)

            experiment_file_path_stem = "/".join(
                [experiment_file_path_stem.parts[-2], experiment_file_path_stem.parts[-1]]
            )
            experiment_paths.append(experiment_file_path_stem)
        return experiment_paths

    @property
    def file_path_input(self) -> FileInput:
        """Return the file path input widget."""
        return self._file_path_input

    @property
    def experiment_file_path_checkbox_group(self) -> CheckboxGroup:
        """Return experiment file path checkboxgroup."""
        return self._experiment_file_path_checkbox_group

    def _click_experiment_file_path_checkbox(self, attr: Any) -> None:
        """
        Click event handler for experiment_file_path_checkbox_group.
        :param attr: Clicked attributes.
        """
        self._file_paths_on_change()

    def add_nuboard_file_to_experiments(self, nuboard_file: NuBoardFile) -> None:
        """
        Add nuboard files to experiments.
        :param nuboard_file: Added nuboard file.
        """
        nuboard_file.current_path = Path(nuboard_file.metric_main_path)
        if nuboard_file not in self.experiment_file_data.file_paths:
            self.experiment_file_data.update_data(file_paths=[nuboard_file])
            self._experiment_file_path_checkbox_group.labels = self.experiment_file_path_stems
            self._experiment_file_path_checkbox_group.active += [len(self.experiment_file_path_stems) - 1]
            self._file_paths_on_change()

    def _add_experiment_file(self, attr: str, old: bytes, new: bytes) -> None:
        """
        Event responds to file change.
        :param attr: Attribute name.
        :param old: Old value.
        :param new: New value.
        """
        if not new:
            return
        try:
            decoded_string = base64.b64decode(new)
            file_stream = io.BytesIO(decoded_string)
            data = pickle.load(file_stream)
            nuboard_file = NuBoardFile.deserialize(data=data)
            self.add_nuboard_file_to_experiments(nuboard_file=nuboard_file)
            file_stream.close()
        except (OSError, IOError) as e:
            logger.info(f"Error loading experiment file. {str(e)}.")

    def _file_paths_on_change(self) -> None:
        """Function to call when we change file paths."""
        for tab in self._tabs:
            tab.file_paths_on_change(
                experiment_file_data=self.experiment_file_data,
                experiment_file_active_index=self._experiment_file_path_checkbox_group.active,
            )
