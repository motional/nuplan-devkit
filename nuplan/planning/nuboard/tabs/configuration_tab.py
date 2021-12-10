import base64
import io
import logging
import pickle
from typing import List

from bokeh.document.document import Document
from bokeh.layouts import column
from bokeh.models import Button, Div, FileInput, Panel, Select
from nuplan.planning.nuboard.base.base_tab import BaseTab
from nuplan.planning.nuboard.base.data_class import NuBoardFile
from nuplan.planning.nuboard.style import configuration_tab_style

logger = logging.getLogger(__name__)


class ConfigurationTab:

    def __init__(self,
                 doc: Document,
                 file_paths: List[NuBoardFile],
                 tabs: List[BaseTab]):
        """
        Metric board to render metrics.
        :param file_paths: Path to a metric pickle file.
        :param tabs: A list of tabs to be updated when configuration is changed.
        """

        self._doc = doc
        self._file_paths = file_paths
        self._tabs = tabs

        # UI.
        main_board_title = Div(text="""<h3 style='margin-left: 18px;
        margin-top: 20px; width: 300px;'>Configuration</h3>""")

        folder_path_section_title = Div(text="""<h4 style='margin-left: 18px;
        margin-top: 20px; width: 300px;'>Add a nuboard file</h3>""")

        self._folder_path_input = FileInput(accept=NuBoardFile.extension(), css_classes=["folder-path-input"],
                                            margin=configuration_tab_style['folder_path_input_margin'])
        self._folder_path_input.on_change('value', self._add_experiment_file)

        self._folder_path_selection = Select(title="Experiment Paths:",
                                             margin=configuration_tab_style['folder_path_selection_margin'],
                                             css_classes=["experiment-path-select"])

        self._folder_path_selection.options = [file_path.main_path for file_path in self._file_paths]
        if len(self._file_paths) > 0:
            self._folder_path_selection.value = self._folder_path_selection.options[0]
        remove_button = Button(label="Remove the nuboard file", css_classes=["folder-path-remove-button"])
        remove_button.on_click(self._remove_button_on_click)
        remove_button_section_title = Div(text="""<h4 style='margin-left: 18px;
        margin-top: 20px; width: 300px;'>Remove a nuboard file</h3>""")

        # Make the width fit to the screen.
        main_board_layout = column([main_board_title, folder_path_section_title, self._folder_path_input,
                                    remove_button_section_title, self._folder_path_selection, remove_button],
                                   css_classes=["configuration"],
                                   height=configuration_tab_style['main_board_layout_height'], width_policy="max")

        self._panel = Panel(title="Configuration", child=main_board_layout)

    def _add_experiment_file(self, attr: str, old: bytes, new: bytes) -> None:
        """
        Event responds to file change.
        :param attr: Attribute name.
        :param old: Old value.
        :param new: New value.
        """

        if new == '':   # type: ignore
            return

        try:
            decoded_string = base64.b64decode(new)
            file_stream = io.BytesIO(decoded_string)
            data = pickle.load(file_stream)
            nuboard_file = NuBoardFile.deserialize(data=data)
            if nuboard_file not in self._file_paths:
                self._file_paths.append(nuboard_file)
                self._folder_path_selection.options = []
                self._folder_path_selection.options = [file_path.main_path for file_path in self._file_paths]
                if len(self._folder_path_selection.value) == 0:
                    self._folder_path_selection.value = ''
                    self._folder_path_selection.value = self._folder_path_selection.options[0]
                self._file_paths_on_change()
            file_stream.close()
        except (OSError, IOError) as e:
            logger.info(f"Error loading experiment file. {str(e)}.")

    def _remove_button_on_click(self) -> None:
        """ Respond to remove button an on_click event. """

        if len(self._folder_path_selection.value) == 0:
            return

        index = self._folder_path_selection.options.index(self._folder_path_selection.value)
        self._file_paths.pop(index)
        self._folder_path_selection.options = []
        self._folder_path_selection.options = [file_path.main_path for file_path in self._file_paths]
        self._folder_path_selection.value = ''
        if len(self._folder_path_selection.options) > 0:
            self._folder_path_selection.value = self._folder_path_selection.options[0]
        else:
            self._folder_path_selection.value = ''
        self._file_paths_on_change()

    def _file_paths_on_change(self) -> None:
        for tab in self._tabs:
            tab.file_paths_on_change(file_paths=self._file_paths)

    @property
    def panel(self) -> Panel:
        """ Retrieve a panel.
        :return A panel.
        """

        return self._panel
