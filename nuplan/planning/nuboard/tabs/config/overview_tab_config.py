from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Optional

OVERVIEW_PLANNER_CHECKBOX_GROUP_NAME = 'overview_planner_checkbox_group'


@dataclass
class OverviewAggregatorData:
    """Aggregator metric data in the overview tab."""

    aggregator_file_name: str  # Aggregator output file name
    aggregator_type: str  # Aggregator type
    planner_name: str  # Planner name
    scenario_type: str  # Scenario type
    num_scenarios: int  # Number of scenarios in the type
    score: float  # The aggregator scores for the scenario type


@dataclass(frozen=True)
class OverviewTabDefaultDataSourceDictConfig:
    """Config for the overview tab default data source tag."""

    experiment: ClassVar[List[str]] = ['-']
    scenario_type: ClassVar[List[str]] = ['-']
    planner: ClassVar[List[str]] = [
        'No metric aggregator results, please add more experiments ' 'or adjust the search filter'
    ]

    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """Get configs as a dict."""
        return {'experiment': cls.experiment, 'scenario_type': cls.scenario_type, 'planner': cls.planner}


@dataclass(frozen=True)
class OverviewTabExperimentTableColumnConfig:
    """Config for the overview tab experiment table column tag."""

    field: ClassVar[str] = 'experiment'
    title: ClassVar[str] = 'Experiment'
    width: ClassVar[int] = 150
    sortable: ClassVar[bool] = False

    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """Get configs as a dict."""
        return {'field': cls.field, 'title': cls.title, 'width': cls.width, 'sortable': cls.sortable}


@dataclass(frozen=True)
class OverviewTabScenarioTypeTableColumnConfig:
    """Config for the overview tab scenario type table column tag."""

    field: ClassVar[str] = 'scenario_type'
    title: ClassVar[str] = 'Scenario Type (Number of Scenarios)'
    width: ClassVar[int] = 200
    sortable: ClassVar[bool] = False

    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """Get configs as a dict."""
        return {'field': cls.field, 'title': cls.title, 'width': cls.width, 'sortable': cls.sortable}


@dataclass(frozen=True)
class OverviewTabPlannerTableColumnConfig:
    """Config for the overview tab planner table column tag."""

    field: ClassVar[str] = 'planner'
    title: ClassVar[str] = 'Evaluation Score'
    sortable: ClassVar[bool] = False

    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """Get configs as a dict."""
        return {'field': cls.field, 'title': cls.title, 'sortable': cls.sortable}


@dataclass(frozen=True)
class OverviewTabDataTableConfig:
    """Config for the overview tab planner data table tag."""

    selectable: ClassVar[bool] = True
    row_height: ClassVar[int] = 80
    index_position: ClassVar[Optional[int]] = None
    name: ClassVar[str] = 'overview_table'
    css_classes: ClassVar[List[str]] = ['overview-table']

    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """Get configs as a dict."""
        return {
            'selectable': cls.selectable,
            'row_height': cls.row_height,
            'index_position': cls.index_position,
            'name': cls.name,
            'css_classes': cls.css_classes,
        }
