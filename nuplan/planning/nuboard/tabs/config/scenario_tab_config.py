from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Tuple


@dataclass(frozen=True)
class ScenarioTabTitleDivConfig:
    """Config for the scenario tab title div tag."""

    text: ClassVar[str] = "-"
    name: ClassVar[str] = 'scenario_title_div'
    css_classes: ClassVar[List[str]] = ['scenario-tab-title-div']

    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """Get configs as a dict."""
        return {'text': cls.text, 'name': cls.name, 'css_classes': cls.css_classes}


@dataclass(frozen=True)
class ScenarioTabScenarioTokenMultiChoiceConfig:
    """Config for scenario tab scenario token multi choice tag."""

    max_items: ClassVar[int] = 1
    option_limit: ClassVar[int] = 10
    height: ClassVar[int] = 40
    placeholder: ClassVar[str] = "Scenario token"
    name: ClassVar[str] = 'scenario_token_multi_choice'
    css_classes: ClassVar[List[str]] = ['scenario-token-multi-choice']

    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """Get configs as a dict."""
        return {
            'max_items': cls.max_items,
            'option_limit': cls.option_limit,
            'height': cls.height,
            'placeholder': cls.placeholder,
            'name': cls.name,
            'css_classes': cls.css_classes,
        }


@dataclass(frozen=True)
class ScenarioTabModalQueryButtonConfig:
    """Config for scenario tab modal query button tag."""

    name: ClassVar[str] = 'scenario_modal_query_btn'
    label: ClassVar[str] = 'Query Scenario'
    css_classes: ClassVar[List[str]] = ['btn', 'btn-primary', 'modal-btn', 'scenario-tab-modal-query-btn']

    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """Get configs as a dict."""
        return {'name': cls.name, 'label': cls.label, 'css_classes': cls.css_classes}


@dataclass(frozen=True)
class ScenarioTabFrameButtonConfig:
    """Config for scenario tab's frame control buttons."""

    label: str
    margin: Tuple[int, int, int, int] = field(default_factory=lambda: (5, 19, 5, 35))  # Top, right, bottom, left
    css_classes: List[str] = field(default_factory=lambda: ["frame-control-button"])
    width: int = field(default_factory=lambda: 56)


# Global config instances
first_button_config = ScenarioTabFrameButtonConfig(label="first")
prev_button_config = ScenarioTabFrameButtonConfig(label="prev")
play_button_config = ScenarioTabFrameButtonConfig(label="play")
next_button_config = ScenarioTabFrameButtonConfig(label="next")
last_button_config = ScenarioTabFrameButtonConfig(label="last")
