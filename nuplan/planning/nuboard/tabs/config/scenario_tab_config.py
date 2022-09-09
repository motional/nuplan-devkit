from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List


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
