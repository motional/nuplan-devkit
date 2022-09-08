from dataclasses import dataclass

from bokeh.models.callbacks import CustomJS


@dataclass(frozen=True)
class ScenarioTabLoadingJSCode:
    """JS when loading simulation in the scenario tab."""

    @classmethod
    def get_js_code(cls) -> CustomJS:
        """Get js code."""
        return CustomJS(
            args={},
            code="""
                cb_obj.tags = [window.outerWidth, window.outerHeight];
                document.getElementById('scenario-loading').style.visibility = 'visible';
                document.getElementById('scenario-plot-section').style.visibility = 'hidden';
                document.getElementById('scenario-setting-form').style.display = 'none';
            """,
        )


@dataclass(frozen=True)
class ScenarioTabUpdateWindowsSizeJSCode:
    """JS when updating window size in the scenario tab."""

    @classmethod
    def get_js_code(cls) -> CustomJS:
        """Get js code."""
        return CustomJS(
            args={},
            code="""
                cb_obj.tags = [window.outerWidth, window.outerHeight];
            """,
        )


@dataclass(frozen=True)
class ScenarioTabLoadingEndJSCode:
    """JS when loading simulation is done in the scenario tab."""

    @classmethod
    def get_js_code(cls) -> CustomJS:
        """Get js code."""
        return CustomJS(
            args={},
            code="""
                document.getElementById('scenario-loading').style.visibility = 'hidden';
                document.getElementById('scenario-plot-section').style.visibility = 'visible';
                document.getElementById('overlay').style.display = 'none';
            """,
        )
