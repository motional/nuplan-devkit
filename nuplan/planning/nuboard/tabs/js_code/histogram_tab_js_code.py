from dataclasses import dataclass

from bokeh.models.callbacks import CustomJS


@dataclass(frozen=True)
class HistogramTabLoadingJSCode:
    """JS when loading in the histogram tab."""

    @classmethod
    def get_js_code(cls) -> CustomJS:
        """Get js code."""
        return CustomJS(
            args={},
            code="""
                cb_obj.tags = [window.outerWidth, window.outerHeight];
                document.getElementById('histogram-loading').style.visibility = 'visible';
                document.getElementById('histogram-plot-section').style.visibility = 'hidden';
                document.getElementById('histogram-setting-form').style.display = 'none';
            """,
        )


@dataclass(frozen=True)
class HistogramTabUpdateWindowsSizeJSCode:
    """JS when updating window size in the histogram tab."""

    @classmethod
    def get_js_code(cls) -> CustomJS:
        """Get js code."""
        return CustomJS(
            args={},
            code="""
                console.log(cb_obj.tags);
                cb_obj.tags = [window.outerWidth, window.outerHeight];
            """,
        )


@dataclass(frozen=True)
class HistogramTabLoadingEndJSCode:
    """JS when loading simulation is done in the histogram tab."""

    @classmethod
    def get_js_code(cls) -> CustomJS:
        """Get js code."""
        return CustomJS(
            args={},
            code="""
                document.getElementById('histogram-loading').style.visibility = 'hidden';
                document.getElementById('histogram-plot-section').style.visibility = 'visible';
                document.getElementById('overlay').style.display = 'none';
            """,
        )
