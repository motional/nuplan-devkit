from dataclasses import dataclass

from bokeh.models import ColumnDataSource, CustomJS, TextInput


@dataclass(frozen=True)
class S3TabLoadingJSCode:
    """JS code when loading."""

    @classmethod
    def get_js_code(cls) -> CustomJS:
        """Get a custom js function."""
        return CustomJS(
            args={},
            code="""
                 document.getElementById('overlay').style.display = 'block';
                 document.getElementById('cloud-input-form').style.display = 'none';
                 document.getElementById('cloud-modal-loading').style.visibility = 'visible';
                 """,
        )


@dataclass(frozen=True)
class S3TabDataTableUpdateJSCode:
    """JS code when updated data table."""

    @classmethod
    def get_js_code(cls) -> CustomJS:
        """Get a custom js function."""
        return CustomJS(
            args={},
            code="""
                 document.getElementById('cloud-modal-loading').style.visibility = 'hidden';
                 document.getElementById('overlay').style.display = 'none';
                 """,
        )


@dataclass(frozen=True)
class S3TabDownloadUpdateJSCode:
    """JS code when downloading finishes."""

    @classmethod
    def get_js_code(cls) -> CustomJS:
        """Get a custom js function."""
        return CustomJS(
            args={},
            code="""
                 if (cb_obj.label == 'Download') {
                    document.getElementById('cloud-modal-loading').style.visibility = 'hidden';
                    document.getElementById('overlay').style.display = 'none';
                 }
                 """,
        )


@dataclass(frozen=True)
class S3TabContentDataSourceOnSelected:
    """JS code when any data table is selected by users."""

    @classmethod
    def get_js_code(cls, selected_column: TextInput, selected_row: TextInput) -> CustomJS:
        """
        Get a custom js function.
        :param selected_column: Selected column text input.
        :param selected_row: Selected row text input.
        """
        return CustomJS(
            args=dict(selected_column=selected_column, selected_row=selected_row),
            code="""
            var grid = document.getElementsByClassName('s3-data-table')[0].children[3].children[2].children[0].children;
            var row, column = '';
            for (var i = 0,max = grid.length; i < max; i++){
                if (grid[i].outerHTML.includes('active')){
                    row = i;
                    for (var j = 0, jmax = grid[i].children.length; j < jmax; j++)
                    {
                        if(grid[i].children[j].outerHTML.includes('active'))
                        {
                            column = j;
                        }
                    }
                }
            }
            selected_column.value = String(column);
            selected_row.value = String(row);
        """,
        )


@dataclass(frozen=True)
class S3TabContentDataSourceOnSelectedLoadingJSCode:
    """JS code about loading when data table selection happens."""

    @classmethod
    def get_js_code(cls, source: ColumnDataSource, selected_column: TextInput) -> CustomJS:
        """
        Get a custom js function.
        :param source: Data source.
        :param selected_column: Selected column text input.
        """
        return CustomJS(
            args=dict(source=source, selected_column=selected_column),
            code="""
                         if (selected_column.value == 0 && source.data['object'][0] != '-') {
                                 document.getElementById('overlay').style.display = 'block';
                                 document.getElementById('cloud-input-form').style.display = 'none';
                                 document.getElementById('cloud-modal-loading').style.visibility = 'visible';
                         }
                         """,
        )
