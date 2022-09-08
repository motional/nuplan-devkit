from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Optional

from bokeh.models import HTMLTemplateFormatter


@dataclass(frozen=True)
class S3TabBucketNameConfig:
    """Config for s3 tab bucket name div tag."""

    text: ClassVar[str] = '-'
    name: ClassVar[str] = 's3_bucket_name'
    css_classes: ClassVar[List[str]] = ['s3-bucket-name', 'h5']

    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """Get configs as a dict."""
        return {'text': cls.text, 'name': cls.name, 'css_classes': cls.css_classes}


@dataclass(frozen=True)
class S3TabDownloadTextInputConfig:
    """Config for s3 tab download input text tag."""

    placeholder: ClassVar[str] = 'Download prefix (without s3://)'
    width: ClassVar[int] = 400
    height: ClassVar[int] = 30
    name: ClassVar[str] = 's3_download_text_input'
    css_classes: ClassVar[List[str]] = ['s3-download-text-input']

    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """Get configs as a dict."""
        return {
            'placeholder': cls.placeholder,
            'width': cls.width,
            'height': cls.height,
            'name': cls.name,
            'css_classes': cls.css_classes,
        }


@dataclass(frozen=True)
class S3TabDownloadButtonConfig:
    """Config for s3 tab download button tag."""

    label: ClassVar[str] = 'Download'
    width: ClassVar[int] = 150
    height: ClassVar[int] = 30
    name: ClassVar[str] = 's3_download_button'
    css_classes: ClassVar[List[str]] = ['btn', 'btn-primary', 'modal-btn', 's3-download-btn']

    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """Get configs as a dict."""
        return {
            'label': cls.label,
            'width': cls.width,
            'height': cls.height,
            'name': cls.name,
            'css_classes': cls.css_classes,
        }


@dataclass(frozen=True)
class S3TabErrorTextConfig:
    """Config for s3 tab error div tag."""

    text: ClassVar[str] = ''
    name: ClassVar[str] = 's3_error_text'
    css_classes: ClassVar[List[str]] = ['s3-error-text', 'h5']

    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """Get configs as a dict."""
        return {'text': cls.text, 'name': cls.name, 'css_classes': cls.css_classes}


@dataclass(frozen=True)
class S3TabBucketTextInputConfig:
    """Config for s3 tab bucket text input tag."""

    placeholder: ClassVar[str] = 'Bucket name'
    name: ClassVar[str] = 's3_bucket_text_input'

    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """Get configs as a dict."""
        return {'placeholder': cls.placeholder, 'name': cls.name}


@dataclass(frozen=True)
class S3TabS3AccessKeyIDTextInputConfig:
    """Config for s3 tab access key id text input tag."""

    value: ClassVar[str] = ''
    placeholder: ClassVar[str] = 'Access key ID'
    name: ClassVar[str] = 's3_access_key_id_text_input'

    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """Get configs as a dict."""
        return {'value': cls.value, 'placeholder': cls.placeholder, 'name': cls.name}


@dataclass(frozen=True)
class S3TabS3SecretAccessKeyPasswordTextInputConfig:
    """Config for s3 tab secret access key password text input tag."""

    value: ClassVar[str] = ''
    placeholder: ClassVar[str] = 'Secret access key'
    name: ClassVar[str] = 's3_secret_access_key_password_input'

    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """Get configs as a dict."""
        return {'value': cls.value, 'placeholder': cls.placeholder, 'name': cls.name}


@dataclass(frozen=True)
class S3TabS3BucketPrefixTextInputConfig:
    """Config for s3 tab bucket prefix text input tag."""

    value: ClassVar[str] = ''
    placeholder: ClassVar[str] = 'prefix e.g. user-name/'
    name: ClassVar[str] = 's3_bucket_prefix_text_input'

    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """Get configs as a dict."""
        return {'value': cls.value, 'placeholder': cls.placeholder, 'name': cls.name}


@dataclass(frozen=True)
class S3TabS3ModalQueryButtonConfig:
    """Config for s3 tab modal query button tag."""

    name: ClassVar[str] = 's3_modal_query_btn'
    label: ClassVar[str] = 'Query S3'
    css_classes: ClassVar[List[str]] = ['btn', 'btn-primary', 'modal-btn', 's3-tab-modal-query-btn']

    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """Get configs as a dict."""
        return {'name': cls.name, 'label': cls.label, 'css_classes': cls.css_classes}


@dataclass(frozen=True)
class S3TabObjectColumnConfig:
    """Config for s3 tab object column tag."""

    field: ClassVar[str] = 'object'
    title: ClassVar[str] = 'Object'
    width: ClassVar[int] = 200
    sortable: ClassVar[bool] = False
    formatter_template: ClassVar[str] = """<a href="#"><%= value %></a>"""

    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """Get configs as a dict."""
        return {
            'field': cls.field,
            'title': cls.title,
            'width': cls.width,
            'sortable': cls.sortable,
            'formatter': HTMLTemplateFormatter(template=cls.formatter_template),
        }


@dataclass(frozen=True)
class S3TabLastModifiedColumnConfig:
    """Config for s3 tab last_modified column tag."""

    field: ClassVar[str] = 'last_modified'
    title: ClassVar[str] = 'Last Modified'
    width: ClassVar[int] = 150
    sortable: ClassVar[bool] = False

    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """Get configs as a dict."""
        return {'field': cls.field, 'title': cls.title, 'width': cls.width, 'sortable': cls.sortable}


@dataclass(frozen=True)
class S3TabTimeStampColumnConfig:
    """Config for s3 tab timestamp column tag."""

    field: ClassVar[str] = 'timestamp'
    title: ClassVar[str] = 'Timestamp'
    width: ClassVar[int] = 150
    sortable: ClassVar[bool] = False

    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """Get configs as a dict."""
        return {'field': cls.field, 'title': cls.title, 'width': cls.width, 'sortable': cls.sortable}


@dataclass(frozen=True)
class S3TabSizeColumnConfig:
    """Config for s3 tab size column tag."""

    field: ClassVar[str] = 'size'
    title: ClassVar[str] = 'Size (KB)'
    width: ClassVar[int] = 150
    sortable: ClassVar[bool] = False

    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """Get configs as a dict."""
        return {'field': cls.field, 'title': cls.title, 'width': cls.width, 'sortable': cls.sortable}


@dataclass(frozen=True)
class S3TabDataTableConfig:
    """Config for s3 tab data table column tag."""

    selectable: ClassVar[bool] = True
    row_height: ClassVar[int] = 80
    index_position: ClassVar[Optional[int]] = None
    name: ClassVar[str] = 's3_data_table'
    css_classes: ClassVar[List[str]] = ["s3-data-table"]

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
