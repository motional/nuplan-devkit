"""
Custom SqlAlchemy types.
https://docs.sqlalchemy.org/en/latest/core/custom_types.html
"""
import pickle
import uuid
from typing import Any, Optional

from sqlalchemy.engine import Dialect
from sqlalchemy.types import BLOB, String, TypeDecorator, TypeEngine

from nuplan.database.common.data_types import RLE, Bbox, CameraIntrinsic, Rotation, Size, Translation, Visibility


class UUID(TypeDecorator):
    """
    Use BLOB(16) for sqlite.(bigint for mysql and uuid for postgresql)
    """

    impl = BLOB

    # Necessary caching flag since SQLAlchemy version 1.4.14
    # https://docs.sqlalchemy.org/en/14/core/custom_types.html#sqlalchemy.types.TypeDecorator
    cache_ok = True

    def load_dialect_impl(self, dialect: Dialect) -> TypeEngine:
        """Inherited, see superclass."""
        return dialect.type_descriptor(BLOB(16))

    def process_bind_param(self, value: Optional[str], dialect: Dialect) -> Optional[bytes]:
        """Inherited, see superclass."""
        if not value:
            return None

        return uuid.UUID(value).bytes

    def process_result_value(self, value: Optional[bytes], dialect: Dialect) -> Optional[str]:
        """Inherited, see superclass."""
        if not value:
            return None
        return value.hex()


class HexLen8(TypeDecorator):
    """
    Use BLOB(16) for sqlite.
    """

    impl = BLOB

    # Necessary caching flag since SQLAlchemy version 1.4.14
    # https://docs.sqlalchemy.org/en/14/core/custom_types.html#sqlalchemy.types.TypeDecorator
    cache_ok = True

    def load_dialect_impl(self, dialect: Any) -> Any:
        """Inherited, see superclass."""
        return dialect.type_descriptor(BLOB(8))

    def process_bind_param(self, value: Any, dialect: Any) -> Optional[bytes]:
        """Inherited, see superclass."""
        if not value:
            return None

        return bytearray.fromhex(value)

    def process_result_value(self, value: Any, dialect: Any) -> Optional[str]:
        """Inherited, see superclass."""
        if not value:
            return None

        return value.hex()  # type: ignore


class SimplePickleType(TypeDecorator):
    """
    Use pickle for dict/list type of objects.
    """

    impl = BLOB
    class_type: Any = None

    def process_bind_param(self, value: Any, dialect: Dialect) -> Optional[bytes]:
        """Inherited, see superclass."""
        if not value:
            return None

        return pickle.dumps(value)

    def process_result_value(self, value: Optional[bytes], dialect: Dialect) -> Any:
        """Inherited, see superclass."""
        if not value:
            return None

        assert self.class_type is not None
        ret = pickle.loads(value)
        return self.class_type(ret)


class SqlRLE(SimplePickleType):
    """Sql type for RLE."""

    class_type = RLE


class SqlTranslation(SimplePickleType):
    """Sql type for Translation."""

    class_type = Translation


class SqlRotation(SimplePickleType):
    """Sql type for Rotation."""

    class_type = Rotation


class SqlBbox(SimplePickleType):
    """Sql type for SimplePickleType."""

    class_type = Bbox


class SqlSize(SimplePickleType):
    """Sql type for Size."""

    class_type = Size


class SqlCameraIntrinsic(SimplePickleType):
    """Sql type for CameraIntrinsic."""

    class_type = CameraIntrinsic


class SqlVisibility(TypeDecorator):
    """Sql type for Visibility."""

    impl = String

    def load_dialect_impl(self, dialect: Dialect) -> TypeEngine:
        """Inherited, see superclass."""
        return dialect.type_descriptor(String(8))

    def process_bind_param(self, value: Optional[Visibility], dialect: Dialect) -> Any:
        """Inherited, see superclass."""
        if not value:
            value = Visibility.unknown

        return value.value

    def process_result_value(self, value: Optional[str], dialect: Dialect) -> Visibility:
        """Inherited, see superclass."""
        if not value:
            return Visibility.unknown

        return Visibility(value)
