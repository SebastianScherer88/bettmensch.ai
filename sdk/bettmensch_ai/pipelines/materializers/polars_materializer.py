"""`PolarsParquetMaterializer`: (de)serializes `polars.DataFrame` values."""

from typing import Any, TypeAlias

import polars

from .base_materializer import BaseMaterializer

PolarsSerializable: TypeAlias = polars.DataFrame


class PolarsParquetMaterializer(BaseMaterializer):
    """Serializes `polars.DataFrame` values to/from a parquet file."""

    name = "polars_parquet"

    @classmethod
    def supports_type(cls, type_hint: Any) -> bool:
        """Reports whether `type_hint` is (a subclass of) `polars.DataFrame`.

        Args:
            type_hint: The declared type to check.

        Returns:
            Whether this materializer can handle `type_hint`.
        """

        return isinstance(type_hint, type) and issubclass(
            type_hint, polars.DataFrame
        )

    def supports(self, value: Any) -> bool:
        """Reports whether `value` is a `polars.DataFrame`.

        Args:
            value: The value to check.

        Returns:
            Whether this materializer can serialize `value`.
        """

        return isinstance(value, polars.DataFrame)

    @property
    def s3_support(self) -> bool:
        """Whether this materializer can hand its output directly to the
        `ArtifactStore`.

        Returns:
            Always `True`.
        """

        return True

    def _load(self, path: str) -> PolarsSerializable:
        """Loads a `polars.DataFrame` from a parquet file.

        Args:
            path: The path to the parquet file to be loaded.

        Returns:
            The deserialized `polars.DataFrame`.
        """

        return polars.read_parquet(path)

    def _save(self, value: PolarsSerializable, path: str) -> None:
        """Exports a `polars.DataFrame` to a parquet file.

        Args:
            value: The `polars.DataFrame` to serialize.
            path: The path to export `value` to.
        """

        value.write_parquet(path)
