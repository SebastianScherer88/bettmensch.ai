from typing import Any, TypeAlias

import polars
from base_materializer import BaseMaterializer

PolarsSerializable: TypeAlias = polars.DataFrame


class PolarsParquetMaterializer(BaseMaterializer):
    def supports(self, value: Any) -> bool:
        return isinstance(value, polars.DataFrame)

    @property
    def s3_support(self):
        return True

    def load(self, path: str) -> PolarsSerializable:
        return polars.read_parquet(path)

    def save(self, value: PolarsSerializable, path: str):

        value.write_parquet(path)
