from .base_materializer import BaseMaterializer
from .default_materializer import DefaultMaterializer
from .polars_materializer import PolarsParquetMaterializer
from .pydantic_materializer import PydanticParquetMaterializer

MATERIALIZER_MAPPING: dict[tuple[type], BaseMaterializer] = {
    (): DefaultMaterializer,
}
