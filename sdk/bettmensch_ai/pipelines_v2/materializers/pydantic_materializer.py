import json
from typing import Any, TypeAlias

from base_materializer import BaseMaterializer
from pydantic import BaseModel
from pydantic_settings import BaseSettings

PydanticSerializable: TypeAlias = BaseSettings | BaseModel


class PydanticParquetMaterializer(BaseMaterializer):
    def __init__(self, model: type[PydanticSerializable] | None = None):
        self.model = model

    def supports(self, value: Any) -> bool:
        return isinstance(value, (BaseSettings, BaseModel))

    @property
    def s3_support(self):
        return False

    def load(
        self,
        path: str,
    ) -> PydanticSerializable:
        """Loads the exported json. If a model is provided, attempts to cast
        the dict to said model, otherwise returns the dict

        Args:
            path (str): The path to the json file to be loaded.

        Returns:
            PydanticSerializable:
        """

        with open(path) as json_file:
            value = json.load(json_file)

        if self.model is not None:
            value = self.model.model_validate(value)

        return value

    def save(self, value: type[PydanticSerializable], path: str):
        """Exports a pydantic model/settings instance to a json file.

        Args:
            value (type[PydanticSerializable]): The value to
                serialize.
            path (str): The path to export the value to as a json file.
        """

        value_json = value.model_dump_json()

        with open(path, "w") as json_file:
            value_json.write(json_file)
