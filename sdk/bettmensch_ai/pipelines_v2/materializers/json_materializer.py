import json
from typing import Any, TypeAlias

from base_materializer import BaseMaterializer

JSONSerializable: TypeAlias = (
    None
    | bool
    | int
    | float
    | str
    | tuple["JSONSerializable"]
    | list["JSONSerializable"]
    | dict[str, "JSONSerializable"]
)


class JsonMaterializer(BaseMaterializer):
    def is_json_serializable(self, value: Any) -> bool:
        if value is None or isinstance(value, (bool, int, float, str)):
            return True

        if isinstance(value, (list, tuple)):
            return all(self.is_json_serializable(x) for x in value)

        if isinstance(value, dict):
            return all(
                isinstance(k, str) and self.is_json_serializable(v)
                for k, v in value.items()
            )

        return False

    def supports(self, value: Any) -> bool:
        return self.is_json_serializable(value)

    @property
    def s3_support(self):
        return False

    def load(
        self,
        path: str,
    ) -> JSONSerializable:
        """Loads the exported json. If a model is provided, attempts to cast
        the dict to said model, otherwise returns the dict

        Args:
            path (str): The path to the json file to be loaded.

        Returns:
            BaseModel | BaseSettings | dict:
        """

        with open(path) as json_file:
            value = json.load(json_file)

        return value

    def save(self, value: JSONSerializable, path: str):
        """Exports a pydantic model/settings instance to a json file.

        Args:
            value (): The value to
                serialize.
            path (str): The path to export the value to as a json file.
        """

        with open(path, "w") as json_file:
            value.write(json_file)
