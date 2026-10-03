"""`JsonMaterializer`: (de)serializes plain JSON-safe values."""

import json
import typing
from typing import Any, TypeAlias

from .base_materializer import BaseMaterializer

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

_JSON_PRIMITIVE_TYPES = (type(None), bool, int, float, str, list, tuple, dict)


class JsonMaterializer(BaseMaterializer):
    """Serializes plain JSON-safe values (primitives, and lists/tuples/
    dicts of them) to/from a JSON file.
    """

    name = "json"

    @classmethod
    def supports_type(cls, type_hint: Any) -> bool:
        """Reports whether `type_hint` is a JSON-safe primitive or
        container type.

        Args:
            type_hint: The declared type to check, e.g. `int`,
                `Dict[str, Any]`, `List[int]`.

        Returns:
            Whether this materializer can handle `type_hint`.
        """

        origin = typing.get_origin(type_hint)
        candidate = origin or type_hint

        return isinstance(candidate, type) and issubclass(
            candidate, _JSON_PRIMITIVE_TYPES
        )

    def is_json_serializable(self, value: Any) -> bool:
        """Recursively checks whether `value` is JSON-serializable.

        Args:
            value: The value to check.

        Returns:
            Whether `value` (and, recursively, everything it contains) is
            a JSON-safe primitive, list/tuple, or dict with string keys.
        """

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
        """Reports whether `value` is JSON-serializable.

        Args:
            value: The value to check.

        Returns:
            Whether this materializer can serialize `value`.
        """

        return self.is_json_serializable(value)

    @property
    def s3_support(self) -> bool:
        """Whether this materializer can hand its output directly to the
        `ArtifactStore`.

        Returns:
            Always `False`.
        """

        return False

    def _load(
        self,
        path: str,
    ) -> JSONSerializable:
        """Loads the exported json value.

        Args:
            path: The path to the json file to be loaded.

        Returns:
            The deserialized value.
        """

        with open(path) as json_file:
            value = json.load(json_file)

        return value

    def _save(self, value: JSONSerializable, path: str) -> None:
        """Exports a json-serializable value to a json file.

        Args:
            value: The value to serialize.
            path: The path to export the value to as a json file.
        """

        with open(path, "w") as json_file:
            json.dump(value, json_file)
