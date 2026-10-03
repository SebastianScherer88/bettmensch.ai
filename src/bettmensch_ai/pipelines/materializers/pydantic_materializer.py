"""`PydanticJsonMaterializer`: (de)serializes pydantic model/settings
values.
"""

import importlib
import json
from typing import Any, Dict, Optional, Type, TypeAlias

from pydantic import BaseModel
from pydantic_settings import BaseSettings

from .base_materializer import BaseMaterializer

PydanticSerializable: TypeAlias = BaseSettings | BaseModel


def _resolve_class(module_name: str, qualname: str) -> Optional[type]:
    """Resolves a class from its `__module__`/`__qualname__`, as recorded
    by `PydanticJsonMaterializer._extra_metadata`.

    `module_name` and `qualname` are kept as two separate metadata fields
    rather than one concatenated string precisely because they can't be
    reliably split back apart afterwards: a nested class's `__qualname__`
    (e.g. `"Outer.Inner"`) itself contains dots, so there is no way to tell,
    from a single dotted string alone, how many trailing components belong
    to the class path versus the module path.

    Depends on `module_name` actually being importable in the current
    process - e.g. because a `CodeBundler` bundle containing it has already
    been extracted onto `sys.path`, which nothing does automatically today.

    Args:
        module_name: The class's `__module__`.
        qualname: The class's `__qualname__`.

    Returns:
        The resolved class, or `None` if `qualname` names a function-local
        class (identifiable by a `<locals>` component - there is no import
        path to such a class, bundled code or not), the module can't be
        imported, or a name in `qualname` doesn't resolve to an attribute.
    """

    if "<locals>" in qualname:
        return None

    try:
        obj: Any = importlib.import_module(module_name)
    except ImportError:
        return None

    try:
        for part in qualname.split("."):
            obj = getattr(obj, part)
    except AttributeError:
        return None

    return obj if isinstance(obj, type) else None


class PydanticJsonMaterializer(BaseMaterializer):
    """Serializes pydantic `BaseModel`/`BaseSettings` instances to/from a
    JSON file.
    """

    name = "pydantic_json"

    def __init__(self, model: Optional[Type[PydanticSerializable]] = None):
        """Initializes the materializer.

        Args:
            model: The pydantic model/settings class to cast a loaded value
                to. If omitted, `load()` returns a plain `dict` instead.
        """

        self.model = model

    @classmethod
    def supports_type(cls, type_hint: Any) -> bool:
        """Reports whether `type_hint` is a pydantic `BaseModel`/
        `BaseSettings` (sub)class.

        Args:
            type_hint: The declared type to check.

        Returns:
            Whether this materializer can handle `type_hint`.
        """

        return isinstance(type_hint, type) and issubclass(
            type_hint, (BaseSettings, BaseModel)
        )

    def supports(self, value: Any) -> bool:
        """Reports whether `value` is a pydantic `BaseModel`/`BaseSettings`
        instance.

        Args:
            value: The value to check.

        Returns:
            Whether this materializer can serialize `value`.
        """

        return isinstance(value, (BaseSettings, BaseModel))

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
    ) -> PydanticSerializable:
        """Loads the exported json. If a model is provided, attempts to cast
        the dict to said model, otherwise returns the dict.

        Args:
            path: The path to the json file to be loaded.

        Returns:
            The deserialized value, cast to `self.model` if one was given.
        """

        with open(path) as json_file:
            value = json.load(json_file)

        if self.model is not None:
            value = self.model.model_validate(value)

        return value

    def _save(self, value: PydanticSerializable, path: str) -> None:
        """Exports a pydantic model/settings instance to a json file.

        Args:
            value: The value to serialize.
            path: The path to export the value to as a json file.
        """

        value_json = value.model_dump_json()

        with open(path, "w") as json_file:
            json_file.write(value_json)

    def _extra_metadata(self, value: PydanticSerializable) -> Dict[str, Any]:
        """Records `value`'s exact runtime class, so `from_metadata` can
        cast back to it later - independent of whether `self.model` was
        ever configured on the instance that did the saving.

        Args:
            value: The value that was just saved.

        Returns:
            `value`'s `__module__`/`__qualname__`, as two separate fields
            (see `_resolve_class` for why they aren't concatenated).
        """

        return {
            "model_module": type(value).__module__,
            "model_qualname": type(value).__qualname__,
        }

    @classmethod
    def from_metadata(
        cls, metadata: Dict[str, Any]
    ) -> "PydanticJsonMaterializer":
        """Reconstructs a materializer configured with the exact pydantic
        model class that produced this artifact, so `load()` returns a real
        model instance instead of a plain `dict`.

        Best-effort: if the recorded class can't be resolved (its module
        isn't importable in this process, it was renamed/removed, or it
        turns out not to actually be a pydantic model/settings class), this
        degrades to an unconfigured materializer rather than raising - the
        same plain-`dict` behaviour `load()` already has when no model is
        known at all, so an artifact stays inspectable either way.

        Args:
            metadata: This artifact's metadata sidecar.

        Returns:
            A materializer with `model` set to the resolved class, or left
            `None` if it couldn't be resolved.
        """

        module_name = metadata.get("model_module")
        qualname = metadata.get("model_qualname")

        if module_name is None or qualname is None:
            return cls()

        model = _resolve_class(module_name, qualname)

        if model is not None and issubclass(model, (BaseSettings, BaseModel)):
            return cls(model=model)

        return cls()
