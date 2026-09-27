"""The `BaseMaterializer` interface and its metadata sidecar contract."""

import json
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Dict

from ..artifact_metadata import METADATA_SCHEMA_VERSION, metadata_key


class BaseMaterializer(ABC):
    """Serializes and deserializes task inputs and outputs.

    Resolved for each task input/output ahead of execution where possible,
    via three separate paths, in increasing order of how little context
    they need:

    1. `supports_type(type_hint)` - static, from a declared type hint. Used
       by the `Assembler` at assembly time.
    2. `supports(value)` - dynamic, from an actual value. Used as a runtime
       fallback when no static type hint is available (e.g. an `Any`-typed
       slot).
    3. The metadata sidecar `save()` always writes, recording this
       materializer's stable `name` - not the live class object, which
       might not even exist by the time someone looks. Used by
       `resolve_materializer_from_artifact` to retrieve an artifact after
       the fact, with no consuming task or type hint around at all: just
       whatever is sitting in the artifact store.
    """

    name: ClassVar[str]
    """This materializer's stable identifier, recorded in the metadata
    sidecar. A plain string rather than the class itself, since the class
    might be renamed/removed by the time an artifact is read back.
    """

    @classmethod
    @abstractmethod
    def supports_type(cls, type_hint: Any) -> bool:
        """Reports whether this materializer handles a statically declared
        type.

        Args:
            type_hint: The declared type to check, e.g. a task parameter's
                or return annotation's type.

        Returns:
            Whether this materializer can (de)serialize values of
            `type_hint`. Used by the `Assembler` to resolve materializers
            ahead of execution.
        """

    @abstractmethod
    def supports(self, value: Any) -> bool:
        """Reports whether this materializer handles a specific value.

        Args:
            value: The value to check.

        Returns:
            Whether this materializer can (de)serialize `value`. Used as a
            runtime fallback when no static type hint is available.
        """

    @property
    @abstractmethod
    def s3_support(self) -> bool:
        """Whether this materializer can hand its serialized representation
        directly to the `ArtifactStore` (e.g. a file-based format), as
        opposed to requiring materialization to a local path first.
        """

    def save(self, value: Any, path: str) -> None:
        """Saves `value` to `path`, then writes a metadata sidecar (see
        `artifact_metadata.metadata_key`) recording this materializer's
        `name` alongside it.

        Subclasses implement `_save`, not this method, so every
        materializer writes that metadata the same way rather than each
        having to remember to.

        Args:
            value: The value to serialize.
            path: The local path to save `value` to.
        """

        self._save(value, path)
        self._write_metadata(value, path)

    def load(self, path: str) -> Any:
        """Loads a previously saved value.

        Args:
            path: The local path a value was previously `save()`d to.

        Returns:
            The deserialized value.
        """

        return self._load(path)

    @abstractmethod
    def _save(self, value: Any, path: str) -> None:
        """Subclass-specific serialization, called by `save()`.

        Args:
            value: The value to serialize.
            path: The local path to save `value` to.
        """

    @abstractmethod
    def _load(self, path: str) -> Any:
        """Subclass-specific deserialization, called by `load()`.

        Args:
            path: The local path a value was previously `_save()`d to.

        Returns:
            The deserialized value.
        """

    def _extra_metadata(self, value: Any) -> Dict[str, Any]:
        """Extra metadata fields this materializer wants recorded in the
        sidecar, on top of the base schema (`schema_version`,
        `materializer`, `value_type`). Overridden by subclasses that need
        more than the stable materializer name to reconstruct themselves
        later - e.g. which exact class to deserialize into. Default: none.

        Args:
            value: The value that was just saved.

        Returns:
            Extra fields to merge into the metadata sidecar. Must not use
            any of the base schema's own keys.
        """

        return {}

    @classmethod
    def from_metadata(cls, metadata: Dict[str, Any]) -> "BaseMaterializer":
        """Reconstructs an instance of this materializer from its own
        metadata sidecar, as read back by `resolve_materializer_from_artifact`
        with no consuming task or type hint around. Overridden by
        subclasses whose `_extra_metadata` records something a no-arg
        constructor can't recover on its own. Default: plain `cls()`.

        Args:
            metadata: This artifact's metadata sidecar (the same dict
                `_write_metadata` wrote, plus the base schema fields).

        Returns:
            A materializer instance configured from `metadata`.
        """

        return cls()

    def _write_metadata(self, value: Any, path: str) -> None:
        """Writes the metadata sidecar for a just-saved value.

        Args:
            value: The value that was just saved, used to record its
                runtime type for human debugging and passed to
                `_extra_metadata` for any subclass-specific fields.
            path: The local path `value` was saved to; the sidecar is
                written to `metadata_key(path)`.
        """

        metadata = {
            "schema_version": METADATA_SCHEMA_VERSION,
            "materializer": self.name,
            "value_type": f"{type(value).__module__}.{type(value).__qualname__}",
            **self._extra_metadata(value),
        }

        with open(metadata_key(path), "w") as metadata_file:
            json.dump(metadata, metadata_file)
