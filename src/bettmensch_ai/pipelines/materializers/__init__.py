"""Materializer resolution: static (by type), dynamic (by value), and
metadata-based (by what an already-stored artifact says about itself).
"""

import json
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Type, Union

from ..artifact_metadata import metadata_key
from ..artifact_store import BaseArtifactStore
from ..exceptions import MaterializerResolutionError
from .base_materializer import BaseMaterializer
from .default_materializer import DefaultMaterializer
from .json_materializer import JsonMaterializer
from .polars_materializer import PolarsParquetMaterializer
from .pydantic_materializer import PydanticJsonMaterializer

if TYPE_CHECKING:
    from ..client import ArtifactClient

# Specialised materializers, tried in order; DefaultMaterializer is always
# the final, catch-all fallback and is therefore excluded from this list.
MATERIALIZER_REGISTRY: List[Type[BaseMaterializer]] = [
    PolarsParquetMaterializer,
    PydanticJsonMaterializer,
    JsonMaterializer,
]

# Keyed by each materializer's stable `name` (not the class object itself,
# which might not even exist by the time someone looks), so an artifact can
# be resolved from nothing but its own metadata sidecar - see
# `resolve_materializer_from_artifact`.
MATERIALIZER_BY_NAME: Dict[str, Type[BaseMaterializer]] = {
    materializer_cls.name: materializer_cls
    for materializer_cls in (*MATERIALIZER_REGISTRY, DefaultMaterializer)
}


def register_materializer(materializer_cls: Type[BaseMaterializer]) -> None:
    """Registers a materializer by name for after-the-fact resolution
    (`resolve_materializer_from_artifact`) without adding it to the
    automatic `MATERIALIZER_REGISTRY` resolution chain.

    For materializers that are only ever selected explicitly - e.g. a
    `Pipeline`'s `default_materializer` override - rather than chosen
    automatically from a type hint or value. Without this, an artifact such
    a materializer produced would record a `materializer` name in its
    metadata sidecar that `resolve_materializer_from_artifact` has no class
    for, and so could never resolve.

    Args:
        materializer_cls: The materializer class to register. Safe to call
            more than once with the same class.
    """

    MATERIALIZER_BY_NAME[materializer_cls.name] = materializer_cls


def resolve_materializer_for_type(
    type_hint: Any,
    default_materializer_cls: Type[BaseMaterializer] = DefaultMaterializer,
) -> BaseMaterializer:
    """Resolves the `BaseMaterializer` to use for a statically declared task
    input/output type, as done by the `Assembler` during assembly.

    Args:
        type_hint: The declared type to resolve a materializer for.
        default_materializer_cls: The materializer to fall back to if none
            of `MATERIALIZER_REGISTRY` supports `type_hint`. Defaults to
            `DefaultMaterializer` (which refuses to serialize); a
            `Pipeline` may override this with e.g. a pickle-based
            materializer it explicitly opts into.

    Returns:
        The first registered materializer whose `supports_type(type_hint)`
        returns `True`, or an instance of `default_materializer_cls` if
        none does.
    """

    for materializer_cls in MATERIALIZER_REGISTRY:
        if materializer_cls.supports_type(type_hint):
            return materializer_cls()

    return default_materializer_cls()


def resolve_materializer_for_value(
    value: Any,
    default_materializer_cls: Type[BaseMaterializer] = DefaultMaterializer,
) -> BaseMaterializer:
    """Resolves the `BaseMaterializer` to use for a runtime value, as a
    fallback for cases where no static type hint was available at
    compilation time.

    Args:
        value: The value to resolve a materializer for.
        default_materializer_cls: The materializer to fall back to if none
            of `MATERIALIZER_REGISTRY` supports `value`. Defaults to
            `DefaultMaterializer` (which refuses to serialize); a
            `Pipeline` may override this with e.g. a pickle-based
            materializer it explicitly opts into.

    Returns:
        The first registered materializer whose `supports(value)` returns
        `True`, or an instance of `default_materializer_cls` if none does.
    """

    for materializer_cls in MATERIALIZER_REGISTRY:
        materializer = materializer_cls()
        if materializer.supports(value):
            return materializer

    return default_materializer_cls()


def resolve_materializer_from_artifact(
    artifact_store: Union[BaseArtifactStore, "ArtifactClient"], key: str
) -> BaseMaterializer:
    """Resolves the `BaseMaterializer` to use for an already-stored
    artifact using only what is recorded in the artifact store - no
    consuming task or declared type hint involved at all.

    This is what makes an artifact inspectable after the fact (e.g. in a
    notebook, months after the run that produced it): `save()` always
    writes a metadata sidecar recording the materializer's stable `name`,
    and this looks it back up from that, independent of whether the
    original `Task`/`Pipeline` definitions still exist in their original
    form.

    Args:
        artifact_store: The store (or `ArtifactClient`) the artifact (and
            its metadata sidecar) live in.
        key: The artifact's own key.

    Returns:
        An instance of the materializer recorded in the artifact's
        metadata sidecar.

    Raises:
        MaterializerResolutionError: If the sidecar names a materializer
            that isn't registered (e.g. defined by code no longer present).
    """

    with tempfile.TemporaryDirectory() as tmp_dir:
        local_path = str(Path(tmp_dir) / "metadata.json")
        artifact_store.download(metadata_key(key), local_path)

        with open(local_path) as metadata_file:
            metadata = json.load(metadata_file)

    materializer_name = metadata["materializer"]
    materializer_cls = MATERIALIZER_BY_NAME.get(materializer_name)

    if materializer_cls is None:
        raise MaterializerResolutionError(
            f"No registered materializer named {materializer_name!r} "
            f"(recorded in the metadata for artifact {key!r})."
        )

    return materializer_cls.from_metadata(metadata)


__all__ = [
    "BaseMaterializer",
    "DefaultMaterializer",
    "JsonMaterializer",
    "PolarsParquetMaterializer",
    "PydanticJsonMaterializer",
    "MATERIALIZER_REGISTRY",
    "MATERIALIZER_BY_NAME",
    "register_materializer",
    "resolve_materializer_for_type",
    "resolve_materializer_for_value",
    "resolve_materializer_from_artifact",
]
