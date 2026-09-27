"""Shared constants/helpers for an artifact's metadata sidecar.

Lives at the top level, alongside ``io_binding.py`` and ``exceptions.py``,
rather than under ``materializers/`` or ``artifact_store/``, because both
packages need it: a ``BaseMaterializer`` writes the sidecar locally
alongside the data it saves, and a ``BaseArtifactStore`` uploads/downloads
that same sidecar under a derived key. Putting it in either package would
make the other depend on it for this alone.
"""

METADATA_SCHEMA_VERSION = 1
METADATA_SUFFIX = ".metadata"


def metadata_key(key: str) -> str:
    """Returns the metadata sidecar reference for the artifact at `key`.

    Works equally for a local filesystem path and a `BaseArtifactStore`
    key: both are just strings, and this is a pure suffix operation that
    does not care which kind it was given.

    Args:
        key: The artifact's own local path or store key.

    Returns:
        The corresponding metadata sidecar's local path or store key.
    """

    return f"{key}{METADATA_SUFFIX}"
