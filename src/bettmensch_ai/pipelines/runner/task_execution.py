"""`execute_task`: resolves a task's inputs by key, calls its function, and
materializes each output by key - the shared execution primitive both
`LocalRunner` (in-process) and the remote task entrypoint (inside a Batch
container or a Lambda invocation) call, so there is exactly one "how does a
task actually run" code path regardless of where it runs.
"""

from __future__ import annotations

import contextlib
import io
import traceback
import warnings
from typing import TYPE_CHECKING, Any, Dict, Optional, Type

from ..artifact_store import BaseArtifactStore
from ..materializers import (
    resolve_materializer_for_type,
    resolve_materializer_for_value,
    resolve_materializer_from_artifact,
)
from .exceptions import MaterializerMismatchWarning

if TYPE_CHECKING:
    from ..materializers.base_materializer import BaseMaterializer
    from ..task.task import Task

# Attached to an exception `execute_task` lets propagate, carrying whatever
# was captured on stdout/stderr (plus the traceback) up to that point - not
# wrapped in a new exception type, which would change what callers observe
# (e.g. a task raising `ValueError` must still surface as `ValueError`).
_LOGS_ATTRIBUTE = "_bettmensch_ai_execution_logs"


def get_captured_logs(exc: BaseException) -> Optional[str]:
    """Reads back whatever `execute_task` captured before `exc` was raised.

    Args:
        exc: The exception `execute_task` raised.

    Returns:
        The captured log text, or `None` if `execute_task` never attached
        any (e.g. `exc` wasn't raised by `execute_task`).
    """

    return getattr(exc, _LOGS_ATTRIBUTE, None)


def execute_task(
    task: "Task",
    artifact_store: BaseArtifactStore,
    static_inputs: Dict[str, Any],
    input_keys: Dict[str, str],
    output_keys: Dict[str, str],
    default_materializer_cls: Type["BaseMaterializer"],
) -> Optional[str]:
    """Resolves `task`'s inputs, calls its function, and materializes each
    output under `output_keys`.

    Takes a bare `Task`, not an `AssembledTask`: materializer resolution is
    a pure function of `(task, default_materializer_cls)`, recomputed
    identically wherever this runs, so nothing about materializers needs to
    cross a process boundary for a remote-executed task.

    Args:
        task: The task to run.
        artifact_store: The store to load bound inputs from and save
            outputs to.
        static_inputs: This call's literal/constant input values, keyed by
            input name - used directly, in memory, never touching the
            store.
        input_keys: Maps each of `task`'s *bound* input names (a
            `TaskOutput`- or `PipelineInput`-sourced one) to the artifact
            key to load its value from. Together with `static_inputs`, must
            cover every parameter `task.func` requires.
        output_keys: Maps each of `task`'s declared output names to the
            artifact key to save it under.
        default_materializer_cls: The pipeline's own `default_materializer`,
            used if nothing in the registry supports a produced value.

    Returns:
        Whatever was captured on stdout/stderr while `task.func` ran, or
        `None` if nothing was.

    Raises:
        Exception: Whatever `task.func` itself raises, unchanged, with
            captured logs (stdout/stderr so far, plus this traceback)
            attached - retrieve them with `get_captured_logs(exc)`.
    """

    log_buffer = io.StringIO()

    try:
        kwargs = {
            input_name: (
                static_inputs[input_name]
                if input_name in static_inputs
                else _load(artifact_store, input_keys[input_name])
            )
            for input_name in task.signature.parameters
        }

        with contextlib.redirect_stdout(log_buffer), contextlib.redirect_stderr(
            log_buffer
        ):
            value = task.func(**kwargs)

        for output_name, key in output_keys.items():
            output_value = _extract_output_value(task, value, output_name)
            materializer = resolve_materializer_for_type(
                task.output_type_hints.get(output_name), default_materializer_cls
            )
            materializer = _reconcile_materializer(
                materializer,
                output_value,
                default_materializer_cls,
                f"{task.base_name}.{output_name}",
            )
            artifact_store.save(materializer, output_value, key)
    except Exception as exc:
        log_buffer.write(traceback.format_exc())
        setattr(exc, _LOGS_ATTRIBUTE, log_buffer.getvalue())
        raise

    return log_buffer.getvalue() or None


def _extract_output_value(task: "Task", value: Any, output_name: str) -> Any:
    """Extracts one named output's value from a task function's return
    value.

    Args:
        task: The task that was just run.
        value: The value `task.func` returned.
        output_name: The output name to extract.

    Returns:
        `value` itself for an ordinary (single-output) task; one field
        (`NamedTuple`) or key (`TypedDict`) of it for a multi-output one.
    """

    if task.is_typed_dict_output:
        return value[output_name]

    if task.is_named_tuple_output:
        return getattr(value, output_name)

    return value


def _reconcile_materializer(
    materializer: "BaseMaterializer",
    value: Any,
    default_materializer_cls: Type["BaseMaterializer"],
    described_by: str,
) -> "BaseMaterializer":
    """Reconciles an assembly-time-resolved materializer against the actual
    value about to be saved.

    Python's type hints aren't enforced at runtime, so the materializer
    resolved from a declared type hint can turn out not to actually support
    what a task produced. Rather than let a mismatched materializer's
    `_save()` fail with a confusing, several-layers-removed error, this
    checks `materializer.supports(value)` first and, if it fails, re-resolves
    from the value itself.

    Args:
        materializer: The materializer resolved from the declared type hint.
        value: The actual value about to be saved.
        default_materializer_cls: The pipeline's own `default_materializer`,
            used if nothing in the registry supports `value` either.
        described_by: A human-readable description of what's being
            materialized, used in the warning message.

    Returns:
        `materializer` unchanged if it already supports `value`; otherwise
        the materializer `resolve_materializer_for_value` resolves instead.
    """

    if materializer.supports(value):
        return materializer

    warnings.warn(
        f"{described_by}'s declared type doesn't match what it "
        f"actually produced (a {type(value).__name__}) - "
        f"{type(materializer).__name__} can't serialize it. Re-resolving "
        "a materializer from the actual value instead. Consider fixing "
        "the type hint.",
        MaterializerMismatchWarning,
        stacklevel=2,
    )

    return resolve_materializer_for_value(value, default_materializer_cls)


def _load(artifact_store: BaseArtifactStore, key: str) -> Any:
    """Loads an already-materialized artifact.

    Resolves the materializer from the artifact's own stored metadata
    rather than from any consuming task's declared type: a pipeline input
    has no such thing to begin with, and reusing whichever materializer
    actually produced the bytes avoids relying on a consumer's
    independently-resolved one just happening to match.

    Args:
        artifact_store: The store to load from.
        key: The artifact's key.

    Returns:
        The deserialized value.
    """

    materializer = resolve_materializer_from_artifact(artifact_store, key)

    return artifact_store.load(materializer, key)
