"""Turns an `AssembledPipeline` into the plain, JSON-able `dag_structure`/
`pipeline_inputs` dicts `BaseMetadataStore` persists, and records them as a
`PipelineAssemblyRecord`.
"""

from __future__ import annotations

import inspect
import json
import textwrap
import uuid
from typing import Any, Dict, Optional, Tuple

from ..io_binding import PipelineInput, TaskOutput
from ..metadata_store import BaseMetadataStore
from ..pipeline.assembled_pipeline import AssembledPipeline


def _jsonable(value: Any) -> Any:
    """Best-effort JSON-safe conversion for a task's static input value.

    A static input can be any python object a user's task happens to be
    called with - not necessarily JSON-serializable. Falling back to `repr`
    keeps the DAG structure always serializable (and still informative in a
    UI) rather than raising or silently dropping the value.
    """

    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)


def _task_source(func: Any) -> Optional[str]:
    """Best-effort source text for a task's underlying function.

    `inspect.getsource` raises `OSError` for a function with no retrievable
    source (defined in a REPL, via `exec`, etc.) and (rarely) `TypeError`
    for a non-function object - either way this degrades to `None` rather
    than failing assembly recording over what's fundamentally a "nice to
    show" detail, not load-bearing structure.
    """

    try:
        return textwrap.dedent(inspect.getsource(func))
    except (OSError, TypeError):
        return None


def _source_to_dict(source: Any) -> Dict[str, Any]:
    """Serializes an `IOBinding`/`PipelineOutput`'s source reference."""

    if isinstance(source, TaskOutput):
        return {"kind": "task_output", "task": source.task_name, "output": source.output_name}

    return {"kind": "pipeline_input", "name": source.name}


def serialize_assembled_pipeline(
    assembled_pipeline: AssembledPipeline,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Serializes `assembled_pipeline` into the `dag_structure`/
    `pipeline_inputs` shape `BaseMetadataStore.record_pipeline_assembly`/
    `register_pipeline` expect.

    This is the backend-agnostic structure a Layer 2 compiler would also
    start from - it describes tasks, their topological rank, their IO
    bindings, and the edges between them, with no backend-specific content.

    Args:
        assembled_pipeline: The pipeline to serialize.

    Returns:
        A `(dag_structure, pipeline_inputs)` pair, both plain, JSON-able
        dicts:

        `dag_structure` is `{"tasks": [...], "edges": [...], "output": ...}`,
        where each task is `{"name", "rank", "inputs", "outputs",
        "output_materializers", "resource_requirements", "uv_requirements",
        "source"}` (`"source"` is the task function's own source text, or
        `None` if it couldn't be retrieved) - `"inputs"` is a list of
        `{"name", "source"}`, `"source"` being
        `{"kind": "static", "value"}`, `{"kind": "pipeline_input", "name"}`,
        or `{"kind": "task_output", "task", "output"}`. `"edges"` is a list
        of `[upstream_task_name, downstream_task_name]` pairs.

        `pipeline_inputs` maps each declared input's name to
        `{"required", "default", "materializer"}`.
    """

    tasks = []
    edges = []

    for rank_index, rank in enumerate(assembled_pipeline.task_ranks):
        for assembled_task in rank:
            bindings = assembled_pipeline.bindings_for(assembled_task.name)
            inputs = [
                {
                    "name": binding.target.input_name,
                    "source": _source_to_dict(binding.source),
                }
                for binding in bindings
            ]
            inputs.extend(
                {"name": name, "source": {"kind": "static", "value": _jsonable(value)}}
                for name, value in assembled_task.static_inputs.items()
            )

            upstream_tasks = {
                binding.source.task_name
                for binding in bindings
                if isinstance(binding.source, TaskOutput)
            }
            edges.extend([upstream, assembled_task.name] for upstream in sorted(upstream_tasks))

            materializers = assembled_task.materializers or {}
            tasks.append(
                {
                    "name": assembled_task.name,
                    "rank": rank_index,
                    "inputs": inputs,
                    "outputs": list(assembled_task.output_names),
                    "output_materializers": {
                        output_name: materializers[output_name].name
                        for output_name in assembled_task.output_names
                        if output_name in materializers
                    },
                    "resource_requirements": {
                        "cpu": assembled_task.resource_requirements.cpu,
                        "memory": assembled_task.resource_requirements.memory,
                        "gpu": assembled_task.resource_requirements.gpu,
                    },
                    "uv_requirements": {
                        "packages": list(assembled_task.uv_requirements.packages),
                        "python": assembled_task.uv_requirements.python,
                    },
                    "source": _task_source(assembled_task.func),
                }
            )

    output = assembled_pipeline.output
    dag_structure = {
        "tasks": tasks,
        "edges": edges,
        "output": (
            {"name": output.name, "source": _source_to_dict(output.source)}
            if output is not None
            else None
        ),
    }

    pipeline_inputs = {
        pipeline_input.name: {
            "required": pipeline_input.required,
            "default": (
                None
                if pipeline_input.required
                else _jsonable(pipeline_input.default)
            ),
            "materializer": assembled_pipeline.input_materializers[pipeline_input.name].name,
        }
        for pipeline_input in assembled_pipeline.inputs
    }

    return dag_structure, pipeline_inputs


def record_assembly(
    metadata_store: BaseMetadataStore, assembled_pipeline: AssembledPipeline
) -> uuid.UUID:
    """Records `assembled_pipeline`'s current structure into `metadata_store`.

    A no-op write, not a strict one: if the most recently recorded assembly
    for this pipeline name already has an identical `dag_structure`/
    `pipeline_inputs`, that existing record's id is reused instead of
    inserting a duplicate - a pipeline's structure doesn't usually change
    between runs, so this keeps assembly history meaningful (one entry per
    actual change) rather than growing one row per run. `LocalRunner` calls
    this automatically before every run; it can also be called standalone to
    record a pipeline that's been assembled but not (yet) run.

    Args:
        metadata_store: The store to record into.
        assembled_pipeline: The pipeline to record.

    Returns:
        The id of the (possibly just-created, possibly pre-existing)
        `PipelineAssemblyRecord` representing `assembled_pipeline`'s current
        structure.
    """

    dag_structure, pipeline_inputs = serialize_assembled_pipeline(assembled_pipeline)

    existing = metadata_store.list_pipeline_assemblies(assembled_pipeline.name)
    if (
        existing
        and existing[0].dag_structure == dag_structure
        and existing[0].pipeline_inputs == pipeline_inputs
    ):
        return existing[0].pipeline_assembly_id

    return metadata_store.record_pipeline_assembly(
        assembled_pipeline.name, dag_structure, pipeline_inputs
    )
