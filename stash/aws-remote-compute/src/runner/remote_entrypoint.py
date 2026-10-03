"""The remote task entrypoint: what actually runs inside an AWS Batch
container or an AWS Lambda invocation to execute one task.

Converges on the same `execute_task` primitive `LocalComputeBackend` calls
in-process - there is exactly one "how does a task actually run" code path
regardless of where it runs. A task placed on `aws_batch(...)`/
`aws_lambda(...)` must be defined at module level (importable by
`module, qualname`) - the same constraint `PydanticJsonMaterializer`
already has for reconstructing an arbitrary class across a process
boundary without pickling.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import tempfile
import traceback
from typing import Any, Dict, Optional

from ..artifact_store import S3ArtifactStore, S3ArtifactStoreConfig
from ..code_bundler import CodeBundler
from ..materializers import register_materializer
from .task_execution import execute_task, get_captured_logs


def _resolve_by_qualname(module_name: str, qualname: str) -> Any:
    """Imports `module_name` and walks `qualname`'s dotted path off it.

    Reuses the same "reconstruct an arbitrary object from `(module,
    qualname)` stored separately" mechanism `PydanticJsonMaterializer`
    already uses to reconstruct a pydantic model class from stored
    metadata, rather than pickling a live object across the boundary.

    Args:
        module_name: The module to import.
        qualname: A dotted attribute path off that module - for a task,
            this resolves to the `Task` object itself (`@task` replaces the
            decorated function's module-level name with it), not the raw
            function.

    Returns:
        The resolved object.
    """

    obj: Any = importlib.import_module(module_name)
    for part in qualname.split("."):
        obj = getattr(obj, part)

    return obj


def _run_task_remotely(
    task_module: str,
    task_qualname: str,
    default_materializer_module: str,
    default_materializer_qualname: str,
    code_bundle_key: str,
    static_inputs: Dict[str, Any],
    input_keys: Dict[str, str],
    output_keys: Dict[str, str],
) -> Dict[str, Any]:
    """Downloads the code bundle, resolves the task and its default
    materializer, and runs the task - the shared body of both `main()`
    (Batch) and `lambda_handler()` (Lambda). Never raises itself.

    Args:
        task_module: The module the task's underlying function is defined
            in.
        task_qualname: That function's qualname within `task_module`.
        default_materializer_module: The module the pipeline's own
            `default_materializer` class is defined in.
        default_materializer_qualname: That class's qualname.
        code_bundle_key: The artifact key this run's code bundle was
            uploaded under.
        static_inputs: This task's literal/constant input values.
        input_keys: Maps each bound input name to the artifact key to load
            its value from.
        output_keys: Maps each output name to the artifact key to save it
            under.

    Returns:
        `{"status": "succeeded", "logs": ...}` or `{"status": "failed",
        "logs": ..., "traceback": ...}`.
    """

    artifact_store = S3ArtifactStore(S3ArtifactStoreConfig())

    with tempfile.TemporaryDirectory() as bundle_dir:
        CodeBundler.download_and_extract(artifact_store, code_bundle_key, bundle_dir)
        sys.path.insert(0, bundle_dir)

        try:
            task = _resolve_by_qualname(task_module, task_qualname)
            default_materializer_cls = _resolve_by_qualname(
                default_materializer_module, default_materializer_qualname
            )
            register_materializer(default_materializer_cls)

            logs = execute_task(
                task,
                artifact_store,
                static_inputs,
                input_keys,
                output_keys,
                default_materializer_cls,
            )
        except Exception as exc:
            return {
                "status": "failed",
                "logs": get_captured_logs(exc),
                "traceback": traceback.format_exc(),
            }
        finally:
            sys.path.remove(bundle_dir)

    return {"status": "succeeded", "logs": logs}


def main(argv: Optional[list] = None) -> int:
    """CLI entrypoint for an AWS Batch container's command.

    Args:
        argv: Arguments to parse, or `None` to use `sys.argv[1:]`.

    Returns:
        `0` on success, `1` if the task failed (so Batch marks the job
        `FAILED`).
    """

    parser = argparse.ArgumentParser(
        description="Run one bettmensch.ai task on remote compute."
    )
    parser.add_argument("--task-module", required=True)
    parser.add_argument("--task-qualname", required=True)
    parser.add_argument("--default-materializer-module", required=True)
    parser.add_argument("--default-materializer-qualname", required=True)
    parser.add_argument("--code-bundle-key", required=True)
    parser.add_argument("--static-inputs", default="{}")
    parser.add_argument("--input-keys", default="{}")
    parser.add_argument("--output-keys", default="{}")
    args = parser.parse_args(argv)

    result = _run_task_remotely(
        args.task_module,
        args.task_qualname,
        args.default_materializer_module,
        args.default_materializer_qualname,
        args.code_bundle_key,
        json.loads(args.static_inputs),
        json.loads(args.input_keys),
        json.loads(args.output_keys),
    )

    if result["status"] == "failed":
        print(
            result.get("traceback") or result.get("logs") or "Task failed.",
            file=sys.stderr,
        )
        return 1

    if result.get("logs"):
        print(result["logs"])

    return 0


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """AWS Lambda handler: runs one task from `event`'s payload.

    Catches (does not raise) any failure, returning a structured
    `{"status": ..., ...}` payload instead of letting Lambda's own
    unhandled-exception response shape leak through -
    `AwsLambdaComputeBackend.run()` reads this directly from the invoke
    response.

    Args:
        event: `{"task_module", "task_qualname",
            "default_materializer_module", "default_materializer_qualname",
            "code_bundle_key", "static_inputs", "input_keys",
            "output_keys"}`.
        context: Unused; accepted for Lambda's own handler signature.

    Returns:
        `{"status": "succeeded", "logs": ...}` or `{"status": "failed",
        "logs": ..., "traceback": ...}`.
    """

    return _run_task_remotely(
        event["task_module"],
        event["task_qualname"],
        event["default_materializer_module"],
        event["default_materializer_qualname"],
        event["code_bundle_key"],
        event.get("static_inputs", {}),
        event.get("input_keys", {}),
        event.get("output_keys", {}),
    )


if __name__ == "__main__":
    sys.exit(main())
