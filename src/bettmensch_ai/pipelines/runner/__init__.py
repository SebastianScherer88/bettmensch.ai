"""Execution: `LocalRunner` (Layer 1's in-process pipeline runner) and the
shared `ExecutionError` hierarchy.

`RegisteredPipeline`/`RemoteRunner` (invoking an already-registered
pipeline's own remote execution) have been stashed, along with the rest of
the AWS Batch/Lambda/Step Functions remote-compute layer - see
`stash/aws-remote-compute/README.md`.
"""

from .exceptions import (
    ExecutionError,
    MaterializerMismatchWarning,
    MissingPipelineInputError,
    UnknownPipelineInputError,
)
from .local_runner import LocalRunner, run_locally

__all__ = [
    "ExecutionError",
    "MaterializerMismatchWarning",
    "MissingPipelineInputError",
    "UnknownPipelineInputError",
    "LocalRunner",
    "run_locally",
]
