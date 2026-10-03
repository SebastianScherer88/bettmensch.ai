"""Compute backends: where an `AssembledTask` actually runs.

`BaseComputeBackend` is the interface; `LocalComputeBackend` (in-process) is
every `AssembledTask`'s default - the only backend currently available.

`AwsBatchComputeBackend`/`AwsLambdaComputeBackend` (placing a task on AWS
Batch/Lambda via `aws_batch`/`aws_lambda` at the pipeline definition's own
call site) have been stashed, pending a redesign of the artifact/metadata
Client abstractions - see `stash/aws-remote-compute/README.md`.
"""

from .base_compute_backend import BaseComputeBackend
from .local_compute_backend import LocalComputeBackend

__all__ = [
    "BaseComputeBackend",
    "LocalComputeBackend",
]
