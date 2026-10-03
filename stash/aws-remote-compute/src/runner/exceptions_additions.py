"""Stashed from `bettmensch_ai/pipelines/runner/exceptions.py` - these three
`ExecutionError` subclasses supported AWS Batch/Lambda/Step Functions remote
execution. Re-add them to that file (and re-export from `runner/__init__.py`/
the top-level `pipelines/__init__.py`) when remote compute comes back.
"""


class RemoteComputeConfigurationError(ExecutionError):
    """Raised when a remote `BaseComputeBackend` is asked to run a task
    without enough configuration to do so - e.g. `AwsBatchConfig.
    job_definition`/`AwsLambdaConfig.function_name` left unset for an
    ad-hoc `LocalRunner`-driven run (which, unlike registration, never
    provisions AWS resources on its own - it can only run against
    infrastructure that already exists), or an artifact store that isn't
    reachable from the remote compute being targeted (e.g. a
    `LocalArtifactStore`, which only exists on this one machine).
    """


class RemoteTaskExecutionError(ExecutionError):
    """Raised when a task run on remote compute (AWS Batch/Lambda) fails on
    the remote compute itself - a non-zero Batch job exit code, a Lambda
    invocation reporting failure - as opposed to the task's own Python code
    raising, which propagates as whatever exception the task itself raised.
    """


class UnsupportedBackendError(ExecutionError):
    """Raised when `RemoteRunner` is given a `RegisteredPipeline` whose
    `backend` it doesn't know how to invoke (only `"aws_stepfunctions"` is
    supported today).
    """
