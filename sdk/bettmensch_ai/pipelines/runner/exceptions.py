"""Errors (and warnings) raised while running an already-assembled
pipeline.
"""

from typing import Sequence


class MaterializerMismatchWarning(RuntimeWarning):
    """Raised when a value about to be materialized doesn't actually match
    the materializer resolved for it at assembly time from a declared type
    hint - i.e. the type hint was inaccurate.

    Not an `ExecutionError`: the run doesn't stop over this by itself, since
    `LocalRunner` re-resolves a materializer from the actual value instead
    (see `LocalRunner._reconcile_materializer`) - Python's type hints aren't
    enforced at runtime, so this is recoverable, not a hard error. Still
    worth surfacing loudly rather than silently correcting it, since it
    signals either a genuine bug or a type hint that should be widened
    (e.g. to a `Union` or `Any`).
    """


class ExecutionError(Exception):
    """Base class for errors raised while running an `AssembledPipeline`.

    Deliberately not a subclass of `AssemblyError`: assembling a pipeline
    (validating its graph, resolving materializers) and running it are
    separate concerns, per the "separate compilation from execution" design
    principle.
    """


class MissingPipelineInputError(ExecutionError):
    """Raised when a run is not given a value for one of the pipeline's
    required inputs.
    """

    def __init__(self, pipeline_name: str, missing_input_names: Sequence[str]):
        """Initializes the error.

        Args:
            pipeline_name: The name of the pipeline being run.
            missing_input_names: The names of the required inputs that
                were not given a value.
        """

        self.pipeline_name = pipeline_name
        self.missing_input_names = tuple(missing_input_names)

        missing = ", ".join(repr(name) for name in self.missing_input_names)
        super().__init__(
            f"Run of pipeline {pipeline_name!r} is missing required "
            f"input(s): {missing}."
        )


class UnknownPipelineInputError(ExecutionError):
    """Raised when a run is given a value for an input the pipeline does
    not declare.
    """

    def __init__(self, pipeline_name: str, unknown_input_names: Sequence[str]):
        """Initializes the error.

        Args:
            pipeline_name: The name of the pipeline being run.
            unknown_input_names: The names of the given values that don't
                match any input the pipeline declares.
        """

        self.pipeline_name = pipeline_name
        self.unknown_input_names = tuple(unknown_input_names)

        unknown = ", ".join(repr(name) for name in self.unknown_input_names)
        super().__init__(
            f"Run of pipeline {pipeline_name!r} received unknown input(s): "
            f"{unknown}."
        )
