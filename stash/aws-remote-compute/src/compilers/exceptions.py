"""Errors raised while compiling an `AssembledPipeline` into a backend-
specific `CompiledPipeline`.
"""


class CompilationError(Exception):
    """Base class for errors raised while compiling an `AssembledPipeline`.

    A sibling to `AssemblyError` and `ExecutionError`, not a subclass of
    either - assembling a pipeline, running it, and compiling it for a
    specific backend are three separate concerns (the same "separate
    concerns" reasoning already applied between `AssemblyError` and
    `ExecutionError`).
    """


class UnsupportedComputeBackendError(CompilationError):
    """Raised when compiling a pipeline that has one or more tasks still on
    `LocalComputeBackend` - a backend-specific compiler needs every task
    placed on a backend it actually knows how to run remotely.
    """
