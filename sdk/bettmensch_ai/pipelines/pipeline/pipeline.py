"""`Pipeline`/`@pipeline`: wraps a plain function into a traceable, then
assemblable, pipeline definition.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Optional, Type, Union, get_type_hints

from ..context import PipelineAssemblyContext
from ..io_binding import DEFAULT_OUTPUT_NAME, NO_DEFAULT, PipelineInput, PipelineOutput, TaskOutput
from ..materializers import BaseMaterializer, DefaultMaterializer, register_materializer
from .assembled_pipeline import AssembledPipeline


class Pipeline:
    """User-facing pipeline definition.

    Wraps a plain python function whose body calls `Task` instances to
    assemble a DAG. `trace()` executes that function once, substituting a
    `PipelineInput` placeholder for every declared parameter (the graph's
    structure never depends on concrete input values, only on which task
    outputs feed which task inputs) and recording the resulting
    `AssembledTask`s and `IOBinding`s; `assemble()` hands that trace to the
    `Assembler` to validate and turn into an `AssembledPipeline`.

    A pipeline, like a `Task`, produces exactly one output: its function
    must return `None` (no output), a `TaskOutput`, or a `PipelineInput`
    (a pass-through of one of its own inputs) - never a dict or a
    `NamedTuple` of several. Fanning out several named values is the
    downstream consumer's job, not the pipeline definition's.

    Turning an `AssembledPipeline` into a backend-specific `CompiledPipeline`
    is a separate, later step - owned by one of Layer 2's backend-specific
    compilers, not by `Pipeline` itself.
    """

    def __init__(
        self,
        func: Callable,
        name: Optional[str] = None,
        default_materializer: Type[BaseMaterializer] = DefaultMaterializer,
    ):
        """Initializes the pipeline.

        Args:
            func: The plain function whose body calls `Task` instances to
                assemble a DAG.
            name: This pipeline's name. Defaults to `func.__name__` if
                omitted. Dasherized (underscores replaced with hyphens).
            default_materializer: The materializer the `Assembler` falls
                back to for any task input/output or pipeline input whose
                type none of `MATERIALIZER_REGISTRY`'s specialised
                materializers support. Defaults to `DefaultMaterializer`,
                which refuses to serialize rather than silently falling
                back to something unsafe; pass e.g. a pickle-based
                materializer here to opt into that risk explicitly, for
                this pipeline only.
        """

        self.func = func
        self.name = (name or func.__name__).replace("_", "-")
        self.signature = inspect.signature(func)
        self.type_hints = get_type_hints(func)
        self.default_materializer = default_materializer
        register_materializer(default_materializer)

    def trace(self) -> PipelineAssemblyContext:
        """Runs `func` once to record its DAG, without validating it.

        Every declared parameter is replaced with a `PipelineInput`
        placeholder, regardless of what a caller would eventually pass -
        the graph's structure never depends on concrete input values, only
        on how task calls wire together.

        Returns:
            The `PipelineAssemblyContext` populated by this trace: its
            `AssembledTask`s, `IOBinding`s, declared inputs, and output -
            not yet validated (see `Assembler`).
        """

        pipeline_inputs = {
            parameter.name: PipelineInput(
                name=parameter.name,
                default=(
                    NO_DEFAULT
                    if parameter.default is inspect.Parameter.empty
                    else parameter.default
                ),
            )
            for parameter in self.signature.parameters.values()
        }

        with PipelineAssemblyContext() as context:
            context.pipeline_inputs = pipeline_inputs
            result = self.func(**pipeline_inputs)
            context.pipeline_output = self._resolve_output(result)

        return context

    def _resolve_output(self, result: Any) -> Optional[PipelineOutput]:
        """Turns `func`'s return value into a `PipelineOutput`.

        Args:
            result: The value `func` returned while being traced.

        Returns:
            `None` if `result` is `None` (the pipeline declares no
            output); otherwise a `PipelineOutput` wrapping it.

        Raises:
            TypeError: If `result` is anything other than `None`, a
                `TaskOutput`, or a `PipelineInput`.
        """

        if result is None:
            return None

        if isinstance(result, (TaskOutput, PipelineInput)):
            return PipelineOutput(name=DEFAULT_OUTPUT_NAME, source=result)

        raise TypeError(
            f"Unsupported pipeline return value {result!r}; a pipeline must "
            "return None, a single TaskOutput, or a single PipelineInput."
        )

    def assemble(self) -> AssembledPipeline:
        """Traces this pipeline's function and hands the resulting graph to
        the `Assembler` to validate, resolve materializers for, and turn
        into a topologically ordered `AssembledPipeline`.

        Returns:
            The validated `AssembledPipeline`.

        Raises:
            AssemblyError: If the traced graph is invalid (see
                `Assembler` for the specific subclasses raised).
        """

        from ..assembler.assembler import Assembler

        context = self.trace()

        return Assembler().assemble(self, context)


def pipeline(
    func: Optional[Callable] = None,
    *,
    name: Optional[str] = None,
    assemble: bool = True,
    default_materializer: Type[BaseMaterializer] = DefaultMaterializer,
) -> Union[
    Pipeline,
    AssembledPipeline,
    Callable[[Callable], Union[Pipeline, AssembledPipeline]],
]:
    """Decorator turning a plain python function into a `Pipeline`.

    By default (`assemble=True`), the decorator immediately traces and
    assembles it, so the decorated name is bound to the resulting
    `AssembledPipeline` rather than the intermediate `Pipeline`. This is
    safe to do eagerly, at decoration time, because assembly never depends
    on concrete input values - only on the wiring between task calls.

    Pass `assemble=False` to opt out and get the lazy `Pipeline` instead,
    e.g. to keep pipeline definition side-effect-free at import time, or to
    call `.assemble()` explicitly later.

    Usage::

        @pipeline
        def my_pipeline(a: int, b: int, c: int = 3):
            ab = add(a, b)
            return add(ab, c)
        # my_pipeline is already an AssembledPipeline

        @pipeline(assemble=False)
        def my_other_pipeline(a: int, b: int):
            return add(a, b)
        # my_other_pipeline is a Pipeline; call my_other_pipeline.assemble()

    Args:
        func: The function to wrap, when used as a bare `@pipeline`.
        name: This pipeline's name, when used as `@pipeline(name=...)`.
            Defaults to the function's own name if omitted.
        assemble: Whether to trace and assemble immediately.
        default_materializer: The materializer to fall back to for any
            input/output type none of the specialised materializers
            support. See `Pipeline.__init__`.

    Returns:
        The `AssembledPipeline` (if `assemble=True`) or `Pipeline` (if
        `assemble=False`) wrapping `func`, or (when called with keyword
        arguments and no `func`) a decorator that will produce one.
    """

    def decorator(fn: Callable) -> Union[Pipeline, AssembledPipeline]:
        built_pipeline = Pipeline(
            fn, name=name, default_materializer=default_materializer
        )

        return built_pipeline.assemble() if assemble else built_pipeline

    if func is not None:
        return decorator(func)

    return decorator
