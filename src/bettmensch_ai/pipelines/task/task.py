"""`Task`/`@task`: wraps a plain function into a pipeline-assemblable task."""

from __future__ import annotations

import inspect
from typing import (
    Any,
    Callable,
    Dict,
    Optional,
    Tuple,
    Union,
    get_type_hints,
    is_typeddict,
)

from ..context import get_active_context
from ..exceptions import MissingRequiredInputError
from ..io_binding import (
    DEFAULT_OUTPUT_NAME,
    IOBinding,
    PipelineInput,
    TaskInput,
    TaskOutput,
)
from .assembled_task import AssembledTask
from .decorators import (
    ResourceRequirements,
    UvRequirements,
    get_resource_requirements,
    get_uv_requirements,
)

# Reserved for per-call-site overrides (see `Task._assemble`) - a task
# function may not declare a parameter with either name, since `_assemble`
# always pops them out of the call's kwargs before binding against the
# function's own signature.
_RESERVED_PARAMETER_NAMES = frozenset({"resources", "uv"})


class ReservedTaskParameterError(TypeError):
    """Raised when a `@task` function declares a parameter named `resources`
    or `uv` - both reserved for per-call-site `resources=`/`uv=` overrides
    (see `Task._assemble`). A `TypeError` subclass (not `AssemblyError`):
    this is a decoration-time function-signature problem, the same
    category of mistake a real `TypeError` would otherwise describe, not a
    pipeline-tracing/assembly one.
    """


def _is_named_tuple_type(type_hint: Any) -> bool:
    """Reports whether `type_hint` is a `NamedTuple` class (not instance).

    Args:
        type_hint: The type to check.

    Returns:
        Whether `type_hint` is a `NamedTuple` subclass, identified the same
        way the standard library itself does: a `tuple` subclass carrying
        the `_fields` attribute every `NamedTuple` class gets.
    """

    return (
        isinstance(type_hint, type)
        and issubclass(type_hint, tuple)
        and hasattr(type_hint, "_fields")
    )


class Task:
    """User-facing task definition.

    Wraps a plain, type-annotated python function. Calling a `Task` while a
    `Pipeline` is being traced records an `AssembledTask` (plus any
    `IOBinding`s implied by its arguments) in the active
    `PipelineAssemblyContext` and returns a symbolic `TaskOutput` reference
    instead of running the function. Calling a `Task` outside of an active
    pipeline trace runs the wrapped function directly, which is useful for
    unit testing task logic in isolation.

    A task produces one output per declared name in `output_names`. For
    almost every task that is a single output, named `DEFAULT_OUTPUT_NAME`
    ("result") - whatever the function returns, unless its return type is
    annotated as a `NamedTuple` or `TypedDict`, in which case each of its
    fields/keys becomes its own independently named, independently
    materialized output instead. Nothing else about the return type
    triggers this: a plain `dict`, a plain `tuple`, or a pydantic
    `BaseModel` all remain one opaque output, exactly like an `int` would.
    """

    def __init__(self, func: Callable, name: Optional[str] = None):
        """Initializes the task.

        Args:
            func: The plain, type-annotated function to wrap.
            name: This task's base name. Defaults to `func.__name__` if
                omitted. Dasherized (underscores replaced with hyphens);
                deduplicated per pipeline by `PipelineAssemblyContext`.
        """

        self.func = func
        self.base_name = (name or func.__name__).replace("_", "-")
        self.signature = inspect.signature(func)

        reserved_collision = _RESERVED_PARAMETER_NAMES & set(self.signature.parameters)
        if reserved_collision:
            raise ReservedTaskParameterError(
                f"Task function {func.__name__!r} declares parameter(s) "
                f"{sorted(reserved_collision)!r}, which are reserved for "
                "per-call-site `resources=`/`uv=` overrides - rename the "
                "function's own parameter(s)."
            )

        self.type_hints = get_type_hints(func)

        return_type = self.type_hints.get("return")
        self.is_named_tuple_output = _is_named_tuple_type(return_type)
        self.is_typed_dict_output = is_typeddict(return_type)
        self.is_multi_output = self.is_named_tuple_output or self.is_typed_dict_output
        self.output_names = self._resolve_output_names(return_type)
        self.output_type_hints = self._resolve_output_type_hints(return_type)

        self.required_input_names = self._resolve_required_input_names()
        self.resource_requirements: ResourceRequirements = (
            get_resource_requirements(func)
        )
        self.uv_requirements: UvRequirements = get_uv_requirements(func)

    def _resolve_output_names(self, return_type: Any) -> Tuple[str, ...]:
        """Determines this task's output name(s) from its return type.

        Args:
            return_type: `self.type_hints.get("return")`.

        Returns:
            `return_type`'s field names, in declaration order, if it's a
            `NamedTuple`; its keys, in declaration order, if it's a
            `TypedDict`; otherwise the single `(DEFAULT_OUTPUT_NAME,)`.
        """

        if self.is_named_tuple_output:
            return tuple(return_type._fields)

        if self.is_typed_dict_output:
            return tuple(get_type_hints(return_type))

        return (DEFAULT_OUTPUT_NAME,)

    def _resolve_output_type_hints(self, return_type: Any) -> Dict[str, Any]:
        """Resolves the type hint for each of this task's output names.

        Args:
            return_type: `self.type_hints.get("return")`.

        Returns:
            A mapping from each name in `output_names` to its own type
            hint: each field's/key's own annotation for a `NamedTuple`/
            `TypedDict` return type, or `{DEFAULT_OUTPUT_NAME: return_type}`
            otherwise.
        """

        if self.is_multi_output:
            return get_type_hints(return_type)

        return {DEFAULT_OUTPUT_NAME: return_type}

    def _resolve_required_input_names(self) -> Tuple[str, ...]:
        """Determines which of `func`'s parameters are required.

        A parameter is required if it has no default and isn't a
        variadic (`*args`/`**kwargs`) catch-all.

        Returns:
            The names of the required parameters, in declaration order.
        """

        variadic = (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        )

        return tuple(
            parameter.name
            for parameter in self.signature.parameters.values()
            if parameter.default is inspect.Parameter.empty
            and parameter.kind not in variadic
        )

    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> Union[TaskOutput, Dict[str, TaskOutput], Any]:
        """Invokes this task.

        Args:
            *args: Positional arguments for `func`.
            **kwargs: Keyword arguments for `func`. While a pipeline is
                being traced, `resources`/`uv` are reserved: pass a
                `ResourceRequirements`/`UvRequirements` instance to override
                this call site's node-level requirements, in place of
                `func`'s own `@resource`/`@uv`-decorated defaults, for this
                node only. Outside a trace, both pass straight through to
                `func` unchanged (there's nothing to override).

        Returns:
            A symbolic reference (see `_assemble`), if called while a
            `Pipeline` is being traced; otherwise `func`'s own return
            value, run eagerly.

        Raises:
            MissingRequiredInputError: If called while tracing, and a
                required input isn't supplied.
        """

        context = get_active_context()

        if context is None:
            return self.func(*args, **kwargs)

        return self._assemble(context, args, kwargs)

    def _assemble(
        self, context, args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> Union[TaskOutput, Dict[str, TaskOutput], Any]:
        """Records this invocation as an `AssembledTask` in `context`.

        Args:
            context: The active `PipelineAssemblyContext` to record into.
            args: Positional arguments this task was called with.
            kwargs: Keyword arguments this task was called with.

        Returns:
            A `TaskOutput` referencing this invocation's (not yet computed)
            single output; or, for a multi-output task, an instance of the
            declared `NamedTuple` (fields holding a `TaskOutput` each) or a
            plain `dict` (for a `TypedDict` return type, keys holding a
            `TaskOutput` each) - so a pipeline body can reference a specific
            output the same way it would read the real return value
            (`.field`/`["key"]`), with autocomplete/type hints matching the
            declared shape even though a `TaskOutput` placeholder, not the
            real value, is what's actually there during tracing.

        Raises:
            MissingRequiredInputError: If a required input isn't supplied.
        """

        kwargs = dict(kwargs)
        resource_override = kwargs.pop("resources", None)
        uv_override = kwargs.pop("uv", None)

        bound = self.signature.bind_partial(*args, **kwargs)
        bound.apply_defaults()

        name = context.generate_unique_name(self.base_name)

        missing = [
            input_name
            for input_name in self.required_input_names
            if input_name not in bound.arguments
        ]
        if missing:
            raise MissingRequiredInputError(
                task_name=name,
                missing_input_names=missing,
                provided_inputs=dict(bound.arguments),
            )

        static_inputs: Dict[str, Any] = {}
        bindings = []

        for input_name, value in bound.arguments.items():
            if isinstance(value, (TaskOutput, PipelineInput)):
                bindings.append(
                    IOBinding(
                        target=TaskInput(task_name=name, input_name=input_name),
                        source=value,
                    )
                )
            else:
                static_inputs[input_name] = value

        assembled_task = AssembledTask(
            name=name,
            task=self,
            static_inputs=static_inputs,
            output_names=self.output_names,
            resource_override=resource_override,
            uv_override=uv_override,
        )
        context.register_task(assembled_task)

        for binding in bindings:
            context.register_binding(binding)

        return self._build_output(name)

    def _build_output(self, name: str) -> Union[TaskOutput, Dict[str, TaskOutput], Any]:
        """Builds the symbolic value returned by a task call while tracing.

        Args:
            name: This invocation's unique (assembled) task name.

        Returns:
            See `_assemble`.
        """

        task_outputs = {
            output_name: TaskOutput(task_name=name, output_name=output_name)
            for output_name in self.output_names
        }

        if self.is_named_tuple_output:
            return_type = self.type_hints["return"]
            return return_type(**task_outputs)

        if self.is_typed_dict_output:
            return task_outputs

        return task_outputs[DEFAULT_OUTPUT_NAME]


def task(
    func: Optional[Callable] = None, *, name: Optional[str] = None
) -> Union[Task, Callable[[Callable], Task]]:
    """Decorator turning a type-annotated python function into a `Task`.

    Usage::

        @task
        def add(a: int, b: int) -> int:
            return a + b

        @task(name="custom-name")
        def subtract(a: int, b: int) -> int:
            return a - b

    Args:
        func: The function to wrap, when used as a bare `@task`.
        name: This task's base name, when used as `@task(name=...)`.
            Defaults to the function's own name if omitted.

    Returns:
        The `Task` wrapping `func`, or (when called with keyword arguments
        and no `func`) a decorator that will produce one.
    """

    def decorator(fn: Callable) -> Task:
        return Task(fn, name=name)

    if func is not None:
        return decorator(func)

    return decorator
