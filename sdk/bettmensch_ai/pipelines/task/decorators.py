"""`@resource`/`@uv`: attach runtime requirements to a task function."""

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence, Tuple, TypeVar, Union

F = TypeVar("F", bound=Callable)

_RESOURCE_REQUIREMENTS_ATTR = "__bettmensch_ai_resource_requirements__"
_UV_REQUIREMENTS_ATTR = "__bettmensch_ai_uv_requirements__"


@dataclass(frozen=True)
class ResourceRequirements:
    """Memory, cpu and gpu requirements for a task's runtime, as attached by
    the `@resource` decorator.
    """

    cpu: Optional[Union[str, float, int]] = None
    memory: Optional[str] = None
    gpu: Optional[int] = None


@dataclass(frozen=True)
class UvRequirements:
    """uv-managed dependencies to be made available in a task's runtime, as
    attached by the `@uv` decorator.
    """

    packages: Tuple[str, ...] = field(default_factory=tuple)
    python: Optional[str] = None


def resource(
    cpu: Optional[Union[str, float, int]] = None,
    memory: Optional[str] = None,
    gpu: Optional[int] = None,
) -> Callable[[F], F]:
    """Attaches `ResourceRequirements` to a task function.

    Must be stacked underneath `@task`, e.g.::

        @task
        @resource(cpu="1", memory="1Gi")
        def my_task(...): ...

    Args:
        cpu: The cpu request/limit, e.g. `"1"` or `0.5`.
        memory: The memory request/limit, e.g. `"1Gi"`.
        gpu: The number of gpus required.

    Returns:
        A decorator that attaches the resulting `ResourceRequirements` to
        the function it's applied to.
    """

    def decorator(func: F) -> F:
        setattr(
            func,
            _RESOURCE_REQUIREMENTS_ATTR,
            ResourceRequirements(cpu=cpu, memory=memory, gpu=gpu),
        )

        return func

    return decorator


def uv(
    packages: Sequence[str] = (), python: Optional[str] = None
) -> Callable[[F], F]:
    """Attaches `UvRequirements` to a task function, declaring the
    uv-managed dependencies that must be available in the task's runtime.

    Must be stacked underneath `@task`, e.g.::

        @task
        @uv(["numpy==2.0.0"])
        def my_task(...): ...

    Args:
        packages: The uv-managed package requirements, e.g.
            `["numpy==2.0.0"]`.
        python: The python version to run the task under, if it must
            differ from the default.

    Returns:
        A decorator that attaches the resulting `UvRequirements` to the
        function it's applied to.
    """

    def decorator(func: F) -> F:
        setattr(
            func,
            _UV_REQUIREMENTS_ATTR,
            UvRequirements(packages=tuple(packages), python=python),
        )

        return func

    return decorator


def get_resource_requirements(func: Callable) -> ResourceRequirements:
    """Reads the `ResourceRequirements` a `@resource` decorator attached to
    `func`.

    Args:
        func: The function to read requirements from.

    Returns:
        The attached `ResourceRequirements`, or a default (empty) instance
        if `@resource` was never applied.
    """

    return getattr(func, _RESOURCE_REQUIREMENTS_ATTR, ResourceRequirements())


def get_uv_requirements(func: Callable) -> UvRequirements:
    """Reads the `UvRequirements` a `@uv` decorator attached to `func`.

    Args:
        func: The function to read requirements from.

    Returns:
        The attached `UvRequirements`, or a default (empty) instance if
        `@uv` was never applied.
    """

    return getattr(func, _UV_REQUIREMENTS_ATTR, UvRequirements())
