"""Errors raised while assembling a traced pipeline into an
`AssembledPipeline`.
"""

from typing import Any, Dict, Sequence

from .io_binding import PipelineInput, TaskOutput

# Lives at the top level, alongside io_binding.py, rather than under
# assembler/, because both `Task` (task/) and `Assembler` (assembler/) need
# to raise these - and assembler/ already depends on task/ (for
# AssembledTask), so task/ depending back on assembler/ for its own
# exceptions would point that dependency both ways.


class AssemblyError(Exception):
    """Base class for errors raised while assembling a traced `Pipeline`
    into an `AssembledPipeline`.
    """


class IOBindingError(AssemblyError):
    """Raised when an `IOBinding` or pipeline output references a task,
    output, or pipeline input that does not exist.
    """


class CyclicGraphError(AssemblyError):
    """Raised when the task dependency graph implied by a pipeline's
    `IOBinding`s contains a cycle.
    """


class MaterializerResolutionError(AssemblyError):
    """Raised when no `BaseMaterializer` can be resolved for a task input or
    output type.
    """


class MissingRequiredInputError(AssemblyError):
    """Raised when a `Task` is invoked, while a `Pipeline` is being traced,
    without one or more of its required inputs.
    """

    def __init__(
        self,
        task_name: str,
        missing_input_names: Sequence[str],
        provided_inputs: Dict[str, Any],
    ):
        """Initializes the error.

        Args:
            task_name: The (unique, assembled) name of the task being
                invoked.
            missing_input_names: The names of the required inputs that
                were not provided.
            provided_inputs: The inputs that *were* provided, mapping each
                name to its value (a literal, a `TaskOutput`, or a
                `PipelineInput`) - used to build a descriptive message.
        """

        self.task_name = task_name
        self.missing_input_names = tuple(missing_input_names)
        self.provided_inputs = dict(provided_inputs)

        super().__init__(self._build_message())

    def _describe(self, value: Any) -> str:
        """Describes a provided input's value for the error message.

        Args:
            value: The value to describe.

        Returns:
            A human-readable description: which task output it came from,
            which pipeline input it came from, or its literal value.
        """

        if isinstance(value, TaskOutput):
            return f"output {value.output_name!r} of task {value.task_name!r}"

        if isinstance(value, PipelineInput):
            return f"pipeline input {value.name!r}"

        return f"literal value {value!r}"

    def _build_message(self) -> str:
        """Builds this error's message from `missing_input_names` and
        `provided_inputs`.

        Returns:
            The formatted error message.
        """

        missing = ", ".join(repr(name) for name in self.missing_input_names)

        if self.provided_inputs:
            provided = ", ".join(
                f"{name}={self._describe(value)}"
                for name, value in self.provided_inputs.items()
            )
        else:
            provided = "none"

        return (
            f"Task {self.task_name!r} is missing required input(s): "
            f"{missing}. Provided input(s): {provided}."
        )
