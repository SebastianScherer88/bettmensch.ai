from typing import List, Union

# from bettmensch_ai.pipelines.constants import ResourceType


class PipelineContext(object):
    """Globally accessible pipeline meta data storage utility."""

    _active: bool = False
    components: List[
        Union["Component", "TorchDDPComponent"]  # noqa: F821
    ] = []
    outputs: List[
        Union["OutputParameter", "OutputArtifact"]  # noqa: F821
    ] = []

    @property
    def active(self):
        return self._active

    def activate(self):
        self._active = True

    def deactivate(self):
        self._active = False

    def get_component_name_counter(self, component_base_name: str) -> int:
        """Utility to get the counter for the name of a potential component to
        ensure uniqueness of identifier in the PipelineContext.

        Args:
            component_base_name (str): The `_base_name` attribute value of a
                Component instance.

        Returns:
            int: The unique counter of that Component w.r.t. the global
                PipelineContext.
        """

        counter = len(
            [
                component
                for component in self.components
                if component_base_name in component.name
            ]
        )

        return counter

    def add_component(
        self, component: Union["Component", "TorchDDPComponent"]  # noqa: F821
    ):
        """Adds the specified Component instance to the global PipelineContext.

        Args:
            component (BaseComponent): The Component instance that will be
                added.

        Raises:
            Exception: Raised if this method was not called within an active
                PipelineContext.
        """

        if self.active:
            component_counter = self.get_component_name_counter(
                component.base_name
            )
            component.name = component.generate_name(component_counter)
            self.components.append(component)
        else:
            raise Exception(
                f"Unable to add component {component.base_name} - pipeline "
                "context is not active."
            )

    def validate_output(
        self, output: Union["OutputParameter", "OutputArtifact"]  # noqa: F821
    ):
        """Utility to validate the suggested output about to be added to the
        global pipeline context.

        Args:
            output (Union[OutputParameter, OutputArtifact]): The output about
            to be added.
        """

        # validate global uniqueness of pipeline output
        present_names = [output.name for output in self.outputs]
        assert output.name not in present_names, (
            f"Output with name {output.name} already present in global"
            " pipeline context."
        )

    def add_output(
        self, output: Union["OutputParameter", "OutputArtifact"]  # noqa: F821
    ):
        """Addes the specified output instance to the global PipelineContext

        Args:
            output (Union[OutputParameter, OutputArtifact]): A Pipeline owned
                output instance that will be added.
        """

        if self.active:
            self.validate_output(output)
            self.outputs.append(output)
        else:
            raise Exception(
                f"Unable to add output {output.name}: {output} - pipeline"
                " context is not active."
            )

    def clear(self):
        """Removes all components from the active PipelineContext. Useful when
        defining a (new) Pipeline and you want to ensure a clean slate.

        Raises:
            Exception: Raised if this method was not called within an active
                PipelineContext.
        """
        if self.active:
            self.components = []
            self.outputs = []
        else:
            raise Exception(
                "Unable to clear components and outputs from context -"
                " pipeline context is not active."
            )

    def __enter__(self):
        self._active = True

        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):

        pass


_pipeline_context = PipelineContext()
