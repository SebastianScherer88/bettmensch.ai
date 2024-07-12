from typing import List


class PipelineContext(object):
    """Globally accessible pipeline meta data storage utility."""

    _active: bool = False
    components: List = []

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

    def add_component(self, component):
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

    def clear(self):
        """Removes all components from the active PipelineContext. Useful when
        defining a (new) Pipeline and you want to ensure a clean slate.

        Raises:
            Exception: Raised if this method was not called within an active
                PipelineContext.
        """
        if self.active:
            self.components = []
        else:
            raise Exception(
                "Unable to clear components from context - pipeline context "
                "is not active."
            )

    def __enter__(self):
        self._active = True

        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):

        pass


_pipeline_context = PipelineContext()
