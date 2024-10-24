from typing import List, Optional

from bettmensch_ai.pipelines.client import hera_client
from hera.workflows import Workflow
from hera.workflows.models import Workflow as WorkflowModel
from hera.workflows.models import (
    WorkflowDeleteResponse as WorkflowDeleteResponseModel,
)


class Flow(object):

    registered_flow: Workflow

    def __init__(self, registered_flow: Workflow):

        self.registered_flow = registered_flow

    @property
    def registered_name(self) -> str:
        return self.registered_flow.name

    @classmethod
    def from_workflow(cls, workflow: Workflow) -> "Flow":
        """Class method to initialize a Flow instance from a
        Workflow instance.

        Args:
            workflow (Workflow): An instance of hera's Workflow class

        Returns:
            Flow: The (registered) Flow instance.
        """

        return cls(registered_flow=workflow)


def get_flow(
    registered_name: str, registered_namespace: Optional[str] = None
) -> Flow:
    """Returns the specified Flow.

    Args:
        registered_name (str): The `name` of the Flow (equivalent to the `name`
            of its underlying Workflow).
        registered_namespace (str): The `namespace` of the Flow (equivalent to
            the `namespace` of its underlying Workflow).
    Returns:
        Flow: The Flow instance.
    """

    workflow_model: WorkflowModel = hera_client.get_workflow(
        namespace=registered_name, name=registered_namespace
    )

    workflow: Workflow = Workflow.from_dict(workflow_model.dict())

    flow: Flow = Flow.from_workflow(workflow)

    return flow


def list_flows(
    registered_namespace: Optional[str] = None,
    label_selector: Optional[str] = None,
    field_selector: Optional[str] = None,
    **kwargs,
) -> List[Flow]:
    """Lists all flows.

    Returns:
        List[Flow]: A list of all Flows that meet the query scope.
    """

    response = hera_client.list_workflows(
        namespace=registered_namespace,
        label_selector=label_selector,
        field_selector=field_selector,
        **kwargs,
    )

    if response.items is not None:
        workflows: List[Workflow] = [
            Workflow.from_dict(workflow_model.dict())
            for workflow_model in response.items
        ]

        flows: List[Flow] = [
            Flow.from_workflow(workflow) for workflow in workflows
        ]
    else:
        flows = []

    return flows


def delete_flow(
    registered_name: str, registered_namespace: Optional[str] = None, **kwargs
) -> WorkflowDeleteResponseModel:
    """Deletes the specified Flow from the server.

    Args:
        registered_name (str): The name of the Flow to delete (equivalent to
            the `name` of its underlying Workflow).
        registered_namespace (Optional[str], optional): The namespace of the
            Flow to delete (equivalent to the `name` of its underlying
            Workflow). Defaults to None.
    """

    delete_response = hera_client.delete_workflow(
        name=registered_name, namespace=registered_namespace, **kwargs
    )

    return delete_response
