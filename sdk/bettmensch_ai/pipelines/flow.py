from typing import Dict, List, Optional

from bettmensch_ai.constants import ARGO_NAMESPACE, FLOW_PHASE
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
        """The unique name of the flow (= argo Workflow)

        Returns:
            str: The name of the flow
        """
        return self.registered_flow.name

    @property
    def pipeline(self) -> str:
        """The unique name of the pipeline (= argo WorkflowTemplate) that the
            flow originates from.

        Returns:
            str: The name of the parent pipeline
        """
        return self.registered_flow.workflow_template_ref.name

    @property
    def phase(self) -> str:
        """The current phase of the flow (= argo Workflow)

        Returns:
            str: The phase of the flow
        """

        return self.registered_flow.status.phase

    @property
    def started_at(self) -> str:
        """The time the flow started (where applicable). Returns None if not
        started yet.

        E.g. "2024-11-05T13:04:19Z"

        Returns:
            str: The phase of the flow
        """

        return self.registered_flow.status.started_at

    @property
    def finished_at(self) -> str:
        """The time the flow finished (where applicable). Returns None if not
        finished yet.

        E.g. "2024-11-05T13:04:19Z"

        Returns:
            str: The phase of the flow
        """

        return self.registered_flow.status.finished_at

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
    registered_name: str, registered_namespace: str = ARGO_NAMESPACE
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
    registered_namespace: str = ARGO_NAMESPACE,
    pipeline: Optional[str] = None,
    phase: Optional[str] = None,
    labels: Dict = {},
    **kwargs,
) -> List[Flow]:
    """List[Flow]: A list of all Flows that meet the query scope.

    Args:
        registered_namespace (Optional[str], optional): The name in which the
            underlying argo Workflow lives. Defaults to ARGO_NAMESPACE.
        pipeline (Optional[str], optional): Optional filter to only consider
            Flows that originate from the specified pipeline. Defaults to None,
            i.e. no pipeline-based filtering.
        phase (Optional[str], optional): Optional filter to only consider Flows
            that are in the specified phase. Defaults to None, i.e. no phase-
            based filtering.
        labels (Dict, optional): Optional filter to only consider Flows whose
            underlying argo Workflow resource contains all of the specified
            labels. Defaults to {}, i.e. no label-based filtering.

    Returns:
        List[Flow]: A list of Flows that meet the filtering specifications.
    """

    # build field selector
    if pipeline is None and phase is None:
        field_selector = None
    else:
        field_selectors = []

        if pipeline is not None:
            field_selectors.append(f"spec.workflowTemplateRef.name={pipeline}")

        if phase is not None:
            assert phase in (
                FLOW_PHASE.error,
                FLOW_PHASE.failed,
                FLOW_PHASE.pending,
                FLOW_PHASE.running,
                FLOW_PHASE.succeeded,
                FLOW_PHASE.unknown,
            ), f"Invalid phase spec: {phase}. Must be one of the constants.FLOW_PHASE levels."  # noqa: E501
            field_selectors.append(f"status.phase={phase}")

        field_selector = ",".join(field_selectors)

    # build label selector
    if labels is not None:
        label_selector = None
    else:
        kv_label_list = list(labels.items())  # [('a',1),('b',2)]
        label_selector = ",".join(
            [f"{k}={v}" for k, v in kv_label_list]
        )  # "a=1,b=2"

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
    registered_name: str, registered_namespace: str = ARGO_NAMESPACE, **kwargs
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
