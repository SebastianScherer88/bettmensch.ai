from typing import Dict, List, Optional

from bettmensch_ai.pipelines.constants import (
    ARGO_NAMESPACE,
    FLOW_LABEL,
    FLOW_PHASE,
)
from bettmensch_ai.pipelines.pipeline.client import hera_client
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
        """The unique name of the Flow (= argo Workflow)

        Returns:
            str: The name of the Flow
        """
        return self.registered_flow.name

    @property
    def registered_namespace(self) -> str:
        """The namespace of the Flow (= argo Workflow)

        Returns:
            str: The namespace of the Flow
        """

        return self.registered_flow.namespace

    @property
    def registered_pipeline(self) -> str:
        """The unique name of the registered pipeline (= argo WorkflowTemplate)
        that the Flow originates from.

        Returns:
            str: The name of the parent Pipeline
        """
        return self.registered_flow.workflow_template_ref.name

    @property
    def phase(self) -> str:
        """The current phase of the Flow (= argo Workflow)

        Returns:
            str: The phase of the Flow
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
    def from_workflow_model(cls, workflow_model: WorkflowModel) -> "Flow":
        """Class method to initialize a Flow instance from a WorkflowModel
        instance.

        Args:
            workflow_model (WorkflowModel): An instance of hera's WorkflowModel
                class

        Returns:
            Flow: The (registered) Flow instance.
        """

        workflow: Workflow = Workflow.from_dict(workflow_model.dict())

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
        namespace=registered_namespace, name=registered_name
    )

    flow: Flow = Flow.from_workflow_model(workflow_model)

    return flow


def list_flows(
    registered_namespace: str = ARGO_NAMESPACE,
    registered_pipeline_name: Optional[str] = None,
    phase: Optional[str] = None,
    labels: Dict = {},
    **kwargs,
) -> List[Flow]:
    """Get all flows that meet the query specifications.

    Args:
        registered_namespace (Optional[str], optional): The namespace in which
            the underlying argo Workflow lives. Defaults to ARGO_NAMESPACE.
        registered_pipeline_name (Optional[str], optional): Optional filter to
            only consider Flows originating from the specified registered
            Pipeline. Defaults to None, i.e. no pipeline-based filtering.
        phase (Optional[str], optional): Optional filter to only consider Flows
            that are in the specified phase. Defaults to None, i.e. no phase-
            based filtering.
        labels (Dict, optional): Optional filter to only consider Flows whose
            underlying argo Workflow resource contains all of the specified
            labels. Defaults to {}, i.e. no label-based filtering.

    Returns:
        List[Flow]: A list of Flows that meet the filtering specifications.
    """

    # build label selector
    if (not labels) and (phase is None) and (registered_pipeline_name is None):
        label_selector = None
    else:
        all_labels = labels.copy()

        # add phase label
        if phase is not None:
            assert phase in (
                FLOW_PHASE.error.value,
                FLOW_PHASE.failed.value,
                FLOW_PHASE.pending.value,
                FLOW_PHASE.running.value,
                FLOW_PHASE.succeeded.value,
                FLOW_PHASE.unknown.value,
            ), f"Invalid phase spec: {phase}. Must be one of the constants.FLOW_PHASE levels."  # noqa: E501
            all_labels.update({FLOW_LABEL.phase.value: phase})

        # add pipeline identifier label
        if registered_pipeline_name is not None:
            all_labels.update(
                {
                    FLOW_LABEL.pipeline_name.value: registered_pipeline_name,
                }
            )

        kv_label_list = list(all_labels.items())  # [('a',1),('b',2)]
        label_selector = ",".join(
            [f"{k}={v}" for k, v in kv_label_list]
        )  # "a=1,b=2"

    response = hera_client.list_workflows(
        namespace=registered_namespace,
        label_selector=label_selector,
        **kwargs,
    )

    if response.items is not None:
        flows: List[Flow] = [
            Flow.from_workflow_model(workflow_model)
            for workflow_model in response.items
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
