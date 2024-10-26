import pytest
from bettmensch_ai.pipelines import delete_flow, list_flows


@pytest.mark.standard
@pytest.mark.ddp
@pytest.mark.delete_flows
@pytest.mark.order(10)
def test_delete(test_namespace):
    """Test the delete_flow function"""

    flows = list_flows(registered_namespace=test_namespace)

    for flow in flows:
        if flow.registered_name.startswith("pipeline-test-"):
            delete_flow(
                registered_name=flow.registered_name,
                registered_namespace=test_namespace,
            )

    test_flows = [
        flow
        for flow in list_flows(registered_namespace=test_namespace)
        if flow.registered_name.startswith("pipeline-test-")
    ]

    assert test_flows == []
