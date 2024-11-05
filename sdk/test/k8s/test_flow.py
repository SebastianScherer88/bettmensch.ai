import pytest
from bettmensch_ai.pipelines import Flow, delete_flow, get_flow, list_flows


@pytest.mark.standard
@pytest.mark.ddp
@pytest.mark.order(10)
def test_get_standard_flow(test_namespace):
    flows = list_flows(registered_namespace=test_namespace)

    for flow in flows:
        flow_reloaded = get_flow(
            registered_name=flow.registered_name,
            registered_namespace=test_namespace,
        )
        assert isinstance(flow_reloaded, Flow)


@pytest.mark.standard
@pytest.mark.ddp
@pytest.mark.delete_flows
@pytest.mark.order(11)
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
