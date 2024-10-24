import pytest
from bettmensch_ai.pipelines import delete_flow, list_flows


@pytest.mark.delete_flows
@pytest.mark.order(10)
def test_delete(test_namespace):
    """Test the pipeline.delete function"""

    flows = list_flows(registered_namespace=test_namespace)

    for flow in flows:
        delete_flow(
            registered_name=flow.registered_name,
            registered_namespace=test_namespace,
        )

    assert list_flows(registered_namespace=test_namespace) == []
