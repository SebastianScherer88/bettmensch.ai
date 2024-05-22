import streamlit as st

st.set_page_config(
    page_title="Flows", page_icon=":arrow_forward:", layout="wide"
)
from typing import Dict, List, Tuple

import pandas as pd
from bettmensch_ai import DagLayoutSetting, DagVisualizationSchema
from bettmensch_ai import RegisteredFlow as Flow
from streamlit_flow import streamlit_flow
from streamlit_flow.interfaces import StreamlitFlowEdge, StreamlitFlowNode
from utils import add_logo  # , FLOW_NODE_EMOJI_MAP
from utils import (
    PIPELINE_NODE_EMOJI_MAP,
    configuration,
    get_colors,
    get_workflows,
)


def get_flow_meta_data(submitted_flows) -> List[Dict]:
    """Retrieves the metadata field from all the ArgoWorkflow CRs obtained by `get_workflows`.

    Args:
        submitted_flows (List[IoArgoprojWorkflowV1alpha1Workflow]): The ArgoWorkflow CRs obtained by `get_workflows`.

    Returns:
        List[Dict]: A list of dictionaries containing the metadata of each ArgoWorkflowTemplate resource.
    """

    return [
        submitted_flow.metadata.to_dict() for submitted_flow in submitted_flows
    ]


def display_flow_summary_table(formatted_flow_data: Dict) -> pd.DataFrame:
    """Generates a summary table displaying the key specs of the submitted flows.

    Args:
        flow_meta_data (List[Dict]): The formatted flow data obtained by `get_formatted_flow_data`.

    Returns:
        pd.DataFrame: The flow summary table shown on the frontend.
    """

    st.markdown("## Submitted flows")

    records = []

    for flow_name in formatted_flow_data["metadata"]:
        records.append(
            dict(
                **formatted_flow_data["state"][flow_name],
                **formatted_flow_data["metadata"][flow_name],
            )
        )

    flow_summary_df = (
        pd.DataFrame(records)[
            ["name", "uid", "creation_timestamp", "progress", "phase"]
        ]
        .rename(
            columns={
                "name": "Name",
                "uid": "ID",
                "creation_timestamp": "Created",
                "progress": "Progress",
                "phase": "Phase",
            }
        )
        .sort_values(by="Created", ignore_index=True)
    )

    st.dataframe(flow_summary_df, hide_index=True)


def get_flow_names(flow_meta_data) -> List[str]:
    """Generates a list of names of all submitted flows.

    Args:
        flow_meta_data (List[Dict]): The flow metadata generated by `get_flow_meta_data`.

    Returns:
        List[str]: The names of all available submitted flows.
    """

    return [resource_meta["name"] for resource_meta in flow_meta_data]


def get_formatted_flow_data(submitted_flows, flow_names) -> Dict:
    """Generates structured flow data for easier frontend useage.

    Args:
        submitted_flows (List[IoArgoprojWorkflowV1alpha1Workflow]): The ArgoWorkflow CRs
            obtained by `get_workflows`
        flow_names (List[str]): The names of flow generated by `get_flow_names`.

    Returns:
        Dict: The formatted flow data.
    """

    formatted_flow_data = {
        "object": {},
        "metadata": {},
        "state": {},
        "inputs": {},
        "dag": {},
        "templates": {},
    }

    for i, (resource_name, submitted_flow) in enumerate(
        zip(flow_names, submitted_flows)
    ):
        try:
            flow_dict = Flow.from_argo_workflow_cr(submitted_flow).model_dump()
            formatted_flow_data["object"][
                resource_name
            ] = Flow.from_argo_workflow_cr(submitted_flow)
            formatted_flow_data["metadata"][resource_name] = flow_dict[
                "metadata"
            ]
            formatted_flow_data["state"][resource_name] = flow_dict["state"]
            # formatted_flow_data['artifact_configuration'][resource_name] = flow_dict['artifact_configuration']
            formatted_flow_data["inputs"][resource_name] = flow_dict["inputs"]
            formatted_flow_data["dag"][resource_name] = flow_dict["dag"]
            formatted_flow_data["templates"][resource_name] = flow_dict[
                "templates"
            ]
        except Exception as e:
            print(e)
            st.write(
                f"Oops! Could not collect data for Flow {resource_name}: {e} Please make sure the Argo Workflow was created with the bettmensch.ai SDK and was submitted successfully."
            )

    return formatted_flow_data


def display_flow_dropdown(flow_names) -> str:
    """Display the flow selection dropdown.

    Args:
        flow_names (List[str]): The names of flow generated by `get_flow_names`.

    Returns:
        str: The name of the user selected flow.
    """

    # display flow selection dropdown
    selected_flow = st.selectbox(f"Select a Flow:", options=flow_names, index=0)

    return selected_flow


def display_flow_dag(
    formatted_flow_data, selected_flow, display_flow_ios
) -> Tuple[DagVisualizationSchema, Dict]:
    """_summary_

    Args:
        formatted_flow_data (_type_): The formatted flow data generated by `get_formatted_flow_data`.
        selected_flow (str): The name of the user selected flow.
        display_flow_ios (bool): The toggle value of the user selected flow dag I/O detail level in the flow chart.

    Returns:
        Tuple[DagVisualizationSchema,Dict]: The specification for the streamlit ReactFlow visualization plugin, and the return of that plugin.
    """

    dag_visualization_schema = formatted_flow_data["object"][
        selected_flow
    ].create_dag_visualization_schema(display_flow_ios)

    dag_visualization_element = streamlit_flow(
        nodes=[
            StreamlitFlowNode(**node.model_dump())
            for node in dag_visualization_schema.nodes
        ],
        edges=[
            StreamlitFlowEdge(**connection.model_dump())
            for connection in dag_visualization_schema.connections
        ],
        **DagLayoutSetting(
            style={
                "backgroundColor": get_colors("custom").secondaryBackgroundColor
            }
        ).model_dump(),
    )

    return dag_visualization_schema, dag_visualization_element


def display_flow_dag_selection(
    formatted_flow_data: Dict,
    selected_flow: str,
    dag_visualization_element,
    tab_container_height: int = 600,
):
    """Generates a tabbed, in depth view of the user selected DAG task node element.

    Args:
        formatted_flow_data (Dict): The formatted flow data as generated by `get_formatted_flow_data`
        selected_flow (str): The user selected flow name.
        dag_visualization_element (_type_): The user selected element from the DAG visuals, as returned by `display_flow_dag`
    """

    try:
        element_type = dag_visualization_element["elementType"]
        element_id = dag_visualization_element["id"]
        element_is_task_node = (element_type == "node") and (
            len(element_id.split("_")) == 1
        )

        if element_is_task_node:
            st.markdown(
                f"### {PIPELINE_NODE_EMOJI_MAP['task']} Task: `{element_id}`"
            )
            (
                task_inputs_tab,
                task_outputs_tab,
                task_script_tab,
                task_state_tab,
            ) = st.tabs(
                ["Task Inputs", "Task Outputs", "Task Script", "Task State"]
            )
            flow = formatted_flow_data["object"][selected_flow]
            task = flow.get_dag_task(element_id).model_dump()

            with st.container(height=tab_container_height, border=False):
                with task_inputs_tab:
                    task_inputs = (
                        task["inputs"]["parameters"]
                        + task["inputs"]["artifacts"]
                    )
                    task_inputs_df = pd.DataFrame(task_inputs)
                    task_inputs_formatted_df = pd.concat(
                        [
                            task_inputs_df.drop(["source"], axis=1),
                            task_inputs_df["source"].apply(pd.Series),
                        ],
                        axis=1,
                    ).rename(
                        columns={
                            "name": "Name",
                            "value": "Value",
                            "value_from": "From",
                            "node": "Upstream Task",
                            "output_name": "Upstream Output",
                            "output_type": "Upstream Type",
                        },
                        inplace=False,
                    )
                    st.dataframe(task_inputs_formatted_df, hide_index=True)

            with st.container(height=tab_container_height, border=False):
                with task_outputs_tab:
                    task_outputs = (
                        task["outputs"]["parameters"]
                        + task["outputs"]["artifacts"]
                    )
                    task_outputs_formatted_df = pd.DataFrame(
                        task_outputs
                    ).rename(
                        columns={
                            "name": "Name",
                            "value": "Value",
                            "value_from": "From",
                        },
                        inplace=False,
                    )
                    st.dataframe(task_outputs_formatted_df, hide_index=True)

            with st.container(height=tab_container_height, border=False):
                with task_script_tab:
                    st.json(
                        flow.get_template(task["template"]).model_dump()[
                            "script"
                        ]
                    )

            with st.container(height=tab_container_height, border=False):
                with task_state_tab:
                    task_state = dict(
                        [
                            (k, v)
                            for k, v in task.items()
                            if k
                            in (
                                "id",
                                "pod_name",
                                "host_node_name",
                                "phase",
                                "logs",
                            )
                        ]
                    )
                    task_state_formatted_df = (
                        pd.DataFrame(task_state, index=[0])
                        .rename(
                            columns={
                                "id": "Flow Task ID",
                                "pod_name": "Pod",
                                "host_node_name": "Host Node",
                                "phase": "Phase",
                                "logs": "Logs",
                            },
                            inplace=False,
                        )
                        .T
                    )
                    st.dataframe(task_state_formatted_df, hide_index=False)

        else:
            st.markdown(f"### {PIPELINE_NODE_EMOJI_MAP['task']} Task: None")
            st.write(
                f"Select a task by clicking on the corresponding {PIPELINE_NODE_EMOJI_MAP['task']} node."
            )

    except TypeError as e:
        st.markdown(f"### {PIPELINE_NODE_EMOJI_MAP['task']} Task: None")
        st.write(
            f"Select a task by clicking on the corresponding {PIPELINE_NODE_EMOJI_MAP['task']} node."
        )


def display_selected_flow(
    formatted_flow_data,
    selected_flow,
    chart_container_height: int = 700,
    col_container_height: int = 750,
    tab_container_height: int = 600,
):
    """Utility to display DAG flow chart and all relevant specs in tabbed layout for a user selected flow.

    Args:
        formatted_flow_data (Dict): The formatted flow data as obtained by `get_formatted_flow_data`.
        selected_flow (str): The name of the user selected flow.
    """

    with st.container(height=chart_container_height):
        display_pipeline_ios = st.toggle(f"Display flow & task I/O")
        dag_visualization_schema, dag_visualization_element = display_flow_dag(
            formatted_flow_data, selected_flow, display_pipeline_ios
        )

    flow_col, task_col = st.columns(2)

    # display flow level data
    with flow_col:
        with st.container(height=col_container_height):
            st.markdown(f"### :arrow_forward: Flow: `{selected_flow}`")

            tab_inputs, tab_metadata, tab_dag, tab_templates = st.tabs(
                [
                    "Flow Inputs",
                    "Flow Meta Data",
                    "Flow DAG",
                    "Flow Templates",
                ]
            )

            with tab_inputs:
                with st.container(height=tab_container_height, border=False):
                    flow_inputs = formatted_flow_data["inputs"][selected_flow]
                    flow_inputs_formatted_df = pd.DataFrame(flow_inputs).rename(
                        columns={
                            "name": "Name",
                            "value": "Default",
                        }
                    )
                    st.dataframe(flow_inputs_formatted_df, hide_index=True)

            with tab_metadata:
                with st.container(height=tab_container_height, border=False):
                    st.markdown("### Spec")
                    st.json(
                        formatted_flow_data["metadata"][selected_flow],
                        expanded=True,
                    )

            with tab_dag:
                with st.container(height=tab_container_height, border=False):
                    st.markdown("### Spec")
                    st.json(
                        formatted_flow_data["dag"][selected_flow], expanded=True
                    )

            with tab_templates:
                with st.container(height=tab_container_height, border=False):
                    st.markdown("### Spec")
                    st.json(
                        formatted_flow_data["templates"][selected_flow],
                        expanded=True,
                    )

    # display task level data
    with task_col:
        with st.container(height=col_container_height):
            display_flow_dag_selection(
                formatted_flow_data,
                selected_flow,
                dag_visualization_element,
                tab_container_height,
            )

    return dag_visualization_schema, dag_visualization_element


def main():
    """Utility function to render the pipeline resources."""

    st.markdown(
        """
        # :arrow_forward: Flows
        
        A `Flow` is the *execution* of a workflow, i.e. it is the running of a `Pipeline`.
        """
    )

    workflows = get_workflows(configuration)

    meta_data = get_flow_meta_data(workflows)
    names = get_flow_names(meta_data)
    formatted_flow_data = get_formatted_flow_data(workflows, names)

    display_flow_summary_table(formatted_flow_data)

    selected_flow = display_flow_dropdown(names)

    dag_visualization_schema, dag_visualization_element = display_selected_flow(
        formatted_flow_data, selected_flow
    )

    with st.sidebar:
        add_logo(sidebar=True)


main()
