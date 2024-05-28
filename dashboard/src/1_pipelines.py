import streamlit as st

st.set_page_config(
    page_title="Pipelines",
    page_icon=":twisted_rightwards_arrows:",
    layout="wide",
)
from typing import Dict, List, Tuple

import pandas as pd
from bettmensch_ai import DagLayoutSetting, DagVisualizationSchema
from bettmensch_ai import RegisteredPipeline as Pipeline
from streamlit_flow import streamlit_flow
from streamlit_flow.interfaces import StreamlitFlowEdge, StreamlitFlowNode
from utils import (
    PIPELINE_NODE_EMOJI_MAP,
    add_logo,
    configuration,
    get_colors,
    get_workflow_templates,
)


def get_pipeline_meta_data(registered_pipelines) -> List[Dict]:
    """Retrieves the metadata field from all the ArgoWorkflowTemplate CRs
    obtained by `get_workflow_templates`.

    Args:
        registered_pipelines (List[IoArgoprojWorkflowV1alpha1WorkflowTemplate]):
            The ArgoWorkflowTemplate CRs obtained by `get_workflow_templates`

    Returns:
        List[Dict]: A list of dictionaries containing the metadata of each
            ArgoWorkflowTemplate resource.
    """

    return [
        registered_pipeline.metadata.to_dict()
        for registered_pipeline in registered_pipelines
    ]


def display_pipeline_summary_table(pipeline_meta_data) -> pd.DataFrame:
    """Generates a summary table displaying the key specs of the registered
    pipelines.

    Args:
        pipeline_meta_data (List[Dict]): The pipeline metadata generated by
            `get_pipeline_meta_data`.

    Returns:
        pd.DataFrame: The pipeline summary table shown on the frontend.
    """

    st.markdown("## Registered pipelines")

    pipeline_summary_df = (
        pd.DataFrame(pipeline_meta_data)[["name", "uid", "creation_timestamp"]]
        .rename(
            columns={
                "name": "Name",
                "uid": "ID",
                "creation_timestamp": "Created",
            }
        )
        .sort_values(by="Created", ignore_index=True)
    )

    st.dataframe(pipeline_summary_df, hide_index=True)


def get_pipeline_names(pipeline_meta_data) -> List[str]:
    """Generates a list of names of all registered pipelines.

    Args:
        pipeline_meta_data (List[Dict]): The pipeline metadata generated by
            `get_pipeline_meta_data`.

    Returns:
        List[str]: The names of all available registered pipelines.
    """

    return [resource_meta["name"] for resource_meta in pipeline_meta_data]


def get_formatted_pipeline_data(registered_pipelines, pipeline_names) -> Dict:
    """Generates structured pipeline data for easier frontend useage.

    Args:
        registered_pipelines (List[IoArgoprojWorkflowV1alpha1WorkflowTemplate]):
            The ArgoWorkflowTemplate CRs obtained by `get_workflow_templates`
        pipeline_names (List[str]): The names of pipelines generated by
            `get_pipeline_names`.

    Returns:
        Dict: The formatted pipeline data.
    """

    formatted_pipeline_data = {
        "object": {},
        "metadata": {},
        "inputs": {},
        "dag": {},
        "templates": {},
    }

    for i, (resource_name, registered_pipeline) in enumerate(
        zip(pipeline_names, registered_pipelines)
    ):
        try:
            pipeline_dict = Pipeline.from_argo_workflow_cr(
                registered_pipeline
            ).model_dump()
            formatted_pipeline_data["object"][
                resource_name
            ] = Pipeline.from_argo_workflow_cr(registered_pipeline)
            formatted_pipeline_data["metadata"][resource_name] = pipeline_dict[
                "metadata"
            ]
            formatted_pipeline_data["inputs"][resource_name] = pipeline_dict[
                "inputs"
            ]
            formatted_pipeline_data["dag"][resource_name] = pipeline_dict["dag"]
            formatted_pipeline_data["templates"][resource_name] = pipeline_dict[
                "templates"
            ]
        except Exception as e:
            raise (e)
            st.write(
                f"Oops! Could not collect data for Pipeline {resource_name}: "
                f"{e} Please make sure the workflow template was created with "
                "the bettmensch.ai SDK and was submitted successfully."
            )

    return formatted_pipeline_data


def display_pipeline_dropdown(pipeline_names) -> str:
    """Display the pipeline selection dropdown.

    Args:
        pipeline_names (List[str]): The names of pipelines generated by
            `get_pipeline_names`.

    Returns:
        str: The name of the user selected pipeline.
    """

    # display pipeline selection dropdown
    selected_pipeline = st.selectbox(
        f"Select a Pipeline:", options=pipeline_names, index=0
    )

    return selected_pipeline


def display_pipeline_dag(
    formatted_pipeline_data, selected_pipeline, display_pipeline_ios
) -> Tuple[DagVisualizationSchema, Dict]:
    """_summary_

    Args:
        formatted_pipeline_data (_type_): The formatted pipeline data generated
            by `get_formatted_pipeline_data`.
        selected_pipeline (str): The name of the user selected pipeline.
        display_pipeline_ios (bool): The toggle value of the user selected
            pipeline dag I/O detail level in the flow chart.

    Returns:
        Tuple[DagVisualizationSchema,Dict]: The specification for the streamlit
            ReactFlow visualization plugin, and the return of that plugin.
    """

    dag_visualization_schema = formatted_pipeline_data["object"][
        selected_pipeline
    ].create_dag_visualization_schema(display_pipeline_ios)

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


def display_pipeline_dag_selection(
    formatted_pipeline_data: Dict,
    selected_pipeline: str,
    dag_visualization_element,
    tab_container_height: int = 600,
):
    """Generates a tabbed, in depth view of the user selected DAG task node
    element.

    Args:
        formatted_pipeline_data (Dict): The formatted pipeline data as
            generated by `get_formatted_pipeline_data`
        selected_pipeline (str): The user selected pipeline name.
        dag_visualization_element (_type_): The user selected element from the
            DAG visuals, as returned by `display_pipeline_dag`
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
            task_inputs_tab, task_outputs_tab, task_script_tab = st.tabs(
                ["Task Inputs", "Task Outputs", "Task Script"]
            )
            pipeline = formatted_pipeline_data["object"][selected_pipeline]
            task = pipeline.get_dag_task(element_id).model_dump()

            with st.container(height=tab_container_height, border=False):
                with task_inputs_tab:

                    # build task input parameters table
                    if task["inputs"]["parameters"]:
                        task_inputs_parameters_df = pd.DataFrame(
                            task["inputs"]["parameters"]
                        )
                        task_inputs_parameters_formatted_df = pd.concat(
                            [
                                task_inputs_parameters_df.drop(
                                    ["source", "value_from"], axis=1
                                ),
                                task_inputs_parameters_df["source"].apply(
                                    pd.Series
                                ),
                            ],
                            axis=1,
                        ).rename(
                            columns={
                                "name": "Name",
                                "value": "Default",
                                "node": "Upstream Task",
                                "output_name": "Upstream Output",
                                "output_type": "Upstream Type",
                            },
                            inplace=False,
                        )
                    else:
                        task_inputs_parameters_formatted_df = pd.DataFrame()

                    # build task input artifact table
                    if task["inputs"]["artifacts"]:
                        task_inputs_artifacts_df = pd.DataFrame(
                            task["inputs"]["artifacts"]
                        )
                        task_inputs_artifacts_formatted_df = pd.concat(
                            [
                                task_inputs_artifacts_df.drop(
                                    ["source"], axis=1
                                ),
                                task_inputs_artifacts_df["source"].apply(
                                    pd.Series
                                ),
                            ],
                            axis=1,
                        ).rename(
                            columns={
                                "name": "Name",
                                "node": "Upstream Task",
                                "output_name": "Upstream Output",
                                "output_type": "Upstream Type",
                            },
                            inplace=False,
                        )
                    else:
                        task_inputs_artifacts_formatted_df = pd.DataFrame()

                    st.write("Parameters")
                    st.dataframe(
                        task_inputs_parameters_formatted_df, hide_index=True
                    )
                    st.write("Artifacts")
                    st.dataframe(
                        task_inputs_artifacts_formatted_df, hide_index=True
                    )

            with st.container(height=tab_container_height, border=False):
                with task_outputs_tab:
                    # build task output parameters table
                    if task["outputs"]["parameters"]:
                        task_outputs_parameters_df = pd.DataFrame(
                            task["outputs"]["parameters"]
                        )
                        task_outputs_parameters_formatted_df = (
                            task_outputs_parameters_df.drop(
                                "value_from", axis=1
                            ).rename(
                                columns={
                                    "name": "Name",
                                },
                                inplace=False,
                            )
                        )
                    else:
                        task_outputs_parameters_formatted_df = pd.DataFrame()

                    # build task output artifact table
                    if task["outputs"]["artifacts"]:
                        task_outputs_artifacts_df = pd.DataFrame(
                            task["outputs"]["artifacts"]
                        )
                        task_outputs_artifacts_formatted_df = (
                            task_outputs_artifacts_df.drop(
                                "path", axis=1
                            ).rename(
                                columns={
                                    "name": "Name",
                                },
                                inplace=False,
                            )
                        )
                    else:
                        task_outputs_artifacts_formatted_df = pd.DataFrame()

                    st.write("Parameters")
                    st.dataframe(
                        task_outputs_parameters_formatted_df, hide_index=True
                    )
                    st.write("Artifacts")
                    st.dataframe(
                        task_outputs_artifacts_formatted_df, hide_index=True
                    )

            with st.container(height=tab_container_height, border=False):
                with task_script_tab:
                    st.json(
                        pipeline.get_template(task["template"]).model_dump()[
                            "script"
                        ]
                    )
        else:
            st.markdown(f"### {PIPELINE_NODE_EMOJI_MAP['task']} Task: None")
            st.write(
                "Select a task by clicking on the corresponding "
                f"{PIPELINE_NODE_EMOJI_MAP['task']} node."
            )

    except TypeError as e:
        st.markdown(f"### {PIPELINE_NODE_EMOJI_MAP['task']} Task: None")
        st.write(
            "Select a task by clicking on the corresponding "
            f"{PIPELINE_NODE_EMOJI_MAP['task']} node."
        )


def display_selected_pipeline(
    formatted_pipeline_data,
    selected_pipeline,
    chart_container_height: int = 550,
    col_container_height: int = 650,
    tab_container_height: int = 600,
):
    """Utility to display DAG flow chart and all relevant specs in tabbed
    layout for a user selected pipeline.

    Args:
        formatted_pipeline_data (Dict): The formatted pipeline data as obtained
            by `get_formatted_pipeline_data`.
        selected_pipeline (str): The name of the user selected pipeline.
    """

    with st.container(height=chart_container_height):
        display_pipeline_ios = st.toggle(f"Display pipeline & task I/O")
        (
            dag_visualization_schema,
            dag_visualization_element,
        ) = display_pipeline_dag(
            formatted_pipeline_data, selected_pipeline, display_pipeline_ios
        )

    pipeline_col, task_col = st.columns(2)

    # display pipeline level data
    with pipeline_col:
        with st.container(height=col_container_height):
            st.markdown(
                "### :twisted_rightwards_arrows: Pipeline: `"
                f"{selected_pipeline}`"
            )

            tab_inputs, tab_metadata, tab_dag, tab_templates = st.tabs(
                [
                    "Pipeline Inputs",
                    "Pipeline Meta Data",
                    "Pipeline DAG",
                    "Pipeline Templates",
                ]
            )

            with tab_inputs:
                with st.container(height=tab_container_height, border=False):
                    pipeline_inputs = formatted_pipeline_data["inputs"][
                        selected_pipeline
                    ]
                    pipeline_inputs_formatted_df = pd.DataFrame(
                        pipeline_inputs
                    ).rename(
                        columns={
                            "name": "Name",
                            "value": "Default",
                        }
                    )
                    st.dataframe(pipeline_inputs_formatted_df, hide_index=True)

            with tab_metadata:
                with st.container(height=tab_container_height, border=False):
                    st.markdown("### Spec")
                    st.json(
                        formatted_pipeline_data["metadata"][selected_pipeline],
                        expanded=True,
                    )

            with tab_dag:
                with st.container(height=tab_container_height, border=False):
                    st.markdown("### Spec")
                    st.json(
                        formatted_pipeline_data["dag"][selected_pipeline],
                        expanded=True,
                    )

            with tab_templates:
                with st.container(height=tab_container_height, border=False):
                    st.markdown("### Spec")
                    st.json(
                        formatted_pipeline_data["templates"][selected_pipeline],
                        expanded=True,
                    )

    # display task level data
    with task_col:
        with st.container(height=col_container_height):
            display_pipeline_dag_selection(
                formatted_pipeline_data,
                selected_pipeline,
                dag_visualization_element,
                tab_container_height,
            )

    return dag_visualization_schema, dag_visualization_element


def main():
    """Utility function to render the pipeline resources."""

    st.markdown(
        """
        # :twisted_rightwards_arrows: Pipelines
        
        A `Pipeline` is the *definition* of a workflow, i.e. it describes a DAG declaring the logic and dependencies that will be executd at runtime.
        """
    )

    workflow_templates = get_workflow_templates(configuration)

    meta_data = get_pipeline_meta_data(workflow_templates)
    display_pipeline_summary_table(meta_data)

    names = get_pipeline_names(meta_data)

    formatted_pipeline_data = get_formatted_pipeline_data(
        workflow_templates, names
    )

    selected_pipeline = display_pipeline_dropdown(names)

    (
        dag_visualization_schema,
        dag_visualization_element,
    ) = display_selected_pipeline(formatted_pipeline_data, selected_pipeline)

    with st.sidebar:
        add_logo(sidebar=True)


main()
