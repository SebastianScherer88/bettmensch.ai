import streamlit as st
st.set_page_config(
    page_title="Pipelines",
    page_icon=":twisted_rightwards_arrows:",
    layout='wide'
)
from streamlit_flow import streamlit_flow
from streamlit_flow.interfaces import StreamlitFlowNode, StreamlitFlowEdge
from pipeline import Pipeline
from utils import add_logo, get_colors, configuration, get_workflow_templates, get_workflows
from typing import Dict, List
import pandas as pd

def get_pipeline_meta_data(registered_pipelines) -> List[Dict]:
    """Retrieves the metadata field from all the ArgoWorkflowTemplate CRs obtained by `get_workflow_templates`.

    Args:
        registered_pipelines (List[IoArgoprojWorkflowV1alpha1WorkflowTemplate]): The ArgoWorkflowTemplate CRs 
            obtained by `get_workflow_templates`

    Returns:
        List[Dict]: A list of dictionaries containing the metadata of each ArgoWorkflowTemplate resource.
    """
    
    return [registered_pipeline.metadata.to_dict() for registered_pipeline in registered_pipelines]

def display_pipeline_summary_table(pipeline_meta_data) -> pd.DataFrame:
    """Generates a summary table displaying the key specs of the registered pipelines.

    Args:
        pipeline_meta_data (List[Dict]): The pipeline metadata generated by `get_pipeline_meta_data`.

    Returns:
        pd.DataFrame: The pipeline summary table shown on the frontend.
    """
    
    st.markdown("## Registered pipelines")
    
    pipeline_summary_df = pd.DataFrame(pipeline_meta_data)[['name','uid','creation_timestamp']]. \
        rename(columns={'name':'Name','uid':'ID','creation_timestamp':'Created'}). \
        sort_values(by='Created',ignore_index=True)
        
    st.dataframe(pipeline_summary_df,hide_index=True)
        
def get_pipeline_names(pipeline_meta_data) -> List[str]:
    """Generates a list of names of all registered pipelines.

    Args:
        pipeline_meta_data (List[Dict]): The pipeline metadata generated by `get_pipeline_meta_data`.

    Returns:
        List[str]: The names of all available registered pipelines.
    """
    
    return [resource_meta['name'] for resource_meta in pipeline_meta_data]
        
def get_formatted_pipeline_data(registered_pipelines, pipeline_meta_data, pipeline_names):
    """Generates structured pipeline data for easier frontend useage.

    Args:
        registered_pipelines (List[IoArgoprojWorkflowV1alpha1WorkflowTemplate]): The ArgoWorkflowTemplate CRs 
            obtained by `get_workflow_templates`
        pipeline_meta_data (_type_): _description_
        pipeline_names (List[str]): The names of pipelines generated by `get_pipeline_names`.

    Returns:
        _type_: _description_
    """
    
    formatted_pipeline_data = {
        'object':{},
        'metadata': {},
        'inputs': {},
        'dag': {},
        'templates': {},
    }

    for i, (resource_name, registered_pipeline) in enumerate(zip(pipeline_names,registered_pipelines)):
        try:
            pipeline_dict = Pipeline.from_argo_workflow_cr(registered_pipeline).model_dump()
            formatted_pipeline_data['object'][resource_name] = Pipeline.from_argo_workflow_cr(registered_pipeline)
            formatted_pipeline_data['metadata'][resource_name] = pipeline_dict['metadata']
            formatted_pipeline_data['inputs'][resource_name] = pipeline_dict['inputs']
            formatted_pipeline_data['dag'][resource_name] = pipeline_dict['dag']
            formatted_pipeline_data['templates'][resource_name] = pipeline_dict['templates']
        except Exception as e:
            print(e)
            st.write(f"Oops! Could not collect data for Pipeline {resource_name}: {e} Please make sure the workflow template was created with the bettmensch.ai SDK and was submitted successfully.")

    return formatted_pipeline_data

def display_pipeline_dropdown(pipeline_names):
    """Display the pipeline selection dropdown.

    Args:
        pipeline_names (List[str]): The names of pipelines generated by `get_pipeline_names`.

    Returns:
        _type_: _description_
    """
    
    # display pipeline selection dropdown
    selected_pipeline = st.selectbox(f'Select a Pipeline:', options=pipeline_names, index=0)
        
    return selected_pipeline#, display_pipeline_ios

def display_pipeline_dag(formatted_pipeline_data, selected_pipeline, display_pipeline_ios):
    """_summary_

    Args:
        formatted_pipeline_data (_type_): The formatted pipeline data generated by `get_formatted_pipeline_data`.
        selected_pipeline (str): The name of the user selected pipeline.
        display_pipeline_ios (bool): The toggle value of the user selected pipeline dag I/O detail level.

    Returns:
        _type_: _description_
    """

    dag_visualization_schema = formatted_pipeline_data['object'][selected_pipeline].create_dag_visualization_schema(display_pipeline_ios)
    
    dag_visualization_element = streamlit_flow(
        nodes = [StreamlitFlowNode(**node.model_dump()) for node in dag_visualization_schema.nodes],
        edges = [StreamlitFlowEdge(**connection.model_dump()) for connection in dag_visualization_schema.connections],
        get_node_on_click=True,
        get_edge_on_click=True,
        fit_view=True,
        direction="down",
        style={"backgroundColor": get_colors('custom').secondaryBackgroundColor},
        layout_vertical_spacing=25,
        layout_horizontal_spacing=100,
    )
        
    return dag_visualization_schema, dag_visualization_element

def display_pipeline_dag_selection(formatted_pipeline_data, selected_pipeline, dag_visualization_element):
    """Generates a tabbed, in depth view of the user selected DAG element.

    Args:
        dag_visualization_element (_type_): The user selected element from the DAG visuals, as returned by `display_pipeline_dag`
    """
    
    try:
        element_type = dag_visualization_element['elementType']
        element_id = dag_visualization_element['id']
        element_is_task_node = (element_type == 'node') and (len(element_id.split('_')) == 1)
        
        if element_is_task_node:
            st.markdown(f"### Selected task: {element_id}")
            task_inputs_tab, task_outputs_tab, task_script_tab = st.tabs(["Inputs","Outputs","Script"])
            pipeline = formatted_pipeline_data['object'][selected_pipeline]
            task = pipeline.get_dag_task(element_id).model_dump()
                
            with task_inputs_tab:
                task_inputs = task['inputs']['parameters'] + task['inputs']['artifacts']
                task_inputs_df = pd.DataFrame(task_inputs)
                task_inputs_formatted_df = pd.concat(
                    [
                        task_inputs_df.drop(['source'],axis=1),
                        task_inputs_df['source'].apply(pd.Series)
                    ],
                    axis=1). \
                    rename(columns={
                            'name':'Name',
                            'value': 'Default',
                            'value_from': 'From',
                            'node':'Upstream Task',
                            'output_name':'Upstream Output',
                            'output_type':'Upstream Type'
                        },inplace=False
                    )
                st.dataframe(task_inputs_formatted_df,hide_index=True)
                
            with task_outputs_tab:
                task_outputs = task['outputs']['parameters'] + task['outputs']['artifacts']
                task_outputs_formatted_df = pd.DataFrame(task_outputs). \
                    rename(columns={
                            'name':'Name',
                            'value': 'Default',
                            'value_from': 'From',
                        },inplace=False
                    )
                st.dataframe(task_outputs_formatted_df,hide_index=True)
                
            with task_script_tab:
                st.json(pipeline.get_template(task['template']).model_dump()['script'])
        else:
            st.markdown(f"### Selected task: None")
            st.write("Select a task by clicking on the corresponding node.")
        
    except TypeError as e:
        st.markdown(f"### Selected task: None")
        st.write("Select a task by clicking on the corresponding node.")
                    
def display_selected_pipeline(formatted_pipeline_data,selected_pipeline):
    """Utility to display DAG flow chart and all relevant specs in tabbed layout for a user selected pipeline.

    Args:
        formatted_pipeline_data (_type_): _description_
        selected_pipeline (_type_): _description_
    """
    
    st.markdown(f"## Selected pipeline: {selected_pipeline}")
    
    tab_dag, tab_metadata, tab_inputs, tab_templates = st.tabs(["DAG", "Meta data", "Inputs", "Templates",])
    
    with tab_dag:
        st.markdown("### Diagram")
        display_pipeline_ios = st.toggle(f'Display pipeline & task I/O') 
        dag_visualization_schema, dag_visualization_element = display_pipeline_dag(
            formatted_pipeline_data,
            selected_pipeline,
            display_pipeline_ios
        )
        
        display_pipeline_dag_selection(formatted_pipeline_data, selected_pipeline, dag_visualization_element)
        
        st.markdown("### Spec")
        st.json(formatted_pipeline_data['dag'][selected_pipeline],expanded=True)
        
    with tab_metadata:
        st.markdown("### Spec")
        st.json(formatted_pipeline_data['metadata'][selected_pipeline],expanded=True)
        
    with tab_inputs:
        pipeline_inputs = formatted_pipeline_data['inputs'][selected_pipeline]
        pipeline_inputs_formatted_df = pd.DataFrame(pipeline_inputs). \
            rename(columns={
                    'name':'Name',
                    'value': 'Default',
                }
            )
        st.dataframe(pipeline_inputs_formatted_df,hide_index=True)
        
    tab_templates.json(formatted_pipeline_data['templates'][selected_pipeline],expanded=True)
        
    return dag_visualization_schema, dag_visualization_element
    

def main():
    """Utility function to render the pipeline resources.
    """
    
    st.markdown(
        """
        # :twisted_rightwards_arrows: Pipelines
        
        A `Pipeline` is the *definition* of the workflow, i.e. its DAG declaring the logic of each node and dependencies on other nodes.
        """
    )
    
    workflow_templates = get_workflow_templates(configuration)
    
    meta_data = get_pipeline_meta_data(workflow_templates)
    display_pipeline_summary_table(meta_data)
    
    names = get_pipeline_names(meta_data)
    
    formatted_pipeline_data = get_formatted_pipeline_data(workflow_templates, meta_data, names)
    
    selected_pipeline = display_pipeline_dropdown(names)
    
    dag_visualization_schema, dag_visualization_element = display_selected_pipeline(formatted_pipeline_data,selected_pipeline)

    
    with st.sidebar:
        add_logo(sidebar=True)

main()