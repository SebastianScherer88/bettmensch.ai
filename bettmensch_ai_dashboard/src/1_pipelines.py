import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_flow import streamlit_flow
from streamlit_flow.interfaces import StreamlitFlowNode, StreamlitFlowEdge
from pipeline import Pipeline
from flow import Flow
from utils import add_logo, get_colors, configuration, get_workflow_templates, get_workflows
from typing import Callable, Dict
import pandas as pd
import sys
sys.path.append('C:\\Users\\bettmensch\\GitReps\\bettmensch.ai\\bettmensch_ai_dashboard\\src')

st.set_page_config(
    page_title="Pipelines",
    page_icon=":twisted_rightwards_arrows:",
)

PIPELINE_OPTION = 'Pipelines'
FLOW_OPTION = 'Flows'
TOGGLE_OPTIONS = [PIPELINE_OPTION, FLOW_OPTION]
RESOURCE_MAP = {
    PIPELINE_OPTION: Pipeline,
    FLOW_OPTION: Flow
}
INTRO_MAP = {
    PIPELINE_OPTION: f"""
    # :twisted_rightwards_arrows: {PIPELINE_OPTION}
    
    A `Pipeline` is the *definition* of the workflow, i.e. its DAG declaring the logic of each node and dependencies on other nodes.
    
    This section displays all registered `Pipeline`s.
    """
    ,
    FLOW_OPTION: f"""
    # :twisted_rightwards_arrows: {FLOW_OPTION}
    
    A `Flow` is the *execution* of a `Pipeline` with specified input parameters and other optional runtime specs that can be used to declare instance selection taints/tolerations, device resources, etc.
    
    This section displays all submitted (pending, running or completed) `Flow`s.
    """
}

def display_pipelines_or_flows(query_resources: Callable, resource_type: str):
    """Utility function to render the pipeline and flow resources.

    Args:
        query_resources (Callable): _description_
        resource_type (str): _description_
        include_outputs (bool): _description_
    """
    
    st.markdown(INTRO_MAP[resource_type])
    
    resources = query_resources(configuration)
    
    resource_meta_data = [resource.metadata.to_dict() for resource in resources]
        
    resource_meta_data_df = pd.DataFrame(resource_meta_data)[['name','uid','creation_timestamp']]. \
        rename(columns={'name':'Name','uid':'ID','creation_timestamp':'Created'}). \
        sort_values(by='Created',ignore_index=True)
    resource_meta_data_df
    resource_names = [resource_meta['name'] for resource_meta in resource_meta_data]

    visualize = {
        'resource':{},
        'metadata': {},
        'inputs': {},
        'dag': {},
        'templates': {},
    }

    for i, (resource_name, resource) in enumerate(zip(resource_names,resources)):
        try:
            resource_dict = RESOURCE_MAP[resource_type].from_argo_workflow_cr(resource).model_dump()
            visualize['resource'][resource_name] = RESOURCE_MAP[resource_type].from_argo_workflow_cr(resource)
            visualize['metadata'][resource_name] = resource_meta_data[i]
            visualize['inputs'][resource_name] = resource_dict['inputs']
            visualize['dag'][resource_name] = resource_dict['dag']
            visualize['templates'][resource_name] = resource_dict['templates']
        except Exception as e:
            print(e)
            st.write(f"Oops! Could not collect data for {resource_type} {resource_name}: {e} Please make sure the workflow (template) was created with the bettmensch.ai SDK and was submitted successfully.")
    
    col1, col2 = st.columns([2,1])

    with col1:
        selected_pipeline = st.selectbox(f'Select a {resource_type[:-1]}:', options=resource_meta_data_df['Name'].tolist(), index=0)
    with col2:
        st.text('')
        st.text('')
        display_pipeline_ios = st.toggle(f'Display pipeline & task I/O')

    dag_visualization_schema = visualize['resource'][selected_pipeline].create_dag_visualization_schema(display_pipeline_ios)
            
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
    
    return dag_visualization_element

with st.sidebar:
    add_logo(sidebar=True)
    selected = option_menu("", TOGGLE_OPTIONS, icons=['diagram-3', 'diagram-3-fill'], menu_icon="twisted_rightwards_arrows", default_index=0,key='display_toggle')

if st.session_state["display_toggle"] in (None,PIPELINE_OPTION):
    dag_visualization_element = display_pipelines_or_flows(get_workflow_templates,resource_type=PIPELINE_OPTION)
elif st.session_state["display_toggle"] in (None,FLOW_OPTION):
    dag_visualization_element = display_pipelines_or_flows(get_workflows,resource_type=FLOW_OPTION)
    
if dag_visualization_element:
    st.write(f"Currently selected {dag_visualization_element['elementType']} {dag_visualization_element['id']}")
    
with st.expander("See explanation",expanded=dag_visualization_element is not None):
    try:
        element_type = dag_visualization_element['elementType']
        element_id = dag_visualization_element['id']
    except TypeError:
        element_type = element_id = None
    st.write(f"Currently selected {element_type} {element_id}")