import streamlit as st
from streamlit_option_menu import option_menu
from utils import add_logo, configuration, get_workflow_templates, get_workflows
from typing import Callable, Dict
import pandas as pd
import dagviz
from dagviz.style.metro import svg_renderer, StyleConfig
import networkx as nx
from IPython.display import HTML
from pipeline import NodeArtifactInput, NodeParameterInput
import sys
sys.path.append('C:\\Users\\bettmensch\\GitReps\\bettmensch.ai\\bettmensch_ai_dashboard\\src')
from pipeline import Pipeline
from flow import Flow
from dag_builder import Block, st_dag_builder, load_schemas, save_schema

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

def display_pipelines_or_flows(query_resources: Callable, resource_type: str, include_outputs: bool):
    """Utility function to render the pipeline and flow resources.

    Args:
        query_resources (Callable): _description_
        resource_type (str): _description_
        include_outputs (bool): _description_
    """
    
    st.markdown(INTRO_MAP[resource_type])
    
    resources = query_resources(configuration)
    
    resource_meta_data = [resource.metadata.to_dict() for resource in resources]
        
    resource_meta_data_df = pd.DataFrame(resource_meta_data)
    resource_meta_data_df
    resource_uids = [resource_meta['uid'] for resource_meta in resource_meta_data]

    visualize = {
        'resource':{},
        'metadata': {},
        'inputs': {},
        'dag': {},
        'templates': {},
    }

    for i, (resource_id, resource) in enumerate(zip(resource_uids,resources)):
        try:
            resource_dict = RESOURCE_MAP[resource_type].from_argo_workflow_cr(resource).model_dump()
            visualize['resource'][resource_id] = RESOURCE_MAP[resource_type].from_argo_workflow_cr(resource)
            visualize['metadata'][resource_id] = resource_meta_data[i]
            visualize['inputs'][resource_id] = resource_dict['inputs']
            visualize['dag'][resource_id] = resource_dict['dag']
            visualize['templates'][resource_id] = resource_dict['templates']
        except Exception as e:
            print(e)
            st.write(f"Oops! Could not collect data for {resource_type} {resource_id}: {e} Please make sure the workflow (template) was created with the bettmensch.ai SDK and was submitted successfully.")
    
    selected_pipeline = st.selectbox(f'Select a {resource_type[:-1]}:', options=resource_meta_data_df['uid'].tolist(), index=0)

    schemas = load_schemas()
    with open('loaded_schemas.yaml','w') as file:
        import yaml
        yaml.dump(schemas['schemas'],file)

    selected_pipeline_dag_schema, node_classes = visualize['resource'][selected_pipeline].create_dag_visualization_assets()
    save_schema(selected_pipeline, selected_pipeline_dag_schema)
            
    from_client = st_dag_builder(base_blocks=node_classes,load_schema=selected_pipeline)

    return

with st.sidebar:
    add_logo(sidebar=True)
    selected = option_menu("", TOGGLE_OPTIONS, icons=['diagram-3', 'diagram-3-fill'], menu_icon="twisted_rightwards_arrows", default_index=0,key='display_toggle')

if st.session_state["display_toggle"] in (None,PIPELINE_OPTION):
    display_pipelines_or_flows(get_workflow_templates,resource_type=PIPELINE_OPTION,include_outputs=False)
elif st.session_state["display_toggle"] in (None,FLOW_OPTION):
    display_pipelines_or_flows(get_workflows,resource_type=FLOW_OPTION,include_outputs=True)
