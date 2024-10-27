import streamlit as st
from utils import ArgoWorkflowsDisplaySettings, add_logo

st.set_page_config(page_title="Argo", page_icon=":books:", layout="wide")

with st.sidebar:
    add_logo(sidebar=True)

st.markdown(
    """
    # :books: Models

    This section displays all registered `Model`s and references to the
     relevant applications (where applicable). For more details on the
     application state, see [the Server section](../Servers)
    """
)

argo_display_settings = ArgoWorkflowsDisplaySettings()
starting_url = (
    f"{argo_display_settings.host}/{argo_display_settings.starting_endpoint}"
)
st.components.v1.iframe(
    starting_url,
    height=argo_display_settings.height,
    width=argo_display_settings.width,
    scrolling=argo_display_settings.scrolling,
)
