import streamlit as st
from utils import MlflowDisplaySettings, add_logo

st.set_page_config(page_title="Models", page_icon=":books:", layout="wide")

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

mlflow_display_settings = MlflowDisplaySettings()
starting_url = f"""{
    mlflow_display_settings.host
    }/{
        mlflow_display_settings.starting_endpoint
    }"""
st.components.v1.iframe(
    starting_url,
    height=mlflow_display_settings.height,
    width=mlflow_display_settings.width,
    scrolling=mlflow_display_settings.scrolling,
)
