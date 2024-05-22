import streamlit as st
from streamlit_option_menu import option_menu
from utils import add_logo

st.set_page_config(
    page_title="Models",
    page_icon=":books:",
)

with st.sidebar:
    add_logo(sidebar=True)
    selected = option_menu(
        "", ["Models"], icons=["arrow-right-square"], menu_icon="books", default_index=0
    )

st.markdown(
    """
    # :books: Models
    
    This section displays all registered `Model`s and references to the relevant applications (where applicable). For more details on the application state, see [the Server section](../Servers)
    """
)
