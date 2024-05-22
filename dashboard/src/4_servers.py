import streamlit as st
from streamlit_option_menu import option_menu
from utils import add_logo

st.set_page_config(
    page_title="Servers",
    page_icon=":rocket:",
)

with st.sidebar:
    add_logo(sidebar=True)
    selected = option_menu(
        "",
        ["Servers"],
        icons=["arrow-right-square"],
        menu_icon="rocket",
        default_index=0,
    )

st.markdown(
    """
    # :rocket: Servers
    
    This section displays definition and state meta data of registered model `Server`s.
    """
)
