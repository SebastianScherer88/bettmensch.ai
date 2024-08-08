import streamlit as st
from utils import add_logo

st.set_page_config(
    page_title="Servers",
    page_icon=":rocket:",
)

with st.sidebar:
    add_logo(sidebar=True)

st.markdown(
    """
    # :rocket: Servers

    This section displays definition and state meta data of registered model
    `Server`s.
    """
)
