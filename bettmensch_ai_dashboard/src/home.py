import streamlit as st
from st_pages import Page, show_pages
from utils import add_logo

st.set_page_config(
    page_title="Bettmensch AI",
    page_icon=":hotel:",
    layout="wide"
)

show_pages(
    [
        Page("src/home.py", "Home", ":hotel:"),
        Page("src/1_pipelines.py", "Pipelines & Flows", ":twisted_rightwards_arrows:"),
        Page("src/2_models.py", "Models", ":books:"),
        Page("src/3_servers.py", "Servers", ":rocket:"),
    ]
)

st.html(
    f"""
    <script>
        var elems = window.parent.document.querySelectorAll('div[class*="stSidebarNav"] p');
        var elem = Array.from(elems).find(x => x.innerText == 'Home');
        elem.style.fontSize = '30px'; // the fontsize you want to set it to
    </script>
    """
)
        
st.markdown(
    """
    # :hotel: Welcome to Bettmensch.AI!
    """
)

add_logo(sidebar=False)
 
st.markdown(
    """
    ## Overview
    
    Bettmensch.AI is a Kubernetes native open source ML Ops framework that allows for tight CI and CD integrations.
    
    This dashboard is purely for querying purposes. To actively manage `Flow`s, `Run`s, `Model`s and `Server`s, please see the respective documentation of `bettmensch.ai` SDK.
    
    This dashboard contains 3 main sections:
    
    1. :twisted_rightwards_arrows: [Pipelines](./Pipelines)
    2. :books: [Models](./Models)
    3. :rocket: [Servers](./Servers)
    """
)

with st.sidebar:
    add_logo(sidebar=True)
