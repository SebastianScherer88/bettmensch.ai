import streamlit as st

st.set_page_config(
    page_title="Bettmensch AI", page_icon=":hotel:", layout="wide"
)
from st_pages import Page, show_pages  # noqa: E402
from utils import add_logo  # noqa: E402

show_pages(
    [
        Page("src/0_home.py", "Home", ":hotel:"),
        Page(
            path="src/1_pipelines.py",
            name="Pipelines",
            icon=":twisted_rightwards_arrows:",
        ),
        Page(path="src/2_flows.py", name="Flows", icon=":arrow_forward:"),
        Page(path="src/3_models.py", name="Models", icon=":books:"),
        Page(path="src/4_servers.py", name="Servers", icon=":rocket:"),
        Page(path="src/5_argo.py", name="Argo", icon=":rocket:"),
    ]
)

st.markdown(
    """
    # :hotel: Welcome to Bettmensch.AI

    :factory: Bettmensch.AI is a Kubernetes native open source ML Ops framework
     that allows for tight CI and CD integrations.

    :eyes: This dashboard is *exclusively for querying* purposes.

    :open_hands: To actively manage `Pipeline`s, `Flow`s, `Run`s, `Model`s and
    `Server`s, please see the respective documentation of `bettmensch.ai` SDK.

    :computer: This dashboard contains 3 main sections:

    - :twisted_rightwards_arrows: `Pipelines & Flows`
      - :twisted_rightwards_arrows: [Pipelines page](./Pipelines)
      - :arrow_forward: [Flows page](./Flows)
    - :books: `Models`
      - :books: [Models page](./Models)
    - :rocket: `Servers`
      - :rocket: [Servers page](./Servers)
    """
)

with st.sidebar:
    add_logo(sidebar=True)
