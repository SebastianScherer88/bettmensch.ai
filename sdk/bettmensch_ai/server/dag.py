from typing import Dict, List, Literal, Tuple, Union

from pydantic import BaseModel


class DagConnection(BaseModel):
    id: str  # connection id
    source: str  # origin node id
    target: str  # target node id
    animated: bool = False
    edge_type: str
    focusable: bool = False
    label: str = ""
    label_show_bg: bool = True
    label_style: Dict = {}
    style: Dict = {}


class DagNode(BaseModel):
    id: str
    pos: Tuple[float, float] = (0, 0)
    data: Dict = {"label": None}
    style: Dict = {
        "backgroundColor": "lightblue",  # roughly the same color as the light
        # blue logo
        "fontWeight": 550,
        "color": "white",
    }
    node_type: str = "default"
    source_position: str = "bottom"
    target_position: str = "top"
    width: int = 50
    height: int = 20
    # height: int = 5


class DagPipelineIONode(DagNode):
    style: Dict = {
        "backgroundColor": "darkgrey",
        "fontWeight": 550,  # font thickness
        "color": "white",  # font color
    }


class DagTaskNode(DagNode):
    style: Dict = {
        "backgroundColor": "darkblue",
        "fontWeight": 550,  # font thickness
        "color": "white",  # font color
    }


class DagTaskIONode(DagNode):
    style: Dict = {
        "backgroundColor": "blue",
        "fontWeight": 550,  # font thickness
        "color": "white",  # font color
    }


class DagVisualizationItems(BaseModel):
    connections: List[DagConnection]
    nodes: List[Union[DagPipelineIONode, DagTaskNode, DagTaskIONode]]


class DagVisualizationSettings(BaseModel):
    get_node_on_click: bool = True
    get_edge_on_click: bool = True
    # layout: Any = TreeLayout(direction="down")
    fit_view: bool = True
    direction: Literal["down", "up", "right", "left"] = "down"
    style: Dict = {}
    layout_vertical_spacing: int = 150
    # layout_horizontal_spacing: int = 100
    height: int = 450
