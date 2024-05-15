from pydantic import BaseModel
from typing import Tuple, List, Dict, Literal
from utils import get_colors

class DagConnection(BaseModel):
    id: str # connection id
    source: str # origin node id
    target: str # target node id
    animated: bool = False
    edge_type: str
    focusable: bool = False
    style: Dict = {}
    
class DagNode(BaseModel):
    id: str
    pos: Tuple[float,float]
    data: Dict= {"label": None}
    style: Dict = {
        'backgroundColor': 'lightblue', # roughly the same color as the light blue logo
        'fontWeight': 550,
        'color':'white'
    }
    node_type: str = "default"
    source_position: str = "bottom"
    target_position: str = "top"
    #height: int = 5

class DagVisualizationSchema(BaseModel):
    connections: List[DagConnection]
    nodes: List[DagNode]
    
class DagLayoutSetting(BaseModel):
    get_node_on_click: bool = True
    get_edge_on_click: bool = True
    fit_view: bool = True
    direction: Literal['down','up','right','left'] = "down"
    style: Dict = {"backgroundColor": get_colors('custom').secondaryBackgroundColor}
    layout_vertical_spacing: int = 25
    layout_horizontal_spacing: int = 100
    height: int = 600