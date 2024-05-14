from pydantic import BaseModel
from typing import Tuple, List, Dict

class DagConnection(BaseModel):
    id: str # connection id
    source: str # origin node id
    target: str # target node id
    animated: bool = False
    edge_type: str
    style: Dict = {}
    
class DagNode(BaseModel):
    id: str
    pos: Tuple[float,float]
    data: Dict= {"label": None}
    style: Dict = {
        'backgroundColor': 'lightblue', # roughly the same color as the light blue logo
        'fontWeight': 750,
        'color':'white'
    }
    node_type: str = "default"
    source_position: str = "bottom"
    target_position: str = "top"

class DagVisualizationSchema(BaseModel):
    connections: List[DagConnection]
    nodes: List[DagNode]