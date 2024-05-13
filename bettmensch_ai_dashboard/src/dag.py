from pydantic import BaseModel, Field
from typing import Tuple, List, Optional, Any, Union, Dict

class DagConnection(BaseModel):
    from_field: str = Field(..., alias='from',serialization_alias="from") # origin node id
    id: str # connection id
    to: str # target node id
    
class DagNodePosition(BaseModel):
    x: int
    y: int
    
class DagNode(BaseModel):
    customClasses: str = ''
    id: str
    interfaces: List[List[Union[str,Dict]]] # an interface is a list/tuple (interface_name, interface_id, interface_value)
    name: str
    state: Dict = {}
    twoColumn: bool = True
    type: str
    width: int = 150
    options: List[List[Any]] = []
    position: DagNodePosition

class DagPanning(BaseModel):
    x: int = 100
    y: int = 250

class DagSchema(BaseModel):
    connections: List[DagConnection]
    nodes: List[DagNode]
    panning: DagPanning
    scaling: float