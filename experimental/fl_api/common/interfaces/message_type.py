from typing import Any, Dict, Optional, Union
from dataclasses import dataclass, field

from pydantic import BaseModel, PositiveInt


class FedModel(BaseModel):
    params_type: Optional[str] = "Full"
    params: Any = dict()
    optimizer_params: Any = dict()
    metrics: Optional[Dict] = dict()
    start_round: Optional[PositiveInt] = 1
    current_round: Optional[PositiveInt] = 1
    total_rounds: Optional[PositiveInt] = 1
    context: Optional[Dict] = dict()
    meta: Optional[Dict] = dict()

@dataclass
class MessageEnvelope:
    payload: Any
    type: Optional[str] = None
    sender: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)


# Type aliases for messaging
MessageType = Union[FedModel, MessageEnvelope]
