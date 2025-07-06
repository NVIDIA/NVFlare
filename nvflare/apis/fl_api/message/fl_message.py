from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field

@dataclass
class FLMessage:
    """
    example usage

    msg = FLMessage(
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict() if optimizer else None,
            context={"algorithm": "FedAvg", "lr": 0.01},
            metrics={"accuracy": 0.91},
            meta={"sender": "client1", "round": 1}
        )
    """
    model_state: Any
    optimizer_state: Optional[Any] = None
    context: Dict[str, Any] = field(default_factory=dict)
    metrics: Optional[Dict[str, float]] = None
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MessageEnvelope:
    payload: Any
    type: Optional[str] = None
    sender: Optional[str] = None
    receiver: Optional[Union[str, List[str]]] = None
    meta: Dict[str, Any] = field(default_factory=dict)

# Type aliases for messaging
MessageType = Union[FLMessage, MessageEnvelope]