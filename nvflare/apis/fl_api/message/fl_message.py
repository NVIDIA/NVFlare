from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field

class FLMessage(BaseModel):
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
    model_state: Any = Field(..., description="Model parameters or serialized object.")
    optimizer_state: Optional[Any] = Field(None, description="Optimizer state if applicable.")
    context: Dict[str, Any] = Field(default_factory=dict, description="Training or algorithm context.")
    metrics: Optional[Dict[str, float]] = Field(default=None, description="Evaluation metrics.")
    meta: Dict[str, Any] = Field(default_factory=dict, description="Bookkeeping info (sender, timestamp, etc.).")

class PeerMessage(BaseModel):
    type: str = Field(..., description="Type of the message, e.g. 'embedding', 'gradient', 'control', etc.")
    payload: Any = Field(..., description="Main content: embedding, gradient, model part, etc.")
    sender: str = Field(..., description="Peer ID of the sender")
    receiver: Optional[Union[str, List[str]]] = Field(
        None, description="Peer ID(s) of the receiver(s); can be a string, a list of strings, or None for broadcast"
    )
    meta: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata, e.g. round, timestamp, etc.")
